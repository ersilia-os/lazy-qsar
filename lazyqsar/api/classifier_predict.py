"""Score one or more saved LazyQSAR models over a list of SMILES.

The entry point the Ersilia Model Hub calls. Both forms of ``model_dir`` — a parent
directory of task subdirectories, or a caller-supplied ``{column_name: directory}``
mapping — resolve to the same list of ``(column, directory)`` pairs and run through the
shared inference runner, so the CLI and the Python API compute the same thing.

The descriptor streaming, chunking and reuse all live in :mod:`lazyqsar.ensemble.runner`;
what remains here is argument handling, column ordering, CSV I/O -- and the one thing the
runner deliberately does not do, which is bound the input.

The runner holds its accumulated per-task channels, and the combined results for every
task, for as long as the call lasts. That is the right trade inside one pass, but it makes
peak memory scale with the number of molecules times the number of endpoints, and
``LAZYQSAR_PREDICT_CHUNK`` does nothing about it -- it bounds the descriptor slice, not the
accumulation. So this module feeds the runner a block of molecules at a time and lets each
block's working set go before starting the next, which is what keeps a large library from
exhausting memory. See :func:`_block_size`.
"""

import csv
import os
import tempfile

import numpy as np
import pandas as pd

from ..ensemble.combine import OUTPUT_NAMES, mask_rows
from ..ensemble.runner import (
    get_chunk_size,
    new_progress,
    persist_descriptors,
    predict_tasks,
    sources_from_mapping,
    sources_from_parent,
)
from ..qsar import invalid_smiles_indices
from ..registry import DESCRIPTORS_MODE, get_descriptor_type
from ..utils.logging import logger

# Kept importable from here: Ersilia model templates and analysis scripts reach for these.
_new_progress = new_progress
_get_chunk_size = get_chunk_size
_persist_descriptors = persist_descriptors

__all__ = [
    "predict",
    "prepare_files",
    "read_smiles",
    "read_output_array",
    "get_task_names",
    "get_featurizer_names",
    "load_featurizer",
]


def prepare_files(
    smiles_list, models: list = None, path: str = None, predict_type: str = "proba"
):
    if path is None:
        path = tempfile.mkdtemp()
    input_csv = os.path.join(path, "_input.csv")
    with open(input_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for s in smiles_list:
            writer.writerow([s])
    output_csv = os.path.join(path, "_output.csv")
    if models is None:
        models_txt = None
    else:
        models_txt = os.path.join(path, "_models.txt")
        with open(models_txt, "w") as f:
            for m in models:
                f.write(m + "\n")
    data = {
        "input_csv": os.path.abspath(input_csv),
        "output_csv": os.path.abspath(output_csv),
        "models_txt": os.path.abspath(models_txt) if models_txt is not None else None,
        "predict_type": predict_type,
    }
    return data


def read_smiles(input_csv):
    smiles_list = []
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            smiles_list += [r[0]]
    return smiles_list


def read_output_array(output_csv):
    df = pd.read_csv(output_csv)
    return np.array(df)


def get_task_names(model_dir):
    task_names = []
    for dn in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir, dn)):
            task_names.append(dn)
    return sorted(task_names)


def get_featurizer_names(model_dir, tasks):
    featurizers_names = []
    for task_name in tasks:
        for dn in os.listdir(os.path.join(model_dir, task_name)):
            if os.path.isdir(os.path.join(model_dir, task_name, dn)):
                featurizers_names += [dn]
    featurizers_names = sorted(set(featurizers_names))
    return featurizers_names


def load_featurizer(model_dir, featurizer_name):
    featurizer = None
    for task_name in get_task_names(model_dir):
        model_subdir = os.path.join(model_dir, task_name, featurizer_name)
        if os.path.isdir(model_subdir):
            featurizer = get_descriptor_type(featurizer_name).load(model_subdir)
            break
    return featurizer


# One gigabyte of working set per block. Not configurable on purpose: a knob here would be
# a second thing to tune beside LAZYQSAR_PREDICT_CHUNK, with worse failure modes -- too
# large and it defeats the point, too small and it reloads featurizers for nothing.
_BLOCK_BUDGET_BYTES = 1 << 30

# The widest descriptor LazyQSAR ships (morgan and chemeleon are both 2048), the most any
# one model can use, and the most channels any request needs. All three are deliberate
# over-estimates: they make the block smaller than strictly necessary, never larger.
#
# The descriptor count is read from the registry rather than written down, because a sixth
# descriptor would otherwise make this silently under-count and the blocks too large --
# the one direction the error must not go. The width is not derivable: no descriptor
# declares its dimensionality without being constructed, and constructing one here would
# pull in RDKit and torch on a path that deliberately avoids both.
_WIDEST_DESCRIPTOR = 2048
_MAX_DESCRIPTORS = max(len(names) for names in DESCRIPTORS_MODE.values())
_MAX_CHANNELS = len(("y", "r", "s", "a"))

# Descriptors that leave an all-NaN row for a molecule RDKit cannot parse, so that the rows
# they flag are a superset of the unparseable ones. `cddd` is deliberately absent: it
# repairs NaN rows from a ChEMBL nearest neighbour, so a row it returns clean is not
# evidence the molecule parsed. When none of these ran, the scan below is skipped and the
# whole list is checked, exactly as it always was.
_NAN_FAITHFUL = frozenset({"morgan", "rdkit", "chemeleon", "clamp"})


def _block_size(n_tasks: int, chunk_size: int) -> int:
    """Molecules to push through the runner at once, derived from a memory budget.

    Per molecule a block costs roughly the descriptor matrix being featurized, the runner's
    accumulated channels, and one ``CombineResult`` per task -- whose ``weights`` and
    ``ranks`` are float64 and, for the CLI, unread. At twenty endpoints and five
    descriptors that is about 11 kB per molecule, so a million molecules in one pass needs
    several gigabytes.

    Always a whole multiple of *chunk_size*, and never smaller than it. That is what keeps
    the result bit-identical: featurization and scoring both step in ``chunk_size`` rows,
    and onnxruntime picks different kernels for different batch sizes, so a block boundary
    that produced a short chunk would move the numbers in the last few digits.
    """
    per_row = (
        4 * _WIDEST_DESCRIPTOR
        + 4 * n_tasks * _MAX_DESCRIPTORS * _MAX_CHANNELS
        + n_tasks * (16 + 16 * _MAX_DESCRIPTORS)
    )
    rows = _BLOCK_BUDGET_BYTES // per_row
    return max(chunk_size, (rows // chunk_size) * chunk_size)


def _unparseable(smiles_list, scan):
    """Positions RDKit cannot parse, re-checking only the rows a descriptor already NaN'd.

    Every descriptor in :data:`_NAN_FAITHFUL` emits an all-NaN row exactly when
    ``Chem.MolFromSmiles`` returns ``None``, and may emit one for its own reasons as well.
    So the rows they flag contain every unparseable molecule and usually a few extra, and
    confirming that handful with RDKit gives the same answer as parsing all N -- which is
    what this used to do, after all five descriptors had already parsed every molecule.

    A clean library means zero parses here instead of one per molecule.
    """
    if not _NAN_FAITHFUL.intersection(scan.get("descriptors", ())):
        return invalid_smiles_indices(smiles_list)
    candidates = sorted(scan.get("nan_rows", ()))
    if not candidates:
        return []
    subset = [smiles_list[i] for i in candidates]
    return [candidates[j] for j in invalid_smiles_indices(subset)]


def _as_column(values):
    """Reduce one output to the single column that goes in the CSV.

    ``binary`` is already per-sample labels; the rest are ``[negative, positive]`` pairs
    and the positive class is the one reported.
    """
    return values if values.ndim == 1 else values[:, 1]


def predict(
    model_dir: str | dict[str, str],
    input_csv: str = None,
    output_csv: str = None,
    models_txt: str = None,
    predict_type: str = "proba",
    smiles: list = None,
) -> tuple[np.ndarray, list[str]]:
    """Score every model in *model_dir* over the given SMILES.

    Parameters
    ----------
    model_dir : str or dict
        A directory whose subdirectories are tasks, or ``{column_name: directory}`` to
        name the output columns explicitly and draw models from unrelated paths. Two
        column names may point at the same directory; both are returned.
    input_csv : str, optional
        CSV with SMILES in the first column and a header. Ignored when *smiles* is given.
    output_csv : str, optional
        Where to write the result. Nothing is written when omitted.
    models_txt : str, optional
        One column name per line. Filters, and its order becomes the column order.
    predict_type : str
        One of ``proba``, ``rank``, ``logit``, ``lift``, ``score``, ``binary``.
    smiles : list of str, optional
        SMILES to score, instead of reading *input_csv*.

    Returns
    -------
    R : ndarray of shape (n_smiles, n_columns)
    header : list of str
        Column names, aligned with the columns of *R*.
    """
    if predict_type not in OUTPUT_NAMES:
        raise ValueError(
            f"Unknown predict_type '{predict_type}'. Choose from: {sorted(OUTPUT_NAMES)}"
        )

    if smiles is not None:
        smiles_list = smiles
        logger.info(f"Using {len(smiles_list)} SMILES from argument")
    else:
        if input_csv is None:
            raise ValueError("Provide either `smiles` or `input_csv`.")
        smiles_list = read_smiles(os.path.abspath(input_csv))
        logger.info(f"Loaded {len(smiles_list)} SMILES from {input_csv}")

    if isinstance(model_dir, dict):
        sources = sources_from_mapping(model_dir, models_txt)
    else:
        sources = sources_from_parent(os.path.abspath(model_dir), models_txt)
    if not sources:
        raise ValueError("No valid models found.")

    header = [s.column_name for s in sources]
    n_total = len(smiles_list)
    chunk_size = get_chunk_size()
    block = _block_size(len(sources), chunk_size)
    n_blocks = max(1, (n_total + block - 1) // block)

    logger.info(
        f"Running prediction | {len(sources)} model(s) | {n_total} molecule(s) | "
        f"{n_blocks} block(s) of up to {block} | "
        f"output: {output_csv} | predict_type: {predict_type}"
    )

    # Filled per block rather than concatenated at the end: holding every block's slice and
    # then joining them would mean two full copies of the output, which is exactly the kind
    # of whole-input allocation this loop exists to remove.
    R = None
    n_bad = 0
    for index, start in enumerate(range(0, n_total, block), start=1):
        end = min(start + block, n_total)
        if n_blocks > 1:
            logger.info(f"Block {index}/{n_blocks} | rows {start}-{end - 1}")
        part = smiles_list[start:end]

        scan: dict = {}
        results = predict_tasks(
            sources,
            part,
            outputs=(predict_type,),
            chunk_size=chunk_size,
            show_progress=True,
            scan=scan,
        )

        # Molecules RDKit cannot parse are blanked rather than dropped or imputed. Their
        # descriptor rows are all-NaN, which the preprocessor's imputer would otherwise
        # fill with the training median -- returning an ordinary-looking score for a
        # string that is not a molecule. Row count and order are untouched, so the output
        # still aligns with the input; the gap is simply visible.
        bad = _unparseable(part, scan)
        if bad:
            n_bad += len(bad)
            shown = [start + i for i in bad[:10]]
            logger.warning(
                f"{len(bad)} SMILES could not be parsed; their predictions are NaN "
                f"(positions: {shown}{' ...' if len(bad) > 10 else ''})"
            )
            for r in results:
                mask_rows(r.values, bad)

        block_R = np.column_stack([_as_column(r.values[predict_type]) for r in results])
        # The block's channels, weights and combined results go here, before the next
        # block allocates its own. This is the whole point of blocking.
        del results, scan

        if R is None:
            R = np.empty((n_total, len(sources)), dtype=block_R.dtype)
        elif block_R.dtype != R.dtype:
            # `binary` is integer labels until some block has a NaN row to carry. Promote
            # what is already filled rather than truncating this block into an int array,
            # so the dtype ends up where a single unblocked pass would have put it.
            R = R.astype(np.result_type(R.dtype, block_R.dtype))
        R[start:end] = block_R
        del block_R

    if R is None:
        # No molecules at all, so the loop never ran. An empty result with the right number
        # of columns, which is what the unblocked column_stack used to produce.
        R = np.empty((0, len(sources)))
    if n_bad:
        logger.warning(f"{n_bad} of {n_total} SMILES could not be parsed in total")

    if output_csv is not None:
        output_csv = os.path.abspath(output_csv)
        pd.DataFrame(R, columns=header).to_csv(output_csv, index=False)
        logger.success(f"Predictions saved to {output_csv}")
    return R, header
