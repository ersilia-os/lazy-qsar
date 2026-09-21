"""Score one or more saved LazyQSAR models over a list of SMILES.

The entry point the Ersilia Model Hub calls. Both forms of ``model_dir`` — a parent
directory of task subdirectories, or a caller-supplied ``{column_name: directory}``
mapping — resolve to the same list of ``(column, directory)`` pairs and run through the
shared inference runner, so the CLI and the Python API compute the same thing.

The descriptor streaming, chunking and reuse all live in :mod:`lazyqsar.ensemble.runner`;
what remains here is argument handling, column ordering and CSV I/O.
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
from ..registry import get_descriptor_type
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

    logger.info(
        f"Running prediction | {len(sources)} model(s) | "
        f"output: {output_csv} | predict_type: {predict_type}"
    )

    results = predict_tasks(
        sources,
        smiles_list,
        outputs=(predict_type,),
        chunk_size=get_chunk_size(),
        show_progress=True,
    )

    # Molecules RDKit cannot parse are blanked rather than dropped or imputed. Their
    # descriptor rows are all-NaN, which the preprocessor's imputer would otherwise fill
    # with the training median -- returning an ordinary-looking score for a string that is
    # not a molecule. Row count and order are untouched, so the output still aligns with
    # the input; the gap is simply visible.
    bad = invalid_smiles_indices(smiles_list)
    if bad:
        logger.warning(
            f"{len(bad)} SMILES could not be parsed; their predictions are NaN "
            f"(positions: {bad[:10]}{' ...' if len(bad) > 10 else ''})"
        )
        for r in results:
            mask_rows(r.values, bad)

    header = [s.column_name for s in sources]
    R = np.column_stack([_as_column(r.values[predict_type]) for r in results])

    if output_csv is not None:
        output_csv = os.path.abspath(output_csv)
        pd.DataFrame(R, columns=header).to_csv(output_csv, index=False)
        logger.success(f"Predictions saved to {output_csv}")
    return R, header
