"""Chunked, multi-task inference over saved LazyQSAR checkpoints.

One loop serves every prediction entry point. It exists because a CLI
``model_dir/<task>/`` directory and a ``LazyClassifierQSAR`` model directory are the same
shape — ``metadata.json`` plus one subdirectory per descriptor — so "score N tasks" and
"score one model" are the same problem with N = 1.

Two reuse properties, both load-bearing, both easy to lose:

*inter-model* — descriptors are computed once for the union of every task, then shared.
Scoring ten models over one library must featurize it once, not ten times.

*intra-model* — every requested output comes from a single pass. Asking for ``proba`` and
``rank`` must not featurize twice.

Featurization is roughly 93% of inference wall clock, so both of these dominate
everything else the runner does.

Memory shape: one descriptor chunk in RAM, one descriptor matrix on disk at a time, and
the accumulated per-task channels (``n_samples`` float32 per channel per task per
descriptor) held until the end, where the row-independent combine step runs.

Imports no RDKit: featurizer classes are resolved lazily through the registry, so this
module loads on a base install even though the descriptors it may instantiate do not.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from ..registry import DESCRIPTOR_TYPES, get_descriptor_type
from ..utils.logging import logger
from .channels import required_channels, score_chunkwise
from .combine import EnsembleSpec, combine

_DEFAULT_CHUNK = 1000


def new_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=True,
    )


def get_chunk_size() -> int:
    """Rows per featurization and inference batch, from ``LAZYQSAR_PREDICT_CHUNK``."""
    try:
        v = int(os.environ.get("LAZYQSAR_PREDICT_CHUNK", str(_DEFAULT_CHUNK)))
        return v if v > 0 else _DEFAULT_CHUNK
    except ValueError:
        return _DEFAULT_CHUNK


def persist_descriptors(
    featurizer,
    smiles_list: list,
    out_path: str,
    chunk_size: int,
    progress: Progress | None = None,
    task_id=None,
) -> None:
    """Compute descriptors in chunks and stream each chunk into a memmap-backed .npy.

    Never materialises an X matrix with more than `chunk_size` rows in RAM. The output is a
    standard .npy file so it can be read back with `np.load(..., mmap_mode='r')`.
    """
    n_total = len(smiles_list)
    if n_total == 0:
        return

    first_end = min(chunk_size, n_total)
    first_chunk = featurizer.transform(smiles_list[:first_end])
    if first_chunk.ndim != 2:
        raise ValueError(
            f"featurizer.transform must return a 2D array; got shape {first_chunk.shape}"
        )
    n_dim = int(first_chunk.shape[1])
    dtype = first_chunk.dtype

    X_mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(n_total, n_dim)
    )
    try:
        X_mm[:first_end] = first_chunk
        del first_chunk
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

        for start in range(first_end, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = featurizer.transform(smiles_list[start:end])
            X_mm[start:end] = chunk
            del chunk
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
        X_mm.flush()
    finally:
        del X_mm
    gc.collect()


@dataclass(frozen=True)
class TaskSource:
    """One output column and the checkpoint directory that produces it.

    Column name and directory are kept as a pair rather than a mapping, so two columns may
    point at the same directory without one of them being lost.
    """

    column_name: str
    task_dir: str


def sources_from_parent(model_dir: str, models_txt: str | None = None):
    """Every task subdirectory of *model_dir*, as sources named after the directories.

    Sorted by default. ``models_txt`` both filters and reorders — its order wins, which is
    how callers pin the column order of the output.
    """
    model_dir = os.path.abspath(model_dir)
    tasks = sorted(
        d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))
    )
    if models_txt is not None:
        with open(models_txt) as f:
            wanted = [line.strip() for line in f if line.strip()]
        tasks = [t for t in wanted if t in tasks]
    return [TaskSource(t, os.path.join(model_dir, t)) for t in tasks]


def sources_from_mapping(col_map: dict, models_txt: str | None = None):
    """Caller-supplied ``{column_name: directory}``, in the caller's order.

    Built as a list so that two column names mapping to one directory both survive.
    """
    sources = [TaskSource(col, os.path.abspath(path)) for col, path in col_map.items()]
    if models_txt is not None:
        with open(models_txt) as f:
            allowed = {line.strip() for line in f if line.strip()}
        sources = [s for s in sources if s.column_name in allowed]
    return sources


def _descriptor_dirs(task_dir: str):
    """Descriptor subdirectories of a checkpoint, in sorted order."""
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Model directory {task_dir!r} does not exist.")
    return sorted(
        d
        for d in os.listdir(task_dir)
        if d in DESCRIPTOR_TYPES and os.path.isdir(os.path.join(task_dir, d))
    )


def _read_metadata(task_dir: str) -> dict:
    import json

    path = os.path.join(task_dir, "metadata.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_artifact(directory: str):
    from ..artifacts.classifier import LazyClassifierArtifact

    return LazyClassifierArtifact.load(directory)


def _load_ad(directory: str):
    ad_dir = os.path.join(directory, "applicability_domain")
    if not os.path.isdir(ad_dir):
        return None
    from ..applicability import ApplicabilityDomainArtifact

    return ApplicabilityDomainArtifact.load(ad_dir)


@dataclass
class _Plan:
    """What one source needs: its active descriptors and its weighting parameters."""

    source: TaskSource
    descriptors: list
    spec: EnsembleSpec
    has_ad: bool


def _plan(source: TaskSource) -> _Plan:
    present = _descriptor_dirs(source.task_dir)
    if not present:
        raise FileNotFoundError(
            f"No descriptor directories found in {source.task_dir!r}"
        )
    spec, active = EnsembleSpec.from_metadata(_read_metadata(source.task_dir), present)
    has_ad = any(
        os.path.isdir(os.path.join(source.task_dir, d, "applicability_domain"))
        for d in active
    )
    return _Plan(source=source, descriptors=active, spec=spec, has_ad=has_ad)


def predict_tasks(
    sources,
    smiles_list,
    *,
    outputs=("proba",),
    chunk_size: int | None = None,
    scratch_dir: str | None = None,
    show_progress: bool = False,
):
    """Score every source over *smiles_list*, sharing one featurization pass.

    Parameters
    ----------
    sources : sequence of TaskSource
        Output columns and their checkpoint directories, in the order results are wanted.
    smiles_list : list of str
        Query compounds.
    outputs : sequence of str
        Which of :data:`~lazyqsar.ensemble.combine.OUTPUT_NAMES` to compute. All of them
        come from the same pass, so asking for several costs no extra featurization.
    chunk_size : int, optional
        Rows per batch. Defaults to :func:`get_chunk_size`.
    scratch_dir : str, optional
        Where descriptor matrices are staged. A temporary directory is created and removed
        when omitted; it is never the model directory, so a crash cannot leave artefacts
        beside the checkpoints.
    show_progress : bool
        Render progress bars.

    Returns
    -------
    list of CombineResult
        Aligned with *sources*, so duplicate column names stay distinguishable.
    """
    sources = list(sources)
    if not sources:
        raise ValueError("No model sources given.")
    if chunk_size is None:
        chunk_size = get_chunk_size()

    plans = [_plan(s) for s in sources]

    # Union across every task: this is what makes scoring N models cost one featurization
    # pass rather than N. Ordered by first appearance for reproducible progress output.
    descriptor_names = []
    for plan in plans:
        for name in plan.descriptors:
            if name not in descriptor_names:
                descriptor_names.append(name)

    owned_scratch = scratch_dir is None
    scratch_dir = scratch_dir or tempfile.mkdtemp(prefix="lazyqsar-predict-")
    store: dict[tuple[int, str], object] = {}
    n_total = len(smiles_list)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    progress = new_progress() if show_progress else None
    try:
        if progress is not None:
            progress.start()
        for name in descriptor_names:
            users = [(i, p) for i, p in enumerate(plans) if name in p.descriptors]
            featurizer = _load_featurizer(name, [p.source.task_dir for _, p in users])
            x_path = os.path.join(scratch_dir, f"X_{name}.npy")

            task_id = (
                progress.add_task(f"[{name}] descriptors", total=n_chunks)
                if progress is not None
                else None
            )
            logger.debug(f"Computing descriptors: {name}")
            persist_descriptors(
                featurizer, smiles_list, x_path, chunk_size, progress, task_id
            )
            del featurizer
            gc.collect()

            for i, plan in users:
                sub = os.path.join(plan.source.task_dir, name)
                artifact = _load_artifact(sub)
                ad = _load_ad(sub) if plan.has_ad else None
                store[(i, name)] = score_chunkwise(
                    artifact,
                    ad,
                    x_path,
                    chunk_size,
                    required_channels(outputs, plan.has_ad),
                    logger=logger,
                )
                del artifact, ad
                gc.collect()

            try:
                os.remove(x_path)
            except OSError:
                pass
    finally:
        if progress is not None:
            progress.stop()
        if owned_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    results = []
    for i, plan in enumerate(plans):
        channels = [store[(i, name)] for name in plan.descriptors]
        results.append(
            combine(
                _stack(channels, "y"),
                _stack(channels, "r"),
                _stack(channels, "s"),
                _stack(channels, "a"),
                spec=plan.spec,
                outputs=outputs,
            )
        )
    return results


def _stack(channels, attr):
    """Column-stack one channel across descriptors, or None if any is missing.

    A partially available channel is not meaningful — mixing a real rank for one
    descriptor with a placeholder for another would silently distort the weighting — so it
    collapses to None and the ensemble falls back, exactly as the non-streaming path does.
    """
    columns = [getattr(c, attr) for c in channels]
    if not columns or any(col is None for col in columns):
        return None
    return np.column_stack(columns)


def _load_featurizer(name, candidate_dirs):
    """Load descriptor *name* from the first checkpoint that carries its config."""
    for directory in candidate_dirs:
        sub = os.path.join(directory, name)
        if os.path.isdir(sub):
            return get_descriptor_type(name).load(sub)
    raise FileNotFoundError(f"No saved {name!r} featurizer among {candidate_dirs!r}")
