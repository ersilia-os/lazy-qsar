"""Fit one LazyQSAR model per task from a directory of labelled CSVs.

The CLI training entry point. Each task becomes a :class:`~lazyqsar.qsar.LazyClassifierQSAR`
— the same object the Python API fits — so a checkpoint means the same thing whichever
way it was produced.

Descriptors are computed once over the union of every task's compounds and sliced per
task. That is the reason this module exists rather than simply looping over
``LazyClassifierQSAR.fit``: for a fifty-task run in slow mode it is the difference
between one featurization pass and fifty, and featurization is roughly 93% of the work.
"""

import csv
import os
import shutil
import tempfile

import numpy as np

from ..descriptors._validate import validate_smiles
from ..ensemble.runner import get_chunk_size, persist_descriptors
from ..qsar import LazyClassifierQSAR
from ..registry import DESCRIPTOR_TYPES, DESCRIPTORS_MODE, get_descriptor_type
from ..utils.logging import logger


def prepare_files(models: list = None, path: str = None):
    if path is None:
        path = tempfile.mkdtemp()
    models_txt = os.path.join(path, "_models.txt")
    if models is not None:
        with open(models_txt, "w") as f:
            for m in models:
                f.write(m + "\n")
    data = {
        "models_txt": os.path.abspath(models_txt) if models is not None else None,
    }
    return data


def read_all_smiles(data_dir):
    smiles_list = []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(data_dir, fn), "r") as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                smiles_list += [r[0]]
    smiles_list = list(set(smiles_list))
    return smiles_list


def get_task_names(data_dir):
    task_names = []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        task_names += [os.path.splitext(fn)[0]]
    task_names = sorted(task_names)
    return task_names


def get_task_data(data_dir, task_name):
    smiles_list = []
    y = []
    with open(os.path.join(data_dir, task_name + ".csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            smiles_list += [r[0]]
            y += [int(r[1])]
    return smiles_list, np.array(y, dtype=int)


def fit(data_dir: str, model_dir: str, models_txt: str = None, mode: str = "slow"):
    """Fit and save one model per CSV in *data_dir*.

    Parameters
    ----------
    data_dir : str
        One CSV per task: SMILES in the first column, a binary label in the second, with
        a header row. The file stem becomes the task name.
    model_dir : str
        Output directory, one subdirectory per task. Must not already exist.
    models_txt : str, optional
        One task name per line, to fit a subset.
    mode : str
        ``"fast"`` (Morgan only) or ``"slow"`` (all five descriptors, pruned by the
        portfolio).
    """
    data_dir = os.path.abspath(data_dir)
    model_dir = os.path.abspath(model_dir)

    if mode not in DESCRIPTORS_MODE:
        raise ValueError(
            f"Unknown mode {mode!r}. Choose from: {sorted(DESCRIPTORS_MODE)}"
        )

    logger.info(
        f"Fitting models in mode '{mode}' | data: {data_dir} | output: {model_dir}"
    )

    if os.path.exists(model_dir):
        raise FileExistsError(
            f"Model directory {model_dir} already exists. Please remove it before running this command."
        )

    task_names = get_task_names(data_dir)
    if models_txt is not None:
        with open(models_txt, "r") as f:
            models = [line.strip() for line in f]
        task_names = [t for t in models if t in task_names]
    if len(task_names) == 0:
        raise ValueError("No valid tasks found in the data directory.")
    logger.info(f"Tasks to fit: {task_names}")

    descriptor_types = DESCRIPTORS_MODE[mode]
    for descriptor_type in descriptor_types:
        if descriptor_type not in DESCRIPTOR_TYPES:
            raise Exception(f"Descriptor type {descriptor_type} is not supported.")

    all_smiles = read_all_smiles(data_dir)
    validate_smiles(all_smiles)
    row_of = {s: i for i, s in enumerate(all_smiles)}
    logger.info(f"Found {len(all_smiles)} unique SMILES across all tasks")

    data = {t: get_task_data(data_dir, t) for t in task_names}

    # Scratch lives outside model_dir: a crash must not strew .npy files among the
    # checkpoints, which is what the previous in-place staging did.
    scratch = os.environ.get("LAZYQSAR_FIT_SCRATCH") or tempfile.mkdtemp(
        prefix="lazyqsar-fit-"
    )
    os.makedirs(scratch, exist_ok=True)
    chunk_size = get_chunk_size()
    try:
        # Phase 1 -- featurize the union once per descriptor.
        for descriptor_type in descriptor_types:
            logger.info(f"Computing descriptors: {descriptor_type}")
            descriptor = get_descriptor_type(descriptor_type)()
            persist_descriptors(
                descriptor,
                all_smiles,
                os.path.join(scratch, f"{descriptor_type}.npy"),
                chunk_size,
            )
            del descriptor

        # Phase 2 -- one model per task, reading its rows out of the staged matrices.
        # The loop is per task rather than per descriptor because the portfolio and the
        # active-descriptor mask are joint decisions across all descriptors of one task.
        for task_name in task_names:
            smiles_list, y = data[task_name]
            rows = [row_of[s] for s in smiles_list]
            precomputed = {
                d: np.load(os.path.join(scratch, f"{d}.npy"), mmap_mode="r")[rows]
                for d in descriptor_types
            }
            model = LazyClassifierQSAR(mode=mode)
            model.fit(smiles_list, y, precomputed=precomputed, validate=False)
            model.save_raw(os.path.join(model_dir, task_name))
            del precomputed, model
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    logger.success(f"All models saved to {model_dir}")
