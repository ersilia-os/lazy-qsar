import os
import csv
import shutil
import tempfile

import numpy as np

from ..agnostic import LazyClassifier
from ..qsar import DESCRIPTOR_TYPES, DESCRIPTORS_MODE, get_descriptor_type
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


def fit(data_dir: str, model_dir: str, models_txt: str = None, mode: str = "default"):

    data_dir = os.path.abspath(data_dir)
    model_dir = os.path.abspath(model_dir)

    logger.info(f"Fitting models in mode '{mode}' | data: {data_dir} | output: {model_dir}")

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

    all_smiles = read_all_smiles(data_dir)
    all_smiles2idx = {s: i for i, s in enumerate(all_smiles)}
    logger.info(f"Found {len(all_smiles)} unique SMILES across all tasks")

    for descriptor_type in descriptor_types:
        if descriptor_type not in DESCRIPTOR_TYPES:
            raise Exception(f"Descriptor type {descriptor_type} is not supported.")
        logger.info(f"Computing descriptors: {descriptor_type}")
        descriptor = get_descriptor_type(descriptor_type)()
        X = descriptor.transform(all_smiles)
        for task_name in task_names:
            model_subdir = os.path.join(model_dir, task_name, descriptor_type)
            if not os.path.exists(model_subdir):
                os.makedirs(model_subdir)
            descriptor.save(model_subdir)
            shutil.copy(
                os.path.join(model_subdir, "featurizer.json"),
                os.path.join(model_dir, f"{descriptor_type}.json"),
            )
        np.save(os.path.join(model_dir, f"{descriptor_type}.npy"), X)

    data = {}
    for task_name in task_names:
        smiles_list, y = get_task_data(data_dir, task_name)
        data[task_name] = (smiles_list, y)

    for descriptor_type in descriptor_types:
        X = np.load(os.path.join(model_dir, f"{descriptor_type}.npy"))
        for task_name in task_names:
            logger.info(f"Fitting task '{task_name}' with descriptor '{descriptor_type}'")
            idxs = [all_smiles2idx[s] for s in data[task_name][0]]
            y = data[task_name][1]
            X_task = X[idxs]
            model = LazyClassifier()
            model.fit(X=X_task, y=y)
            model_subdir = os.path.join(model_dir, task_name, descriptor_type)
            model.save(model_subdir)
            shutil.copy(
                os.path.join(model_dir, f"{descriptor_type}.json"),
                os.path.join(model_subdir, "featurizer.json"),
            )
        os.remove(os.path.join(model_dir, f"{descriptor_type}.json"))
        os.remove(os.path.join(model_dir, f"{descriptor_type}.npy"))
    logger.success(f"All models saved to {model_dir}")
