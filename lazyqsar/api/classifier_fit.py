import json
import os
import csv
import shutil
import tempfile

import numpy as np

from ..agnostic import LazyClassifier
from ..descriptors._validate import validate_smiles
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

    all_smiles = read_all_smiles(data_dir)
    validate_smiles(all_smiles)
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

    # Collect per-(task, descriptor) metadata to build the task-level metadata.json
    task_descriptor_meta = {task: {} for task in task_names}

    for descriptor_type in descriptor_types:
        X = np.load(os.path.join(model_dir, f"{descriptor_type}.npy"))
        for task_name in task_names:
            logger.info(
                f"Fitting task '{task_name}' with descriptor '{descriptor_type}'"
            )
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
            inner = model._model
            task_descriptor_meta[task_name][descriptor_type] = {
                "oof_auc": model.oof_auc_,
                "train_auc": model.train_auc_,
                "decision_cutoff_raw": inner.decision_cutoff_raw_,
                "decision_cutoff_proba": inner.decision_cutoff_proba_,
                "decision_cutoff_rank": inner.decision_cutoff_rank_,
                "portfolio": inner.portfolio,
                "num_batches": len(inner.models),
            }
        os.remove(os.path.join(model_dir, f"{descriptor_type}.json"))
        os.remove(os.path.join(model_dir, f"{descriptor_type}.npy"))

    # Write task-level metadata.json (aggregated across descriptors)
    for task_name in task_names:
        _, y = data[task_name]
        desc_meta = task_descriptor_meta[task_name]
        population_prior = float(np.mean(y == 1))

        avg_raw = float(np.mean([m["decision_cutoff_raw"] for m in desc_meta.values()]))
        avg_proba = float(
            np.mean([m["decision_cutoff_proba"] for m in desc_meta.values()])
        )
        avg_rank = float(
            np.mean([m["decision_cutoff_rank"] for m in desc_meta.values()])
        )
        _p_clip = float(np.clip(avg_proba, 1e-7, 1.0 - 1e-7))

        meta = {
            "mode": mode,
            "descriptor_types": descriptor_types,
            "n_compounds": int(len(y)),
            "n_actives": int((y == 1).sum()),
            "ratio_actives": population_prior,
            "population_prior": population_prior,
            "portfolio": desc_meta[descriptor_types[0]]["portfolio"],
            "num_batches": {d: m["num_batches"] for d, m in desc_meta.items()},
            "decision_cutoff_raw": avg_raw,
            "decision_cutoff_proba": avg_proba,
            "decision_cutoff_rank": avg_rank,
            "decision_cutoff_logit": float(np.log(_p_clip / (1.0 - _p_clip))),
            "decision_cutoff_lift": float(avg_proba / population_prior)
            if population_prior > 0
            else None,
            "oof_aucs": {d: m["oof_auc"] for d, m in desc_meta.items()},
            "train_aucs": {d: m["train_auc"] for d, m in desc_meta.items()},
        }
        task_dir = os.path.join(model_dir, task_name)
        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

    logger.success(f"All models saved to {model_dir}")
