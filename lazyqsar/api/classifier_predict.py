import csv
import os
import tempfile

import numpy as np
import pandas as pd

from ..agnostic import LazyClassifier
from ..qsar import get_descriptor_type
from ..utils.logging import logger

_PREDICT_DISPATCH = {
    "proba":  lambda model, X: model.predict_proba(X)[:, 1],
    "rank":   lambda model, X: model.predict_rank(X)[:, 1],
    "logit":  lambda model, X: model.predict_logit(X)[:, 1],
    "lift":   lambda model, X: model.predict_lift(X)[:, 1],
    "score":  lambda model, X: model.predict_score(X)[:, 1],
    "binary": lambda model, X: model.predict(X),
}


def prepare_files(smiles_list, models: list = None, path: str = None, predict_type: str = "proba"):
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


def _predict_from_dict(
    model_dir: dict[str, str],
    input_csv: str,
    output_csv: str,
    models_txt: str | None,
    predict_type: str,
) -> None:
    if predict_type not in _PREDICT_DISPATCH:
        raise ValueError(
            f"Unknown predict_type '{predict_type}'. "
            f"Choose from: {sorted(_PREDICT_DISPATCH)}"
        )

    col_map = {os.path.abspath(p): col for p, col in model_dir.items()}
    input_csv = os.path.abspath(input_csv)
    output_csv = os.path.abspath(output_csv)

    logger.info(
        f"Running dict prediction | {len(col_map)} models | input: {input_csv} | "
        f"output: {output_csv} | predict_type: {predict_type}"
    )

    smiles_list = read_smiles(input_csv)
    logger.info(f"Loaded {len(smiles_list)} SMILES from {input_csv}")

    if models_txt is not None:
        with open(models_txt) as f:
            allowed = {line.strip() for line in f}
        col_map = {p: c for p, c in col_map.items() if c in allowed}
        logger.info(f"Filtered to {len(col_map)} models via {models_txt}")
    if not col_map:
        raise ValueError("No valid models found.")

    all_featurizers = sorted({
        dn
        for p in col_map
        for dn in os.listdir(p)
        if os.path.isdir(os.path.join(p, dn))
    })
    logger.info(f"Featurizers found: {all_featurizers}")

    _predict_fn = _PREDICT_DISPATCH[predict_type]
    results: dict[tuple[str, str], np.ndarray] = {}

    for featurizer_name in all_featurizers:
        featurizer = None
        for p in col_map:
            feat_dir = os.path.join(p, featurizer_name)
            if os.path.isdir(feat_dir):
                featurizer = get_descriptor_type(featurizer_name).load(feat_dir)
                break
        if featurizer is None:
            continue
        logger.info(f"Computing descriptors: {featurizer_name}")
        X = featurizer.transform(smiles_list)
        for p, col_name in col_map.items():
            model_subdir = os.path.join(p, featurizer_name)
            if os.path.isdir(model_subdir):
                logger.debug(f"Predicting '{col_name}' with '{featurizer_name}'")
                model = LazyClassifier.load(model_subdir)
                results[(col_name, featurizer_name)] = _predict_fn(model, X)

    aggregated: dict[str, np.ndarray] = {}
    for col_name in col_map.values():
        vals = [v for (c, _), v in results.items() if c == col_name]
        if vals:
            aggregated[col_name] = np.average(np.array(vals), axis=0)

    cols_ordered = list(col_map.values())
    R = np.array([aggregated[c] for c in cols_ordered]).T
    pd.DataFrame(R, columns=cols_ordered).to_csv(output_csv, index=False)
    logger.success(f"Predictions saved to {output_csv}")


def predict(
    model_dir: str | dict[str, str],
    input_csv: str,
    output_csv: str,
    models_txt: str = None,
    predict_type: str = "proba",
):
    if isinstance(model_dir, dict):
        return _predict_from_dict(model_dir, input_csv, output_csv, models_txt, predict_type)

    if predict_type not in _PREDICT_DISPATCH:
        raise ValueError(
            f"Unknown predict_type '{predict_type}'. "
            f"Choose from: {sorted(_PREDICT_DISPATCH)}"
        )

    model_dir = os.path.abspath(model_dir)
    input_csv = os.path.abspath(input_csv)
    output_csv = os.path.abspath(output_csv)

    logger.info(
        f"Running prediction | model: {model_dir} | input: {input_csv} | output: {output_csv} | predict_type: {predict_type}"
    )

    smiles_list = read_smiles(input_csv)
    logger.info(f"Loaded {len(smiles_list)} SMILES from {input_csv}")

    tasks = get_task_names(model_dir)
    logger.info(f"Found tasks: {tasks}")
    if models_txt is not None:
        with open(models_txt, "r") as f:
            models = [line.strip() for line in f]
        tasks = [t for t in models if t in tasks]
        logger.info(f"Filtered to tasks: {tasks}")
    if len(tasks) == 0:
        raise ValueError("No valid tasks found in the model directory.")

    featurizers = get_featurizer_names(model_dir, tasks)
    _predict = _PREDICT_DISPATCH[predict_type]

    results = {}
    for featurizer_name in featurizers:
        logger.info(f"Computing descriptors: {featurizer_name}")
        featurizer = load_featurizer(model_dir, featurizer_name)
        X = featurizer.transform(smiles_list)
        for task_name in tasks:
            model_subdir = os.path.join(model_dir, task_name, featurizer_name)
            if os.path.isdir(model_subdir):
                logger.debug(
                    f"Predicting task '{task_name}' with descriptor '{featurizer_name}'"
                )
                model = LazyClassifier.load(model_subdir)
                results[(task_name, featurizer_name)] = _predict(model, X)

    aggregated_results = {}
    for task_name in tasks:
        R = []
        for k, v in results.items():
            if k[0] == task_name:
                R += [v]
        aggregated_results[task_name] = np.average(np.array(R), axis=0)

    R = []
    for task in tasks:
        R += [aggregated_results[task]]
    R = np.array(R).T

    df = pd.DataFrame(R, columns=tasks)
    df.to_csv(output_csv, index=False)
    logger.success(f"Predictions saved to {output_csv}")
