import csv
import gc
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from ..agnostic import LazyClassifier
from ..qsar import get_descriptor_type
from ..utils.logging import logger


def _new_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=True,
    )


def _get_chunk_size() -> int:
    try:
        v = int(os.environ.get("LAZYQSAR_PREDICT_CHUNK", "200"))
        return v if v > 0 else 1000
    except ValueError:
        return 1000


def _persist_descriptors(
    featurizer,
    smiles_list: list,
    out_path: str,
    chunk_size: int,
    progress: Progress | None = None,
    task_id=None,
) -> None:
    """Compute descriptors in chunks and stream each chunk into a memmap-backed .npy at `out_path`.

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


def _predict_from_persisted(
    model, x_path: str, predict_fn, chunk_size: int
) -> np.ndarray:
    """Run `predict_fn(model, X_chunk)` over mmapped chunks of the persisted descriptor matrix."""
    X_mm = np.load(x_path, mmap_mode="r")
    n_total = X_mm.shape[0]
    parts: list[np.ndarray] = []
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        X_chunk = np.ascontiguousarray(X_mm[start:end])
        parts.append(predict_fn(model, X_chunk))
        del X_chunk
    del X_mm
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


_PREDICT_DISPATCH = {
    "proba": lambda model, X: model.predict_proba(X)[:, 1],
    "rank": lambda model, X: model.predict_rank(X)[:, 1],
    "logit": lambda model, X: model.predict_logit(X)[:, 1],
    "lift": lambda model, X: model.predict_lift(X)[:, 1],
    "score": lambda model, X: model.predict_score(X)[:, 1],
    "binary": lambda model, X: model.predict(X),
}


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


def _predict_from_dict(
    model_dir: dict[str, str],
    input_csv: str | None,
    output_csv: str | None,
    models_txt: str | None,
    predict_type: str,
    smiles: list | None = None,
) -> tuple[np.ndarray, list[str]]:
    if predict_type not in _PREDICT_DISPATCH:
        raise ValueError(
            f"Unknown predict_type '{predict_type}'. "
            f"Choose from: {sorted(_PREDICT_DISPATCH)}"
        )

    col_map = {os.path.abspath(p): col for col, p in model_dir.items()}
    if input_csv is not None:
        input_csv = os.path.abspath(input_csv)
    if output_csv is not None:
        output_csv = os.path.abspath(output_csv)

    logger.info(
        f"Running dict prediction | {len(col_map)} models | input: {input_csv} | "
        f"output: {output_csv} | predict_type: {predict_type}"
    )

    if smiles is not None:
        smiles_list = smiles
        logger.info(f"Using {len(smiles_list)} SMILES from argument")
    else:
        smiles_list = read_smiles(input_csv)
        logger.info(f"Loaded {len(smiles_list)} SMILES from {input_csv}")

    if models_txt is not None:
        with open(models_txt) as f:
            allowed = {line.strip() for line in f}
        col_map = {p: c for p, c in col_map.items() if c in allowed}
        logger.info(f"Filtered to {len(col_map)} models via {models_txt}")
    if not col_map:
        raise ValueError("No valid models found.")

    all_featurizers = sorted(
        {
            dn
            for p in col_map
            for dn in os.listdir(p)
            if os.path.isdir(os.path.join(p, dn))
        }
    )
    logger.info(f"Featurizers found: {all_featurizers}")

    _predict_fn = _PREDICT_DISPATCH[predict_type]
    results: dict[tuple[str, str], np.ndarray] = {}

    chunk_size = _get_chunk_size()
    n_total = len(smiles_list)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    scratch_dir = tempfile.mkdtemp(prefix="lazyqsar-predict-")
    try:
        for featurizer_name in all_featurizers:
            featurizer = None
            for p in col_map:
                feat_dir = os.path.join(p, featurizer_name)
                if os.path.isdir(feat_dir):
                    featurizer = get_descriptor_type(featurizer_name).load(feat_dir)
                    break
            if featurizer is None:
                continue

            cols_with_models = [
                (p, c)
                for p, c in col_map.items()
                if os.path.isdir(os.path.join(p, featurizer_name))
            ]
            if not cols_with_models:
                continue

            logger.info(f"Computing descriptors: {featurizer_name}")
            x_path = os.path.join(scratch_dir, f"X_{featurizer_name}.npy")

            with _new_progress() as progress:
                desc_task = progress.add_task(
                    f"[{featurizer_name}] descriptors", total=n_chunks
                )
                _persist_descriptors(
                    featurizer,
                    smiles_list,
                    x_path,
                    chunk_size,
                    progress=progress,
                    task_id=desc_task,
                )
                del featurizer
                gc.collect()

                pred_task = progress.add_task(
                    f"[{featurizer_name}] predicting", total=len(cols_with_models)
                )
                for p, col_name in cols_with_models:
                    model_subdir = os.path.join(p, featurizer_name)
                    model = LazyClassifier.load(model_subdir)
                    results[(col_name, featurizer_name)] = _predict_from_persisted(
                        model, x_path, _predict_fn, chunk_size
                    )
                    del model
                    gc.collect()
                    progress.update(pred_task, advance=1)

            try:
                os.remove(x_path)
            except OSError:
                pass
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    aggregated: dict[str, np.ndarray] = {}
    for col_name in col_map.values():
        vals = [v for (c, _), v in results.items() if c == col_name]
        if vals:
            aggregated[col_name] = np.average(np.array(vals), axis=0)

    cols_ordered = list(col_map.values())
    R = np.array([aggregated[c] for c in cols_ordered]).T
    if output_csv is not None:
        pd.DataFrame(R, columns=cols_ordered).to_csv(output_csv, index=False)
        logger.success(f"Predictions saved to {output_csv}")
    return R, cols_ordered


def predict(
    model_dir: str | dict[str, str],
    input_csv: str = None,
    output_csv: str = None,
    models_txt: str = None,
    predict_type: str = "proba",
    smiles: list = None,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(model_dir, dict):
        return _predict_from_dict(
            model_dir, input_csv, output_csv, models_txt, predict_type, smiles
        )

    if predict_type not in _PREDICT_DISPATCH:
        raise ValueError(
            f"Unknown predict_type '{predict_type}'. "
            f"Choose from: {sorted(_PREDICT_DISPATCH)}"
        )

    model_dir = os.path.abspath(model_dir)
    if input_csv is not None:
        input_csv = os.path.abspath(input_csv)
    if output_csv is not None:
        output_csv = os.path.abspath(output_csv)

    logger.info(
        f"Running prediction | model: {model_dir} | input: {input_csv} | output: {output_csv} | predict_type: {predict_type}"
    )

    if smiles is not None:
        smiles_list = smiles
        logger.info(f"Using {len(smiles_list)} SMILES from argument")
    else:
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

    chunk_size = _get_chunk_size()

    results = {}
    n_total = len(smiles_list)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    scratch_dir = tempfile.mkdtemp(prefix="lazyqsar-predict-")
    try:
        for featurizer_name in featurizers:
            tasks_with_models = [
                t
                for t in tasks
                if os.path.isdir(os.path.join(model_dir, t, featurizer_name))
            ]
            if not tasks_with_models:
                continue

            logger.info(f"Computing descriptors: {featurizer_name}")
            featurizer = load_featurizer(model_dir, featurizer_name)
            x_path = os.path.join(scratch_dir, f"X_{featurizer_name}.npy")

            with _new_progress() as progress:
                desc_task = progress.add_task(
                    f"[{featurizer_name}] descriptors", total=n_chunks
                )
                _persist_descriptors(
                    featurizer,
                    smiles_list,
                    x_path,
                    chunk_size,
                    progress=progress,
                    task_id=desc_task,
                )
                del featurizer
                gc.collect()

                pred_task = progress.add_task(
                    f"[{featurizer_name}] predicting", total=len(tasks_with_models)
                )
                for task_name in tasks_with_models:
                    model_subdir = os.path.join(model_dir, task_name, featurizer_name)
                    model = LazyClassifier.load(model_subdir)
                    results[(task_name, featurizer_name)] = _predict_from_persisted(
                        model, x_path, _predict, chunk_size
                    )
                    del model
                    gc.collect()
                    progress.update(pred_task, advance=1)

            try:
                os.remove(x_path)
            except OSError:
                pass
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

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

    if output_csv is not None:
        pd.DataFrame(R, columns=tasks).to_csv(output_csv, index=False)
        logger.success(f"Predictions saved to {output_csv}")
    return R, tasks
