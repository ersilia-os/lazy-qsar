"""Build real ONNX checkpoints for the pipeline tier.

These are genuine artifacts -- a real ``LazyClassifier`` fit and a real ONNX export per
descriptor -- because the point of the pipeline tests is to pin what the ONNX readers
actually return. That makes them the most expensive thing in the suite, which is why the
fixtures that call these are session-scoped and hand back paths rather than loaded models.
"""

import json
import os

import numpy as np


def build_checkpoint(root, task, descriptors, smiles, y, seed=0):
    """Fit and save one task's checkpoint, laid out as ``root/task/descriptor/``.

    Mirrors what ``api.classifier_fit`` produces: one ``LazyClassifier`` per descriptor plus
    a task-level ``metadata.json``.
    """
    from lazyqsar.agnostic import LazyClassifier
    from lazyqsar.registry import get_descriptor_type

    task_dir = os.path.join(root, task)
    per_descriptor = {}
    for d in descriptors:
        sub = os.path.join(task_dir, d)
        os.makedirs(sub, exist_ok=True)
        feat = get_descriptor_type(d)()
        X = feat.transform(smiles)
        model = LazyClassifier()
        model.fit(X=X, y=np.asarray(y, dtype=int))
        model.save(sub)
        feat.save(sub)
        inner = model._model
        per_descriptor[d] = {
            "oof_auc": float(model.oof_auc_),
            "train_auc": float(model.train_auc_),
            "decision_cutoff_raw": float(inner.decision_cutoff_raw_),
            "decision_cutoff_proba": float(inner.decision_cutoff_proba_),
            "decision_cutoff_rank": float(inner.decision_cutoff_rank_),
            "portfolio": inner.portfolio,
            "num_batches": len(inner.models),
        }

    y_arr = np.asarray(y, dtype=int)
    prior = float((y_arr == 1).mean())
    meta = {
        "mode": "fast",
        "descriptor_types": list(descriptors),
        "n_compounds": len(y_arr),
        "n_actives": int((y_arr == 1).sum()),
        "ratio_actives": prior,
        "population_prior": prior,
        "portfolio": per_descriptor[descriptors[0]]["portfolio"],
        "num_batches": {d: m["num_batches"] for d, m in per_descriptor.items()},
        "decision_cutoff_raw": float(
            np.mean([m["decision_cutoff_raw"] for m in per_descriptor.values()])
        ),
        "decision_cutoff_proba": float(
            np.mean([m["decision_cutoff_proba"] for m in per_descriptor.values()])
        ),
        "decision_cutoff_rank": float(
            np.mean([m["decision_cutoff_rank"] for m in per_descriptor.values()])
        ),
        "oof_aucs": {d: m["oof_auc"] for d, m in per_descriptor.items()},
        "train_aucs": {d: m["train_auc"] for d, m in per_descriptor.items()},
    }
    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return task_dir


def build_multitask_checkpoint(root, register, tasks=("taskA", "taskB", "taskC")):
    """Three tasks over one stubbed descriptor, sharing part of their compound pool.

    The overlap is deliberate: it is what makes "featurize the union once" a meaningful
    assertion rather than a tautology.
    """
    from .smiles import make_smiles

    counter = register("morgan")
    rng = np.random.default_rng(0)
    smiles = make_smiles(90)
    tasks = list(tasks)
    for i, task in enumerate(tasks):
        subset = smiles[i * 10 : i * 10 + 60]
        y = rng.integers(0, 2, len(subset))
        y[:6] = 1
        y[-6:] = 0
        build_checkpoint(root, task, ["morgan"], subset, y, seed=i)
    counter.reset()
    return root, tasks, smiles


def build_streaming_checkpoint(
    root, register, tasks=("taskE", "taskF"), descriptors=("morgan", "rdkit")
):
    """Two tasks over the same two stubbed descriptors.

    The shape the streaming invariants need, and the smallest one that works.
    ``build_multitask_checkpoint`` has a single descriptor, so in it both "one descriptor
    matrix on disk at a time" and "one ONNX artifact in memory at a time" hold by
    construction and the tests pass while asserting nothing. Two tasks over two
    descriptors is the minimum at which neither holds for free -- going below this makes
    the streaming tests vacuous rather than merely weaker.
    """
    from .smiles import make_smiles

    tasks = list(tasks)
    descriptors = list(descriptors)
    counter = register(*descriptors)
    rng = np.random.default_rng(2)
    smiles = make_smiles(90)
    for i, task in enumerate(tasks):
        subset = smiles[i * 15 : i * 15 + 40]
        y = rng.integers(0, 2, len(subset))
        y[:8] = 1
        y[-8:] = 0
        build_checkpoint(root, task, descriptors, subset, y, seed=i)
    counter.reset()
    return root, tasks, smiles, descriptors
