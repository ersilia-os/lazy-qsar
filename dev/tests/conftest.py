"""Shared fixtures: a chemistry-free descriptor stub and synthetic ONNX checkpoints.

CI installs only the ``fit`` extra, so RDKit, torch and the deep-learning descriptors are
absent. Every test that exercises the fit or predict pipelines therefore has to supply its
own featurizer. ``StubFeaturizer`` is a deterministic, dependency-free stand-in registered
through :data:`lazyqsar.registry.DESCRIPTOR_TYPES`, which is the single place both paths
resolve descriptor classes — so one patch covers the CLI api, the Python api and the
shared runner alike.
"""

import hashlib
import json
import os
import sys
import types
from typing import ClassVar

import numpy as np
import pytest

STUB_MODULE = "lazyqsar_stub_descriptors"
STUB_DIM = 24


class StubFeaturizer:
    """Deterministic pseudo-descriptor: one fixed vector per (descriptor, SMILES) pair.

    The row is derived from a hash of the SMILES *and* ``salt``, so a compound gets the
    same vector wherever it appears — which is what lets the reuse tests compare row
    counts across calls — while two descriptors registered under different names produce
    genuinely different features. Without the salt every stubbed descriptor would yield
    the same matrix, hence the same model and the same predictions, which would quietly
    defeat any test about combining descriptors.
    """

    featurizer_name = "stub"
    salt = ""

    def __init__(self):
        self.n_dim = STUB_DIM
        self.features = [f"stub_{i}" for i in range(STUB_DIM)]

    def transform(self, smiles_list):
        out = np.empty((len(smiles_list), STUB_DIM), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            digest = hashlib.md5(f"{self.salt}|{smi}".encode()).digest()[:8]
            seed = int.from_bytes(digest, "little")
            out[i] = np.random.default_rng(seed).normal(size=STUB_DIM)
        return out

    def is_applicable(self, smiles_list):
        return True

    def save(self, dir_name):
        with open(os.path.join(dir_name, "featurizer.json"), "w") as f:
            json.dump({"featurizer": self.featurizer_name, "n_dim": STUB_DIM}, f)

    @classmethod
    def load(cls, dir_name):
        return cls()


class CountingStub(StubFeaturizer):
    """StubFeaturizer that records the size of every ``transform`` call.

    Counts live on the class, not the instance, because the pipelines construct their own
    featurizer objects; the tests care how many *rows* were featurized in total, which is
    the quantity the descriptor-reuse optimisation exists to keep down.
    """

    calls: ClassVar[list[int]] = []

    def transform(self, smiles_list):
        CountingStub.calls.append(len(smiles_list))
        return super().transform(smiles_list)

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def total_rows(cls):
        return sum(cls.calls)


@pytest.fixture
def stub_descriptors(monkeypatch):
    """Register CountingStub under the given descriptor names, and hand back the counter.

    Yields a callable ``register(*names)``; call it with the descriptor names the test
    wants stubbed. ``DESCRIPTOR_TYPES`` is patched rather than any single call site, so
    this keeps working wherever ``get_descriptor_type`` is called from.
    """
    mod = types.ModuleType(STUB_MODULE)
    mod.CountingStub = CountingStub
    mod.StubFeaturizer = StubFeaturizer
    sys.modules[STUB_MODULE] = mod

    # A stubbed descriptor means the pipeline never hands these strings to RDKit, so the
    # RDKit-backed SMILES check is neither available on a base install nor meaningful here.
    # The strings are still valid SMILES, so this only removes the dependency, not the check.
    import lazyqsar.qsar as _qsar
    from lazyqsar import registry

    monkeypatch.setattr(_qsar, "validate_smiles", lambda smiles_list: None)

    def register(*names):
        for name in names:
            cls_name = f"CountingStub_{name}"
            setattr(
                mod,
                cls_name,
                type(
                    cls_name, (CountingStub,), {"salt": name, "featurizer_name": name}
                ),
            )
            monkeypatch.setitem(
                registry.DESCRIPTOR_TYPES, name, (STUB_MODULE, cls_name)
            )
        CountingStub.reset()
        return CountingStub

    yield register

    CountingStub.reset()
    sys.modules.pop(STUB_MODULE, None)


def make_smiles(n):
    """n distinct, genuinely valid SMILES (simple ethers and alcohols).

    They have to parse: ``api.classifier_fit`` runs ``validate_smiles`` over the union
    before featurizing, so placeholder strings would be rejected before the stub is ever
    reached. Chemistry is irrelevant here — only validity and distinctness.
    """
    if n > 90:
        raise ValueError("make_smiles supports up to 90 distinct strings")
    return [("C" * (i // 10 + 1)) + "O" + ("C" * (i % 10)) for i in range(n)]


def build_checkpoint(root, task, descriptors, smiles, y, seed=0):
    """Fit and save a real ONNX checkpoint laid out as ``root/task/descriptor/``.

    Mirrors what ``api.classifier_fit`` produces today: one ``LazyClassifier`` per
    descriptor plus a task-level ``metadata.json``. Real artifacts, because the point of
    these tests is to pin what the ONNX readers actually return.
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


@pytest.fixture
def multitask_checkpoint(tmp_path, stub_descriptors):
    """Three tasks over one stubbed descriptor, sharing part of their compound pool.

    Returns ``(root, tasks, smiles, counter)``. The counter is reset after the build, so a
    test measures only what its own calls featurize.
    """
    register = stub_descriptors
    counter = register("morgan")
    rng = np.random.default_rng(0)
    smiles = make_smiles(90)
    root = str(tmp_path / "models")
    tasks = ["taskA", "taskB", "taskC"]
    for i, task in enumerate(tasks):
        subset = smiles[i * 10 : i * 10 + 60]
        y = rng.integers(0, 2, len(subset))
        y[:6] = 1
        y[-6:] = 0
        build_checkpoint(root, task, ["morgan"], subset, y, seed=i)
    counter.reset()
    return root, tasks, smiles, counter
