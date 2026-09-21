"""A chemistry-free descriptor stand-in, and the chem-function bindings it must neutralise.

Only the ``chem`` tier installs RDKit, so every test that exercises fit or predict has to
supply its own featurizer. :class:`StubFeaturizer` is a deterministic, dependency-free
descriptor registered through :data:`lazyqsar.registry.DESCRIPTOR_TYPES` -- the single place
both entry points resolve descriptor classes, so one patch covers the CLI api, the Python api
and the shared runner alike.

Registering a stub is necessary but not sufficient. The pipelines also call RDKit *directly*
to validate SMILES, and those calls have to be neutralised at every binding -- see
:data:`STUBBED_BINDINGS`.
"""

import hashlib
import importlib
import json
import os
import sys
import types
from typing import ClassVar

import numpy as np

STUB_MODULE = "lazyqsar_stub_descriptors"
STUB_DIM = 24

# Every name through which the RDKit-backed SMILES checks are reachable.
#
# Patching `lazyqsar.qsar` alone is not enough: `api/classifier_fit.py` and
# `api/classifier_predict.py` bind these by from-import, which copies the function object at
# import time. Rebinding the definition site leaves those copies pointing at the original,
# which imports RDKit at module scope -- so the `fit` tier, which has no RDKit, would fail.
#
# `packaging/test_stub_bindings.py` asserts this list stays exhaustive, so a new re-export
# fails loudly instead of quietly making the suite require RDKit.
STUBBED_BINDINGS = (
    ("lazyqsar.qsar", "validate_smiles"),
    ("lazyqsar.qsar", "invalid_smiles_indices"),
    ("lazyqsar.api.classifier_fit", "validate_smiles"),
    ("lazyqsar.api.classifier_predict", "invalid_smiles_indices"),
)

# The names those bindings refer to, for the meta-test that looks for new ones.
CHEM_FUNCTION_NAMES = ("validate_smiles", "invalid_smiles_indices")

# What each stub returns: validation passes, and nothing is unparseable.
_STUB_IMPLEMENTATIONS = {
    "validate_smiles": lambda smiles_list: None,
    "invalid_smiles_indices": lambda smiles_list: [],
}


class StubFeaturizer:
    """Deterministic pseudo-descriptor: one fixed vector per (descriptor, SMILES) pair.

    The row is derived from a hash of the SMILES *and* ``salt``, so a compound gets the same
    vector wherever it appears -- which is what lets the reuse tests compare row counts
    across calls -- while two descriptors registered under different names produce genuinely
    different features. Without the salt every stubbed descriptor would yield the same
    matrix, hence the same model and the same predictions, which would quietly defeat any
    test about combining descriptors.
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
    featurizer objects; the tests care how many *rows* were featurized in total, which is the
    quantity the descriptor-reuse optimisation exists to keep down. Because the state is
    class-level, isolation is enforced centrally by an autouse fixture rather than trusted to
    each test.
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


def patch_chem_functions(monkeypatch):
    """Neutralise every binding in :data:`STUBBED_BINDINGS`.

    Stubbed descriptors never hand their strings to RDKit, so the RDKit-backed checks are
    neither available on a base install nor meaningful here. The strings the suite uses are
    valid SMILES, so this removes the dependency, not the check.
    """
    for module_name, attr in STUBBED_BINDINGS:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, attr, _STUB_IMPLEMENTATIONS[attr])


def install_stub_registry(monkeypatch):
    """Patch the chem functions and return a ``register(*names)`` callable.

    ``register`` installs a distinctly-salted :class:`CountingStub` subclass under each given
    descriptor name and returns the counter class. ``DESCRIPTOR_TYPES`` is patched rather
    than any single call site, so this keeps working wherever ``get_descriptor_type`` is
    called from.
    """
    from lazyqsar import registry

    mod = sys.modules.get(STUB_MODULE)
    if mod is None:
        mod = types.ModuleType(STUB_MODULE)
        sys.modules[STUB_MODULE] = mod
    mod.CountingStub = CountingStub
    mod.StubFeaturizer = StubFeaturizer

    patch_chem_functions(monkeypatch)

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

    return register
