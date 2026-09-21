"""Tier gating, shared checkpoints and per-test hygiene.

The suite is organised so that a test's *directory* determines which dependencies it may
use. ``tests/unit`` and ``tests/packaging`` must run on a base install -- numpy, onnxruntime,
pandas -- because that is the environment Ersilia Model Hub templates deploy into.
``tests/fit`` and ``tests/pipeline`` need the ``fit`` extra, and ``tests/chem`` needs RDKit.

Tier markers are applied here rather than in each file, and a missing dependency becomes a
*skip that names the module*, not a collection error.
"""

import contextlib
import dataclasses
import os
import pathlib
import shutil

import numpy as np
import pytest

from _helpers.stubs import CountingStub

_TESTS_ROOT = pathlib.Path(__file__).parent

# The registry as shipped, captured at import time -- before any fixture can patch it.
#
# The stub fixtures register descriptors by mutating `DESCRIPTOR_TYPES` in place, so while
# a pipeline test runs, the live dict maps "morgan" at a synthetic stub module. A test that
# asserts on what the package actually ships has to read this snapshot instead, or it
# passes or fails depending on what ran before it.
from lazyqsar.registry import DESCRIPTOR_TYPES as _LIVE_DESCRIPTOR_TYPES  # noqa: E402

PRISTINE_DESCRIPTOR_TYPES = dict(_LIVE_DESCRIPTOR_TYPES)

from _helpers.tiers import TIER_DIRS, TIER_MODULES, missing_for_tier  # noqa: F401


def pytest_collection_modifyitems(config, items):
    missing = {tier: missing_for_tier(tier) for tier in TIER_MODULES}
    for item in items:
        try:
            parts = pathlib.Path(str(item.fspath)).relative_to(_TESTS_ROOT).parts
        except ValueError:  # pragma: no cover - a test collected from outside tests/
            continue
        for tier, dirs in TIER_DIRS.items():
            if parts and parts[0] in dirs:
                item.add_marker(getattr(pytest.mark, tier))
        for tier, gone in missing.items():
            if gone and item.get_closest_marker(tier) is not None:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"needs the [{tier}] tier; missing: {', '.join(gone)}"
                    )
                )


# --------------------------------------------------------------------- stub registry


@pytest.fixture
def shipped_descriptor_types():
    """The registry as shipped, unaffected by whatever the stub fixtures have patched."""
    return PRISTINE_DESCRIPTOR_TYPES


@contextlib.contextmanager
def stubbed_registry():
    """Install the stub descriptors for the duration of the block, then restore.

    Deliberately a context manager rather than a session-scoped fixture. The stubs work by
    mutating ``DESCRIPTOR_TYPES``, which is global, so a patch that lives for the whole
    session is visible to every test in it -- including ``tests/chem``, whose entire purpose
    is to exercise the *real* descriptors. Keeping the patch scoped to the block that needs
    it means the expensive thing (building a checkpoint) is still done once per session
    while the global mutation lasts only as long as it is wanted.
    """
    from _helpers.stubs import install_stub_registry

    mp = pytest.MonkeyPatch()
    try:
        yield install_stub_registry(mp)
    finally:
        mp.undo()


@pytest.fixture
def stub_descriptors(monkeypatch):
    """Function-scoped stub registry, for tests that register their own descriptor names."""
    from _helpers.stubs import install_stub_registry

    register = install_stub_registry(monkeypatch)
    yield register
    CountingStub.reset()


# ------------------------------------------------------------------ shared checkpoints
#
# The invariant that makes session scoping safe: these return paths and plain data, never a
# loaded model. `_ensemble_cache` lives on ArtifactWrapper/LazyClassifierQSAR instances and is
# keyed by the MD5 of the SMILES list, so a shared wrapper would make the second test to
# predict the same compounds see zero featurizations -- and pass a reuse assertion vacuously.
# Fitting is what is expensive; constructing an ONNX session is not.


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    root: str
    tasks: list
    smiles: list


@pytest.fixture(scope="session")
def _checkpoint_build(tmp_path_factory):
    """Build the shared checkpoint once. The stub patch does not outlive the build."""
    from _helpers.checkpoints import build_multitask_checkpoint

    root = str(tmp_path_factory.mktemp("ckpt") / "models")
    with stubbed_registry() as register:
        _, tasks, smiles = build_multitask_checkpoint(root, register)
    return Checkpoint(root=root, tasks=tasks, smiles=smiles)


@pytest.fixture
def checkpoint(_checkpoint_build, stub_descriptors):
    """Read-only view of the session checkpoint (3 tasks, 1 stubbed descriptor).

    Registers the stub for *this test only*. The checkpoint on disk was written by a stub
    featurizer, so reading it needs the same stub registered -- but only while the test that
    reads it is running, so the real descriptors stay available to ``tests/chem``.
    """
    stub_descriptors("morgan")
    return _checkpoint_build


@pytest.fixture
def mutable_checkpoint(_checkpoint_build, tmp_path):
    """A private copy, for tests that write into a checkpoint.

    Copying a few MB costs milliseconds; refitting costs seconds.
    """
    dst = tmp_path / "models"
    shutil.copytree(_checkpoint_build.root, dst)
    return dataclasses.replace(_checkpoint_build, root=str(dst))


@pytest.fixture
def load_model(checkpoint):
    """Factory returning a *fresh* wrapper per call, so ``_ensemble_cache`` starts cold."""
    from lazyqsar.qsar import LazyClassifierQSAR

    return lambda task: LazyClassifierQSAR.load(os.path.join(checkpoint.root, task))


@pytest.fixture
def multitask_checkpoint(checkpoint):
    """``(root, tasks, smiles, counter)`` -- the shape the ported tests expect.

    Kept tuple-shaped deliberately. These tests were written to pin behaviour through the
    3.5.0 refactor, so porting them verbatim is what makes them worth having; rewriting their
    unpacking would mean re-reading every assertion to be sure nothing shifted. The important
    change is underneath: this is now a view onto a checkpoint built once per session rather
    than a fresh set of model fits per test.
    """
    return checkpoint.root, checkpoint.tasks, checkpoint.smiles, CountingStub


@pytest.fixture(scope="session")
def _multidescriptor_build(tmp_path_factory):
    from _helpers.checkpoints import build_multidescriptor_checkpoint

    root = str(tmp_path_factory.mktemp("multi") / "models")
    with stubbed_registry() as register:
        root, task, smiles, descriptors = build_multidescriptor_checkpoint(
            root, register
        )
    return {"root": root, "task": task, "smiles": smiles, "descriptors": descriptors}


@pytest.fixture
def multidescriptor_checkpoint(_multidescriptor_build, stub_descriptors):
    """One task over three stubbed descriptors, for combining and weighting tests."""
    stub_descriptors(*_multidescriptor_build["descriptors"])
    return _multidescriptor_build


# ------------------------------------------------------------------------- hygiene


@pytest.fixture(autouse=True)
def _reset_stub_counter():
    """``CountingStub.calls`` is class-level by design, so isolation is enforced here."""
    CountingStub.reset()
    yield
    CountingStub.reset()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """A leaked env var from one test must not change another test's numbers."""
    for var in ("LAZYQSAR_PREDICT_CHUNK", "LAZYQSAR_FIT_SCRATCH"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _scramble_global_rng(request):
    """Seed the global numpy RNG differently for every test.

    This is the opposite of the usual advice and it is deliberate. ``ApplicabilityDomain``
    passes ``random_state`` to PCA precisely because ``svd_solver="auto"`` resolves to
    ``"randomized"`` at these dimensionalities and otherwise draws from global state. A fixed
    ambient seed would make a regression there invisible; a varying one turns it into a
    failure.
    """
    np.random.seed(abs(hash(request.node.nodeid)) % (2**32))
