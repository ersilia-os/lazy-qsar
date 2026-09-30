"""``tests/unit``, ``tests/packaging`` and the machinery they load must run on a base install.

The tier markers in ``tests/conftest.py`` are applied by directory, so a test under
``tests/unit`` is *declared* to need nothing beyond numpy, onnxruntime and pandas. Nothing
enforces that declaration at runtime: a module-scope ``import sklearn`` would make the whole
file fail to collect, and the base CI job would go red for a reason the marker system was
supposed to prevent.

This reads the test files rather than importing them, so it reports every offender at once
instead of dying on the first.
"""

import ast
import os

import pytest

from _helpers.stubs import STUB_MODULE  # noqa: F401  (import sanity)

TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ``_helpers`` is in here because ``tests/conftest.py`` imports ``_helpers.stubs`` and
# ``_helpers.tiers`` at module level, and conftest is loaded for *every* test. A tiered
# import added under ``_helpers/`` therefore errors the entire base tier -- 308 tests at
# once -- while a guard that scanned only the test directories stayed green.
BASE_TIER_DIRS = ("unit", "packaging", "_helpers")

# Scanned as well as the directories above, for the same reason: it is imported before
# every test in the suite.
BASE_TIER_FILES = ("conftest.py",)

# Anything outside the core install. `tests/conftest.py` may reference these by *name* for
# the tier gating, but no base-tier module may import one.
TIERED = {
    "sklearn",
    "xgboost",
    "skl2onnx",
    "onnxmltools",
    "onnxconverter_common",
    "scipy",
    "joblib",
    "rdkit",
    "torch",
    "chemprop",
    "chemeleon",
    "FPSim2",
}


def _base_tier_files():
    for directory in BASE_TIER_DIRS:
        root = os.path.join(TESTS_ROOT, directory)
        for name in sorted(os.listdir(root)):
            if name.endswith(".py"):
                yield os.path.join(root, name)
    for name in BASE_TIER_FILES:
        yield os.path.join(TESTS_ROOT, name)


def _module_level_imports(path):
    tree = ast.parse(open(path).read())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "path", list(_base_tier_files()), ids=lambda p: os.path.relpath(p, TESTS_ROOT)
)
def test_base_tier_module_imports_nothing_tiered(path):
    offenders = _module_level_imports(path) & TIERED
    assert not offenders, (
        f"{os.path.relpath(path, TESTS_ROOT)} imports {sorted(offenders)} at module level. "
        "Base-tier tests must import on a core install; move the import inside the test, or "
        "move the file to tests/fit or tests/chem."
    )


def test_the_guard_found_some_files():
    """A typo in the directory names would make every assertion above vacuous."""
    assert len(list(_base_tier_files())) >= 5
