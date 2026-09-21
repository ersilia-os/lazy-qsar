"""Which optional dependencies each test tier needs, and how a directory opts out.

The suite is tiered by directory: ``tests/unit`` and ``tests/packaging`` must run on a base
install, ``tests/fit`` and ``tests/pipeline`` need the ``fit`` extra, ``tests/chem`` needs
RDKit. This module is the single place that mapping lives, so the root conftest (which
applies the markers) and the per-directory conftests (which gate collection) cannot drift.
"""

import importlib.util

TIER_MODULES = {
    "fit": ("sklearn", "xgboost", "skl2onnx", "onnxmltools", "scipy", "joblib"),
    "chem": ("rdkit",),
    "deep": ("torch", "chemprop", "chemeleon"),
}

# Which directories carry which tier. Anything not listed is base tier.
TIER_DIRS = {"fit": ("fit", "pipeline"), "chem": ("chem",), "deep": ("deep",)}


def missing_for_tier(tier):
    """The tier's dependencies that are not importable here."""
    return [m for m in TIER_MODULES[tier] if importlib.util.find_spec(m) is None]


def skip_directory_if_tier_unavailable(tier):
    """``(collect_ignore_glob, missing)`` for a tiered directory's ``conftest.py``.

    Markers alone are not enough. ``pytest_collection_modifyitems`` runs *after* a module is
    imported, so a test file with ``from sklearn... import`` at module scope raises during
    collection on an install without it -- a hard error, before any marker could skip it.

    Requiring every tiered module to defer its imports into function bodies would also work,
    but that is per-file discipline nothing enforces, and one slip breaks the base CI job.
    Gating collection makes the tier boundary a property of the layout instead. Inside an
    environment that does have the dependencies, the markers still drive ``-m`` selection.
    """
    missing = missing_for_tier(tier)
    return (["*"] if missing else []), missing
