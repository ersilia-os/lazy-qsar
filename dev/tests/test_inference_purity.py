"""The inference path must import under numpy + onnxruntime alone.

LazyQSAR models are deployed inside Ersilia Model Hub templates that install neither
scikit-learn, XGBoost, RDKit nor torch. A stray import in an artifact module would only
surface at deploy time, so it is guarded here instead.

The guard hooks ``find_spec``, not ``find_module``. ``MetaPathFinder.find_module`` was
removed in Python 3.12 — the version CI pins — so a blocker defining only that method is
never consulted and the whole test silently passes no matter what gets imported.
"""

import subprocess
import sys
import textwrap

_BLOCKER = """
    import sys

    BANNED = {"sklearn", "xgboost", "rdkit", "torch", "chemprop"}

    class Blocker:
        # find_spec, not find_module: the latter is not an import hook on 3.12+.
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BANNED:
                raise ImportError(f"banned import at inference time: {name}")
            return None

    sys.meta_path.insert(0, Blocker())
"""

_ARTIFACTS = textwrap.dedent(
    _BLOCKER
    + """
    import numpy as np
    from lazyqsar.utils.ranking import rank_from_knots
    import lazyqsar.artifacts.xgboost      # noqa: F401
    import lazyqsar.artifacts.linear       # noqa: F401
    import lazyqsar.artifacts.classifier   # noqa: F401
    import lazyqsar.registry               # noqa: F401
    import lazyqsar.applicability          # noqa: F401

    assert rank_from_knots(0.5, np.array([0.0, 0.5, 1.0])) > 0
    print("OK")
    """
)

# api.classifier_predict is the module the Ersilia Hub actually calls. It needs pandas
# and rich (both core dependencies) but must not drag in RDKit, which it would if it
# took the descriptor registry from qsar.py rather than lazyqsar.registry.
_PREDICT_API = textwrap.dedent(
    _BLOCKER
    + """
    import lazyqsar.api.classifier_predict  # noqa: F401
    print("OK")
    """
)

# The blocker itself must work, or every assertion above is vacuous.
_SELF_TEST = textwrap.dedent(
    _BLOCKER
    + """
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("OK")
    else:
        raise AssertionError("blocker did not block a banned import")
    """
)


def _run(script):
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_blocker_actually_blocks():
    """Guard the guard: a no-op blocker would make every other test here pass."""
    _run(_SELF_TEST)


def test_artifacts_import_without_training_dependencies():
    _run(_ARTIFACTS)


def test_predict_api_imports_without_training_dependencies():
    _run(_PREDICT_API)
