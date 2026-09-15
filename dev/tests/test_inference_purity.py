"""The inference path must import under numpy + onnxruntime alone.

LazyQSAR models are deployed inside Ersilia Model Hub templates that install neither
scikit-learn, XGBoost, RDKit nor torch. A stray import in an artifact module would only
surface at deploy time, so it is guarded here instead.
"""

import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import sys

    BANNED = {"sklearn", "xgboost", "rdkit", "torch", "chemprop"}

    class Blocker:
        def find_module(self, name, path=None):
            if name.split(".")[0] in BANNED:
                raise ImportError(f"banned import at inference time: {name}")
            return None

    sys.meta_path.insert(0, Blocker())

    import numpy as np
    from lazyqsar.utils.ranking import rank_from_knots
    import lazyqsar.artifacts.xgboost      # noqa: F401
    import lazyqsar.artifacts.linear       # noqa: F401
    import lazyqsar.artifacts.classifier   # noqa: F401

    assert rank_from_knots(0.5, np.array([0.0, 0.5, 1.0])) > 0
    print("OK")
    """
)


def test_artifacts_import_without_training_dependencies():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
