# Default variables
ONNX_TARGET_OPSET = 16
ONNX_IR_VERSION = 10

import sys as _sys  # noqa: E402

if _sys.platform == "darwin":
    import os as _os  # noqa: E402

    # On macOS, PyTorch and XGBoost each ship their own libomp. When both are
    # loaded in the same process, OpenMP initialization can segfault. Setting
    # these env vars before either library is imported prevents the conflict.
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

from .utils.logging import logger as _logger  # noqa: E402


def set_verbosity(verbose: bool) -> None:
    """Enable (True) or disable (False) verbose logging globally."""
    _logger.set_verbosity(verbose)
