# Default variables
ONNX_TARGET_OPSET = 16
ONNX_IR_VERSION = 10

import atexit as _atexit  # noqa: E402
import os as _os  # noqa: E402
import shutil as _shutil  # noqa: E402
import sys as _sys  # noqa: E402
import tempfile as _tempfile  # noqa: E402

if "MPLCONFIGDIR" not in _os.environ:
    _mpl_dir = _tempfile.mkdtemp(prefix="lazyqsar_mpl_")
    _os.environ["MPLCONFIGDIR"] = _mpl_dir
    _atexit.register(lambda: _shutil.rmtree(_mpl_dir, ignore_errors=True))

if _sys.platform == "darwin":
    # On macOS, PyTorch and XGBoost each ship their own libomp. When both are
    # loaded in the same process, OpenMP initialization can segfault. Setting
    # these env vars before either library is imported prevents the conflict.
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

from .utils.logging import logger as _logger  # noqa: E402


def set_verbosity(verbose: bool) -> None:
    """Enable (True) or disable (False) verbose logging globally."""
    _logger.set_verbosity(verbose)
