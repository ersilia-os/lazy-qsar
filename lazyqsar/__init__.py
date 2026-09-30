# Default variables
ONNX_TARGET_OPSET = 16
ONNX_IR_VERSION = 10

import atexit as _atexit  # noqa: E402
import os as _os  # noqa: E402
import shutil as _shutil  # noqa: E402
import sys as _sys  # noqa: E402
import tempfile as _tempfile  # noqa: E402

if "MPLCONFIGDIR" not in _os.environ:
    # Matplotlib needs a writable config dir and falls over on a read-only HOME, which is
    # what a container running an Ersilia model has. Pointing it at the system temp dir
    # fixes that -- but a fresh mkdtemp per process also threw the font cache away every
    # run, so every invocation paid the rebuild and printed "Matplotlib is building the
    # font cache" to stderr. A stable per-user path keeps the read-only-HOME fix and lets
    # the cache survive; the uid is in the name so two users on one machine cannot collide
    # on a directory neither of them can write.
    _mpl_dir = _os.path.join(
        _tempfile.gettempdir(),
        f"lazyqsar-mpl-{_os.getuid() if hasattr(_os, 'getuid') else 'shared'}",
    )
    try:
        _os.makedirs(_mpl_dir, exist_ok=True)
    except (
        OSError
    ):  # pragma: no cover - fall back to a throwaway dir we know we can make
        _mpl_dir = _tempfile.mkdtemp(prefix="lazyqsar_mpl_")
        _atexit.register(lambda: _shutil.rmtree(_mpl_dir, ignore_errors=True))
    _os.environ["MPLCONFIGDIR"] = _mpl_dir

if _sys.platform == "darwin":
    # On macOS, PyTorch and XGBoost each ship their own libomp. When both are
    # loaded in the same process, OpenMP initialization can segfault. Setting
    # these env vars before either library is imported prevents the conflict.
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

from .utils.logging import logger as _logger  # noqa: E402

# Read from the installed distribution rather than hardcoded, so it cannot drift from
# pyproject.toml -- which the release workflow already asserts against the git tag. There
# was no `__version__` at all before 3.6, which mattered precisely because 3.6 changed
# what `rank` means: a user holding a checkpoint and a set of numbers had no way to ask
# which version produced them.
try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("lazyqsar")
    except _PackageNotFoundError:  # pragma: no cover - running from a source tree
        __version__ = "unknown"
except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.11+
    __version__ = "unknown"


def set_verbosity(verbose: bool) -> None:
    """Enable (True) or disable (False) verbose logging globally."""
    _logger.set_verbosity(verbose)


__all__ = ["__version__", "set_verbosity"]
