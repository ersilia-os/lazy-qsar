# Default variables
ONNX_TARGET_OPSET = 16
ONNX_IR_VERSION = 10

from .utils.logging import logger as _logger


def set_verbosity(verbose: bool) -> None:
    """Enable (True) or disable (False) verbose logging globally."""
    _logger.set_verbosity(verbose)
