from .utils.logging import logger
try:
    from .qsar_v1 import LazyBinaryQSAR
except Exception as e:
    logger.warning(
        "You are not using the full version of lazy-qsar which has descriptors pipeline!"
    )
    logger.warning(e)
    pass
from .agnostic_v1 import LazyBinaryClassifier


# Default variables
ONNX_TARGET_OPSET = 16
ONNX_IR_VERSION = 10