from .utils.logging import logger
try:
    from .qsar import LazyBinaryQSAR
except Exception as e:
    logger.warning(
        "You are not using the full version of lazy-qsar which has descriptors pipeline!"
    )
    logger.warning(e)
    pass
from .agnostic import LazyBinaryClassifier
