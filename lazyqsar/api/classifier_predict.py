"""Former name of :mod:`lazyqsar.api.predict`, from before regression; an alias of it.

The module object is the same, so code that imports or patches
``lazyqsar.api.classifier_predict`` reaches exactly what runs.
"""

import sys

from . import predict as _module

sys.modules[__name__] = _module
