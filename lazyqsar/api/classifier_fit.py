"""Former name of :mod:`lazyqsar.api.fit`, from before regression; an alias of it.

The module object is the same, so code that imports or patches
``lazyqsar.api.classifier_fit`` reaches exactly what runs.
"""

import sys

from . import fit as _module

sys.modules[__name__] = _module
