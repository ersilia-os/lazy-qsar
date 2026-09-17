"""Molecular descriptors.

Classes resolve lazily. Importing a single module from this package — the portfolio, say —
used to import every descriptor with it, and each of them imports RDKit at module scope. So
anything that merely wanted to *name* a descriptor inherited RDKit, torch and chemprop.

The public names below still work; they are just imported when first touched.
"""

__all__ = ["MorganFingerprint", "RDKitDescriptor"]

_LAZY = {
    "MorganFingerprint": (".morgan", "MorganFingerprint"),
    "RDKitDescriptor": (".rdkit_descriptors", "RDKitDescriptor"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name, __name__), attr)


def __dir__():
    return sorted(__all__)
