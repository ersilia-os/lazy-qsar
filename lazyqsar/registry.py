"""Descriptor registry: names, modes, and lazy class lookup.

Kept separate from :mod:`lazyqsar.qsar` so the inference stack can learn *which*
descriptors a checkpoint uses without importing any of them. ``qsar.py`` pulls in
``descriptors._validate``, which imports RDKit at module scope, so anything importing
the registry from there inherits a hard RDKit dependency — including
``api/classifier_predict.py``, which needs the names long before it needs a featurizer.

This module must stay importable on the base install: standard library only, no RDKit,
no scikit-learn, no NumPy. ``get_descriptor_type`` imports the concrete class on demand,
so the heavy dependency is paid at featurization time and only for the descriptors a
model actually uses.

``lazyqsar.qsar`` re-exports all three names, so existing
``from lazyqsar.qsar import get_descriptor_type`` imports keep working.
"""

DESCRIPTOR_TYPES = {
    "chemeleon": ("lazyqsar.descriptors.chemeleon", "ChemeleonDescriptor"),
    "morgan": ("lazyqsar.descriptors.morgan", "MorganFingerprint"),
    "rdkit": ("lazyqsar.descriptors.rdkit_descriptors", "RDKitDescriptor"),
    "cddd": ("lazyqsar.descriptors.cddd", "ContinuousDataDrivenDescriptor"),
    "clamp": ("lazyqsar.descriptors.clamp", "ClampDescriptor"),
}

DESCRIPTORS_MODE = {
    "fast": ["morgan"],
    "slow": ["chemeleon", "morgan", "rdkit", "cddd", "clamp"],
}

DESCRIPTORS_MODE = {k: sorted(v) for k, v in DESCRIPTORS_MODE.items()}


def get_descriptor_type(descriptor_name):
    """Return the descriptor class registered under *descriptor_name*.

    The class is imported on first use rather than at module load, which is what keeps
    this module free of RDKit and torch.

    Parameters
    ----------
    descriptor_name : str
        A key of :data:`DESCRIPTOR_TYPES`.

    Returns
    -------
    type
        The descriptor class, not an instance.
    """
    module_name, class_name = DESCRIPTOR_TYPES[descriptor_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)
