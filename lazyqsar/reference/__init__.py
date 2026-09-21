"""The fixed reference library `predict_rank` reports percentiles against.

Fit-time only. A trained checkpoint carries its reference as knots in ``metadata.json``, so
scoring molecules never needs anything here -- which is what keeps inference on numpy and
onnxruntime alone.
"""

from .identity import DEFAULT_N, REFERENCE_ID, default_n, reference_dir
from .store import (
    ReferenceIntegrityError,
    ReferenceUnavailable,
    is_available,
    iter_chunks,
    load,
    reference_smiles,
    status,
)

__all__ = [
    "DEFAULT_N",
    "default_n",
    "REFERENCE_ID",
    "ReferenceIntegrityError",
    "ReferenceUnavailable",
    "is_available",
    "iter_chunks",
    "load",
    "reference_dir",
    "reference_smiles",
    "status",
]
