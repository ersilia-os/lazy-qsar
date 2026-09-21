"""Shared ensemble arithmetic for every LazyQSAR prediction path.

``combine`` folds per-descriptor predictions into a model's final outputs. It is the one
implementation of that arithmetic: the Python API, the CLI api and the fit-time estimator
all call it, so they cannot drift apart again.

Dependency-free by design — numpy only. The Ersilia Model Hub deploys LazyQSAR models
into templates that install neither scikit-learn nor RDKit, and this module sits on the
path those deployments run.
"""

from .combine import (
    OUTPUT_NAMES,
    CombineResult,
    EnsembleSpec,
    build_weight_matrix,
    combine,
    mask_rows,
)
from .reference import build_pooled_rank_knots

__all__ = [
    "OUTPUT_NAMES",
    "build_pooled_rank_knots",
    "CombineResult",
    "EnsembleSpec",
    "build_weight_matrix",
    "combine",
    "mask_rows",
]
