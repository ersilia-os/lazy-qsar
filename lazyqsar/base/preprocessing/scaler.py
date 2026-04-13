"""
Scaler selection and construction for base preprocessing.
"""

from sklearn.preprocessing import (
    MaxAbsScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from .inspector import PreprocessingProfile


def select_scaler(profile: PreprocessingProfile) -> str:
    if profile.is_sparse_counts:
        return "max_abs"
    if profile.binary_feature_fraction >= 0.8:
        return "max_abs"
    if profile.sparsity > 0.5:
        return "max_abs"
    if profile.outlier_fraction > 0.3:
        return "robust"
    if profile.median_feature_skewness > 1.5:
        return "power"
    return "standard"


_SCALER_FACTORIES = {
    "standard": StandardScaler,
    "robust":   RobustScaler,
    "power":    lambda: PowerTransformer(method="yeo-johnson"),
    "max_abs":  MaxAbsScaler,
}


def build_scaler(scaler_name: str):
    if scaler_name not in _SCALER_FACTORIES:
        raise ValueError(
            f"Unknown scaler {scaler_name!r}. "
            f"Valid options: {sorted(_SCALER_FACTORIES)}"
        )
    return _SCALER_FACTORIES[scaler_name]()
