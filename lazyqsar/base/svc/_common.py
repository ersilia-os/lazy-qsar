"""What the SVM classifier and regressor share: limits, cost model and the size guard.

A kernel SVM's ONNX graph stores every support vector, so its size grows with the
training set. Both heads fit the winning preset, then shrink an oversized model by
halving C (fewer margin violations, fewer support vectors), and as a last resort refit
the linear variant, whose graph holds one weight per feature.
"""

from __future__ import annotations

from lazyqsar.utils.logging import logger

_VAL_FRACTION = 0.1
_VAL_MIN_ROWS = 200
_RANDOM_STATE = 42
# Skip a preset whose estimated cost exceeds this multiple of the default preset's.
_MAX_COST_MULTIPLIER = 20
# ONNX size budget for kernel models (5 MB); linear models are O(n_features).
_ONNX_MAX_BYTES = 5_000_000
_ONNX_MAX_RETRIES = 3


def _fold_jobs(k: int, params: dict) -> int:
    """Threads for the k OOF fold fits.

    libsvm releases the GIL and its kernel solver draws no random numbers, so fold fits
    run concurrently with unchanged results. liblinear (LinearSVC/LinearSVR) shuffles
    with the process-wide C ``rand()``, which concurrent fits interleave: those folds
    stay serial to keep the fit deterministic.
    """
    import os

    if params.get("use_linear", False):
        return 1
    return max(1, min(k, os.cpu_count() or 1))


def svm_cost(params: dict, n_tr: int, n_features: int) -> float:
    """Relative training cost: O(n p) for the linear solver, O(n^2 p) for a kernel."""
    if params.get("use_linear", False):
        return float(n_tr * n_features)
    return float(n_tr**2 * n_features)


def onnx_size_bytes(model, n_features: int) -> int:
    """ONNX size of a kernel model (support vectors x float32); 0 for linear models."""
    if not hasattr(model, "support_vectors_"):
        return 0
    return model.support_vectors_.shape[0] * n_features * 4


class _SVMHead:
    """Final fit with the ONNX size guard, shared by both SVM heads.

    A subclass names the fitted-model attribute (``_fitted_attr``) and provides
    ``_make(params)`` and ``_linear_fallback_params(profile)``.
    """

    # A kernel sums over every support vector and feature in float32. On RDKit
    # descriptors (wide value ranges) that rounding reached 2.4e-4 of the output scale on
    # real data; every other head family stays below 1.1e-6 (dev/benchmarks measurement).
    _onnx_tolerance = 1e-3

    def _fit_final(self, X, target, profile) -> None:
        """Fit ``self.params_`` on all rows; shrink the model if its graph is too big."""
        n_features = profile.n_features
        model = self._make(self.params_).fit(X, target)
        if not self._use_linear_:
            for _retry in range(_ONNX_MAX_RETRIES):
                size = onnx_size_bytes(model, n_features)
                if size <= _ONNX_MAX_BYTES:
                    break
                new_C = self.params_["C"] / 2.0
                logger.warning(
                    f"ONNX size guard: {size / 1e6:.1f} MB > "
                    f"{_ONNX_MAX_BYTES / 1e6:.0f} MB — halving C: "
                    f"{self.params_['C']:.4g} → {new_C:.4g}"
                )
                self.params_["C"] = new_C
                model = self._make(self.params_).fit(X, target)
            else:
                size = onnx_size_bytes(model, n_features)
                if size > _ONNX_MAX_BYTES:
                    logger.warning(
                        f"ONNX size guard exhausted after {_ONNX_MAX_RETRIES} retries "
                        f"({size / 1e6:.1f} MB > {_ONNX_MAX_BYTES / 1e6:.0f} MB); "
                        "switching to the linear model."
                    )
                    self.params_ = self._linear_fallback_params(profile)
                    model = self._make(self.params_).fit(X, target)
                    self._use_linear_ = True
                    self.preset_name_ = self.preset_name_ + "+linear_guard"
        setattr(self, self._fitted_attr, model)
