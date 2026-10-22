"""
Sklearn-compatible support-vector regressor with one evidence-based preset.

The regression twin of BaseSVCClassifier: an RBF ``SVR`` with ``C=10``, ``epsilon=0.05``
and ``gamma="scale"``, fitted on internally standardised targets so C and epsilon mean
the same on every target scale (pIC50, logS, ...). The ONNX graph emits standardised
values; predictions and the artifact map them back.

Why one preset and no race: on LazyQSAR's regression benchmark (TDC ADMET, Biogen ADME,
30 MoleculeACE ChEMBL sets; random and scaffold splits; Morgan fingerprints after the
standard preprocessor) this setting was within 0.3% of the per-task best RMSE on
average and the best of six candidates on 50 of 86 task splits -- against 3.8% for
sklearn's defaults (C=1) and 44% for a LinearSVR. van Tilborg et al. (2022) reach the
same setting (C=10) on most MoleculeACE sets by grid search. Racing presets on holdout
splits cost more fit time than it won back.

After the fit, the ONNX size guard shared with the SVC (support vectors x features x 4
bytes <= 5 MB) halves C up to three times and, as a last resort, falls back to a
LinearSVR, whose graph is one weight per feature.

OOF -- quantile-stratified k-fold with the same preset, for the ensemble gate.
"""

from __future__ import annotations

import json
import os
import time as _time

import numpy as np

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.model_selection import train_test_split
    from sklearn.utils.validation import check_array, check_is_fitted

    _FIT_DEPS_AVAILABLE = True
except ImportError:

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass

    train_test_split = check_array = check_is_fitted = None  # type: ignore[assignment]
    _FIT_DEPS_AVAILABLE = False

from .inspector import inspect as _inspect, DatasetProfile
from ._common import _SVR_ONNX_MAX_BYTES, _SVMHead, _fold_jobs
from lazyqsar.base._head import RegressorHead, require_fit_deps
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import probe_rows
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.splits import make_quantile_oof_splits


# epsilon-tube width in standardised target units.
_EPSILON = 0.05


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def svr_rbf_params(profile: DatasetProfile) -> dict:
    """The preset: RBF kernel, C=10, epsilon=0.05 (standardised y), gamma='scale'."""
    return {
        "C": 10.0,
        "kernel": "rbf",
        "gamma": "scale",
        "epsilon": _EPSILON,
        "tol": 1e-3,
        "use_linear": False,
    }


def svr_linear_params(profile: DatasetProfile) -> dict:
    """LinearSVR preset; C scaled by n as in the SVC linear preset."""
    n = profile.n_samples
    if n < 500:
        C = 0.1
    elif n < 2_000:
        C = 1.0
    else:
        C = 10.0
    return {
        "C": C,
        "epsilon": 0.0,
        "max_iter": max(1_000, min(10_000, n * 5)),
        "tol": 1e-3,
        "random_state": 42,
        "use_linear": True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_svr(params: dict):
    """Build a (not yet fitted) SVR or LinearSVR from a preset params dict."""
    if params.get("use_linear", False):
        from sklearn.svm import LinearSVR

        return LinearSVR(
            C=params["C"],
            epsilon=params.get("epsilon", 0.0),
            loss="squared_epsilon_insensitive",
            dual="auto",
            max_iter=params.get("max_iter", 5_000),
            tol=params.get("tol", 1e-3),
            random_state=params.get("random_state", 42),
        )
    from sklearn.svm import SVR

    # No iteration cap for the kernel solver: libsvm converges on these problem sizes,
    # and a cap (the SVC presets carry one) can stop it short of the optimum.
    return SVR(
        C=params["C"],
        epsilon=params.get("epsilon", _EPSILON),
        kernel=params.get("kernel", "rbf"),
        gamma=params.get("gamma", "scale"),
        max_iter=-1,
        tol=params.get("tol", 1e-3),
    )


def _svr_onnx_bytes(svr, n_features: int) -> bytes:
    """SVR or LinearSVR as a serialised ONNX graph (standardised output).

    An RBF SVR is written as matrix products (:func:`_rbf_svr_onnx_bytes`); a LinearSVR
    (or any other kernel) goes through skl2onnx.
    """
    if getattr(svr, "kernel", None) == "rbf" and hasattr(svr, "support_vectors_"):
        return _rbf_svr_onnx_bytes(svr, n_features)
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    return convert_sklearn(
        svr, initial_types=initial_types, target_opset=15
    ).SerializeToString()


def _rbf_svr_onnx_bytes(svr, n_features: int) -> bytes:
    """An RBF SVR as standard ONNX ops: ``exp(-gamma * |x - sv|^2) @ dual + b``.

    skl2onnx emits onnxruntime's ``SVMRegressor`` operator, which evaluates the kernel
    support vector by support vector, without BLAS: ~2 ms per row at 1,200 support
    vectors x 2,048 features, so a fit spent seconds predicting its own training rows
    and every later prediction paid the same. Here the squared distance is expanded as
    ``|x|^2 - 2 x.sv + |sv|^2`` around one ``MatMul`` -- about 100x faster on the
    benchmark -- and the support vectors are a packed float32 initializer (4 bytes per
    value against the attribute list's 5). Rounding can take a squared distance a hair
    below zero, so it is clamped at 0. The output matches skl2onnx's: ``variable``,
    float32, ``(n, 1)``; agreement with the native model is checked at bind time like
    any graph (measured gap no larger than ``SVMRegressor``'s).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    sv = np.asarray(svr.support_vectors_, dtype=np.float64)
    dual = np.asarray(svr.dual_coef_, dtype=np.float64).ravel()

    def const(name, values):
        return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name)

    initializers = [
        const("sv_t", sv.T),  # (p, m)
        const("sv_sq", (sv**2).sum(axis=1)[None, :]),  # (1, m)
        const("dual", dual[:, None]),  # (m, 1)
        const("intercept", [float(np.ravel(svr.intercept_)[0])]),
        const("neg_gamma", [-float(svr._gamma)]),
        const("minus_two", [-2.0]),
        const("zero", [0.0]),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "feature_axis"),
    ]
    nodes = [
        helper.make_node("Mul", ["float_input", "float_input"], ["x_x"]),
        helper.make_node("ReduceSum", ["x_x", "feature_axis"], ["x_sq"], keepdims=1),
        helper.make_node("MatMul", ["float_input", "sv_t"], ["x_sv"]),
        helper.make_node("Mul", ["x_sv", "minus_two"], ["minus_2x_sv"]),
        helper.make_node("Add", ["minus_2x_sv", "x_sq"], ["partial"]),
        helper.make_node("Add", ["partial", "sv_sq"], ["d2_raw"]),
        helper.make_node("Max", ["d2_raw", "zero"], ["d2"]),
        helper.make_node("Mul", ["d2", "neg_gamma"], ["exponent"]),
        helper.make_node("Exp", ["exponent"], ["kernel"]),
        helper.make_node("MatMul", ["kernel", "dual"], ["weighted"]),
        helper.make_node("Add", ["weighted", "intercept"], ["variable"]),
    ]
    graph = helper.make_graph(
        nodes,
        "rbf_svr",
        [helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])],
        initializers,
    )
    # IR 8 / opset 15: what skl2onnx writes, and what every supported onnxruntime reads.
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)], ir_version=8
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


# ---------------------------------------------------------------------------
# BaseSVR
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BaseSVR
# ---------------------------------------------------------------------------


class BaseSVR(_SVMHead, RegressorHead, BaseEstimator, RegressorMixin):
    """
    RBF support-vector regressor with the benchmark-selected preset (see the module).

    Parameters
    ----------
    portfolio : bool
        Kept for API compatibility; there is a single preset, so it has no effect.
    oof : bool
        If True (default), collect quantile-stratified OOF predictions.
    random_state : int
        Random seed for reproducibility.

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict
    preset_name_ : str         — "rbf_c10" (+ "+linear_guard" if the size guard fell back)
    portfolio_scores_ : dict   — empty: there is no preset race
    svr_ : SVR or LinearSVR    — fitted on standardised targets
    y_mean_, y_std_ : float    — target standardisation
    oof_preds_ : ndarray (n,)  — OOF predictions in original units
    oof_rho_ : float or None
    train_rho_ : float or None
    timing_ : dict
    """

    _fitted_attr = "svr_"
    # The epsilon tube, not C, sets an SVR's support-vector count: halving C changed it by
    # about 1% on the benchmark, so an oversized kernel goes straight to the linear model.
    _size_guard_retries = 0
    _onnx_max_bytes = _SVR_ONNX_MAX_BYTES
    # The RBF graph stores support vectors as a packed float32 tensor.
    _onnx_bytes_per_value = 4

    def __init__(
        self,
        *,
        portfolio: bool = True,
        oof: bool = True,
        random_state: int = 42,
    ):
        self.portfolio = portfolio
        self.oof = oof
        self.random_state = random_state

    def fit(self, X, y) -> "BaseSVR":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        if self.oof:
            return self.collect_oof(X, y)
        return self._fit_raw(X, y)

    @staticmethod
    def _make(params: dict):
        return _make_svr(params)

    def _linear_fallback_params(self, profile) -> dict:
        return dict(
            svr_linear_params(profile),
            # The C three halvings of the kernel's C=10 used to reach.
            C=min(self.params_["C"] / 8.0, 10.0),
            random_state=self.random_state,
        )

    def _fit_raw(self, X, y) -> "BaseSVR":
        X = check_array(X, dtype="numeric", accept_sparse=False)
        y = np.asarray(y, dtype=float).ravel()
        self._onnx_probe_ = probe_rows(X)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        z = (y - self.y_mean_) / self.y_std_

        profile = _inspect(X, y, task="regression")
        self.profile_ = profile
        self.timing_ = {}

        logger.rule("BaseSVR")
        logger.profile_summary(profile)

        best_name, best_params = "rbf_c10", svr_rbf_params(profile)
        self.portfolio_scores_ = {}
        self.preset_name_ = best_name
        self.params_ = dict(best_params)
        self._use_linear_ = bool(best_params.get("use_linear", False))

        _t_p2 = _time.perf_counter()
        self._fit_final(X, z, profile)
        self.timing_["phase2_refit"] = _time.perf_counter() - _t_p2

        self.n_features_in_ = profile.n_features
        self._bind_onnx()
        self.train_rho_ = spearman_score(y, self.predict(X))
        logger.rule("Done")
        logger.success(
            f"preset={self.preset_name_} | "
            f"{'linear' if self._use_linear_ else 'kernel'} | C={self.params_['C']}"
        )
        return self

    def collect_oof(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseSVR":
        """Full fit (portfolio runs once), then quantile-stratified k-fold OOF."""
        X = check_array(X, dtype="numeric", accept_sparse=False)
        y = np.asarray(y, dtype=float).ravel()
        k, fold_splits = make_quantile_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseSVR.collect_oof: full fit on n={len(y)} (portfolio runs once)"
        )
        self._fit_raw(X, y)

        def fit_predict(train_idx, val_idx):
            y_tr = y[train_idx]
            mu, sd = float(y_tr.mean()), float(y_tr.std()) or 1.0
            est = _make_svr(self.params_).fit(X[train_idx], (y_tr - mu) / sd)
            return est.predict(X[val_idx]) * sd + mu

        self._finish_oof(
            y,
            self._oof_loop(
                fold_splits, fit_predict, n_jobs=_fold_jobs(len(fold_splits), self.params_)
            ),
        )
        return self

    def _raw_native(self, X) -> np.ndarray:
        """Standardised predictions from the scikit-learn estimator itself."""
        X = check_array(X, dtype="numeric", accept_sparse=False)
        return self.svr_.predict(X)

    def _onnx_bytes(self) -> bytes:
        return _svr_onnx_bytes(self.svr_, self.n_features_in_)

    def predict(self, X) -> np.ndarray:
        """Return predictions in original target units, shape (n_samples,)."""
        check_is_fitted(self, "svr_")
        X = check_array(X, dtype="numeric", accept_sparse=False)
        return self._raw(X) * self.y_std_ + self.y_mean_

    def save(self, directory: str) -> None:
        """Write ``svr.onnx`` and ``svr.json`` (fit metadata incl. target scaling)."""
        check_is_fitted(self, "svr_")
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "svr.onnx"))
        metadata = {
            "task": "regression",
            "format": "onnx",
            "preset_name": self.preset_name_,
            "params": self.params_,
            "use_linear": self._use_linear_,
            "n_features_in": self.n_features_in_,
            "portfolio_scores": self.portfolio_scores_,
            "y_mean": self.y_mean_,
            "y_std": self.y_std_,
        }
        metadata.update(self._task_metadata())
        with open(os.path.join(directory, "svr.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# Inference: one implementation per head, in lazyqsar.artifacts.
from lazyqsar.artifacts.svc import SVRArtifact as BaseSVRArtifact  # noqa: E402,F401
