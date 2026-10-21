"""
Sklearn-compatible regressor backed by automatic SVR hyperparameter selection.

The regression twin of BaseSVCClassifier. Training uses sklearn.svm.SVR (RBF
kernel) or sklearn.svm.LinearSVR depending on the dataset profile and winning
preset.

Targets are standardised internally (z-score) before fitting, so that C and
epsilon carry the same meaning on every target scale (pIC50, logS, ...).
Predictions are mapped back to the original units; the ONNX graph emits
standardised values and the artifact applies the inverse transform.

On .fit(), the estimator runs a two-phase portfolio process:

  Phase 1 – Portfolio selection (n >= 200):
    Three preset configurations are evaluated on a 90/10 holdout split
    (stratified on quantile bins of y) and scored by RMSE on the standardised target:
      1. heuristic – rule-based params (same kernel and C rules as the SVC)
      2. default   – sklearn SVR defaults (C=1, RBF, gamma='scale', epsilon=0.1)
      3. linear    – LinearSVR, C scaled by n (compact ONNX, fast on fingerprints)
    Non-default presets win only if they beat the default by at least an
    adaptive threshold max(0.005, coef / sqrt(n)).

  Phase 2 – Retrain the winner on 100% of the data, then the same ONNX size
    guard as the SVC: halve C up to 3 times, else fall back to LinearSVR.

  OOF – Quantile-stratified k-fold with the winning preset (no per-fold
    search) to collect out-of-fold predictions for the ensemble gate.
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
from .params import get_params as _svc_get_params
from ._common import (
    _MAX_COST_MULTIPLIER,
    _RANDOM_STATE,
    _VAL_FRACTION,
    _VAL_MIN_ROWS,
    _SVMHead,
    svm_cost as _svr_cost,
)
from lazyqsar.base._head import RegressorHead, require_fit_deps
from lazyqsar.base._race import min_gain_threshold
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import probe_rows
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.ranking import prepare_knots, rank_from_knots
from lazyqsar.utils.splits import make_quantile_oof_splits, quantile_bins


# Same as the XGBoost preset race: below this many training rows, average the holdout
# score over several random splits.
_STAGE1_MULTI_SPLIT_THRESHOLD = 2_000
_STAGE1_MULTI_SPLITS = 3
# epsilon-tube width in standardised target units (sklearn's SVR default).
_EPSILON = 0.1


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def svr_heuristic_params(profile: DatasetProfile) -> dict:
    """
    Rule-based preset: the SVC kernel and C rules, without class weighting.

    Kernel choice (linear for sparse fingerprints or n > 5000, RBF otherwise)
    and the C schedule transfer directly because y is standardised.
    """
    p = _svc_get_params(profile)
    p.pop("class_weight", None)
    p["epsilon"] = _EPSILON
    return p


def svr_default_params(profile: DatasetProfile) -> dict:
    """sklearn SVR out-of-the-box defaults (C=1, RBF, gamma='scale', epsilon=0.1)."""
    n = profile.n_samples
    return {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "epsilon": _EPSILON,
        "max_iter": max(1_000, min(10_000, n * 5)),
        "tol": 1e-3,
        "random_state": 42,
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


def _min_gain_threshold(n: int) -> float:
    """Relative-RMSE margin a challenger preset must win by; noise scales as 1/sqrt(n)."""
    return min_gain_threshold(n, n)


def _validation_split(X, z: np.ndarray, random_state: int = _RANDOM_STATE):
    """90/10 holdout stratified on quantile bins of the (standardised) target."""
    n = len(z)
    if n < _VAL_MIN_ROWS:
        return X, X, z, z, False
    bins = quantile_bins(z, 5)
    try:
        X_tr, X_val, z_tr, z_val = train_test_split(
            X, z, test_size=_VAL_FRACTION, stratify=bins, random_state=random_state
        )
        return X_tr, X_val, z_tr, z_val, True
    except ValueError:
        return X, X, z, z, False


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _portfolio_select_svr(
    X, z: np.ndarray, profile: DatasetProfile, random_state: int = _RANDOM_STATE
):
    """Fit each preset on 90%, score RMSE on the standardised 10%, pick the best.

    Scored by RMSE -- the metric every regression head's preset race uses -- on the
    standardised target, so the scale is comparable across datasets. Below
    ``_STAGE1_MULTI_SPLIT_THRESHOLD`` training rows the score is averaged over several
    random splits: a 10% holdout of a few hundred molecules is too noisy to rank
    presets on (the same reasoning, and threshold, as the XGBoost race).

    A non-default preset must beat the default by a *relative* RMSE margin of
    ``_min_gain_threshold(n)``. Kernel presets are no longer skipped up front on a
    worst-case ONNX size: the post-fit support-vector guard in ``_fit_raw`` handles
    size on the model that was actually fitted.
    """
    n_features = profile.n_features
    candidates = [
        ("heuristic", svr_heuristic_params(profile)),
        ("default", svr_default_params(profile)),
        ("linear", svr_linear_params(profile)),
    ]
    first = _validation_split(X, z, random_state)
    if not first[-1]:
        return "heuristic", svr_heuristic_params(profile), {}

    n_tr = len(first[2])
    default_cost = _svr_cost(dict(candidates)["default"], n_tr, n_features)
    budget = _MAX_COST_MULTIPLIER * default_cost
    n_splits = _STAGE1_MULTI_SPLITS if n_tr < _STAGE1_MULTI_SPLIT_THRESHOLD else 1
    splits = [first] + [
        _validation_split(X, z, random_state + 97 * k) for k in range(1, n_splits)
    ]

    logger.rule("SVR Portfolio — Stage 1")
    scores: dict = {}
    for name, params in candidates:
        cost = _svr_cost(params, n_tr, n_features)
        if name not in ("default", "heuristic") and cost > budget:
            scores[name] = float("nan")
            continue
        try:
            _t = _time.perf_counter()
            rmses = []
            for X_tr, X_val, z_tr, z_val, _ in splits:
                est = _make_svr(params).fit(X_tr, z_tr)
                rmses.append(_rmse(z_val, est.predict(X_val)))
            scores[name] = -float(np.mean(rmses))
            logger.info(
                f"[portfolio] {name:12s}: rmse={-scores[name]:.4f}  "
                f"splits={len(splits)}  t={_time.perf_counter() - _t:.1f}s  "
                f"{'linear' if params.get('use_linear') else 'kernel'}"
            )
        except Exception as exc:
            logger.warning(f"[portfolio] {name}: failed ({exc})")
            scores[name] = float("nan")

    default_score = scores.get("default", float("nan"))
    best_name, best_score = None, float("-inf")
    for name, _ in candidates:
        s = scores.get(name, float("nan"))
        if s == s and s > best_score:
            best_name, best_score = name, s
    if best_name is None:
        best_name = "default"
    elif best_name != "default" and default_score == default_score:
        relative_gain = (best_score - default_score) / max(abs(default_score), 1e-12)
        if relative_gain < _min_gain_threshold(n_tr):
            best_name = "default"
    logger.info(f"Portfolio winner: {best_name}")
    return best_name, dict(candidates)[best_name], scores


def _svr_onnx_bytes(svr, n_features: int) -> bytes:
    """SVR or LinearSVR as a serialised ONNX graph (standardised output)."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    return convert_sklearn(
        svr, initial_types=initial_types, target_opset=15
    ).SerializeToString()


# ---------------------------------------------------------------------------
# BaseSVR
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BaseSVR
# ---------------------------------------------------------------------------


class BaseSVR(_SVMHead, RegressorHead, BaseEstimator, RegressorMixin):
    """
    Regressor with automatically selected SVR hyperparameters.

    Parameters
    ----------
    portfolio : bool
        If True (default), evaluate the preset configurations on a holdout
        split and select the best. If False, use the heuristic preset only.
    oof : bool
        If True (default), collect quantile-stratified OOF predictions.
    random_state : int
        Random seed for reproducibility.

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict
    preset_name_ : str         — "heuristic", "default" or "linear" (+ "+linear_guard")
    portfolio_scores_ : dict   — negative holdout RMSE (standardised) per preset
    svr_ : SVR or LinearSVR    — fitted on standardised targets
    y_mean_, y_std_ : float    — target standardisation
    oof_preds_ : ndarray (n,)  — OOF predictions in original units
    oof_rho_ : float or None
    train_rho_ : float or None
    timing_ : dict
    """

    _fitted_attr = "svr_"

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
            C=min(self.params_["C"], 10.0),
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

        _t_ps = _time.perf_counter()
        if profile.n_samples >= _VAL_MIN_ROWS and self.portfolio:
            best_name, best_params, scores = _portfolio_select_svr(
                X, z, profile, random_state=self.random_state
            )
            self.portfolio_scores_ = scores
        else:
            best_name, best_params = "heuristic", svr_heuristic_params(profile)
            self.portfolio_scores_ = {}
        self.timing_["portfolio_select"] = _time.perf_counter() - _t_ps

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

        self._finish_oof(y, self._oof_loop(fold_splits, fit_predict))
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


class BaseSVRArtifact:
    """
    Inference-only loader for a model saved by BaseSVR.save().

    Only requires numpy and onnxruntime. Applies the inverse target
    standardisation to the ONNX output.
    """

    def __init__(self):
        self._session = None
        self._input_name = ""
        self.metadata: dict = {}
        self.task = "regression"
        self._ranker = None
        self._y_mean = 0.0
        self._y_std = 1.0

    @classmethod
    def load(cls, directory: str) -> "BaseSVRArtifact":
        json_path = os.path.join(directory, "svr.json")
        onnx_path = os.path.join(directory, "svr.onnx")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"svr.json not found in {directory!r}")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"svr.onnx not found in {directory!r}")
        self = cls()
        with open(json_path) as f:
            self.metadata = json.load(f)
        self._ranker = self.metadata.get("ranker", None)
        self._y_mean = float(self.metadata["y_mean"])
        self._y_std = float(self.metadata["y_std"])
        import onnxruntime as rt

        self._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        return self

    def run(self, X) -> np.ndarray:
        """Return predictions in original target units, shape (n_samples,)."""
        X_f32 = np.asarray(X, dtype=np.float32)
        out = self._session.run(None, {self._input_name: X_f32})[0]
        return np.asarray(out, dtype=np.float64).ravel() * self._y_std + self._y_mean

    def predict_rank(self, X) -> np.ndarray:
        """Map predictions to [0, 1] via the OOF ECDF, shape (n_samples,)."""
        if self._ranker is None:
            raise RuntimeError("No ranker stored in this artifact.")
        if getattr(self, "_ranker_prepared", None) is None:
            self._ranker_prepared = prepare_knots(self._ranker["knots"])
        return rank_from_knots(self.run(X), prepared=self._ranker_prepared)
