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
    (stratified on quantile bins of y) and scored by Spearman ρ:
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
from lazyqsar.base._onnx_binding import OnnxBoundHead
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import check_values, probe_rows
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.ranking import prepare_knots, rank_from_knots, subsample_knots
from lazyqsar.utils.splits import make_quantile_oof_splits, quantile_bins


_VAL_FRACTION = 0.1
_VAL_MIN_ROWS = 200
_PORTFOLIO_MIN_GAIN = 0.005
_MAX_COST_MULTIPLIER = 20
_ONNX_MAX_BYTES = 5_000_000
_ONNX_MAX_RETRIES = 3
_RANKER_MAX_KNOTS = 10_000
_RANDOM_STATE = 42
# epsilon-tube width in standardised target units (sklearn's SVR default).
_EPSILON = 0.1


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

    return SVR(
        C=params["C"],
        epsilon=params.get("epsilon", _EPSILON),
        kernel=params.get("kernel", "rbf"),
        gamma=params.get("gamma", "scale"),
        max_iter=params.get("max_iter", -1),
        tol=params.get("tol", 1e-3),
    )


def _svr_cost(params: dict, n_tr: int, n_features: int) -> float:
    if params.get("use_linear", False):
        return float(n_tr * n_features)
    return float(n_tr**2 * n_features)


def _min_gain_threshold(n: int) -> float:
    coef = 0.3 if n < 2_000 else 0.1
    return max(_PORTFOLIO_MIN_GAIN, coef / max(1, n) ** 0.5)


def _onnx_size_bytes(svr, n_features: int) -> int:
    if not hasattr(svr, "support_vectors_"):
        return 0
    return svr.support_vectors_.shape[0] * n_features * 4


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


def _portfolio_select_svr(
    X, z: np.ndarray, profile: DatasetProfile, random_state: int = _RANDOM_STATE
):
    """Single-stage portfolio: fit each preset on 90%, score Spearman ρ on 10%."""
    n_features = profile.n_features
    candidates = [
        ("heuristic", svr_heuristic_params(profile)),
        ("default", svr_default_params(profile)),
        ("linear", svr_linear_params(profile)),
    ]
    X_tr, X_val, z_tr, z_val, did_split = _validation_split(X, z, random_state)
    if not did_split:
        return "heuristic", svr_heuristic_params(profile), {}

    n_tr = len(z_tr)
    default_cost = _svr_cost(dict(candidates)["default"], n_tr, n_features)
    budget = _MAX_COST_MULTIPLIER * default_cost

    logger.rule("SVR Portfolio — Stage 1")
    scores: dict = {}
    for name, params in candidates:
        if not params.get("use_linear", False):
            worst_case = profile.n_samples * n_features * 4
            if worst_case > _ONNX_MAX_BYTES:
                logger.debug(
                    f"[portfolio] {name:12s}: skipped (worst-case ONNX "
                    f"{worst_case / 1e6:.1f} MB > {_ONNX_MAX_BYTES / 1e6:.0f} MB)"
                )
                scores[name] = float("nan")
                continue
        cost = _svr_cost(params, n_tr, n_features)
        if name not in ("default", "heuristic") and cost > budget:
            scores[name] = float("nan")
            continue
        try:
            _t = _time.perf_counter()
            est = _make_svr(params)
            est.fit(X_tr, z_tr)
            rho = spearman_score(z_val, est.predict(X_val))
            scores[name] = float("nan") if rho is None else rho
            logger.info(
                f"[portfolio] {name:12s}: rho={scores[name]:.4f}  "
                f"t={_time.perf_counter() - _t:.1f}s  "
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
        if best_score - default_score < _min_gain_threshold(n_tr):
            best_name = "default"
    logger.info(f"Portfolio winner: {best_name}")
    return best_name, dict(candidates)[best_name], scores


def _to_onnx(svr, path: str, n_features: int) -> None:
    """Export SVR or LinearSVR to ONNX; single output of standardised predictions."""
    with open(path, "wb") as f:
        f.write(_svr_onnx_bytes(svr, n_features))


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


class BaseSVR(OnnxBoundHead, BaseEstimator, RegressorMixin):
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
    portfolio_scores_ : dict   — holdout Spearman ρ per preset
    svr_ : SVR or LinearSVR    — fitted on standardised targets
    y_mean_, y_std_ : float    — target standardisation
    oof_preds_ : ndarray (n,)  — OOF predictions in original units
    oof_rho_ : float or None
    train_rho_ : float or None
    timing_ : dict
    """

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
        if not _FIT_DEPS_AVAILABLE:
            raise ImportError(
                "Training requires scikit-learn. Install with: pip install 'lazyqsar[fit]'"
            )
        if self.oof:
            return self.collect_oof(X, y)
        return self._fit_raw(X, y)

    def _fit_raw(self, X, y) -> "BaseSVR":
        X = check_array(X, dtype="numeric", accept_sparse=False)
        y = np.asarray(y, dtype=float).ravel()
        self._onnx_probe_ = probe_rows(X)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        z = (y - self.y_mean_) / self.y_std_

        profile = _inspect(X, y, task="regression")
        self.profile_ = profile
        n_features = profile.n_features
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
        self.svr_ = _make_svr(self.params_).fit(X, z)

        # ONNX size guard (kernel SVR only), as in BaseSVCClassifier.
        if not self._use_linear_:
            for _retry in range(_ONNX_MAX_RETRIES):
                size = _onnx_size_bytes(self.svr_, n_features)
                if size <= _ONNX_MAX_BYTES:
                    break
                new_C = self.params_["C"] / 2.0
                logger.warning(
                    f"ONNX size guard: {size / 1e6:.1f} MB > "
                    f"{_ONNX_MAX_BYTES / 1e6:.0f} MB — halving C: "
                    f"{self.params_['C']:.4g} → {new_C:.4g}"
                )
                self.params_["C"] = new_C
                self.svr_ = _make_svr(self.params_).fit(X, z)
            else:
                size = _onnx_size_bytes(self.svr_, n_features)
                if size > _ONNX_MAX_BYTES:
                    logger.warning(
                        f"ONNX size guard exhausted after {_ONNX_MAX_RETRIES} retries; "
                        "switching to LinearSVR."
                    )
                    self.params_ = dict(
                        svr_linear_params(profile),
                        C=min(self.params_["C"], 10.0),
                        random_state=self.random_state,
                    )
                    self.svr_ = _make_svr(self.params_).fit(X, z)
                    self._use_linear_ = True
                    self.preset_name_ = self.preset_name_ + "+linear_guard"
        self.timing_["phase2_refit"] = _time.perf_counter() - _t_p2

        self.n_features_in_ = n_features
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
        n = len(y)
        k, fold_splits = make_quantile_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(f"BaseSVR.collect_oof: full fit on n={n} (portfolio runs once)")
        self._fit_raw(X, y)

        oof = np.full(n, np.nan, dtype=float)
        fold_times = []
        for train_idx, val_idx in fold_splits:
            _t_fold = _time.perf_counter()
            y_tr = y[train_idx]
            mu, sd = float(y_tr.mean()), float(y_tr.std()) or 1.0
            est = _make_svr(self.params_).fit(X[train_idx], (y_tr - mu) / sd)
            oof[val_idx] = est.predict(X[val_idx]) * sd + mu
            fold_times.append(_time.perf_counter() - _t_fold)

        self.timing_["oof_folds"] = fold_times
        self.timing_["oof_total"] = sum(fold_times)
        self.oof_preds_ = oof
        self.oof_y_ = y.copy()
        self.oof_rho_ = spearman_score(y, oof)
        self._ranker_knots = subsample_knots(np.sort(oof), _RANKER_MAX_KNOTS)
        self._ranker_prepared_ = prepare_knots(self._ranker_knots)
        logger.success(
            f"OOF collected ({k} folds) | rho_oof={self.oof_rho_} | "
            f"rho_train={self.train_rho_}"
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

    def predict_rank(self, X) -> np.ndarray:
        """Map predictions to [0, 1] via the OOF ECDF, shape (n_samples,)."""
        check_is_fitted(self, "_ranker_knots")
        prepared = getattr(self, "_ranker_prepared_", None)
        if prepared is None:
            prepared = self._ranker_prepared_ = prepare_knots(self._ranker_knots)
        return rank_from_knots(self.predict(X), prepared=prepared)

    def to_onnx(self, path: str) -> None:
        """Export to ONNX. The graph emits standardised predictions (see y_mean/y_std)."""
        check_is_fitted(self, "svr_")
        self._write_onnx(path)
        probe = getattr(self, "_onnx_probe_", None)
        if probe is not None:
            check_values(path, probe, self._raw_native(probe), "BaseSVR")

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
            "oof_rho": getattr(self, "oof_rho_", None),
            "train_rho": getattr(self, "train_rho_", None),
        }
        if hasattr(self, "_ranker_knots"):
            metadata["ranker"] = {"knots": self._ranker_knots.tolist()}
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
