"""
Sklearn-compatible estimator backed by automatic SVC hyperparameter selection.

Training uses sklearn.svm.SVC (kernel) or sklearn.svm.LinearSVC depending on
the dataset profile and winning preset.

On .fit(), the estimator runs a two-phase portfolio process:

  Phase 1 – Portfolio selection (when a genuine validation split exists, n >= 200):
    Four preset configurations are evaluated on a 90/10 validation split scored
    by AUC (ROC):
      1. heuristic   – rule-based params from params.py (dataset-profiling based)
      2. default     – sklearn SVC defaults (C=1, RBF, gamma='scale')
      3. linear      – linear kernel (compact ONNX, fast for fingerprints)
      4. balanced_rbf – class-weighted RBF with C scaled by minority count

    Non-default presets win only if they beat the default by at least
    _min_gain_threshold (noise-adaptive, same formula as XGB).

  Phase 2 – Retrain the winner on 100% of the training data.

  Fallback – When n < 200, the heuristic preset is used directly.

ONNX export notes
-----------------
  LinearSVC → always compact: ONNX stores only weights + bias (O(n_features)).
  Kernel SVC → ONNX stores all support vectors (O(n_sv * n_features)).

  ONNX size pre-flight: kernel presets are skipped during portfolio evaluation
    when n_samples × n_features × 4 bytes already exceeds _ONNX_MAX_BYTES.
    This avoids fitting models that cannot be exported within budget.

  After phase 2, the ONNX size guard checks kernel SVC models:
    if n_sv * n_features * 4 > _ONNX_MAX_BYTES (5 MB):
      Halve C and refit, up to _ONNX_MAX_RETRIES (3) times.
      If still too large: retrain with LinearSVC (always fits in budget).

  The ONNX model outputs decision function scores (NOT calibrated probabilities).
  At inference, sigmoid is applied to map scores to [0, 1], then the OOF-fitted
  calibrator is applied to obtain final calibrated class probabilities.

The winning preset name is stored in .preset_name_ after .fit().
The chosen parameters are accessible via .params_.
"""

from __future__ import annotations

import json
import os
import time as _time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted

from .inspector import inspect as _inspect, DatasetProfile
from .params import get_params as _get_params
from .presets import (
    svc_heuristic_params,
    svc_default_params,
    svc_linear_params,
    svc_balanced_rbf_params,
)
from lazyqsar.utils.logging import logger
from lazyqsar.utils.splits import make_stratified_oof_splits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VAL_FRACTION    = 0.1
_VAL_MIN_ROWS    = 200
_VAL_MIN_MINORITY = 15

# Minimum AUC gain over default for a non-default preset to win portfolio
_PORTFOLIO_MIN_GAIN = 0.005

# Skip preset if cost > this multiple of the default preset cost
_MAX_COST_MULTIPLIER = 20

# ONNX size budget in bytes (5 MB).  Only applies to kernel SVC; LinearSVC
# is always negligibly small (O(n_features) weights only).
_ONNX_MAX_BYTES = 5_000_000
_ONNX_MAX_RETRIES = 3

_CALIBRATION_ISOTONIC_MIN_MINORITY = 500
_DEFAULT_DECISION_CUTOFF = 0.5
_RANKER_MAX_KNOTS = 10_000
_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _make_svc(params: dict):
    """
    Build a (not yet fitted) sklearn SVC or LinearSVC from a preset params dict.

    The 'use_linear' key controls which estimator is created; it is not passed
    to the sklearn estimator.  For SVC, probability=False is always used so
    that the ONNX model exports raw decision function values; calibration is
    applied externally.
    """
    use_linear = params.get("use_linear", False)
    common = {
        "C": params["C"],
        "class_weight": params.get("class_weight", "balanced"),
        "max_iter": params.get("max_iter", 5_000),
        "tol": params.get("tol", 1e-3),
    }
    if use_linear:
        from sklearn.svm import LinearSVC
        return LinearSVC(
            **common,
            random_state=params.get("random_state", 42),
            dual="auto",
        )
    else:
        from sklearn.svm import SVC
        return SVC(
            **common,
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            probability=False,   # export raw scores; calibration is external
            random_state=params.get("random_state", 42),
        )


def _decision_scores(svc, X: np.ndarray) -> np.ndarray:
    """Return shape-(n,) decision function scores; positive → class 1."""
    scores = svc.decision_function(X)
    return np.asarray(scores, dtype=np.float64).ravel()


def _svc_cost(params: dict, n_tr: int, n_features: int) -> float:
    """
    Estimate relative training cost for cost-budget filtering.

    LinearSVC:  O(n * p)   → n_tr * n_features
    Kernel SVC: O(n^2 * p) → n_tr^2 * n_features (approximation)
    """
    if params.get("use_linear", False):
        return float(n_tr * n_features)
    return float(n_tr ** 2 * n_features)


def _min_gain_threshold(profile: DatasetProfile, y_train: np.ndarray) -> float:
    """
    Adaptive minimum-gain threshold for portfolio selection (same formula as XGB).

    Formula: max(_PORTFOLIO_MIN_GAIN, coef / sqrt(n_effective))
      classification: n_effective = minority count in training set
    """
    n_eff = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    coef = 0.3 if len(y_train) < 2_000 else 0.1
    return max(_PORTFOLIO_MIN_GAIN, coef / max(1, n_eff) ** 0.5)


def _validation_split(X, y: np.ndarray, profile: DatasetProfile,
                      random_state: int = _RANDOM_STATE):
    """
    Create a stratified 90/10 holdout split.

    Returns (X_tr, X_val, y_tr, y_val, did_split).
    did_split=False means n was too small; train == full.
    The validation fraction is raised when the minority class is small so that
    at least _VAL_MIN_MINORITY minority samples appear in the validation set.
    """
    n = len(y)
    if n < _VAL_MIN_ROWS:
        return X, X, y, y, False
    minority_n = int(np.bincount(np.asarray(y, dtype=int)).min())
    val_frac = max(_VAL_FRACTION, _VAL_MIN_MINORITY / max(minority_n, 1))
    val_frac = min(val_frac, 0.5)
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_frac, stratify=y, random_state=random_state
        )
        # Safety: ensure enough minority samples on both sides
        if np.bincount(np.asarray(y_val, dtype=int)).min() < 2:
            return X, X, y, y, False
        return X_tr, X_val, y_tr, y_val, True
    except ValueError:
        return X, X, y, y, False


def _learn_balanced_accuracy_cutoff(y_true: np.ndarray,
                                    p1: np.ndarray) -> tuple[float, str]:
    """Learn a decision threshold that maximises balanced accuracy on OOF scores."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(p1, dtype=float)
    mask = np.isfinite(p_arr)
    if mask.sum() == 0 or len(np.unique(y_arr[mask])) < 2:
        return _DEFAULT_DECISION_CUTOFF, "default_0.5"
    probs = p_arr[mask]
    labels = y_arr[mask]
    unique = np.unique(probs)
    if unique.size == 0:
        return _DEFAULT_DECISION_CUTOFF, "default_0.5"
    candidates = np.unique(np.concatenate([
        unique,
        np.array([np.nextafter(unique[0], -np.inf),
                  _DEFAULT_DECISION_CUTOFF,
                  np.nextafter(unique[-1], np.inf)], dtype=float),
    ]))
    best_threshold = _DEFAULT_DECISION_CUTOFF
    best_key = None
    for thr in candidates:
        score = balanced_accuracy_score(labels, (probs >= thr).astype(int))
        key = (-float(score), abs(float(thr) - _DEFAULT_DECISION_CUTOFF), float(thr))
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(thr)
    return best_threshold, "oof_balanced_accuracy"


def _apply_calibrator_artifact(proba: np.ndarray, cal: dict) -> np.ndarray:
    raw_p1 = np.asarray(proba[:, 1], dtype=np.float64)
    if cal["method"] == "isotonic":
        p1 = np.clip(
            np.interp(raw_p1, cal["X_thresholds"], cal["y_thresholds"]),
            0, 1,
        )
    else:
        A, B = cal["coef"], cal["intercept"]
        p1 = 1.0 / (1.0 + np.exp(-(A * raw_p1 + B)))
    return np.column_stack([1 - p1, p1])


def _portfolio_select_svc(X, y: np.ndarray, profile: DatasetProfile,
                          random_state: int = _RANDOM_STATE):
    """
    Single-stage portfolio selection for SVC.

    Unlike XGB, SVC has no 'n_estimators' / early stopping — each fit is a
    single call.  We therefore run a simpler single-stage evaluation:

      1. Split 90/10 (stratified).
      2. Fit each preset on the 90% split and score AUC on the 10% holdout.
         Presets whose estimated cost > _MAX_COST_MULTIPLIER × default cost
         are skipped ('default' and 'heuristic' are never skipped).
      3. Pick winner.  Non-default presets must beat the default by at least
         _min_gain_threshold(profile, y_tr) to avoid fitting on noise.

    Returns (best_name, best_params, scores_dict).
    """
    n_features = profile.n_features
    candidates = [
        ("heuristic",    svc_heuristic_params(profile)),
        ("default",      svc_default_params(profile)),
        ("linear",       svc_linear_params(profile)),
        ("balanced_rbf", svc_balanced_rbf_params(profile)),
    ]

    X_tr, X_val, y_tr, y_val, did_split = _validation_split(X, y, profile,
                                                              random_state=random_state)
    if not did_split:
        # Too small: use heuristic directly
        params = svc_heuristic_params(profile)
        return "heuristic", params, {}

    n_tr = len(y_tr)
    default_cost = _svc_cost(dict(candidates)[1][1] if False else
                             next(p for n, p in candidates if n == "default"),
                             n_tr, n_features)
    budget = _MAX_COST_MULTIPLIER * default_cost

    logger.rule("SVC Portfolio — Stage 1")
    logger.info(
        f"n_tr={n_tr} | n_features={n_features} | "
        f"budget={budget:.2e} ({_MAX_COST_MULTIPLIER}× default cost={default_cost:.2e})"
    )

    scores: dict = {}
    skipped: list = []

    for name, params in candidates:
        # Pre-flight ONNX size check: skip kernel presets whose worst-case ONNX
        # (all n_samples as support vectors) already exceeds the budget.
        # Uses profile.n_samples (full dataset) because the guard targets the
        # final model, not just the 90% validation split.
        if not params.get("use_linear", False):
            worst_case = profile.n_samples * n_features * 4
            if worst_case > _ONNX_MAX_BYTES:
                logger.debug(
                    f"[portfolio] {name:12s}: skipped "
                    f"(worst-case ONNX {worst_case / 1e6:.1f} MB "
                    f"> budget {_ONNX_MAX_BYTES / 1e6:.0f} MB)"
                )
                scores[name] = float("nan")
                skipped.append(name)
                continue

        # Cost filter: skip non-mandatory presets that exceed the budget
        cost = _svc_cost(params, n_tr, n_features)
        if name not in ("default", "heuristic") and cost > budget:
            logger.debug(
                f"[portfolio] {name:12s}: skipped "
                f"(cost={cost:.2e} > budget={budget:.2e})"
            )
            scores[name] = float("nan")
            skipped.append(name)
            continue

        try:
            _t = _time.perf_counter()
            svc = _make_svc(params)
            svc.fit(X_tr, y_tr)
            df_val = _decision_scores(svc, X_val)
            auc = roc_auc_score(y_val, df_val)
            scores[name] = auc
            elapsed = _time.perf_counter() - _t
            use_linear = params.get("use_linear", False)
            logger.info(
                f"[portfolio] {name:12s}: AUC={auc:.4f}  "
                f"cost={cost:.2e}  t={elapsed:.1f}s  "
                f"{'linear' if use_linear else 'kernel'}"
            )
        except Exception as exc:
            logger.warning(f"[portfolio] {name}: failed ({exc})")
            scores[name] = float("nan")

    # Pick winner
    default_score = scores.get("default", float("-inf"))
    best_name = None
    best_score = float("-inf")
    for name, _ in candidates:
        s = scores.get(name, float("nan"))
        if s != s:  # nan
            continue
        if s > best_score:
            best_score = s
            best_name = name

    if best_name is None:
        best_name = "default"
        best_score = float("nan")
    elif best_name != "default":
        gain = best_score - default_score
        threshold = _min_gain_threshold(profile, y_tr)
        if gain < threshold:
            logger.info(
                f"[portfolio] {best_name} gain={gain:+.4f} < "
                f"threshold={threshold:.4f} → default wins"
            )
            best_name = "default"
            best_score = default_score

    logger.info(f"Portfolio winner: {best_name}  (AUC={best_score:.4f})")
    best_params = next(p for n, p in candidates if n == best_name)
    return best_name, best_params, scores


def _onnx_size_bytes(svc, n_features: int) -> int:
    """Estimate ONNX size for kernel SVC (support vectors × float32)."""
    if not hasattr(svc, "support_vectors_"):
        return 0  # LinearSVC: negligible size
    n_sv = svc.support_vectors_.shape[0]
    return n_sv * n_features * 4


def _to_onnx(svc, path: str, n_features: int) -> None:
    """
    Export SVC or LinearSVC to ONNX via skl2onnx.

    The exported model outputs raw decision function scores (NOT probabilities).
    At inference, sigmoid is applied in the artifact to obtain [0,1] scores,
    then the stored calibrator converts these to calibrated probabilities.

    Output layout (opset 15):
      output[0]: int64 labels, shape (n,)
      output[1]: float32 scores — column 1 (or the only column) = class-1 score

    Options differ by estimator type:
      SVC (kernel)  — zipmap=False: suppresses dict output, returns float array
      LinearSVC     — raw_scores=True: returns decision function scores (not labels)
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.svm import LinearSVC as _LinearSVC

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    if isinstance(svc, _LinearSVC):
        options = {id(svc): {"raw_scores": True}}
    else:
        options = {id(svc): {"zipmap": False}}
    import onnx
    onnx_model = convert_sklearn(
        svc,
        initial_types=initial_types,
        target_opset=15,
        options=options,
    )
    for output in onnx_model.graph.output:
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                dim.ClearField("dim_value")
                dim.dim_param = "batch_size"
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


# ---------------------------------------------------------------------------
# BaseSVCClassifier
# ---------------------------------------------------------------------------

class BaseSVCClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary classifier with automatically selected SVC hyperparameters.

    Parameters
    ----------
    portfolio : bool
        If True (default), evaluate four preset configurations on a holdout
        split and select the best.  If False, use the heuristic preset only.
    calibrated : bool
        If True (default), run the full calibration workflow: portfolio
        selection once, k-fold OOF to collect held-out decision scores,
        Platt/isotonic calibrator.
    random_state : int
        Random seed for reproducibility.

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict             — winning preset's hyperparameters
    preset_name_ : str         — "heuristic", "default", "linear", or "balanced_rbf"
    portfolio_scores_ : dict   — validation AUC per preset (empty when portfolio=False)
    svc_ : SVC or LinearSVC    — fitted sklearn estimator
    classes_ : ndarray
    _use_linear_ : bool        — True when LinearSVC was used
    calibrator_ : fitted calibrator (if calibrated=True)
    calibrator_method_ : str   — "isotonic" or "platt"
    oof_probas_ : ndarray      — calibrated OOF class-1 probabilities
    oof_y_ : ndarray
    decision_cutoff_ : float
    decision_cutoff_source_ : str
    timing_ : dict
    """

    def __init__(
        self,
        *,
        portfolio: bool = True,
        calibrated: bool = True,
        random_state: int = 42,
    ):
        self.portfolio = portfolio
        self.calibrated = calibrated
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the classifier (full calibration workflow when calibrated=True)."""
        if self.calibrated:
            y_arr = np.asarray(y, dtype=int)
            if np.bincount(y_arr).min() >= 2:
                return self.calibrate(X, y)
        return self._fit_raw(X, y)

    # ------------------------------------------------------------------
    # Internal raw training
    # ------------------------------------------------------------------

    def _fit_raw(self, X, y) -> "BaseSVCClassifier":
        X = check_array(X, dtype="numeric", accept_sparse=False)
        y = np.asarray(y, dtype=int).ravel()
        profile = _inspect(X, y, task="classification")
        self.profile_ = profile
        n_features = profile.n_features
        self.timing_ = {}

        logger.rule("BaseSVCClassifier")
        logger.profile_summary(profile)
        logger.info(f"portfolio={self.portfolio}")

        _t_ps = _time.perf_counter()
        if profile.n_samples >= _VAL_MIN_ROWS and self.portfolio:
            best_name, best_params, scores = _portfolio_select_svc(
                X, y, profile, random_state=self.random_state
            )
            self.timing_["portfolio_select"] = _time.perf_counter() - _t_ps
            self.portfolio_scores_ = scores
        else:
            best_params = svc_heuristic_params(profile)
            best_name = "heuristic"
            self.portfolio_scores_ = {}
            self.timing_["portfolio_select"] = 0.0

        self.preset_name_ = best_name
        self.params_ = dict(best_params)
        self._use_linear_ = bool(best_params.get("use_linear", False))

        # Phase 2: retrain on full data
        _t_p2 = _time.perf_counter()
        logger.rule("Phase 2 — full retraining")
        logger.info(
            f"preset={best_name} | n={profile.n_samples} | "
            f"{'linear' if self._use_linear_ else 'kernel'} | C={best_params['C']}"
        )
        svc = _make_svc(self.params_)
        svc.fit(X, y)
        self.svc_ = svc
        self.timing_["phase2_refit"] = _time.perf_counter() - _t_p2

        # ONNX size guard (kernel SVC only).
        # Halve C up to _ONNX_MAX_RETRIES times; if still over budget, switch to
        # LinearSVC which is O(n_features) and always fits within any budget.
        if not self._use_linear_:
            for _retry in range(_ONNX_MAX_RETRIES):
                size = _onnx_size_bytes(self.svc_, n_features)
                if size <= _ONNX_MAX_BYTES:
                    break
                new_C = self.params_["C"] / 2.0
                logger.warning(
                    f"ONNX size guard: {size / 1e6:.1f} MB > "
                    f"{_ONNX_MAX_BYTES / 1e6:.0f} MB — "
                    f"halving C: {self.params_['C']:.4g} → {new_C:.4g}"
                )
                self.params_["C"] = new_C
                new_svc = _make_svc(self.params_)
                new_svc.fit(X, y)
                self.svc_ = new_svc
            else:
                size = _onnx_size_bytes(self.svc_, n_features)
                if size > _ONNX_MAX_BYTES:
                    logger.warning(
                        f"ONNX size guard exhausted after {_ONNX_MAX_RETRIES} retries "
                        f"({size / 1e6:.1f} MB > {_ONNX_MAX_BYTES / 1e6:.0f} MB). "
                        f"Switching to LinearSVC."
                    )
                    linear_params = {
                        "use_linear": True,
                        "C": min(self.params_["C"], 10.0),
                        "class_weight": self.params_.get("class_weight", "balanced"),
                        "max_iter": self.params_.get("max_iter", 5_000),
                        "tol": self.params_.get("tol", 1e-3),
                        "random_state": self.random_state,
                    }
                    linear_svc = _make_svc(linear_params)
                    linear_svc.fit(X, y)
                    self.svc_ = linear_svc
                    self.params_ = linear_params
                    self._use_linear_ = True
                    self.preset_name_ = self.preset_name_ + "+linear_guard"

        self.decision_cutoff_raw_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_raw_source_ = "default_0.5"
        self.decision_cutoff_proba_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_rank_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_logit_ = 0.0
        self.classes_ = np.array([0, 1])
        logger.rule("Done")
        logger.success(
            f"preset={self.preset_name_} | "
            f"n_sv={'N/A (linear)' if self._use_linear_ else self.svc_.support_vectors_.shape[0]}"
        )
        logger.info(
            f"decision cutoff: {self.decision_cutoff_raw_:.4f} | "
            f"source={self.decision_cutoff_raw_source_}"
        )
        return self

    # ------------------------------------------------------------------
    # Calibration (OOF + Platt/isotonic)
    # ------------------------------------------------------------------

    def calibrate(self, X, y, n_splits: int | None = None,
                  random_state: int = 42) -> "BaseSVCClassifier":
        """
        Full calibration workflow.

        Portfolio selection runs ONCE via _fit_raw().  k-fold OOF then reuses
        the winning params without re-running portfolio comparison per fold.
        The calibrator is fitted on sigmoid(decision_function) OOF scores.
        """
        X = check_array(X, dtype="numeric", accept_sparse=False)
        y = np.asarray(y, dtype=int).ravel()
        n = len(y)
        k, fold_splits = make_stratified_oof_splits(y, n_splits=n_splits,
                                                     random_state=random_state)

        logger.info(
            f"BaseSVCClassifier.calibrate: full fit on n={n} (portfolio runs once)"
        )
        self._fit_raw(X, y)  # sets self.params_, self.svc_, self.preset_name_

        # k-fold OOF: collect raw decision function scores
        oof_raw = np.full(n, np.nan, dtype=float)
        logger.info(
            f"calibrate: {k}-fold OOF  preset={self.preset_name_}  "
            f"{'linear' if self._use_linear_ else 'kernel'}"
        )
        fold_times = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            logger.debug(
                f"  Fold {fold_idx + 1}/{k}: train={len(train_idx)}  val={len(val_idx)}"
            )
            _t_fold = _time.perf_counter()
            fold_svc = _make_svc(self.params_)
            fold_svc.fit(X[train_idx], y[train_idx])
            oof_raw[val_idx] = _decision_scores(fold_svc, X[val_idx])
            fold_times.append(_time.perf_counter() - _t_fold)

        self.timing_["calibration_folds"] = fold_times
        self.timing_["calibration_total"] = sum(fold_times)

        # Convert decision function scores to [0,1] for calibrator fitting
        oof_sigmoid = _sigmoid(oof_raw)

        # Fit calibrator
        minority_count = int(np.bincount(y).min())
        if minority_count >= _CALIBRATION_ISOTONIC_MIN_MINORITY:
            cal = IsotonicRegression(out_of_bounds="clip")
            self.oof_probas_ = cal.fit_transform(oof_sigmoid, y)
            self.calibrator_method_ = "isotonic"
        else:
            cal = LogisticRegression(C=1.0, solver="lbfgs")
            cal.fit(oof_sigmoid.reshape(-1, 1), y)
            self.oof_probas_ = cal.predict_proba(oof_sigmoid.reshape(-1, 1))[:, 1]
            self.calibrator_method_ = "platt"
        self.calibrator_ = cal
        self.decision_cutoff_raw_, self.decision_cutoff_raw_source_ = (
            _learn_balanced_accuracy_cutoff(y, oof_sigmoid)
        )
        if self.calibrator_method_ == "isotonic":
            self.decision_cutoff_proba_ = float(np.clip(
                self.calibrator_.predict(np.array([self.decision_cutoff_raw_]))[0], 0.0, 1.0
            ))
        else:
            self.decision_cutoff_proba_ = float(
                self.calibrator_.predict_proba(np.array([[self.decision_cutoff_raw_]]))[:, 1][0]
            )
        self.oof_y_ = y.copy()

        # Ranker knots (ECDF for predict_rank)
        sorted_scores = np.sort(oof_sigmoid)
        n_r = len(sorted_scores)
        if n_r > _RANKER_MAX_KNOTS:
            idx = np.round(np.linspace(0, n_r - 1, _RANKER_MAX_KNOTS)).astype(int)
            self._ranker_knots = sorted_scores[idx]
        else:
            self._ranker_knots = sorted_scores
        n_k = len(self._ranker_knots)
        self.decision_cutoff_rank_ = float(np.interp(
            self.decision_cutoff_raw_, self._ranker_knots, np.linspace(0.0, 1.0, n_k)
        ))
        _p = np.clip(self.decision_cutoff_proba_, 1e-7, 1.0 - 1e-7)
        self.decision_cutoff_logit_ = float(np.log(_p / (1.0 - _p)))
        logger.success(
            f"Calibrator fitted ({self.calibrator_method_}, "
            f"minority={minority_count}) on {k}-fold OOF."
        )
        logger.info(
            f"cutoff: {self.decision_cutoff_raw_:.4f} | "
            f"source={self.decision_cutoff_raw_source_}"
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _raw_sigmoid(self, X: np.ndarray) -> np.ndarray:
        """Return shape-(n,) sigmoid(decision_function) — pre-calibration [0,1]."""
        check_is_fitted(self, "svc_")
        return _sigmoid(_decision_scores(self.svc_, X))

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated class probabilities, shape (n_samples, 2)."""
        check_is_fitted(self, "svc_")
        X = check_array(X, dtype="numeric", accept_sparse=False)
        p_raw = self._raw_sigmoid(X)
        if hasattr(self, "calibrator_"):
            if self.calibrator_method_ == "isotonic":
                p1 = np.clip(self.calibrator_.predict(p_raw), 0, 1)
            else:
                p1 = self.calibrator_.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        else:
            p1 = p_raw
        return np.column_stack([1 - p1, p1])

    def predict_score(self, X) -> np.ndarray:
        """Return raw (pre-calibration) sigmoid scores, shape (n_samples, 2)."""
        check_is_fitted(self, "svc_")
        X = check_array(X, dtype="numeric", accept_sparse=False)
        p1 = self._raw_sigmoid(X)
        return np.column_stack([1 - p1, p1])

    def predict_logit(self, X) -> np.ndarray:
        """Return logit of calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_rank(self, X) -> np.ndarray:
        """Map sigmoid scores to [0,1] ranks via OOF ECDF, shape (n_samples, 2)."""
        check_is_fitted(self, "_ranker_knots")
        scores = self.predict_score(X)[:, 1]
        n_k = len(self._ranker_knots)
        rank_1 = np.interp(scores, self._ranker_knots, np.linspace(0.0, 1.0, n_k))
        return np.column_stack([1 - rank_1, rank_1])

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Return binary 0/1 predictions."""
        threshold = self.decision_cutoff_raw_ if cutoff is None else float(cutoff)
        return (self.predict_score(X)[:, 1] >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_onnx(self, path: str) -> None:
        """
        Export the trained model to ONNX.

        Output layout (binary classification, zipmap=False):
          output[0]: int64 labels, shape (n,)
          output[1]: float32 scores, shape (n, 2) — col 1 = class-1 decision score

        At inference, BaseSVCArtifact applies sigmoid to output[1][:, 1] to
        obtain [0,1] pre-calibration scores, then applies the stored calibrator.
        """
        check_is_fitted(self, "svc_")
        _to_onnx(self.svc_, path, self.profile_.n_features)

    def save(self, directory: str) -> None:
        """
        Save the trained model to a directory.

        Always writes svc.onnx + svc.json.  The ONNX size guard in _fit_raw()
        guarantees the model fits within _ONNX_MAX_BYTES — either by reducing C
        or by switching to LinearSVC — so joblib is never needed.
        """
        import dataclasses

        check_is_fitted(self, "svc_")
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "svc.onnx"))
        fmt = "onnx"

        profile_dict = (
            dataclasses.asdict(self.profile_)
            if dataclasses.is_dataclass(self.profile_)
            else dict(vars(self.profile_))
        )
        metadata = {
            "task": "classification",
            "format": fmt,
            "preset_name": self.preset_name_,
            "use_linear": self._use_linear_,
            "n_features_in": self.profile_.n_features,
            "classes": [0, 1],
            "params": self.params_,
            "profile": profile_dict,
            "portfolio_scores": self.portfolio_scores_,
            "decision_cutoff_raw": float(
                getattr(self, "decision_cutoff_raw_", _DEFAULT_DECISION_CUTOFF)
            ),
            "decision_cutoff_raw_source": getattr(
                self, "decision_cutoff_raw_source_", "default_0.5"
            ),
            "decision_cutoff_proba": float(
                getattr(self, "decision_cutoff_proba_", _DEFAULT_DECISION_CUTOFF)
            ),
            "decision_cutoff_rank": float(
                getattr(self, "decision_cutoff_rank_", _DEFAULT_DECISION_CUTOFF)
            ),
            "decision_cutoff_logit": float(
                getattr(self, "decision_cutoff_logit_", 0.0)
            ),
        }
        if hasattr(self, "calibrator_"):
            if self.calibrator_method_ == "isotonic":
                metadata["calibrator"] = {
                    "method": "isotonic",
                    "X_thresholds": self.calibrator_.X_thresholds_.tolist(),
                    "y_thresholds": self.calibrator_.y_thresholds_.tolist(),
                }
            else:  # platt
                metadata["calibrator"] = {
                    "method": "platt",
                    "coef": float(self.calibrator_.coef_[0][0]),
                    "intercept": float(self.calibrator_.intercept_[0]),
                }
        if hasattr(self, "_ranker_knots"):
            metadata["ranker"] = {"knots": self._ranker_knots.tolist()}

        with open(os.path.join(directory, "svc.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# BaseSVCArtifact — inference only (numpy + onnxruntime, no sklearn)
# ---------------------------------------------------------------------------

class BaseSVCArtifact:
    """
    Inference-only loader for a model saved by BaseSVCClassifier.save().

    Only requires numpy and onnxruntime — no sklearn at inference time.

    The ONNX model outputs raw decision function scores.  This artifact applies:
      1. sigmoid(score)          → pre-calibration [0,1] probability (predict_score)
      2. calibrator(sigmoid)     → calibrated probability (run / predict_proba)
    """

    def __init__(self):
        self._session = None
        self._input_name = ""
        self.metadata: dict = {}
        self._cal = None
        self._ranker = None
        self.decision_cutoff_raw = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_proba = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_rank = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_logit = 0.0

    @classmethod
    def load(cls, directory: str) -> "BaseSVCArtifact":
        json_path = os.path.join(directory, "svc.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"svc.json not found in {directory!r}")
        self = cls.__new__(cls)
        with open(json_path) as f:
            self.metadata = json.load(f)
        fmt = self.metadata.get("format", "onnx")
        if fmt != "onnx":
            raise ValueError(
                f"Unsupported SVC artifact format {fmt!r} in {directory!r}. "
                "Only 'onnx' is supported."
            )
        self._cal = self.metadata.get("calibrator", None)
        self._ranker = self.metadata.get("ranker", None)
        self.decision_cutoff_raw = float(
            self.metadata.get("decision_cutoff_raw", _DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_proba = float(
            self.metadata.get("decision_cutoff_proba", _DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_rank = float(
            self.metadata.get("decision_cutoff_rank", _DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_logit = float(
            self.metadata.get("decision_cutoff_logit", 0.0)
        )
        import onnxruntime as rt
        onnx_path = os.path.join(directory, "svc.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"svc.onnx not found in {directory!r}")
        self._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Return shape-(n,) raw decision function scores from ONNX."""
        X_f32 = np.asarray(X, dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: X_f32})
        raw = np.asarray(outputs[1], dtype=np.float64)
        if raw.ndim == 1:
            return raw
        if raw.ndim == 2 and raw.shape[1] == 1:
            return raw.ravel()
        if raw.ndim == 2:
            return raw[:, 1]
        return raw.ravel()

    def run(self, X) -> np.ndarray:
        """
        Return calibrated class probabilities, shape (n_samples, 2).

        Pipeline: raw_score → sigmoid → calibrator.
        """
        X = np.asarray(X, dtype=np.float32)
        df = self._raw_scores(X)
        p_raw = 1.0 / (1.0 + np.exp(-np.clip(df, -500.0, 500.0)))
        proba = np.column_stack([1 - p_raw, p_raw])
        if self._cal is not None:
            proba = _apply_calibrator_artifact(proba, self._cal)
        return proba

    def predict_score(self, X) -> np.ndarray:
        """Return sigmoid(decision_function) pre-calibration scores, shape (n_samples, 2)."""
        X = np.asarray(X, dtype=np.float32)
        df = self._raw_scores(X)
        p_raw = 1.0 / (1.0 + np.exp(-np.clip(df, -500.0, 500.0)))
        return np.column_stack([1 - p_raw, p_raw])

    def predict_logit(self, X) -> np.ndarray:
        """Return logit of calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.run(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_rank(self, X) -> np.ndarray:
        """Map pre-calibration scores to [0,1] ranks via ECDF, shape (n_samples, 2)."""
        if self._ranker is None:
            raise RuntimeError("No ranker stored in this artifact.")
        knots = np.asarray(self._ranker["knots"])
        rank_1 = np.interp(
            self.predict_score(X)[:, 1],
            knots,
            np.linspace(0.0, 1.0, len(knots)),
        )
        return np.column_stack([1 - rank_1, rank_1])

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Return binary 0/1 predictions."""
        threshold = self.decision_cutoff_raw if cutoff is None else float(cutoff)
        return (self.run(X)[:, 1] >= threshold).astype(int)
