"""
BaseLinearClassifier and BaseLinearRegressor: linear models with auto feature selection.

Both handle three dataset regimes automatically:
  - standard  (small CV cost, p <= n): LogisticRegressionCV / RidgeCV
  - high_dim  (small CV cost, p >  n): ElasticNet + SelectFromModel pre-filter
  - large     (high CV cost)        : SGD estimator, alpha tuned on subsample

Note: feature scaling is NOT applied internally — callers should standardize their
data before passing it to fit() / predict().

Literature basis:
  Fan & Lin (2008) - LIBLINEAR coordinate descent for L1
  Zou & Hastie (2005) - ElasticNet removes L1 feature-count ceiling at p>n
  Friedman et al. (2010) - ElasticNet grouping effect for correlated features
  Bottou (2010) - SGD for large-scale linear models
"""

from __future__ import annotations

import inspect
import json
import os
import time as _time

import numpy as np

# ---------------------------------------------------------------------------
# Fit-time dependencies — NOT required for ONNX inference.
# Guarded so lean [descriptors]-only installs can import this module to reach
# the artifact (inference) classes without errors.
# ---------------------------------------------------------------------------
try:
    from sklearn.base import BaseEstimator
    from sklearn.feature_selection import (
        SelectFromModel,
        SelectKBest,
        VarianceThreshold,
        f_classif,
    )
    from sklearn.linear_model import (
        ElasticNetCV,
        Lasso,
        LogisticRegression,
        LogisticRegressionCV,
        RidgeCV,
        SGDClassifier,
        SGDRegressor,
    )
    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
    from sklearn.model_selection import (
        GridSearchCV,
        KFold,
        StratifiedKFold,
        StratifiedShuffleSplit,
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.validation import check_array, check_is_fitted

    _FIT_DEPS_AVAILABLE = True
except ImportError:

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    SelectFromModel = SelectKBest = VarianceThreshold = f_classif = None  # type: ignore[assignment,misc]
    ElasticNetCV = Lasso = LogisticRegression = LogisticRegressionCV = None  # type: ignore[assignment,misc]
    RidgeCV = SGDClassifier = SGDRegressor = None  # type: ignore[assignment,misc]
    balanced_accuracy_score = r2_score = roc_auc_score = None  # type: ignore[assignment]
    GridSearchCV = KFold = StratifiedKFold = StratifiedShuffleSplit = None  # type: ignore[assignment,misc]
    LabelEncoder = None  # type: ignore[assignment,misc]
    check_array = check_is_fitted = None  # type: ignore[assignment]
    _FIT_DEPS_AVAILABLE = False

from lazyqsar.utils.logging import logger
from lazyqsar.utils.splits import (
    auto_stratified_oof_n_splits,
    make_stratified_oof_splits,
)


def _sklearn_kwargs(estimator, **kwargs):
    if "use_legacy_attributes" in inspect.signature(estimator).parameters:
        kwargs["use_legacy_attributes"] = False
    return kwargs


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


def _auto_n_splits(y: np.ndarray) -> int:
    """Backward-compatible wrapper around the shared stratified OOF helper."""
    return auto_stratified_oof_n_splits(y)


# Minimum minority-class OOF samples required to use isotonic calibration.
# Below this threshold, Platt scaling (2-parameter sigmoid) is used instead.
_CALIBRATION_ISOTONIC_MIN_MINORITY = 500
_RANKER_MAX_KNOTS = 10_000

_CLASSIFIER_STANDARD_MAX_CV_WORK = 2_000_000
_CLASSIFIER_HIGHDIM_MAX_CV_WORK = 1_000_000
_SGD_VALIDATION_FRACTION = 0.1
_SGD_N_ITER_NO_CHANGE = 10
_DEFAULT_DECISION_CUTOFF = 0.5


def _detect_classifier_regime(n: int, p: int) -> str:
    """
    Choose the classifier regime with a simple cost-aware proxy.

    LogisticRegressionCV becomes disproportionately expensive once n*p is large,
    especially in the high-dimensional path where a SelectFromModel pre-fit runs
    before the CV search. We therefore reserve the CV-heavy regimes for datasets
    whose dense feature work stays below a conservative threshold and fall back
    to the SGD-based large regime earlier than the old n>50k rule.
    """
    work = n * p
    if p > n:
        if n > 50_000 or work > _CLASSIFIER_HIGHDIM_MAX_CV_WORK:
            return "large"
        return "high_dim"
    if n > 50_000 or work > _CLASSIFIER_STANDARD_MAX_CV_WORK:
        return "large"
    return "standard"


def _classifier_solver_label(regime: str, calibrated: bool = False) -> str:
    if regime == "standard":
        return "LogisticRegression" if calibrated else "LogisticRegressionCV"
    if regime == "high_dim":
        if calibrated:
            return "LogisticRegression + SelectKBest(f_classif)"
        return "LogisticRegressionCV + SelectKBest(f_classif)"
    if regime == "large":
        return "SGDClassifier"
    return "unknown"


def _learn_balanced_accuracy_cutoff(
    y_true: np.ndarray, p1: np.ndarray
) -> tuple[float, str]:
    """Learn a deterministic balanced-accuracy cutoff from OOF class-1 probabilities."""
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

    candidates = np.unique(
        np.concatenate(
            [
                unique,
                np.array(
                    [
                        np.nextafter(unique[0], -np.inf),
                        _DEFAULT_DECISION_CUTOFF,
                        np.nextafter(unique[-1], np.inf),
                    ],
                    dtype=float,
                ),
            ]
        )
    )

    best_threshold = _DEFAULT_DECISION_CUTOFF
    best_key = None
    for thr in candidates:
        score = balanced_accuracy_score(labels, (probs >= thr).astype(int))
        key = (-float(score), abs(float(thr) - _DEFAULT_DECISION_CUTOFF), float(thr))
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(thr)

    return best_threshold, "oof_balanced_accuracy"


def _detect_regime(n: int, p: int) -> str:
    if n > 50_000:
        return "large"
    if p > n:
        return "high_dim"
    return "standard"


# ---------------------------------------------------------------------------
# Hyperparameter heuristics
# ---------------------------------------------------------------------------


def _default_grid_size(n: int, p: int) -> int:
    """Adaptive grid size: keep small problems thorough, trim expensive searches."""
    work = n * p
    if work <= 200_000:
        return 20
    if work <= 2_000_000:
        return 12
    return 8


def _default_C_grid(n: int, p: int, n_grid: int | None = None) -> np.ndarray:
    """Adaptive C grid centered on lasso-theory optimal C* ~ sqrt(n)/p."""
    if n_grid is None:
        n_grid = _default_grid_size(n, p)
    C_center = max(1e-4, (n**0.5) / p)
    return np.logspace(np.log10(C_center) - 2, np.log10(C_center) + 2, n_grid)


def _default_alpha_grid(n: int, p: int, n_grid: int | None = None) -> np.ndarray:
    """SGD alpha = 1/(C*n); derived from C grid."""
    if n_grid is None:
        n_grid = _default_grid_size(n, p)
    C_grid = _default_C_grid(n, p, n_grid)
    return np.clip(1.0 / (C_grid * n), 1e-7, 1.0)


def _default_cv(n: int, y: np.ndarray) -> int:
    """Adaptive fold count: more folds when n is small."""
    min_class = int(np.bincount(y).min())
    if n < 200:
        return min(min_class, 10)
    if n < 1000:
        return min(min_class, 5)
    return min(min_class, 3)


def _default_l1_ratio(n: int, p: int) -> float:
    """ElasticNet mixing: more grouping (lower l1_ratio) when p/n is large."""
    ratio = p / max(n, 1)
    if ratio > 10:
        return 0.5
    if ratio > 2:
        return 0.7
    return 0.9


def _sfm_max_features(n: int, p: int) -> int:
    """SelectFromModel cap: at most 2n features (Zou & Hastie 2005 bound), at least 10."""
    return max(10, min(p, 2 * n))


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


class BaseLinearClassifier(BaseEstimator):
    """
    Binary logistic regression with embedded L1/ElasticNet feature selection.

    Automatically selects solver and regularization regime based on (n, p) and
    estimated CV cost:
      - standard  (small n*p, p<=n): LogisticRegressionCV (saga, L1)
      - high_dim  (small n*p, p>n):  SelectFromModel + LogisticRegressionCV
      - large     (large n*p):       SGDClassifier + ElasticNet, subsample alpha tuning

    Feature scaling is NOT applied internally. Standardize X before calling fit().

    Parameters
    ----------
    regime : str or None
        Force a specific regime ("standard", "high_dim", "large").
        None = auto-detect from data shape.
    C_values : array-like or None
        Override the adaptive C grid for standard/high_dim regimes.
    alpha_values : array-like or None
        Override the adaptive alpha grid for the large regime.
    l1_ratio : float or None
        ElasticNet mixing (0=L2, 1=L1). None = auto from p/n heuristic.
    cv : int or None
        Number of CV folds. None = auto from n and class balance.
    class_weight : str or dict
        Passed to all estimators. "balanced" is strongly recommended for
        imbalanced bioactivity data.
    max_iter : int
        Max iterations for liblinear/saga solvers.
    n_jobs : int
        Parallelism for CV and some solvers.
    random_state : int or None
        Reproducibility seed.
    variance_threshold : float
        VarianceThreshold cutoff. 0.0 removes only constant features.
    tuning_subsample : int
        Max rows used for alpha-grid tuning in the large regime.
    verbose : bool
        If True, log rule banners and regime info.
    """

    def __init__(
        self,
        *,
        regime: str | None = None,
        C_values: list | np.ndarray | None = None,
        alpha_values: list | np.ndarray | None = None,
        l1_ratio: float | None = None,
        cv: int | None = None,
        class_weight: str | dict = "balanced",
        max_iter: int = 20_000,
        n_jobs: int = -1,
        random_state: int | None = 42,
        variance_threshold: float = 0.0,
        tuning_subsample: int = 10_000,
        calibrated: bool = True,
    ):
        self.regime = regime
        self.C_values = C_values
        self.alpha_values = alpha_values
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.variance_threshold = variance_threshold
        self.tuning_subsample = tuning_subsample
        self.calibrated = calibrated

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "BaseLinearClassifier":
        """Fit the model. If calibrated=True, runs calibrate() for OOF calibration."""
        if not _FIT_DEPS_AVAILABLE:
            raise ImportError(
                "Training requires scikit-learn. Install with: pip install 'lazyqsar[fit]'"
            )
        if self.calibrated:
            y_arr = np.asarray(y, dtype=int)
            if np.bincount(y_arr).min() >= 2:
                return self.calibrate(X, y)
        return self._fit_raw(X, y)

    def _fit_raw(self, X, y) -> "BaseLinearClassifier":
        if not _FIT_DEPS_AVAILABLE:
            raise ImportError(
                "Training requires scikit-learn. Install with: pip install 'lazyqsar[fit]'"
            )
        logger.rule("BaseLinearClassifier")

        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y)

        # --- Label encoding ---
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)

        if len(np.unique(y_enc)) < 2:
            raise ValueError(
                "Training data contains only one class. "
                "Binary classification requires both classes."
            )

        n, p = X.shape
        self.n_features_in_ = p

        # --- Regime ---
        self.regime_ = (
            self.regime if self.regime is not None else _detect_classifier_regime(n, p)
        )

        # --- Heuristics ---
        effective_l1_ratio = (
            self.l1_ratio if self.l1_ratio is not None else _default_l1_ratio(n, p)
        )
        effective_cv = self.cv if self.cv is not None else _default_cv(n, y_enc)
        scoring = "roc_auc"

        logger.info(
            "BaseLinearClassifier.fit: "
            f"regime={self.regime_} | solver={_classifier_solver_label(self.regime_)} "
            f"| n={n:,} | p={p:,} | cv={effective_cv} | scoring={scoring}"
        )

        skf = StratifiedKFold(
            n_splits=effective_cv, shuffle=True, random_state=self.random_state
        )

        # --- Preprocessing: VarianceThreshold only (scaling is caller's responsibility) ---
        self._vt = VarianceThreshold(threshold=self.variance_threshold)
        try:
            X_vt = self._vt.fit_transform(X)
        except ValueError:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        if X_vt.shape[1] == 0:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        # --- Dispatch ---
        _t_hp = _time.perf_counter()
        if self.regime_ == "standard":
            self._fit_standard(X_vt, y_enc, n, p, skf, scoring)
        elif self.regime_ == "high_dim":
            self._fit_high_dim(X_vt, y_enc, n, p, skf, scoring, effective_l1_ratio)
        elif self.regime_ == "large":
            self._fit_large(X_vt, y_enc, n, p, skf, scoring, effective_l1_ratio)
        else:
            raise ValueError(
                f"Unknown regime: {self.regime_!r}. Expected 'standard', 'high_dim', or 'large'."
            )
        self.timing_ = {"hparam_search": _time.perf_counter() - _t_hp}

        # --- Consolidated feature mask ---
        vt_support = self._vt.get_support()
        if self._sfm is not None:
            sfm_support = self._sfm.get_support()
            combined = np.zeros(p, dtype=bool)
            idx = np.where(vt_support)[0]
            combined[idx[sfm_support]] = True
            self.feature_mask_ = combined
        else:
            self.feature_mask_ = vt_support

        # --- Coef extraction ---
        if hasattr(self._estimator, "coef_"):
            self.coef_ = self._estimator.coef_
        else:
            self.coef_ = None

        self.classes_ = self._label_encoder.classes_
        self.decision_cutoff_raw_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_raw_source_ = "default_0.5"
        self.decision_cutoff_proba_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_rank_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_logit_ = 0.0
        logger.info(
            "decision cutoff: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_raw_source_}"
        )

        logger.rule("Done")
        return self

    def predict_logit(self, X) -> np.ndarray:
        """Return logit of calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_score(self, X) -> np.ndarray:
        """Return raw (pre-calibration) probabilities, shape (n_samples, 2)."""
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        X_t = self._transform(X)
        return self._estimator.predict_proba(X_t)

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        threshold = self.decision_cutoff_raw_ if cutoff is None else float(cutoff)
        proba = self.predict_score(X)[:, 1]
        y_enc = (proba >= threshold).astype(int)
        return self._label_encoder.inverse_transform(y_enc)

    def predict_rank(self, X) -> np.ndarray:
        """Map raw scores to [0, 1] ranks via OOF ECDF, shape (n_samples, 2)."""
        check_is_fitted(self, attributes=["_ranker_knots"])
        scores = self.predict_score(X)[:, 1]
        n_k = len(self._ranker_knots)
        rank_1 = np.interp(scores, self._ranker_knots, np.linspace(0.0, 1.0, n_k))
        return np.column_stack([1 - rank_1, rank_1])

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        X_t = self._transform(X)
        proba = self._estimator.predict_proba(X_t)
        if hasattr(self, "calibrator_"):
            if self.calibrator_method_ == "isotonic":
                p1 = np.clip(self.calibrator_.predict(proba[:, 1]), 0, 1)
            else:  # platt
                p1 = self.calibrator_.predict_proba(proba[:, 1].reshape(-1, 1))[:, 1]
            proba = np.column_stack([1 - p1, p1])
        return proba

    def calibrate(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseLinearClassifier":
        """
        Collect out-of-fold predicted probabilities via stratified k-fold CV,
        then fit an isotonic calibrator on them.

        Hyperparameter selection (C-search / alpha-grid) runs ONCE on the full
        data via self.fit().  Each fold reuses the pre-selected hyperparameter
        with a single model fit — no inner LogisticRegressionCV or GridSearchCV.

        After this call
        ---------------
        self.oof_probas_ : ndarray, shape (n,)
            Calibrated out-of-fold probabilities for class 1, in the same
            row order as the input X / y.
        self.oof_y_ : ndarray, shape (n,)
            Original y labels (0/1), same order as X.
        self.calibrator_ : IsotonicRegression
            Fitted isotonic calibrator.  predict_proba() will apply this
            layer to new-data predictions.
        """
        if not _FIT_DEPS_AVAILABLE:
            raise ImportError(
                "Training requires scikit-learn. Install with: pip install 'lazyqsar[fit]'"
            )
        from sklearn.isotonic import IsotonicRegression

        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=int)
        n = len(y)

        k, fold_splits = make_stratified_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )

        # Step 1: full fit — hyperparameter search runs ONCE here
        logger.info(
            f"BaseLinearClassifier.calibrate: full fit on n={n} (C/alpha search runs once)"
        )
        self._fit_raw(X, y)  # sets self.regime_, self._vt, self._sfm, self._estimator

        # Extract best hyperparameter from the full-data fit
        regime = self.regime_
        if regime in ("standard", "high_dim"):
            best_C = float(self._estimator.C_[0])
            logger.info(f"calibrate: regime={regime}  best_C={best_C:.4g}")
        else:  # large
            best_alpha = float(self._estimator.alpha)
            best_l1_ratio = float(self._estimator.l1_ratio)
            logger.info(f"calibrate: regime={regime}  best_alpha={best_alpha:.4g}")

        # Step 2: k-fold OOF — reuse best hyperparam, no inner CV per fold
        oof_raw = np.full(n, np.nan, dtype=float)
        logger.info(
            "calibrate: "
            f"{k}-fold OOF (no inner CV in folds) | fold_solver={_classifier_solver_label(regime, calibrated=True)}"
        )

        fold_times = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            logger.debug(
                f"  Fold {fold_idx + 1}/{k}: train={len(train_idx)}  val={len(val_idx)}"
            )
            _t_fold = _time.perf_counter()
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            # Refit VarianceThreshold on fold training data (cheap)
            vt = VarianceThreshold(threshold=self.variance_threshold)
            X_tr_vt = vt.fit_transform(X_tr)
            X_val_vt = vt.transform(X_val)

            if regime == "standard":
                fold_est = LogisticRegression(
                    **_sklearn_kwargs(
                        LogisticRegression,
                        C=best_C,
                        solver="saga",
                        penalty="l1",
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    )
                )
                fold_est.fit(X_tr_vt, y_tr)
                oof_raw[val_idx] = fold_est.predict_proba(X_val_vt)[:, 1]

            elif regime == "high_dim":
                # Refit fast univariate F-test on fold data (O(n*p), no inner model)
                fold_sfm = SelectKBest(f_classif, k=self._sfm.k)
                fold_sfm.fit(X_tr_vt, y_tr)
                X_tr_sfm = fold_sfm.transform(X_tr_vt)
                X_val_sfm = fold_sfm.transform(X_val_vt)
                fold_est = LogisticRegression(
                    **_sklearn_kwargs(
                        LogisticRegression,
                        C=best_C,
                        solver="saga",
                        penalty="elasticnet",
                        l1_ratio=self._estimator.l1_ratios[0],
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    )
                )
                fold_est.fit(X_tr_sfm, y_tr)
                oof_raw[val_idx] = fold_est.predict_proba(X_val_sfm)[:, 1]

            else:  # large
                fold_est = SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    l1_ratio=best_l1_ratio,
                    alpha=best_alpha,
                    class_weight=self.class_weight,
                    max_iter=self.max_iter,
                    early_stopping=True,
                    validation_fraction=_SGD_VALIDATION_FRACTION,
                    n_iter_no_change=_SGD_N_ITER_NO_CHANGE,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                fold_est.fit(X_tr_vt, y_tr)
                oof_raw[val_idx] = fold_est.predict_proba(X_val_vt)[:, 1]

            fold_times.append(_time.perf_counter() - _t_fold)

        self.timing_["calibration_folds"] = fold_times
        self.timing_["calibration_total"] = sum(fold_times)

        # Step 3: fit calibrator — method chosen by minority class size
        minority_count = int(np.bincount(y).min())
        if minority_count >= _CALIBRATION_ISOTONIC_MIN_MINORITY:
            cal = IsotonicRegression(out_of_bounds="clip")
            self.oof_probas_ = cal.fit_transform(oof_raw, y)
            self.calibrator_method_ = "isotonic"
        else:
            cal = LogisticRegression(C=1.0, solver="lbfgs")
            cal.fit(oof_raw.reshape(-1, 1), y)
            self.oof_probas_ = cal.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
            self.calibrator_method_ = "platt"
        self.calibrator_ = cal
        self.decision_cutoff_raw_, self.decision_cutoff_raw_source_ = (
            _learn_balanced_accuracy_cutoff(y, oof_raw)
        )
        if self.calibrator_method_ == "isotonic":
            self.decision_cutoff_proba_ = float(
                np.clip(
                    self.calibrator_.predict(np.array([self.decision_cutoff_raw_]))[0],
                    0.0,
                    1.0,
                )
            )
        else:
            self.decision_cutoff_proba_ = float(
                self.calibrator_.predict_proba(np.array([[self.decision_cutoff_raw_]]))[
                    :, 1
                ][0]
            )
        self.oof_y_ = y.copy()
        # Uncalibrated [0,1] out-of-fold probabilities (the calibrator's input), kept alongside the
        # calibrated oof_probas_ so downstream can report the raw score too.
        self.oof_raw_ = oof_raw.copy()
        sorted_scores = np.sort(oof_raw)
        n_r = len(sorted_scores)
        if n_r > _RANKER_MAX_KNOTS:
            idx = np.round(np.linspace(0, n_r - 1, _RANKER_MAX_KNOTS)).astype(int)
            self._ranker_knots = sorted_scores[idx]
        else:
            self._ranker_knots = sorted_scores
        n_k = len(self._ranker_knots)
        self.decision_cutoff_rank_ = float(
            np.interp(
                self.decision_cutoff_raw_,
                self._ranker_knots,
                np.linspace(0.0, 1.0, n_k),
            )
        )
        _p = np.clip(self.decision_cutoff_proba_, 1e-7, 1.0 - 1e-7)
        self.decision_cutoff_logit_ = float(np.log(_p / (1.0 - _p)))
        logger.success(
            f"Calibrator fitted ({self.calibrator_method_}, minority={minority_count}) "
            f"on OOF predictions."
        )
        logger.info(
            "calibration cutoff learned from OOF scores: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_raw_source_}"
        )
        return self

    def score(self, X, y) -> float:
        proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_mask_"])
        return np.where(self.feature_mask_)[0].astype(str)

    def to_onnx(self, path: str) -> None:
        """
        Export the trained model to an ONNX file.

        Accepts a float32 input named ``"float_input"`` with shape
        ``(n_samples, n_features_in_)`` (raw, unprocessed features).
        Produces:
          - ``"output_label"``       int64  (n_samples,)   — predicted class
          - ``"output_probability"`` float32 (n_samples, 2) — [P(class_0), P(class_1)]

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model.onnx"``.
        """
        check_is_fitted(self, attributes=["_estimator"])
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import Pipeline

        steps = [("vt", self._vt)]
        if self._sfm is not None:
            steps.append(("sfm", self._sfm))
        steps.append(("clf", self._estimator))

        pipeline = Pipeline(steps)
        initial_types = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def save(self, directory: str) -> None:
        """
        Save the trained model to a directory.

        Always writes ``linear.json`` (fit metadata) and ``linear.onnx``.

        Parameters
        ----------
        directory : str
            Destination directory (created if it does not exist).
        """
        check_is_fitted(self, attributes=["_estimator"])
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "linear.onnx"))
        metadata = {
            "task": "classification",
            "format": "onnx",
            "regime": self.regime_,
            "n_features_in": self.n_features_in_,
            "classes": self._label_encoder.classes_.tolist(),
            "feature_mask": self.feature_mask_.tolist(),
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
        with open(os.path.join(directory, "linear.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Internal: transform helpers
    # ------------------------------------------------------------------

    def _transform(self, X) -> np.ndarray:
        X_vt = self._vt.transform(X)
        if self._sfm is not None:
            return self._sfm.transform(X_vt)
        return X_vt

    # ------------------------------------------------------------------
    # Internal: regime-specific fit methods
    # ------------------------------------------------------------------

    def _fit_standard(self, X, y, n, p, skf, scoring):
        self._sfm = None
        C_grid = (
            np.asarray(self.C_values)
            if self.C_values is not None
            else _default_C_grid(n, p)
        )

        self._estimator = LogisticRegressionCV(
            **_sklearn_kwargs(
                LogisticRegressionCV,
                Cs=C_grid,
                cv=skf,
                solver="saga",
                penalty="l1",
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                scoring=scoring,
                n_jobs=self.n_jobs,
                refit=True,
                random_state=self.random_state,
            )
        )
        self._estimator.fit(X, y)

    def _fit_high_dim(self, X, y, n, p, skf, scoring, l1_ratio):
        # Pre-filter with fast univariate F-test (O(n*p), no inner model)
        max_feat = min(_sfm_max_features(n, p), X.shape[1])
        self._sfm = SelectKBest(f_classif, k=max_feat)
        self._sfm.fit(X, y)
        X_sfm = self._sfm.transform(X)

        if X_sfm.shape[1] == 0:
            # Pathological fallback: keep at least 1 feature
            self._sfm = SelectKBest(f_classif, k=1)
            self._sfm.fit(X, y)
            X_sfm = self._sfm.transform(X)

        C_grid = (
            np.asarray(self.C_values)
            if self.C_values is not None
            else _default_C_grid(n, X_sfm.shape[1])
        )

        self._estimator = LogisticRegressionCV(
            **_sklearn_kwargs(
                LogisticRegressionCV,
                Cs=C_grid,
                cv=skf,
                solver="saga",
                penalty="elasticnet",
                l1_ratios=[l1_ratio],
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                scoring=scoring,
                n_jobs=self.n_jobs,
                refit=True,
                random_state=self.random_state,
            )
        )
        self._estimator.fit(X_sfm, y)

    def _fit_large(self, X, y, n, p, skf, scoring, l1_ratio):
        self._sfm = None
        alpha_grid = (
            np.asarray(self.alpha_values)
            if self.alpha_values is not None
            else _default_alpha_grid(n, p)
        )

        # Tune alpha on a stratified subsample
        sub_size = min(self.tuning_subsample, n)
        if sub_size < n:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=None,
                train_size=sub_size,
                random_state=self.random_state,
            )
            sub_idx, _ = next(sss.split(X, y))
            X_sub, y_sub = X[sub_idx], y[sub_idx]
        else:
            X_sub, y_sub = X, y

        base_sgd = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=_SGD_VALIDATION_FRACTION,
            n_iter_no_change=_SGD_N_ITER_NO_CHANGE,
            random_state=self.random_state,
        )

        # Adjust CV folds for subsample size
        sub_min_class = int(np.bincount(y_sub).min())
        sub_cv = min(skf.n_splits, sub_min_class)
        if sub_cv < 2:
            sub_cv = 2
        sub_skf = StratifiedKFold(
            n_splits=sub_cv, shuffle=True, random_state=self.random_state
        )

        gs = GridSearchCV(
            estimator=base_sgd,
            param_grid={"alpha": alpha_grid},
            cv=sub_skf,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=False,
        )
        gs.fit(X_sub, y_sub)
        best_alpha = gs.best_params_["alpha"]

        # Refit on full data with best alpha
        self._estimator = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            alpha=best_alpha,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=_SGD_VALIDATION_FRACTION,
            n_iter_no_change=_SGD_N_ITER_NO_CHANGE,
            random_state=self.random_state,
        )
        self._estimator.fit(X, y)


# ---------------------------------------------------------------------------
# Regression helper
# ---------------------------------------------------------------------------


def _cv_from_n(n: int) -> int:
    """Adaptive CV folds for regression (no class constraint)."""
    if n < 200:
        return min(n // 10, 10)
    if n < 1000:
        return 5
    return 3


def _regression_scoring(y: np.ndarray) -> str:
    """Auto-select CV metric based on target skewness.

    Skewness > 1.0 (moderately skewed) → neg_mean_absolute_error, which is
    more robust to outliers than R². Otherwise R².
    """
    from scipy.stats import skew as scipy_skew

    sk = abs(float(scipy_skew(y)))
    return "neg_mean_absolute_error" if sk > 1.0 else "r2"


def _regression_sample_weight(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Inverse-frequency sample weights for continuous targets.

    Bins ``y`` into ``n_bins`` quantile bins and assigns each sample a weight
    proportional to 1 / (bin_count / n_samples), so that rare target regions
    are up-weighted — analogous to ``class_weight='balanced'`` for classifiers.
    """
    from sklearn.utils.class_weight import compute_sample_weight

    # Quantile binning handles skewed distributions better than equal-width.
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y, quantiles)
    bin_edges = np.unique(bin_edges)  # collapse duplicates (discrete-ish targets)
    n_unique_bins = len(bin_edges) - 1
    if n_unique_bins < 2:
        # Target has essentially one value; all weights equal.
        return np.ones(len(y), dtype=float)
    bin_ids = np.digitize(y, bin_edges[1:-1])  # 0-based bin index
    return compute_sample_weight("balanced", bin_ids)


# ---------------------------------------------------------------------------
# Linear regressor
# ---------------------------------------------------------------------------


class BaseLinearRegressor(BaseEstimator):
    """
    Linear regression with embedded feature selection.

    Automatically selects solver and regularization regime based on (n, p):
      - standard  (n<=50K, p<=n): RidgeCV (L2),               k-fold CV
      - high_dim  (n<=50K, p>n):  ElasticNetCV,               SelectFromModel(Lasso) pre-filter
      - large     (n>50K):        SGDRegressor + ElasticNet,  subsample alpha tuning

    Feature scaling is NOT applied internally. Standardize X before calling fit().
    CV scoring is auto-selected based on target skewness (R² or neg_MAE).

    Imbalanced target distributions are handled automatically: samples in rare
    target regions are up-weighted via inverse-frequency quantile binning,
    analogous to ``class_weight='balanced'`` for classifiers.

    Parameters
    ----------
    regime : str or None
        Force a specific regime. None = auto-detect from data shape.
    alpha_values : array-like or None
        Override the adaptive regularization strength grid.
    l1_ratio : float or None
        ElasticNet mixing (0=L2, 1=L1). None = auto from p/n heuristic.
        Only used in high_dim and large regimes.
    cv : int or None
        Number of CV folds. None = auto from n.
    max_iter : int
        Max iterations for iterative solvers.
    n_jobs : int
        Parallelism for CV.
    random_state : int or None
        Reproducibility seed.
    variance_threshold : float
        VarianceThreshold cutoff. 0.0 removes only constant features.
    tuning_subsample : int
        Max rows used for alpha-grid tuning in the large regime.
    verbose : bool
        If True, log rule banners and regime info.
    """

    def __init__(
        self,
        *,
        regime: str | None = None,
        alpha_values: list | np.ndarray | None = None,
        l1_ratio: float | None = None,
        cv: int | None = None,
        max_iter: int = 20_000,
        n_jobs: int = -1,
        random_state: int | None = 42,
        variance_threshold: float = 0.0,
        tuning_subsample: int = 10_000,
    ):
        self.regime = regime
        self.alpha_values = alpha_values
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.variance_threshold = variance_threshold
        self.tuning_subsample = tuning_subsample

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "BaseLinearRegressor":
        if not _FIT_DEPS_AVAILABLE:
            raise ImportError(
                "Training requires scikit-learn. Install with: pip install 'lazyqsar[fit]'"
            )
        logger.rule("BaseLinearRegressor")

        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float)

        n, p = X.shape
        self.n_features_in_ = p

        # --- Regime ---
        self.regime_ = self.regime if self.regime is not None else _detect_regime(n, p)

        # --- Heuristics ---
        effective_l1_ratio = (
            self.l1_ratio if self.l1_ratio is not None else _default_l1_ratio(n, p)
        )
        effective_cv = self.cv if self.cv is not None else _cv_from_n(n)
        scoring = _regression_scoring(y)

        kf = KFold(
            n_splits=max(2, effective_cv), shuffle=True, random_state=self.random_state
        )

        # --- Imbalance weights: inverse-frequency over quantile bins ---
        sample_weight = _regression_sample_weight(y)

        # --- Preprocessing: VarianceThreshold only ---
        self._vt = VarianceThreshold(threshold=self.variance_threshold)
        try:
            X_vt = self._vt.fit_transform(X)
        except ValueError:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        if X_vt.shape[1] == 0:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        # --- Dispatch ---
        if self.regime_ == "standard":
            self._fit_standard(X_vt, y, n, p, kf, scoring, sample_weight)
        elif self.regime_ == "high_dim":
            self._fit_high_dim(
                X_vt, y, n, p, kf, scoring, effective_l1_ratio, sample_weight
            )
        elif self.regime_ == "large":
            self._fit_large(
                X_vt, y, n, p, kf, scoring, effective_l1_ratio, sample_weight
            )
        else:
            raise ValueError(f"Unknown regime: {self.regime_!r}.")

        # --- Consolidated feature mask ---
        vt_support = self._vt.get_support()
        if self._sfm is not None:
            sfm_support = self._sfm.get_support()
            combined = np.zeros(p, dtype=bool)
            idx = np.where(vt_support)[0]
            combined[idx[sfm_support]] = True
            self.feature_mask_ = combined
        else:
            self.feature_mask_ = vt_support

        # --- Coef extraction ---
        if hasattr(self._estimator, "coef_"):
            self.coef_ = self._estimator.coef_
        else:
            self.coef_ = None

        logger.rule("Done")
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict(self._transform(X))

    def score(self, X, y) -> float:
        return r2_score(y, self.predict(X))

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_mask_"])
        return np.where(self.feature_mask_)[0].astype(str)

    def to_onnx(self, path: str) -> None:
        """
        Export the trained model to an ONNX file.

        Accepts a float32 input named ``"float_input"`` with shape
        ``(n_samples, n_features_in_)`` (raw, unprocessed features).
        Produces a single output ``"variable"`` (float32, shape [n_samples, 1]).

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model.onnx"``.
        """
        check_is_fitted(self, attributes=["_estimator"])
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import Pipeline

        steps = [("vt", self._vt)]
        if self._sfm is not None:
            steps.append(("sfm", self._sfm))
        steps.append(("reg", self._estimator))

        pipeline = Pipeline(steps)
        initial_types = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def save(self, directory: str) -> None:
        """
        Save the trained model to a directory.

        Always writes ``linear.json`` (fit metadata) and ``linear.onnx``.

        Parameters
        ----------
        directory : str
            Destination directory (created if it does not exist).
        """
        check_is_fitted(self, attributes=["_estimator"])
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "linear.onnx"))
        metadata = {
            "task": "regression",
            "format": "onnx",
            "regime": self.regime_,
            "n_features_in": self.n_features_in_,
            "feature_mask": self.feature_mask_.tolist(),
        }
        with open(os.path.join(directory, "linear.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transform(self, X) -> np.ndarray:
        X_vt = self._vt.transform(X)
        if self._sfm is not None:
            return self._sfm.transform(X_vt)
        return X_vt

    def _fit_standard(self, X, y, n, p, kf, scoring, sample_weight):
        self._sfm = None
        alpha_grid = (
            np.asarray(self.alpha_values)
            if self.alpha_values is not None
            else _default_alpha_grid(n, p)
        )

        self._estimator = RidgeCV(
            alphas=alpha_grid,
            cv=kf,
            scoring=scoring,
        )
        self._estimator.fit(X, y, sample_weight=sample_weight)

    def _fit_high_dim(self, X, y, n, p, kf, scoring, l1_ratio, sample_weight):
        # Pre-filter with Lasso to reduce dimensionality
        max_feat = _sfm_max_features(n, p)
        pre_lasso = Lasso(alpha=1.0, max_iter=self.max_iter)
        self._sfm = SelectFromModel(
            estimator=pre_lasso, max_features=max_feat, threshold=-np.inf
        )
        self._sfm.fit(X, y, sample_weight=sample_weight)
        X_sfm = self._sfm.transform(X)

        if X_sfm.shape[1] == 0:
            self._sfm = SelectFromModel(
                estimator=pre_lasso, max_features=1, threshold=-np.inf
            )
            self._sfm.fit(X, y, sample_weight=sample_weight)
            X_sfm = self._sfm.transform(X)

        alpha_grid = (
            np.asarray(self.alpha_values)
            if self.alpha_values is not None
            else _default_alpha_grid(n, X_sfm.shape[1])
        )

        self._estimator = ElasticNetCV(
            alphas=alpha_grid,
            l1_ratio=[l1_ratio],
            cv=kf,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        )
        self._estimator.fit(X_sfm, y, sample_weight=sample_weight)

    def _fit_large(self, X, y, n, p, kf, scoring, l1_ratio, sample_weight):
        self._sfm = None
        alpha_grid = (
            np.asarray(self.alpha_values)
            if self.alpha_values is not None
            else _default_alpha_grid(n, p)
        )

        # Tune alpha on a random subsample
        sub_size = min(self.tuning_subsample, n)
        if sub_size < n:
            rng = np.random.default_rng(self.random_state)
            sub_idx = rng.choice(n, size=sub_size, replace=False)
            X_sub, y_sub = X[sub_idx], y[sub_idx]
            sw_sub = sample_weight[sub_idx]
        else:
            X_sub, y_sub = X, y
            sw_sub = sample_weight

        base_sgd = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=_SGD_VALIDATION_FRACTION,
            n_iter_no_change=_SGD_N_ITER_NO_CHANGE,
            random_state=self.random_state,
        )

        sub_cv = min(kf.n_splits, max(2, sub_size // 50))
        sub_kf = KFold(n_splits=sub_cv, shuffle=True, random_state=self.random_state)

        gs = GridSearchCV(
            estimator=base_sgd,
            param_grid={"alpha": alpha_grid},
            cv=sub_kf,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=False,
        )
        gs.fit(X_sub, y_sub, sample_weight=sw_sub)
        best_alpha = gs.best_params_["alpha"]

        self._estimator = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            alpha=best_alpha,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=_SGD_VALIDATION_FRACTION,
            n_iter_no_change=_SGD_N_ITER_NO_CHANGE,
            random_state=self.random_state,
        )
        self._estimator.fit(X, y, sample_weight=sample_weight)


# ---------------------------------------------------------------------------
# Artifact: load a saved model for inference
# ---------------------------------------------------------------------------


def _apply_calibrator_artifact(proba: np.ndarray, cal: dict) -> np.ndarray:
    """Apply a saved calibrator dict to a (n, 2) probability array."""
    raw_p1 = proba[:, 1]
    if cal["method"] == "isotonic":
        p1 = np.clip(np.interp(raw_p1, cal["X_thresholds"], cal["y_thresholds"]), 0, 1)
    else:  # platt
        A, B = cal["coef"], cal["intercept"]
        p1 = 1.0 / (1.0 + np.exp(-(A * raw_p1 + B)))
    return np.column_stack([1 - p1, p1])


class BaseLinearArtifact:
    """
    Load a saved linear model for forward inference.

    Reads the files written by BaseLinearClassifier.save() or BaseLinearRegressor.save():
      - linear.onnx  — model binary
      - linear.json  — fit metadata (task, format, regime, …)

    Usage
    -----
    artifact = BaseLinearArtifact.load("path/to/directory")
    predictions = artifact.run(X)
    """

    def __init__(self):
        self._session = None  # onnxruntime.InferenceSession
        self.metadata: dict = {}
        self.task: str = ""
        self._cal = None
        self.decision_cutoff_raw: float = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_raw_source: str = "default_0.5"
        self.decision_cutoff_proba: float = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_rank: float = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_logit: float = 0.0

    @classmethod
    def load(cls, directory: str) -> "BaseLinearArtifact":
        """Load a model saved with BaseLinearClassifier.save() or BaseLinearRegressor.save()."""
        json_path = os.path.join(directory, "linear.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"No metadata found at {json_path!r}")

        artifact = cls()
        with open(json_path) as f:
            artifact.metadata = json.load(f)
        artifact.task = artifact.metadata["task"]

        artifact._cal = artifact.metadata.get("calibrator", None)
        artifact.decision_cutoff_raw = float(
            artifact.metadata.get("decision_cutoff_raw", _DEFAULT_DECISION_CUTOFF)
        )
        artifact.decision_cutoff_raw_source = artifact.metadata.get(
            "decision_cutoff_raw_source", "default_0.5"
        )
        artifact.decision_cutoff_proba = float(
            artifact.metadata.get("decision_cutoff_proba", _DEFAULT_DECISION_CUTOFF)
        )
        artifact.decision_cutoff_rank = float(
            artifact.metadata.get("decision_cutoff_rank", _DEFAULT_DECISION_CUTOFF)
        )
        artifact.decision_cutoff_logit = float(
            artifact.metadata.get("decision_cutoff_logit", 0.0)
        )

        import onnxruntime as rt

        onnx_path = os.path.join(directory, "linear.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"No ONNX model found at {onnx_path!r}")
        artifact._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        return artifact

    def run(self, X) -> np.ndarray:
        """
        Run forward inference on X.

        Returns
        -------
        Classification : ndarray of shape (n_samples, 2)
            Columns are [P(class=0), P(class=1)], matching predict_proba.
        Regression : ndarray of shape (n_samples,)
            Predicted continuous values.
        """
        if self._session is None:
            raise RuntimeError("No model loaded. Call BaseLinearArtifact.load() first.")

        X_f32 = np.asarray(X, dtype=np.float32)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: X_f32})
        if self.task == "classification":
            # skl2onnx classifier pipeline: outputs[0]=labels, outputs[1]=prob dict/array
            prob_raw = outputs[1]
            if isinstance(prob_raw, list):
                # list of dicts [{0: p0, 1: p1}, …]
                proba = np.array(
                    [[d[k] for k in sorted(d)] for d in prob_raw], dtype=np.float64
                )
            else:
                proba = np.asarray(prob_raw, dtype=np.float64)
                if proba.ndim == 1:
                    proba = np.column_stack([1 - proba, proba])
            if self._cal is not None:
                proba = _apply_calibrator_artifact(proba, self._cal)
            return proba
        else:
            return np.asarray(outputs[0], dtype=np.float64).ravel()

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Return class predictions using the stored decision cutoff by default."""
        if self.task != "classification":
            raise RuntimeError(
                "predict() is only available for classification artifacts."
            )
        threshold = self.decision_cutoff_raw if cutoff is None else float(cutoff)
        classes = np.asarray(self.metadata.get("classes", [0, 1]))
        return classes[(self.run(X)[:, 1] >= threshold).astype(int)]
