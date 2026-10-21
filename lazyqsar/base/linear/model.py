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
        f_regression,
    )
    from sklearn.linear_model import (
        ElasticNet,
        ElasticNetCV,
        Lasso,
        LogisticRegression,
        LogisticRegressionCV,
        Ridge,
        RidgeCV,
        SGDClassifier,
        SGDRegressor,
    )
    from sklearn.metrics import r2_score, roc_auc_score
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

    SelectFromModel = SelectKBest = VarianceThreshold = f_classif = f_regression = None  # type: ignore[assignment,misc]
    ElasticNet = ElasticNetCV = Lasso = LogisticRegression = LogisticRegressionCV = None  # type: ignore[assignment,misc]
    Ridge = RidgeCV = SGDClassifier = SGDRegressor = None  # type: ignore[assignment,misc]
    r2_score = roc_auc_score = None  # type: ignore[assignment]
    GridSearchCV = KFold = StratifiedKFold = StratifiedShuffleSplit = None  # type: ignore[assignment,misc]
    LabelEncoder = None  # type: ignore[assignment,misc]
    check_array = check_is_fitted = None  # type: ignore[assignment]
    _FIT_DEPS_AVAILABLE = False

from lazyqsar.base._head import ClassifierHead, RegressorHead, require_fit_deps
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import probe_rows
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.splits import make_quantile_oof_splits, make_stratified_oof_splits


def _linear_pipeline_onnx(model, last_step: str) -> bytes:
    """VarianceThreshold, optional selector and estimator, as one ONNX graph."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import Pipeline

    from lazyqsar.base.preprocessing.reducer import (
        _register_correlation_filter_onnx_converter,
    )

    # The Gather converter for VarianceThreshold (skl2onnx's own breaks on one column).
    _register_correlation_filter_onnx_converter()
    steps = [("vt", model._vt)]
    if model._sfm is not None:
        steps.append(("sfm", model._sfm))
    steps.append((last_step, model._estimator))
    initial_types = [("float_input", FloatTensorType([None, model.n_features_in_]))]
    return convert_sklearn(
        Pipeline(steps), initial_types=initial_types
    ).SerializeToString()


def _sklearn_kwargs(estimator, **kwargs):
    if "use_legacy_attributes" in inspect.signature(estimator).parameters:
        kwargs["use_legacy_attributes"] = False
    return kwargs


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


_CLASSIFIER_STANDARD_MAX_CV_WORK = 2_000_000
_CLASSIFIER_HIGHDIM_MAX_CV_WORK = 1_000_000
_SGD_VALIDATION_FRACTION = 0.1
_SGD_N_ITER_NO_CHANGE = 10


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


class _LinearHead(BaseEstimator):
    """The linear-specific half of both linear heads.

    Shared: the VarianceThreshold front end, the consolidated feature mask, the
    pipeline export (VarianceThreshold, optional selector, estimator), the fold loop
    (VarianceThreshold refit per fold) and the common part of ``save``. A subclass
    fits its regime-specific estimator and provides
    ``_fit_predict_fold(X_tr_vt, y_tr, X_val_vt)``.
    """

    _task = ""
    _onnx_last_step = ""
    _fitted_attr = "_estimator"

    def _fit_variance_threshold(self, X):
        """VarianceThreshold only; scaling is the caller's responsibility."""
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
        return X_vt

    def _finish_fit(self, p: int) -> None:
        """Consolidated feature mask, coefficients, and binding to the ONNX graph."""
        vt_support = self._vt.get_support()
        if self._sfm is not None:
            sfm_support = self._sfm.get_support()
            combined = np.zeros(p, dtype=bool)
            idx = np.where(vt_support)[0]
            combined[idx[sfm_support]] = True
            self.feature_mask_ = combined
        else:
            self.feature_mask_ = vt_support
        self.coef_ = getattr(self._estimator, "coef_", None)
        self._bind_onnx()

    def _transform(self, X) -> np.ndarray:
        X_vt = self._vt.transform(X)
        if self._sfm is not None:
            return self._sfm.transform(X_vt)
        return X_vt

    def _oof(self, X, y, fold_splits) -> np.ndarray:
        def fit_predict(train_idx, val_idx):
            # Refit VarianceThreshold on fold training data (cheap)
            vt = VarianceThreshold(threshold=self.variance_threshold)
            X_tr_vt = vt.fit_transform(X[train_idx])
            X_val_vt = vt.transform(X[val_idx])
            return self._fit_predict_fold(X_tr_vt, y[train_idx], X_val_vt)

        return self._oof_loop(fold_splits, fit_predict)

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_mask_"])
        return np.where(self.feature_mask_)[0].astype(str)

    def _onnx_bytes(self) -> bytes:
        return _linear_pipeline_onnx(self, self._onnx_last_step)

    def _save_extras(self) -> dict:
        return {}

    def save(self, directory: str) -> None:
        """
        Save the trained model to a directory.

        Always writes ``linear.json`` (fit metadata) and ``linear.onnx``.

        Parameters
        ----------
        directory : str
            Destination directory (created if it does not exist).
        """
        check_is_fitted(self, attributes=[self._fitted_attr])
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "linear.onnx"))
        metadata = {
            "task": self._task,
            "format": "onnx",
            "regime": self.regime_,
            "n_features_in": self.n_features_in_,
            **self._save_extras(),
            "feature_mask": self.feature_mask_.tolist(),
        }
        metadata.update(self._task_metadata())
        with open(os.path.join(directory, "linear.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


class BaseLinearClassifier(_LinearHead, ClassifierHead):
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

    _task = "classification"
    _onnx_last_step = "clf"

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

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "BaseLinearClassifier":
        """Fit the model. If calibrated=True, runs calibrate() for OOF calibration."""
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        if self._wants_calibration(y):
            return self.calibrate(X, y)
        return self._fit_raw(X, y)

    def _fit_raw(self, X, y) -> "BaseLinearClassifier":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        logger.rule("BaseLinearClassifier")

        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y)
        self._onnx_probe_ = probe_rows(X)

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
        X_vt = self._fit_variance_threshold(X)

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
        self._finish_fit(p)

        self.classes_ = self._label_encoder.classes_
        self._set_default_cutoffs()
        logger.info(
            "decision cutoff: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_raw_source_}"
        )

        logger.rule("Done")
        return self

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        return self._label_encoder.inverse_transform(super().predict(X, cutoff))

    def _raw_native(self, X) -> np.ndarray:
        """Raw P(y=1) from the scikit-learn pipeline itself."""
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict_proba(self._transform(X))[:, 1]

    def calibrate(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseLinearClassifier":
        """
        Collect out-of-fold predicted probabilities via stratified k-fold CV,
        then fit the calibrator on them.

        Hyperparameter selection (C-search / alpha-grid) runs ONCE on the full
        data via self.fit().  Each fold reuses the pre-selected hyperparameter
        with a single model fit — no inner LogisticRegressionCV or GridSearchCV.
        """
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=int)
        k, fold_splits = make_stratified_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )

        # Step 1: full fit — hyperparameter search runs ONCE here
        logger.info(
            f"BaseLinearClassifier.calibrate: full fit on n={len(y)} (C/alpha search runs once)"
        )
        self._fit_raw(X, y)  # sets self.regime_, self._vt, self._sfm, self._estimator

        regime = self.regime_
        if regime in ("standard", "high_dim"):
            logger.info(
                f"calibrate: regime={regime}  best_C={float(np.ravel(self._estimator.C_)[0]):.4g}"
            )
        else:  # large
            logger.info(
                f"calibrate: regime={regime}  best_alpha={float(self._estimator.alpha):.4g}"
            )
        logger.info(
            "calibrate: "
            f"{k}-fold OOF (no inner CV in folds) | fold_solver={_classifier_solver_label(regime, calibrated=True)}"
        )
        oof_raw = self._oof(X, y, fold_splits)
        self._finish_calibration(y, oof_raw)
        return self

    def _fit_predict_fold(self, X_tr_vt, y_tr, X_val_vt) -> np.ndarray:
        """One fold with the full fit's hyperparameter frozen (no inner CV)."""
        regime = self.regime_
        if regime == "standard":
            fold_est = LogisticRegression(
                **_sklearn_kwargs(
                    LogisticRegression,
                    C=float(np.ravel(self._estimator.C_)[0]),
                    solver="saga",
                    penalty="l1",
                    class_weight=self.class_weight,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
            )
            fold_est.fit(X_tr_vt, y_tr)
            return fold_est.predict_proba(X_val_vt)[:, 1]
        if regime == "high_dim":
            # Refit fast univariate F-test on fold data (O(n*p), no inner model)
            fold_sfm = SelectKBest(f_classif, k=self._sfm.k)
            fold_sfm.fit(X_tr_vt, y_tr)
            fold_est = LogisticRegression(
                **_sklearn_kwargs(
                    LogisticRegression,
                    C=float(np.ravel(self._estimator.C_)[0]),
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=self._estimator.l1_ratios[0],
                    class_weight=self.class_weight,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
            )
            fold_est.fit(fold_sfm.transform(X_tr_vt), y_tr)
            return fold_est.predict_proba(fold_sfm.transform(X_val_vt))[:, 1]
        fold_est = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=float(self._estimator.l1_ratio),
            alpha=float(self._estimator.alpha),
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=_SGD_VALIDATION_FRACTION,
            n_iter_no_change=_SGD_N_ITER_NO_CHANGE,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        fold_est.fit(X_tr_vt, y_tr)
        return fold_est.predict_proba(X_val_vt)[:, 1]

    def score(self, X, y) -> float:
        proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)

    def _save_extras(self) -> dict:
        return {"classes": self._label_encoder.classes_.tolist()}

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


def _ridge_alpha_grid(n: int) -> np.ndarray:
    """Ridge penalties on the scale sklearn's ``Ridge`` uses.

    ``Ridge`` minimises ``||y - Xw||^2 + alpha * ||w||^2`` with the loss *summed* over
    samples, so for standardised features ``X^T X`` is ~``n`` on its diagonal and a
    meaningful ``alpha`` scales with ``n``. The grid spans ``alpha / n`` from 1e-4
    (barely regularised) to 1e2 (shrunk hard) in 13 log steps; the leave-one-out path
    makes the full grid essentially free.
    """
    return float(n) * np.logspace(-4, 2, 13)


def _elasticnet_alpha_grid(X: np.ndarray, y: np.ndarray, l1_ratio: float) -> np.ndarray:
    """ElasticNet's own regularisation path: 20 values down from ``alpha_max``.

    ``alpha_max`` is the smallest penalty at which every coefficient is zero (glmnet's
    lambda_max); the path runs three decades below it.
    """
    n = X.shape[0]
    y_c = np.asarray(y, dtype=float) - float(np.mean(y))
    X_c = np.asarray(X, dtype=float) - np.asarray(X, dtype=float).mean(axis=0)
    alpha_max = float(np.abs(X_c.T @ y_c).max()) / (n * max(l1_ratio, 1e-3))
    alpha_max = max(alpha_max, 1e-6)
    return np.logspace(np.log10(alpha_max) - 3.0, np.log10(alpha_max), 20)


# ---------------------------------------------------------------------------
# Linear regressor
# ---------------------------------------------------------------------------


class BaseLinearRegressor(_LinearHead, RegressorHead):
    """
    Linear regression with embedded feature selection.

    Automatically selects solver and regularization regime based on (n, p):
      - standard  (n<=50K, p<=n): RidgeCV (L2), efficient leave-one-out over
                                  ``alpha = n * logspace(-4, 2, 13)``
      - high_dim  (n<=50K, p>n):  ElasticNetCV over its own alpha path, after a
                                  univariate ``SelectKBest(f_regression)`` to <= 2n
      - large     (n>50K):        SGDRegressor + ElasticNet,  subsample alpha tuning

    Feature scaling is NOT applied internally. Standardize X before calling fit().
    Model selection minimises squared error throughout (the regression metric
    LazyQSAR selects presets by), and every sample is weighted equally.

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
    oof : bool
        If True (default), collect out-of-fold predictions after the full fit
        (quantile-stratified k-fold, hyperparameters frozen from the full fit).
        These feed the ensemble gate and the OOF ECDF ranker.

    Attributes (after .fit())
    --------------------------
    timing_ : dict
    oof_preds_ : ndarray (n,)   — out-of-fold predictions (when oof=True)
    oof_rho_ : float or None    — Spearman ρ of OOF predictions
    train_rho_ : float or None  — Spearman ρ of in-sample predictions
    """

    _task = "regression"
    _onnx_last_step = "reg"

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
        oof: bool = True,
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
        self.oof = oof

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "BaseLinearRegressor":
        """Fit the regressor, then collect OOF predictions when ``oof=True``."""
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        if self.oof:
            return self.collect_oof(X, y)
        return self._fit_raw(X, y)

    def _fit_raw(self, X, y) -> "BaseLinearRegressor":
        logger.rule("BaseLinearRegressor")
        _t_hp = _time.perf_counter()

        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float)
        self._onnx_probe_ = probe_rows(X)

        n, p = X.shape
        self.n_features_in_ = p

        # --- Regime ---
        self.regime_ = self.regime if self.regime is not None else _detect_regime(n, p)

        # --- Heuristics ---
        effective_l1_ratio = (
            self.l1_ratio if self.l1_ratio is not None else _default_l1_ratio(n, p)
        )
        effective_cv = self.cv if self.cv is not None else _cv_from_n(n)
        scoring = "neg_mean_squared_error"

        kf = KFold(
            n_splits=max(2, effective_cv), shuffle=True, random_state=self.random_state
        )

        # Every sample counts equally. (A former "inverse-frequency" weighting over
        # quantile bins was ~1 for every sample by construction -- quantile bins hold
        # equal counts -- and real inverse-density weighting trades bulk accuracy for
        # the extremes, which is not what QSAR ranking needs.)
        sample_weight = None
        X_vt = self._fit_variance_threshold(X)

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
        self._finish_fit(p)

        self.timing_ = {"hparam_search": _time.perf_counter() - _t_hp}
        self.train_rho_ = spearman_score(y, self.predict(X))
        logger.rule("Done")
        return self

    def _fold_estimator(self):
        """Fresh estimator with the hyperparameters frozen from the full fit."""
        if self.regime_ == "standard":
            return Ridge(alpha=float(self._estimator.alpha_))
        if self.regime_ == "high_dim":
            return ElasticNet(
                alpha=float(self._estimator.alpha_),
                l1_ratio=float(self._estimator.l1_ratio_),
                max_iter=self.max_iter,
            )
        from sklearn.base import clone

        return clone(self._estimator)

    def _fit_predict_fold(self, X_tr_t, y_tr, X_val_t) -> np.ndarray:
        if self._sfm is not None:
            # The same univariate selector as the full fit, refit on fold data.
            n_keep = int(self._sfm.get_support().sum())
            fold_sfm = SelectKBest(f_regression, k=min(n_keep, X_tr_t.shape[1]))
            X_tr_t = fold_sfm.fit_transform(X_tr_t, y_tr)
            X_val_t = fold_sfm.transform(X_val_t)
        fold_est = self._fold_estimator()
        fold_est.fit(X_tr_t, y_tr)
        return fold_est.predict(X_val_t)

    def collect_oof(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseLinearRegressor":
        """
        Full fit (alpha search runs once), then quantile-stratified k-fold OOF
        predictions reusing the selected regularization — no inner CV per fold.
        """
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float).ravel()
        k, fold_splits = make_quantile_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseLinearRegressor.collect_oof: full fit on n={len(y)} (alpha search runs once)"
        )
        self._fit_raw(X, y)
        logger.info(f"collect_oof: {k}-fold OOF | regime={self.regime_}")
        self._finish_oof(y, self._oof(X, y, fold_splits))
        return self

    def _raw_native(self, X) -> np.ndarray:
        """Predictions from the scikit-learn pipeline itself."""
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict(self._transform(X))

    def score(self, X, y) -> float:
        return r2_score(y, self.predict(X))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_standard(self, X, y, n, p, kf, scoring, sample_weight):
        self._sfm = None
        alpha_grid = (
            np.asarray(self.alpha_values)
            if self.alpha_values is not None
            else _ridge_alpha_grid(n)
        )
        # cv=None selects RidgeCV's closed-form leave-one-out path: exact, and far
        # cheaper than refitting per fold.
        self._estimator = RidgeCV(alphas=alpha_grid, cv=None)
        self._estimator.fit(X, y)

    def _fit_high_dim(self, X, y, n, p, kf, scoring, l1_ratio, sample_weight):
        # A univariate F-test selector: deterministic, O(n*p), and the same selector
        # the out-of-fold folds refit -- so OOF predictions describe this model.
        self._sfm = SelectKBest(f_regression, k=min(_sfm_max_features(n, p), p))
        X_sfm = self._sfm.fit_transform(X, y)

        alpha_grid = (
            np.asarray(self.alpha_values)
            if self.alpha_values is not None
            else _elasticnet_alpha_grid(X_sfm, y, l1_ratio)
        )
        self._estimator = ElasticNetCV(
            alphas=alpha_grid,
            l1_ratio=[l1_ratio],
            cv=kf,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        )
        self._estimator.fit(X_sfm, y)

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
        else:
            X_sub, y_sub = X, y

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
        gs.fit(X_sub, y_sub)
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
        self._estimator.fit(X, y)


# ---------------------------------------------------------------------------
# Artifact: one inference implementation, in lazyqsar.artifacts
# ---------------------------------------------------------------------------

from lazyqsar.artifacts.linear import LinearArtifact as BaseLinearArtifact  # noqa: E402,F401
