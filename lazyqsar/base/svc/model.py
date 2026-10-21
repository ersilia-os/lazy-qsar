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

# ---------------------------------------------------------------------------
# Fit-time dependencies — NOT required for ONNX inference.
# Guarded so lean [descriptors]-only installs can import this module to reach
# BaseSVCArtifact without errors.
# ---------------------------------------------------------------------------
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.validation import check_array, check_is_fitted

    _FIT_DEPS_AVAILABLE = True
except ImportError:

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class ClassifierMixin:  # type: ignore[no-redef]
        pass

    roc_auc_score = None  # type: ignore[assignment]
    train_test_split = check_array = check_is_fitted = None  # type: ignore[assignment]
    _FIT_DEPS_AVAILABLE = False

from .inspector import inspect as _inspect, DatasetProfile
from .presets import (
    svc_heuristic_params,
    svc_default_params,
    svc_linear_params,
    svc_balanced_rbf_params,
)
from ._common import (
    _MAX_COST_MULTIPLIER,
    _ONNX_MAX_BYTES,
    _RANDOM_STATE,
    _VAL_FRACTION,
    _VAL_MIN_ROWS,
    _SVMHead,
    _fold_jobs,
    svm_cost as _svc_cost,
)
from lazyqsar.base._head import (
    ClassifierHead,
    require_fit_deps,
)
from lazyqsar.base._race import min_gain_threshold
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import probe_rows
from lazyqsar.utils.splits import make_stratified_oof_splits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VAL_MIN_MINORITY = 15

# Deterministic probe used to record the true sign of decision_function at save time,
# so the artifact can identify which ONNX score column to read; defined once, with the
# artifact that reads it (SVCArtifact._resolve_score_column).
from lazyqsar.artifacts.svc import _SCORE_CANARY_ROWS, _SCORE_CANARY_SEED  # noqa: E402


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
    to the sklearn estimator.  For SVC, Platt probabilities are never requested so
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

        # `probability` is deliberately not passed. This code has always wanted
        # probability=False -- raw decision values are what gets exported, and calibration
        # is applied externally -- and False is sklearn's default, so omitting it builds
        # the identical estimator. sklearn 1.9 deprecated the parameter outright and
        # removes it in 1.11: passing it emits a FutureWarning on every fit today and
        # would be a TypeError then. Omitting it is correct on every supported version.
        return SVC(
            **common,
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            random_state=params.get("random_state", 42),
        )


def _decision_scores(svc, X: np.ndarray) -> np.ndarray:
    """Return shape-(n,) decision function scores; positive → class 1."""
    scores = svc.decision_function(X)
    return np.asarray(scores, dtype=np.float64).ravel()


def _min_gain_threshold(profile: DatasetProfile, y_train: np.ndarray) -> float:
    """AUC margin a challenger preset must win by; noise scales with the minority count."""
    n_eff = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    return min_gain_threshold(n_eff, len(y_train))


def _validation_split(
    X, y: np.ndarray, profile: DatasetProfile, random_state: int = _RANDOM_STATE
):
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


def _portfolio_select_svc(
    X, y: np.ndarray, profile: DatasetProfile, random_state: int = _RANDOM_STATE
):
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
        ("heuristic", svc_heuristic_params(profile)),
        ("default", svc_default_params(profile)),
        ("linear", svc_linear_params(profile)),
        ("balanced_rbf", svc_balanced_rbf_params(profile)),
    ]

    X_tr, X_val, y_tr, y_val, did_split = _validation_split(
        X, y, profile, random_state=random_state
    )
    if not did_split:
        # Too small: use heuristic directly
        params = svc_heuristic_params(profile)
        return "heuristic", params, {}

    n_tr = len(y_tr)
    default_cost = _svc_cost(
        dict(candidates)[1][1]
        if False
        else next(p for n, p in candidates if n == "default"),
        n_tr,
        n_features,
    )
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


def _svc_onnx_bytes(svc, n_features: int) -> bytes:
    """
    Export SVC or LinearSVC to ONNX via skl2onnx.

    The exported model outputs raw decision function scores (NOT probabilities).
    At inference, sigmoid is applied in the artifact to obtain [0,1] scores,
    then the stored calibrator converts these to calibrated probabilities.

    Output layout (opset 15), the same for both estimator types:
      output[0]: int64 labels, shape (n,)
      output[1]: float32 scores, shape (n, 2) — two mutually-negated columns,
                 [decision_function, -decision_function] in SOME order

    The two converters do not agree on that order: decision_function is column 1
    for LinearSVC and column 0 for kernel SVC. Never index this output by position.
    BaseSVCArtifact._resolve_score_column works out which column it is by checking
    which one's sign reproduces output[0], since a binary SVC predicts class 1
    exactly when decision_function > 0.

    Options differ by estimator type:
      SVC (kernel)  — zipmap=False: suppresses the dict output, returns a float array
      LinearSVC     — raw_scores=True: output[1] carries decision scores rather than
                      probabilities (output[0] is still the int64 label)
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
    del onnx_model.graph.value_info[:]
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    return onnx_model.SerializeToString()


# ---------------------------------------------------------------------------
# BaseSVCClassifier
# ---------------------------------------------------------------------------


class BaseSVCClassifier(_SVMHead, ClassifierHead, BaseEstimator, ClassifierMixin):
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

    _onnx_kind = "decision"
    # A kernel sums over every support vector and feature in float32: ~1e-4 of rounding.
    _onnx_tolerance = 1e-4
    _fitted_attr = "svc_"

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
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        if self._wants_calibration(y):
            return self.calibrate(X, y)
        return self._fit_raw(X, y)

    # ------------------------------------------------------------------
    # Internal raw training
    # ------------------------------------------------------------------

    @staticmethod
    def _make(params: dict):
        return _make_svc(params)

    def _linear_fallback_params(self, profile) -> dict:
        return {
            "use_linear": True,
            "C": min(self.params_["C"], 10.0),
            "class_weight": self.params_.get("class_weight", "balanced"),
            "max_iter": self.params_.get("max_iter", 5_000),
            "tol": self.params_.get("tol", 1e-3),
            "random_state": self.random_state,
        }

    def _fit_raw(self, X, y) -> "BaseSVCClassifier":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        X = check_array(X, dtype="numeric", accept_sparse=False)
        y = np.asarray(y, dtype=int).ravel()
        self._onnx_probe_ = probe_rows(X)
        profile = _inspect(X, y, task="classification")
        self.profile_ = profile
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

        # Phase 2: retrain on full data (with the ONNX size guard for kernel SVC).
        _t_p2 = _time.perf_counter()
        logger.rule("Phase 2 — full retraining")
        logger.info(
            f"preset={best_name} | n={profile.n_samples} | "
            f"{'linear' if self._use_linear_ else 'kernel'} | C={best_params['C']}"
        )
        self._fit_final(X, y, profile)
        self.timing_["phase2_refit"] = _time.perf_counter() - _t_p2

        self._set_default_cutoffs()
        self.classes_ = np.array([0, 1])
        self._bind_onnx()
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

    def calibrate(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseSVCClassifier":
        """
        Full calibration workflow.

        Portfolio selection runs ONCE via _fit_raw().  k-fold OOF then reuses
        the winning params without re-running portfolio comparison per fold.
        The calibrator is fitted on sigmoid(decision_function) OOF scores.
        """
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        X = check_array(X, dtype="numeric", accept_sparse=False)
        y = np.asarray(y, dtype=int).ravel()
        k, fold_splits = make_stratified_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseSVCClassifier.calibrate: full fit on n={len(y)} (portfolio runs once)"
        )
        self._fit_raw(X, y)  # sets self.params_, self.svc_, self.preset_name_
        logger.info(
            f"calibrate: {k}-fold OOF  preset={self.preset_name_}  "
            f"{'linear' if self._use_linear_ else 'kernel'}"
        )

        def fit_predict(train_idx, val_idx):
            fold_svc = _make_svc(self.params_)
            fold_svc.fit(X[train_idx], y[train_idx])
            return _decision_scores(fold_svc, X[val_idx])

        # The calibrator, cutoffs and ranks live on sigmoid(decision_function).
        oof_raw = self._oof_loop(
            fold_splits, fit_predict, n_jobs=_fold_jobs(len(fold_splits))
        )
        self._finish_calibration(y, _sigmoid(oof_raw))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _raw_native(self, X) -> np.ndarray:
        """``decision_function`` from the scikit-learn estimator itself."""
        return _decision_scores(self.svc_, X)

    def _onnx_bytes(self) -> bytes:
        return _svc_onnx_bytes(self.svc_, self.profile_.n_features)

    def _check_X(self, X):
        return check_array(X, dtype="numeric", accept_sparse=False)

    def _score1(self, X) -> np.ndarray:
        """sigmoid(decision_function): the pre-calibration [0, 1] score."""
        return _sigmoid(self._raw(X))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

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

        profile_dict = (
            dataclasses.asdict(self.profile_)
            if dataclasses.is_dataclass(self.profile_)
            else dict(vars(self.profile_))
        )
        metadata = {
            "task": "classification",
            "format": "onnx",
            "preset_name": self.preset_name_,
            "use_linear": self._use_linear_,
            "n_features_in": self.profile_.n_features,
            "classes": [0, 1],
            "params": self.params_,
            "profile": profile_dict,
            "portfolio_scores": self.portfolio_scores_,
        }
        metadata.update(self._task_metadata())

        # skl2onnx emits the binary-SVC score output as two mutually-negated columns and
        # their order is not contractual across versions. Reading the wrong one silently
        # inverts every prediction this head makes, so record decision_function on a
        # deterministic probe and let the artifact match against it.
        probe = np.random.default_rng(_SCORE_CANARY_SEED).standard_normal(
            (_SCORE_CANARY_ROWS, int(self.profile_.n_features))
        )
        # The column the fitted head was bound to; the artifact must resolve the same one.
        if getattr(self, "_onnx_column_", None) is not None:
            metadata["score_column"] = int(self._onnx_column_)
        metadata["score_canary"] = {
            "seed": _SCORE_CANARY_SEED,
            "n_rows": _SCORE_CANARY_ROWS,
            "decision": _decision_scores(self.svc_, probe).tolist(),
        }

        with open(os.path.join(directory, "svc.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# Inference: one implementation per head, in lazyqsar.artifacts.
from lazyqsar.artifacts.svc import SVCArtifact as BaseSVCArtifact  # noqa: E402,F401
