"""
Sklearn-compatible estimators backed by automatic XGBoost parameter selection.

Training uses the low-level xgb.train() API with QuantileDMatrix so that
quantile boundaries are computed once on the training split and reused for
the validation split via the ref= parameter.

On .fit(), both estimators run a two-phase portfolio process:

  Phase 2 – Portfolio selection (when a genuine validation split exists):
    Four preset configurations are trained in parallel on the 90% training
    split with early stopping against the 10% validation split:
      1. heuristic  – rule-based params from params.py (dataset-profiling based)
      2. default    – XGBoost out-of-the-box defaults (lr=0.3, max_depth=6)
      3. flaml      – FLAML: 1-NN portfolio selection on meta-features
                      (microsoft/FLAML, flaml/default/xgboost/*.json, MIT license)
      4. autogluon  – AutoGluon tabular XGBoost defaults (lr=0.1, max_depth=6)
                      (autogluon/autogluon tabular/.../xgboost, Apache-2 license)
    The preset with the highest validation metric score wins.

  Phase 3 – Retrain winner on 100% of the data for exactly best_iteration
    rounds (no early stopping), so the final model sees all training samples
    at the round count calibrated on 90%.

  Fallback – When the dataset is too small to split (n < 200), the heuristic
    preset is used directly (no portfolio comparison).

The winning preset name is stored in .preset_name_ after .fit().
The chosen parameters are accessible via .params_.
The best boosting round is in .best_iteration_.
"""

import dataclasses
import json
import os
import time as _time

import numpy as np

# ---------------------------------------------------------------------------
# Fit-time dependencies — NOT required for ONNX inference.
# Guarded so lean [descriptors]-only installs can import this module to reach
# BaseXGBArtifact without errors.
# ---------------------------------------------------------------------------
try:
    from joblib import Parallel, delayed
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.model_selection import train_test_split
    from sklearn.utils.validation import check_is_fitted
    import xgboost as xgb

    _FIT_DEPS_AVAILABLE = True
except ImportError:
    Parallel = delayed = None  # type: ignore[assignment,misc]

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class ClassifierMixin:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass

    train_test_split = check_is_fitted = None  # type: ignore[assignment]
    xgb = None  # type: ignore[assignment]
    _FIT_DEPS_AVAILABLE = False

from .inspector import inspect as _inspect, DatasetProfile
from .params import get_params as _get_params
from .presets import (
    probst_regression_params,
    xgb_default_params,
    flaml_params,
    autogluon_params,
    MAXIMIZE_METRICS,
)
from lazyqsar.base._head import ClassifierHead, RegressorHead, require_fit_deps
from lazyqsar.base._race import min_gain_threshold
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import probe_rows
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.splits import (
    make_quantile_oof_splits,
    make_stratified_oof_splits,
    quantile_strata,
)


_VAL_FRACTION = 0.1
_VAL_MIN_ROWS = 200
# Below this many rows a regression fit has no held-out split for early stopping.
_REGRESSION_MIN_ES_ROWS = 50


_VAL_MIN_MINORITY = 15  # minimum minority-class samples in the validation split
_RANDOM_STATE = 42

# Keys that guide training but are not native XGBoost parameters
_META_KEYS = frozenset({"n_estimators", "early_stopping_rounds"})

# Minimum boosting rounds for phase 2 regardless of early-stopping result.
_PHASE2_MIN_ROUNDS = 100


# Maximum allowed cost for a non-default preset, expressed as a multiple of
# the default preset's cost on the same data.  This scales automatically with
# dataset size and learning rate so that slow presets (e.g. FLAML with lr=0.007
# training for 2000 rounds) are filtered out relative to how long the baseline
# default preset would take.  "default" and "heuristic" are never filtered.
_MAX_COST_MULTIPLIER = 20

# Maximum tree depth used when estimating Stage-1 cost and when running Stage-1
# fast evaluations. Deep presets can otherwise dominate the cost model.
# Stage 2 and Phase 2 still use the preset's true max_depth, so the final model
# is unaffected.
#
# Value of 10 (1024 leaves, cost ratio ≈ 14×):
# Cost at depth 10 stays within the Stage-1 budget for the supported presets.
_STAGE1_MAX_DEPTH = 10

# Number of repeated random 90/10 splits used to estimate best_iteration for
# the winning preset only.  The ranking stage uses a single fast split (see
# _PORTFOLIO_FAST_ROUNDS / _PORTFOLIO_FAST_PATIENCE), so _CV_REPEATS only
# applies to one preset rather than all candidates.
_CV_REPEATS = 3

# Cost ratio above which Stage 2 (best_iteration calibration) is skipped
# entirely and replaced by an analytical heuristic.  Above this ratio Stage 2
# dominates total training time more than Phase 2 itself; the heuristic
# best_iter = patience × (0.1 / lr) is a reliable upper bound that Phase 2
# clips to _PHASE2_MIN_ROUNDS.  Below this ratio, actually training on a 90%
# split gives a more accurate round estimate and is worth the cost.
_STAGE2_SKIP_COST_RATIO = 15

# Budget caps for the fast ranking stage of portfolio selection.
# All presets are compared on a single split with these reduced limits so that
# slower presets don't dominate the comparison time. The winner is then
# re-evaluated with its full original params to get an accurate best_iteration
# for phase 2.
_PORTFOLIO_FAST_ROUNDS = 300
_PORTFOLIO_FAST_PATIENCE = 30

# For small training sets the single 90/10 validation split has very few val
# samples (e.g. n=380 → 38 val rows), making AUC estimates noisy enough that
# the wrong preset can win by chance.  When n_train < this threshold we run
# Stage 1 over multiple random splits and average the scores, reducing ranking
# noise at negligible extra wall-clock cost (small n means each split is fast).
_STAGE1_MULTI_SPLIT_THRESHOLD = 2_000
_STAGE1_MULTI_SPLITS = 3

# Cached GPU availability check (None = not yet tested)
_GPU_AVAILABLE: bool | None = None


def _resolve_device(device: str) -> str:
    """
    Resolve 'auto' to 'gpu' or 'cpu' based on CUDA availability.

    The check is performed once and cached.  'cpu' and 'gpu' are returned
    unchanged.  When device='auto', a single 1-round XGBoost training is
    attempted on CUDA; if it succeeds, 'gpu' is returned for all subsequent
    calls.
    """
    if xgb is None:
        return "cpu"
    global _GPU_AVAILABLE
    if device != "auto":
        return device
    if _GPU_AVAILABLE is None:
        try:
            dm = xgb.DMatrix([[1.0]], label=[0])
            xgb.train(
                {"tree_method": "hist", "device": "cuda", "verbosity": 0},
                dm,
                num_boost_round=1,
            )
            _GPU_AVAILABLE = True
            logger.debug("device=auto: CUDA detected, using GPU")
        except Exception:
            _GPU_AVAILABLE = False
            logger.debug("device=auto: no CUDA, using CPU")
    return "gpu" if _GPU_AVAILABLE else "cpu"


class _XGBHead(BaseEstimator):
    """The booster-specific half of both XGBoost heads: race, refit, folds, export, save."""

    _task = ""
    _fitted_attr = "booster_"

    def _fit_booster(self, X, y) -> DatasetProfile:
        """Preset race (or the heuristic below the split minimum), then the full refit."""
        profile = _inspect(X, y, task=self._task)
        self.profile_ = profile
        self._onnx_probe_ = probe_rows(X)
        device = _resolve_device(self.device)

        logger.rule(type(self).__name__)
        logger.profile_summary(profile)
        logger.info(f"device={device} | portfolio={self.portfolio}")

        self.timing_ = {}

        if self._task == "regression":
            final_booster, best_iter = self._fit_regression_preset(
                X, y, profile, device
            )
        elif profile.n_samples >= _VAL_MIN_ROWS:
            if self.portfolio:
                _t_ps = _time.perf_counter()
                best_name, best_params, best_iter, scores = _portfolio_select(
                    X, y, profile, device, self.nthread
                )
                self.timing_["portfolio_select"] = _time.perf_counter() - _t_ps
                self.preset_name_ = best_name
                self.params_ = best_params
                self.portfolio_scores_ = scores
            else:
                X_train, X_val, y_train, y_val, _ = _validation_split(
                    X, y, profile, stratify=self._task == "classification"
                )
                logger.debug(
                    f"Train split: {len(y_train)} rows | Val split: {len(y_val)} rows"
                )
                best_params = xgb_default_params(
                    profile, device=device, nthread=self.nthread
                )
                _, best_iter, _ = _train_phase1(
                    X_train, y_train, X_val, y_val, best_params, verbose=False
                )
                self.preset_name_ = "default"
                self.params_ = best_params
                self.portfolio_scores_ = {}
            logger.debug(
                f"objective={best_params['objective']} | "
                f"eval_metric={best_params['eval_metric']} | "
                f"lr={best_params['learning_rate']} | "
                f"colsample_bytree={best_params['colsample_bytree']}"
            )
            _t_p2 = _time.perf_counter()
            final_booster, best_iter = _train_phase2(
                X, y, best_params, best_iter, max_rounds=self.max_rounds
            )
            self.timing_["phase2_refit"] = _time.perf_counter() - _t_p2
        else:
            # Dataset too small to split: use heuristic preset directly.
            params = _get_params(profile, device=device, nthread=self.nthread)
            self.params_ = params
            self.preset_name_ = "heuristic"
            self.portfolio_scores_ = {}
            _t_p2 = _time.perf_counter()
            final_booster, best_iter, _ = _train_phase1(
                X, y, X, y, params, verbose=False
            )
            self.timing_["phase2_refit"] = _time.perf_counter() - _t_p2

        self.booster_ = final_booster
        self.best_iteration_ = best_iter
        self._bind_onnx()
        return profile

    def _fit_regression_preset(self, X, y, profile, device):
        """One preset, its round count set by early stopping on a held-out split.

        Regression does not race presets: the benchmark behind
        :func:`~lazyqsar.base.xgboost.presets.probst_regression_params` found one that
        is within ~1% of the per-task best on average, and the race cost more fit time
        than it won back. The held-out split is quantile-stratified on y (10%, or 20%
        below 200 rows); under 50 rows there is nothing to hold out and a fixed round
        count is used.
        """
        params = probst_regression_params(profile, device=device, nthread=self.nthread)
        self.preset_name_ = "probst"
        self.params_ = params
        n = len(y)
        t0 = _time.perf_counter()
        if n >= _REGRESSION_MIN_ES_ROWS:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X,
                y,
                test_size=_VAL_FRACTION if n >= _VAL_MIN_ROWS else 0.2,
                random_state=_RANDOM_STATE,
                # Two rows per stratum at least, or the split cannot place one on each
                # side; tied targets merge strata rather than crash.
                stratify=quantile_strata(y, 5, min_count=2),
            )
            _, best_iter, score = _train_phase1(
                X_tr, y_tr, X_val, y_val, params, verbose=False
            )
            self.portfolio_scores_ = {"probst": float(score)}
        else:
            best_iter = _PHASE2_MIN_ROUNDS
            self.portfolio_scores_ = {}
        self.timing_["portfolio_select"] = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        booster, best_iter = _train_phase2(
            X, y, params, best_iter, max_rounds=self.max_rounds
        )
        self.timing_["phase2_refit"] = _time.perf_counter() - t0
        return booster, best_iter

    def _oof(self, X, y, fold_splits) -> np.ndarray:
        """Each fold trains the chosen preset for the full fit's round count.

        The folds train concurrently, each with an equal share of the threads: XGBoost
        on a few thousand rows scales poorly across all cores, and its result does not
        depend on the thread count, so this changes only the wall time.
        """
        k = len(fold_splits)
        total = int(self.params_.get("nthread", 0) or (os.cpu_count() or 1))
        fold_params = dict(self.params_)
        if "nthread" in fold_params:
            fold_params["nthread"] = max(1, total // k)

        def fit_predict(train_idx, val_idx):
            fold_booster, fold_best_iter = _train_phase2(
                X[train_idx],
                y[train_idx],
                fold_params,
                self.best_iteration_,
                max_rounds=self.max_rounds,
                label=None,  # suppress banner; fold context logged by the caller
            )
            return fold_booster.predict(
                xgb.DMatrix(X[val_idx]), iteration_range=(0, fold_best_iter + 1)
            )

        logger.info(
            f"{k}-fold OOF  preset={self.preset_name_}  rounds={self.best_iteration_}"
        )
        return self._oof_loop(fold_splits, fit_predict, n_jobs=k if total >= k else 1)

    def _raw_native(self, X) -> np.ndarray:
        """Output of the booster itself, over the rounds ``predict`` uses."""
        return self.booster_.predict(
            xgb.DMatrix(X), iteration_range=(0, self.best_iteration_ + 1)
        )

    def _onnx_bytes(self) -> bytes:
        booster = _rounds_used(self.booster_, self.best_iteration_)
        wrapper = _booster_to_sklearn_wrapper(booster, task=self._task)
        return _onnx_model_bytes(wrapper, self.profile_.n_features)

    def save(self, directory: str) -> None:
        """
        Save the trained model to a directory.

        Always writes ``xgboost.json`` (fit metadata) and ``xgboost.onnx``.

        Parameters
        ----------
        directory : str
            Path to the output directory (created if it does not exist).
        """
        check_is_fitted(self, self._fitted_attr)
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "xgboost.onnx"))
        metadata = {
            "task": self._task,
            "format": "onnx",
            "preset_name": self.preset_name_,
            "best_iteration": self.best_iteration_,
            "params": self.params_,
            "profile": dataclasses.asdict(self.profile_),
            "portfolio_scores": self.portfolio_scores_,
        }
        metadata.update(self._task_metadata())
        with open(os.path.join(directory, "xgboost.json"), "w") as f:
            json.dump(metadata, f, indent=2, cls=_NumpyEncoder)


class BaseXGBClassifier(_XGBHead, ClassifierHead, ClassifierMixin):
    """
    Binary classifier with automatically selected XGBoost hyperparameters.

    Parameters
    ----------
    device : str
        "cpu", "gpu", or "auto".  "auto" detects CUDA availability at the
        first .fit() call and uses GPU when available, CPU otherwise.
    portfolio : bool
        If True (default), train all five preset configurations on a validation
        split and select the best.  If False, use the XGBoost default preset
        only (faster; useful as a no-tuning baseline).
    nthread : int
        Number of parallel threads for XGBoost.  -1 (default) lets XGBoost
        use all available CPU cores.

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict        — hyperparameters of the winning preset
    preset_name_ : str    — which of the 4 presets won ("heuristic", "default",
                            "flaml", or "autogluon")
    portfolio_scores_ : dict — val scores for every preset (empty when
                               portfolio=False or dataset too small to split)
    booster_ : xgb.Booster
    best_iteration_ : int
    classes_ : ndarray
    """

    _task = "classification"

    def __init__(
        self,
        device: str = "cpu",
        portfolio: bool = True,
        nthread: int = -1,
        calibrated: bool = True,
        max_rounds: int | None = None,
    ):
        self.device = device
        self.portfolio = portfolio
        self.nthread = nthread
        self.calibrated = calibrated
        self.max_rounds = max_rounds

    def fit(self, X, y):
        """Fit the classifier.

        When ``calibrated=True`` (default), runs the full calibration
        workflow: portfolio selection once, stratified k-fold OOF to collect
        held-out probabilities, isotonic calibrator.

        When ``calibrated=False``, runs the raw training only (no OOF pass).
        """
        require_fit_deps(_FIT_DEPS_AVAILABLE, "xgboost and scikit-learn")
        if self._wants_calibration(y):
            return self.calibrate(X, y)
        return self._fit_raw(X, y)

    def _fit_raw(self, X, y):
        require_fit_deps(_FIT_DEPS_AVAILABLE, "xgboost and scikit-learn")
        y = np.asarray(y).ravel()
        self._fit_booster(X, y)
        self._set_default_cutoffs()
        logger.rule("Done")
        logger.success(
            f"preset={self.preset_name_} | best_iteration={self.best_iteration_}"
        )
        logger.info(
            "decision cutoff: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_raw_source_}"
        )
        self.classes_ = np.array([0, 1])
        return self

    def calibrate(
        self, X, y, n_splits=None, random_state: int = 42
    ) -> "BaseXGBClassifier":
        """
        Collect out-of-fold predicted probabilities via stratified k-fold CV,
        then fit the calibrator on them.

        Hyperparameter selection (portfolio) runs ONCE on the full data via
        _fit_raw().  Each fold then trains for exactly self.best_iteration_
        rounds using the pre-selected preset — no per-fold portfolio search.

        Parameters
        ----------
        n_splits : int or None
            Number of CV folds. None = auto from minority class size
            (min 2, max 5, roughly minority_count // 10).
        """
        require_fit_deps(_FIT_DEPS_AVAILABLE, "xgboost and scikit-learn")
        import scipy.sparse as sp

        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        k, fold_splits = make_stratified_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseXGBClassifier.calibrate: full fit on n={len(y)} (portfolio runs once)"
        )
        self._fit_raw(X, y)
        oof_raw = self._oof(X, y.astype(float), fold_splits)
        self._finish_calibration(y, oof_raw)
        return self


class BaseXGBRegressor(_XGBHead, RegressorHead, RegressorMixin):
    """
    Regressor with automatically selected XGBoost hyperparameters.

    Every preset trains with squared error (the objective the RMSE race scores);
    a different objective can be passed explicitly through the parameters and is
    exported faithfully.

    Parameters
    ----------
    device : str
        "cpu", "gpu", or "auto".  "auto" detects CUDA availability at the
        first .fit() call and uses GPU when available, CPU otherwise.
    portfolio : bool
        If True (default), train all preset configurations on a validation
        split and select the best.  If False, use the XGBoost default preset
        only (faster; useful as a no-tuning baseline).
    nthread : int
        Number of parallel threads for XGBoost.  -1 (default) lets XGBoost
        use all available CPU cores.
    oof : bool
        If True (default), collect out-of-fold predictions after the full fit
        (quantile-stratified k-fold with the pre-selected preset).  These feed
        the ensemble gate and the OOF ECDF ranker.  The regression analogue of
        the classifier's ``calibrated`` flag; no calibrator is fitted.
    max_rounds : int or None
        Optional cap on boosting rounds for the full refit and OOF folds.

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict
    preset_name_ : str
    portfolio_scores_ : dict
    booster_ : xgb.Booster
    best_iteration_ : int
    timing_ : dict
    oof_preds_ : ndarray (n,)   — out-of-fold predictions (when oof=True)
    oof_rho_ : float or None    — Spearman ρ of OOF predictions
    train_rho_ : float or None  — Spearman ρ of in-sample predictions
    """

    _task = "regression"

    def __init__(
        self,
        device: str = "cpu",
        portfolio: bool = True,
        nthread: int = -1,
        oof: bool = True,
        max_rounds: int | None = None,
    ):
        self.device = device
        self.portfolio = portfolio
        self.nthread = nthread
        self.oof = oof
        self.max_rounds = max_rounds

    def fit(self, X, y):
        """Fit the regressor, then collect OOF predictions when ``oof=True``."""
        require_fit_deps(_FIT_DEPS_AVAILABLE, "xgboost and scikit-learn")
        if self.oof:
            return self.collect_oof(X, y)
        return self._fit_raw(X, y)

    def _fit_raw(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        self._fit_booster(X, y)
        self.train_rho_ = spearman_score(y, self.predict(X))
        logger.rule("Done")
        logger.success(
            f"preset={self.preset_name_} | best_iteration={self.best_iteration_}"
        )
        return self

    def collect_oof(
        self, X, y, n_splits=None, random_state: int = 42
    ) -> "BaseXGBRegressor":
        """
        Full fit (portfolio runs once), then quantile-stratified k-fold OOF
        predictions with the pre-selected preset and fixed round count.
        """
        import scipy.sparse as sp

        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float).ravel()
        k, fold_splits = make_quantile_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseXGBRegressor.collect_oof: full fit on n={len(y)} (portfolio runs once)"
        )
        self._fit_raw(X, y)
        self._finish_oof(y, self._oof(X, y, fold_splits))
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to native Python types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dmatrix_threads(params: dict) -> int:
    """Threads for building a DMatrix: the training thread count.

    The default (-1) resolves to OpenMP's maximum, which is 1 under the
    ``OMP_NUM_THREADS=1`` set on macOS, so the quantile sketch ran single-threaded.
    The cuts and the trained booster are the same at any thread count.
    """
    return int(params.get("nthread", -1) or -1)


def _train_phase1(X_train, y_train, X_val, y_val, params: dict, verbose: bool):
    """
    Phase 1: train on X_train/y_train with early stopping against X_val/y_val.

    Returns (booster, best_iteration, comparable_score) where comparable_score
    is normalised so that higher is always better (AUC/AUCPR are returned
    as-is; RMSE and other minimisation metrics are negated).
    """
    max_bin = params.get("max_bin", 256)
    num_boost_round = params["n_estimators"]
    early_stopping_rounds = params["early_stopping_rounds"]

    xgb_params = {k: v for k, v in params.items() if k not in _META_KEYS}

    nthread = _dmatrix_threads(params)
    dtrain = xgb.QuantileDMatrix(
        X_train, label=y_train, max_bin=max_bin, nthread=nthread
    )
    dval = xgb.QuantileDMatrix(
        X_val, label=y_val, ref=dtrain, max_bin=max_bin, nthread=nthread
    )

    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
    )
    best_iter = booster.best_iteration
    metric = xgb_params.get("eval_metric", "rmse")
    score = booster.best_score
    if metric not in MAXIMIZE_METRICS:
        score = -score  # normalise: higher is always better for comparison

    return booster, best_iter, score


def _train_phase2(
    X_full,
    y_full,
    params: dict,
    best_iter: int,
    max_rounds: int | None = None,
    label: str | None = "Phase 2 — full retraining",
):
    """
    Phase 2: retrain on the full dataset for at least _PHASE2_MIN_ROUNDS rounds.

    The round count is max(best_iter, early_stopping_rounds, _PHASE2_MIN_ROUNDS),
    optionally capped at max_rounds.  Use max_rounds to limit training time when
    a preset would otherwise produce a very large model.

    label controls the banner emitted via logger.rule().  Pass None to suppress
    the banner (e.g. when called from within calibration folds).

    No early stopping — the round count was calibrated in phase 1.
    The final model therefore sees 100% of the data.
    """
    import time as _time

    max_bin = params.get("max_bin", 256)
    min_iter = params.get("early_stopping_rounds", 0)
    xgb_params = {k: v for k, v in params.items() if k not in _META_KEYS}

    num_rounds = max(best_iter, min_iter, _PHASE2_MIN_ROUNDS)
    if max_rounds is not None:
        num_rounds = min(num_rounds, max_rounds)
    n = len(y_full)
    if label is not None:
        logger.rule(label)
    logger.info(
        f"n={n:,} samples | {num_rounds} rounds | "
        f"lr={params.get('learning_rate')} | max_bin={max_bin}"
    )
    _t0 = _time.perf_counter()
    dfull = xgb.QuantileDMatrix(
        X_full, label=y_full, max_bin=max_bin, nthread=_dmatrix_threads(params)
    )
    booster = xgb.train(
        xgb_params,
        dfull,
        num_boost_round=num_rounds,
        verbose_eval=False,
    )
    logger.debug(f"Phase 2: done in {_time.perf_counter() - _t0:.1f}s")
    # Return the last round index (0-based) so predict uses the full booster.
    return booster, num_rounds - 1


def _min_gain_threshold(profile: DatasetProfile, y_train: np.ndarray) -> float:
    """AUC margin a challenger preset must win by (see ``min_gain_threshold``).

    The classifier's preset race only: the effective size is the minority-class count,
    which sets the noise of an AUC. (Regression fits a single preset.)
    """
    n_eff = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    return min_gain_threshold(n_eff, len(y_train))


def _training_cost(params: dict, n_train: int) -> float:
    """
    Estimate the computational cost of a single phase-1 training run.

    cost_proxy = n_train × expected_rounds × max_leaves × num_parallel_tree

    expected_rounds is a heuristic upper bound: early stopping typically fires
    ~patience rounds after the best round, which is itself reached after roughly
    patience × (lr_ref / lr) rounds (lr_ref = 0.1).  Formula:
        expected_rounds = min(n_estimators, patience × (1 + lr_ref / lr))

    max_leaves is derived from the tree structure:
      - lossguide (FLAML): uses the explicit max_leaves parameter
      - depthwise (all others): 2 ** max_depth
    """
    lr = float(params.get("learning_rate", 0.3))
    patience = int(params.get("early_stopping_rounds", 50))
    n_est = int(params.get("n_estimators", 2000))
    n_par = int(params.get("num_parallel_tree", 1))

    expected_rounds = min(n_est, int(patience * (1 + 0.1 / max(lr, 1e-9))))

    if params.get("grow_policy") == "lossguide":
        max_leaves = int(params.get("max_leaves", 64))
    else:
        max_leaves = 2 ** int(params.get("max_depth", 6))

    return float(n_train * expected_rounds * max_leaves * n_par)


def _eval_preset_rep(X_tr, y_tr, X_val, y_val, params: dict, nthread: int) -> int:
    """
    Evaluate one Stage-2 calibration rep with a reduced thread count.

    Identical contract to _eval_preset_fast but uses the winner's full params
    (no budget cap) so the returned best_iteration is accurate for phase 2.
    Returns best_iteration (int).
    """
    p = dict(params)
    p["nthread"] = nthread
    max_bin = p.get("max_bin", 256)
    xgb_params = {k: v for k, v in p.items() if k not in _META_KEYS}
    dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, max_bin=max_bin, nthread=nthread)
    dval = xgb.QuantileDMatrix(
        X_val, label=y_val, ref=dtrain, max_bin=max_bin, nthread=nthread
    )
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=p["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=p["early_stopping_rounds"],
        verbose_eval=False,
    )
    return booster.best_iteration


def _eval_preset_fast(X_tr, y_tr, X_val, y_val, params: dict, nthread: int) -> tuple:
    """
    Evaluate one preset with a reduced thread count for parallel Stage 1.

    Each call builds its own QuantileDMatrix from the raw split arrays.
    Although this repeats the quantile computation, all N preset evaluations
    run concurrently in threads (XGBoost releases the GIL for both DMatrix
    construction and training), so the wall-clock cost is that of the slowest
    single preset rather than the sum.

    Returns (best_iteration, comparable_score) where score is normalised so
    that higher is always better (minimisation metrics are negated).
    """
    p = dict(params)
    p["nthread"] = nthread  # reduce threads so N parallel jobs ≈ 1 full core set
    max_bin = p.get("max_bin", 256)
    xgb_params = {k: v for k, v in p.items() if k not in _META_KEYS}
    dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, max_bin=max_bin, nthread=nthread)
    dval = xgb.QuantileDMatrix(
        X_val, label=y_val, ref=dtrain, max_bin=max_bin, nthread=nthread
    )
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=p["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=p["early_stopping_rounds"],
        verbose_eval=False,
    )
    metric = xgb_params.get("eval_metric", "rmse")
    score = booster.best_score
    if metric not in MAXIMIZE_METRICS:
        score = -score
    return booster.best_iteration, score


def _portfolio_select(
    X, y: np.ndarray, profile: DatasetProfile, device: str, nthread: int = -1
):
    """
    Two-stage portfolio selection returning the best preset for this dataset.

    Stage 1 — Fast parallel ranking:
      All four presets are evaluated on a 90/10 validation split with a capped
      budget (n_estimators=_PORTFOLIO_FAST_ROUNDS, patience=_PORTFOLIO_FAST_PATIENCE).
      Tree depth is also capped at _STAGE1_MAX_DEPTH so that deep presets stay
      within the cost budget.
      Presets whose estimated Stage-1 cost exceeds _MAX_COST_MULTIPLIER ×
      default cost are skipped.

      For small datasets (n < _STAGE1_MULTI_SPLIT_THRESHOLD) the Stage-1 score
      is averaged over _STAGE1_MULTI_SPLITS independent splits to reduce
      ranking noise from tiny (~30-sample) validation folds.

    Stage 2 — Best-iteration calibration (winner only):
      The winning preset is re-trained with its original full params (no depth
      cap, full patience) across 1–3 random 90/10 splits to obtain a stable
      best_iteration estimate for phase 2.  When the winner is very expensive
      (cost_ratio > _STAGE2_SKIP_COST_RATIO), Stage 2 is skipped and
      best_iteration is estimated analytically from patience and learning rate.

    A non-default preset wins only if its Stage-1 score exceeds the default's
    by at least _min_gain_threshold(profile, y).  This noise-aware threshold
    prevents overfitting the preset selection on small validation folds.

    Returns (best_preset_name, best_params, mean_best_iteration, scores_dict).
    """
    # FLAML and AutoGluon were calibrated on datasets with p ≤ ~158 (FLAML center=28,
    # scale=130) and p ≤ ~150 (AutoGluon OpenML benchmark).  Both are out-of-distribution
    # for p > 200; skip them to avoid poorly-matched presets and excess compute.
    _calibration_skip = {"flaml", "autogluon"} if profile.n_features > 200 else set()

    candidates = [
        ("heuristic", _get_params(profile, device, nthread=nthread)),
        ("default", xgb_default_params(profile, device, nthread=nthread)),
        ("flaml", flaml_params(profile, device, nthread=nthread)),
        ("autogluon", autogluon_params(profile, device, nthread=nthread)),
    ]
    params_map = {name: p for name, p in candidates}
    stratify = profile.task == "classification"

    # ------------------------------------------------------------------
    # Stage 1: fast parallel ranking
    #
    # For large datasets (n_tr >= _STAGE1_MULTI_SPLIT_THRESHOLD) a single
    # 90/10 split gives enough val samples for reliable AUC estimates, so
    # we use one split (cheap, deterministic).
    #
    # For small datasets the val set can be only 30–60 samples, making AUC
    # estimates noisy enough that the wrong preset wins by chance.  We
    # average scores across _STAGE1_MULTI_SPLITS random splits to reduce
    # ranking noise.  Wall-clock cost stays low because n is small and
    # all presets still run in parallel within each split.
    # ------------------------------------------------------------------
    # Use the first split to determine n_tr and cost budget.
    X_tr, X_val, y_tr, y_val, did_split = _validation_split(
        X,
        y,
        profile,
        stratify=stratify,
        random_state=_RANDOM_STATE,
    )

    fast_scores: dict = {}
    n_tr = len(y_tr) if did_split else len(y)
    default_cost = _training_cost(params_map["default"], n_tr)
    budget = _MAX_COST_MULTIPLIER * default_cost

    # Phase-2 cost budget: skip presets whose *full-params* estimated cost
    # exceeds _MAX_COST_MULTIPLIER × the default's full-params cost.
    # This catches slow FLAML configs (e.g. lr=0.007 → ~2000 rounds vs ~100
    # for default) that pass the Stage-1 filter (which uses capped fast_p).
    default_phase2_cost = _training_cost(params_map["default"], n_tr)
    phase2_budget = _MAX_COST_MULTIPLIER * default_phase2_cost

    # Adaptive fast budget: larger datasets need fewer rounds to rank presets
    # because each tree gets better gradient estimates (lower variance).
    # Scale by sqrt(n_ref / n_tr), clamped to [_PORTFOLIO_FAST_ROUNDS/6, full].
    # n_ref = 5000 (typical small drug dataset).  Examples:
    #   n=  1k → 1.0 → rounds=300, patience=30  (unchanged)
    #   n=  5k → 1.0 → rounds=300, patience=30
    #   n= 10k → 0.71 → rounds=212, patience=21
    #   n= 50k → 0.32 → rounds= 95, patience=10
    #   n=100k → 0.22 → rounds= 67, patience=10 (floor)
    _n_ref = 5_000
    _scale = min(1.0, (_n_ref / max(n_tr, 1)) ** 0.5)
    fast_rounds = max(
        _PORTFOLIO_FAST_ROUNDS // 6, int(round(_PORTFOLIO_FAST_ROUNDS * _scale))
    )
    fast_patience = max(
        _PORTFOLIO_FAST_PATIENCE // 3, int(round(_PORTFOLIO_FAST_PATIENCE * _scale))
    )
    logger.debug(
        f"[portfolio] Stage 1 budget: rounds={fast_rounds}, patience={fast_patience} "
        f"(n_tr={n_tr}, scale={_scale:.2f})"
    )

    # Filter candidates and build fast-budget params.
    # fast_p caps max_depth at _STAGE1_MAX_DEPTH so the Stage-1 comparison
    # stays within budget. Stage 2 and Phase 2 always use the original
    # uncapped params from params_map[best_name].
    to_run: list = []  # (name, fast_p)
    fast_params_map: dict = {}  # name → fast_p (for logging depth info)
    skipped_names: list = []
    for name, params in candidates:
        if name in _calibration_skip:
            logger.debug(
                f"[portfolio] {name:10s}: skipped (p={profile.n_features} > 200, outside calibration regime)"
            )
            fast_scores[name] = float("nan")
            skipped_names.append(name)
            continue
        fast_p = dict(params)
        fast_p["n_estimators"] = fast_rounds
        fast_p["early_stopping_rounds"] = fast_patience
        if fast_p.get("max_depth", 0) > _STAGE1_MAX_DEPTH:
            fast_p["max_depth"] = _STAGE1_MAX_DEPTH
        if name not in ("default", "heuristic"):
            # Phase-2 cost check (full params): skip presets that would make
            # Phase 2 (and calibration folds) >> default training cost.
            phase2_cost = _training_cost(params, n_tr)
            if phase2_cost > phase2_budget:
                logger.debug(
                    f"[portfolio] {name:10s}: skipped "
                    f"(phase2_cost={phase2_cost:.2e} > budget={phase2_budget:.2e} "
                    f"[{_MAX_COST_MULTIPLIER}× default phase-2])"
                )
                fast_scores[name] = float("nan")
                skipped_names.append(name)
                continue
            # Stage-1 cost check (fast_p with capped rounds)
            cost = _training_cost(fast_p, n_tr)
            if cost > budget:
                logger.debug(
                    f"[portfolio] {name:10s}: skipped "
                    f"(cost={cost:.2e} > budget={budget:.2e} "
                    f"[{_MAX_COST_MULTIPLIER}× default])"
                )
                fast_scores[name] = float("nan")
                skipped_names.append(name)
                continue
        fast_params_map[name] = fast_p
        to_run.append((name, fast_p))

    import time as _time

    # Divide CPU cores across parallel jobs so total threads ≈ all cores.
    n_jobs = len(to_run)
    n_cores = os.cpu_count() or 1
    nthread_each = max(1, n_cores // n_jobs) if n_jobs > 1 else n_cores

    # Number of Stage-1 splits: more splits on small datasets to average out
    # the noise from tiny val sets.
    n_stage1_splits = (
        _STAGE1_MULTI_SPLITS if n_tr < _STAGE1_MULTI_SPLIT_THRESHOLD else 1
    )

    logger.rule("Portfolio — Stage 1")
    logger.info(
        f"{n_jobs} presets × {n_stage1_splits} split(s) | "
        f"rounds={fast_rounds}, patience={fast_patience}, "
        f"nthread_each={nthread_each}"
        + (f" | skipped={skipped_names}" if skipped_names else "")
    )
    _t1 = _time.perf_counter()

    # Accumulate scores across splits; average at the end.
    # Each split uses a different random seed so the val sets are independent.
    accum: dict = {name: [] for name, _ in to_run}
    splits_used = 0
    for split_idx in range(n_stage1_splits):
        rs = _RANDOM_STATE + split_idx * 97  # prime stride → well-separated seeds
        if split_idx == 0:
            Xs_tr, Xs_val, ys_tr, ys_val = X_tr, X_val, y_tr, y_val
        else:
            Xs_tr, Xs_val, ys_tr, ys_val, ok = _validation_split(
                X,
                y,
                profile,
                stratify=stratify,
                random_state=rs,
            )
            if not ok:
                break
        splits_used += 1

        # Run all presets in parallel threads.  Each thread builds its own
        # QuantileDMatrix from the (read-only) numpy arrays.
        raw = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_eval_preset_fast)(
                Xs_tr, ys_tr, Xs_val, ys_val, fast_p, nthread_each
            )
            for _, fast_p in to_run
        )
        for (name, _), (_, score) in zip(to_run, raw):
            accum[name].append(score)

    logger.info(
        f"Stage 1: done in {_time.perf_counter() - _t1:.1f}s "
        f"({splits_used} split(s) averaged)"
    )

    for name, scores in accum.items():
        s = float(np.mean(scores)) if scores else float("nan")
        fast_scores[name] = s

    # Pick winner from fast scores
    default_score = fast_scores.get("default", float("-inf"))

    best_name = None
    best_score = float("-inf")
    for name, _ in candidates:
        s = fast_scores.get(name, float("nan"))
        if s != s:  # nan
            continue
        if s > best_score:
            best_score = s
            best_name = name

    if best_name is None:
        best_name = "default"
        best_score = float("nan")
        threshold = _min_gain_threshold(profile, y)
    elif best_name != "default":
        threshold = _min_gain_threshold(profile, y)
        gain = best_score - default_score
        if gain < threshold:
            best_name = "default"
            best_score = default_score
    else:
        threshold = _min_gain_threshold(profile, y)

    # Rich portfolio table (only when verbose=True)
    logger.portfolio_table(
        fast_scores=fast_scores,
        params_map=fast_params_map,
        winner=best_name,
        threshold=threshold,
        default_score=default_score,
        n_tr=n_tr,
        n_splits=splits_used,
        skipped=skipped_names,
    )
    logger.info(f"Portfolio winner: {best_name}  (score={best_score:+.4f})")

    # ------------------------------------------------------------------
    # Stage 2: calibrate best_iteration for the winner
    #
    # When the winning preset is very expensive relative to the XGBoost
    # default (cost_ratio > _STAGE2_SKIP_COST_RATIO), training a full 90%
    # split just to estimate best_iter costs almost as much as Phase 2
    # itself.  In that regime we use a closed-form heuristic instead:
    #
    #   heuristic_iter = patience × (lr_ref / lr),  lr_ref = 0.1
    #
    # Intuition: with early_stopping_rounds ∝ 1/lr, the model typically
    # reaches its optimum after ~patience × (lr_ref/lr) rounds and fires
    # early stopping patience rounds later.  The heuristic gives the
    # estimated best_iter (before the plateau).  _train_phase2 then
    # applies its own floor (max of best_iter, patience, _PHASE2_MIN_ROUNDS).
    #
    # Accuracy: at lr=0.02 the heuristic is ≈ 250×5 = 1250 rounds; the
    # actual optimum is typically 800–1500 for large dense datasets, so
    # Phase 2 is well-calibrated.  At lr=0.1 (sparse fingerprints) it
    # gives 50 rounds, clamped up to _PHASE2_MIN_ROUNDS=100 by Phase 2.
    #
    # For moderate-cost winners (cost_ratio ≤ _STAGE2_SKIP_COST_RATIO),
    # the number of repeated splits is reduced as cost grows so that Stage
    # 2 never dominates total training time.  Cheap presets (≤ 3×) get
    # the full _CV_REPEATS; multiple reps run in parallel threads.
    # ------------------------------------------------------------------
    winner_params = params_map[best_name]
    winner_cost = _training_cost(winner_params, n_tr)
    default_cost = _training_cost(params_map["default"], n_tr)
    cost_ratio = winner_cost / max(default_cost, 1.0)
    _t2 = _time.perf_counter()

    logger.rule("Portfolio — Stage 2")
    if cost_ratio > _STAGE2_SKIP_COST_RATIO:
        # Heuristic path: skip Stage 2 training entirely.
        lr = float(winner_params.get("learning_rate", 0.1))
        patience = int(winner_params.get("early_stopping_rounds", 50))
        heuristic_iter = int(round(patience * 0.1 / max(lr, 1e-9)))
        best_iter = max(heuristic_iter, _PHASE2_MIN_ROUNDS - 1)
        logger.info(
            f"Stage 2: SKIPPED (cost_ratio={cost_ratio:.1f}x > {_STAGE2_SKIP_COST_RATIO}x); "
            f"heuristic best_iter={best_iter} "
            f"(lr={lr}, patience={patience})"
        )
    else:
        stage2_repeats = 1 if cost_ratio > 3 else _CV_REPEATS
        mode = "parallel" if stage2_repeats > 1 else "single-rep"
        logger.info(
            f"Stage 2: winner={best_name}, {stage2_repeats} rep(s) [{mode}] "
            f"(cost_ratio={cost_ratio:.1f}x, lr={winner_params.get('learning_rate')}, "
            f"patience={winner_params.get('early_stopping_rounds')})"
        )

        # Pre-generate all splits for Stage 2 so we can run them in parallel.
        splits2: list = []
        for rep in range(stage2_repeats):
            X_tr2, X_val2, y_tr2, y_val2, ok = _validation_split(
                X,
                y,
                profile,
                stratify=stratify,
                random_state=_RANDOM_STATE + rep,
            )
            if not ok:
                break
            splits2.append((X_tr2, X_val2, y_tr2, y_val2))

        rep_iters: list = []
        if len(splits2) > 1:
            # Parallel calibration: divide cores across reps so total threads ≈ all cores.
            nthread_s2 = max(1, (os.cpu_count() or 1) // len(splits2))
            par_results = Parallel(n_jobs=len(splits2), prefer="threads")(
                delayed(_eval_preset_rep)(
                    X_tr2, y_tr2, X_val2, y_val2, winner_params, nthread_s2
                )
                for X_tr2, X_val2, y_tr2, y_val2 in splits2
            )
            for rep, b_iter in enumerate(par_results):
                rep_iters.append(b_iter)
                logger.debug(f"Stage 2: rep={rep} best_iter={b_iter}")
        else:
            for rep, (X_tr2, X_val2, y_tr2, y_val2) in enumerate(splits2):
                try:
                    _, b_iter, _ = _train_phase1(
                        X_tr2, y_tr2, X_val2, y_val2, winner_params, verbose=False
                    )
                    rep_iters.append(b_iter)
                    logger.debug(f"Stage 2: rep={rep} best_iter={b_iter}")
                except Exception as exc:
                    logger.debug(f"Stage 2: rep={rep} failed: {exc}")

        best_iter = int(round(np.mean(rep_iters))) if rep_iters else 0
        logger.info(
            f"Stage 2: done in {_time.perf_counter() - _t2:.1f}s  "
            f"best_iter={best_iter} (reps={rep_iters})"
        )

    return best_name, winner_params, best_iter, fast_scores


def _rounds_used(booster: "xgb.Booster", best_iteration: int) -> "xgb.Booster":
    """The booster cut to exactly the boosting rounds ``predict`` uses.

    ``predict`` evaluates ``iteration_range=(0, best_iteration + 1)`` -- *rounds*, each
    of ``num_parallel_tree`` trees. onnxmltools, given an early-stopped booster, keeps
    ``best_iteration + 1`` *trees* instead, which drops trees whenever a round holds more
    than one. Exporting a booster sliced to the used rounds removes the ambiguity: it
    carries no early-stopping attribute, so every tree in it is exported.
    """
    n_rounds = best_iteration + 1
    if n_rounds >= booster.num_boosted_rounds():
        return booster
    return booster[:n_rounds]


def _booster_objective(booster: "xgb.Booster") -> str:
    """The training objective recorded in the booster's own configuration."""
    config = json.loads(booster.save_config())
    return config["learner"]["objective"]["name"]


def _booster_to_sklearn_wrapper(booster: "xgb.Booster", task: str):
    """
    Wrap a raw Booster in an sklearn-compatible object for onnxmltools export.

    Injects the booster via the internal ``_Booster`` attribute to avoid
    calling ``load_model()``, which triggers a sklearn ≥1.6 / xgboost
    version incompatibility in ``_load_model_attributes``.

    The wrapper is built with the booster's own objective. onnxmltools reads the
    objective from the wrapper's parameters, not from the booster, so a default-built
    wrapper would export a ``reg:tweedie`` model as if it were squared error -- without
    the log link, and with silently wrong predictions.
    """
    objective = _booster_objective(booster)
    if task == "classification":
        wrapper = xgb.XGBClassifier(objective=objective)
        wrapper._Booster = booster
        wrapper.n_classes_ = 2
    else:
        wrapper = xgb.XGBRegressor(objective=objective)
        wrapper._Booster = booster
    return wrapper


def _export_onnx(model, path: str, n_features: int) -> None:
    """Convert an XGBoost sklearn estimator to ONNX and write to path."""
    with open(path, "wb") as f:
        f.write(_onnx_model_bytes(model, n_features))


def _onnx_model_bytes(model, n_features: int) -> bytes:
    """Convert an XGBoost sklearn estimator to a serialised ONNX graph."""
    import onnx
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx_model = convert_xgboost(
        model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
    )
    # onnxmltools fixes the second dim of probabilities to 2; onnxruntime infers
    # {N,1} from the missing n_targets attribute → conflict → MergeShapeInfo warning.
    # Clearing the fixed value makes the dim fully dynamic, resolving the conflict.
    for _out in onnx_model.graph.output:
        if _out.name == "probabilities":
            _dims = _out.type.tensor_type.shape.dim
            if len(_dims) >= 2 and _dims[1].dim_value != 0:
                _dims[1].ClearField("dim_value")
            break
    for output in onnx_model.graph.output:
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                dim.ClearField("dim_value")
                dim.dim_param = "batch_size"
    del onnx_model.graph.value_info[:]
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    return onnx_model.SerializeToString()


def _validation_split(
    X, y, profile: DatasetProfile, stratify: bool, random_state: int = _RANDOM_STATE
):
    """
    Split off a small validation set for early stopping.
    Falls back to reusing the full set when n_samples is very small.

    For binary classification, the validation fraction is dynamically raised
    when the minority class is small, ensuring at least _VAL_MIN_MINORITY
    minority samples in the validation set.  With too few minority samples,
    AUC estimates are noisy and early stopping can trigger at a suboptimal
    round (e.g. HIA_Hou has ~62 minority samples in training; 10% val gives
    only 6, making each rank-swap change AUC by ~0.017 — far too noisy).

    Returns (X_train, X_val, y_train, y_val, did_split).
    did_split=False means the dataset was too small to split; train==full.
    """
    if profile.n_samples < _VAL_MIN_ROWS:
        return X, X, y, y, False

    val_fraction = _VAL_FRACTION
    if stratify and profile.task == "classification":
        ratio = profile.imbalance_ratio
        minority_count = int(profile.n_samples * min(ratio, 1.0) / (1.0 + ratio))
        if minority_count > 0:
            needed = _VAL_MIN_MINORITY / minority_count
            val_fraction = max(_VAL_FRACTION, min(0.25, needed))

    strat = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=random_state,
        stratify=strat,
    )
    return X_train, X_val, y_train, y_val, True


# ---------------------------------------------------------------------------
# Artifact: one inference implementation, in lazyqsar.artifacts
# ---------------------------------------------------------------------------

from lazyqsar.artifacts.xgboost import XGBoostArtifact as BaseXGBArtifact  # noqa: E402,F401
