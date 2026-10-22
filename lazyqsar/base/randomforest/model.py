from __future__ import annotations

import json
import os
import time as _time
from contextlib import contextmanager

import numpy as np

# ---------------------------------------------------------------------------
# Fit-time dependencies — NOT required for ONNX inference.
# Guarded so lean [descriptors]-only installs can import this module to reach
# BaseRFArtifact without errors.
# ---------------------------------------------------------------------------
try:
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.validation import check_array, check_is_fitted

    _FIT_DEPS_AVAILABLE = True
except ImportError:

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    RandomForestClassifier = RandomForestRegressor = None  # type: ignore[assignment,misc]
    roc_auc_score = None  # type: ignore[assignment]
    check_array = check_is_fitted = None  # type: ignore[assignment]
    _FIT_DEPS_AVAILABLE = False

from lazyqsar.base._head import (
    ClassifierHead,
    RegressorHead,
    require_fit_deps,
)
from lazyqsar.base._race import min_gain_threshold
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import probe_rows
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.splits import make_quantile_oof_splits, make_stratified_oof_splits
from .params import (
    _HEURISTIC,
    profile_rf_dataset,
    heuristic_rf_params,
    default_rf_params,
    flaml_rf_params,
    autogluon_rf_params,
    rf_leaf_cap,
)


_IMBALANCE_BALANCED_SUBSAMPLE_RATIO = 3.0
# Regression: above this many rows the forest's OOB predictions serve as its OOF.
_OOB_OOF_MIN_N = 5_000
_PORTFOLIO_MIN_N = (
    200  # below this skip OOB comparison; OOB is unreliable on tiny datasets
)
# FLAML was calibrated on datasets with p ≤ ~158 (center=28, scale=130 from
# binary.json); AutoGluon on OpenML datasets with typical p ≤ 150.  Both are
# out-of-distribution beyond p≈200, regardless of sparsity.
_EXTERNAL_PRESETS_MAX_P = 200


def _forest_onnx(model) -> bytes:
    """The fitted forest as a serialised ONNX graph."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, model.n_features_in_]))]
    est = model._estimator
    # Probabilities as a plain tensor, not ZipMap's list of per-row dicts that every
    # prediction then unpacked in Python; the float32 values are the same.
    options = {id(est): {"zipmap": False}} if hasattr(est, "classes_") else None
    with _rf_onnx_bool_attr_compat():
        onnx_model = convert_sklearn(est, initial_types=initial_types, options=options)
    return onnx_model.SerializeToString()


@contextmanager
def _rf_onnx_bool_attr_compat():
    """
    `skl2onnx` may emit Python bools inside the TreeEnsemble
    `nodes_missing_value_tracks_true` int attribute. Recent ONNX builds
    reject that mix, so coerce the values to 0/1 during RF conversion.
    """
    import skl2onnx.common.tree_ensemble as tree_ensemble

    original_add_node = tree_ensemble.add_node

    def add_node_compat(*args, **kwargs):
        attr_pairs = args[0]
        before = len(attr_pairs["nodes_missing_value_tracks_true"])
        out = original_add_node(*args, **kwargs)
        if len(attr_pairs["nodes_missing_value_tracks_true"]) > before:
            value = attr_pairs["nodes_missing_value_tracks_true"][-1]
            attr_pairs["nodes_missing_value_tracks_true"][-1] = int(bool(value))
        return out

    tree_ensemble.add_node = add_node_compat
    try:
        yield
    finally:
        tree_ensemble.add_node = original_add_node


class _RFHead(BaseEstimator):
    """The forest-specific half of both RF heads: export, folds, save, and the OOB race.

    The OOB preset race (:meth:`_candidate_presets`, :meth:`_fit_forest`) is the
    classifier's; it needs ``_make_estimator(params, oob_score)``, ``_oob_score(est, y)``
    (higher is better), ``_gain(score, default_score)`` and ``_min_gain(y)``. The
    regressor fits one preset.
    """

    _task = ""
    _fitted_attr = "_estimator"

    def _resolve_params(self, preset: dict) -> dict:
        """Merge a preset dict with any explicit user overrides from __init__."""
        out = dict(preset)
        if self.n_estimators != _HEURISTIC:
            out["n_estimators"] = int(self.n_estimators)
        if self.max_depth != _HEURISTIC:
            out["max_depth"] = self.max_depth
        if self.min_samples_leaf != _HEURISTIC:
            out["min_samples_leaf"] = int(self.min_samples_leaf)
        if self.max_features != _HEURISTIC:
            out["max_features"] = self.max_features
        return out

    def _candidate_presets(self, profile, n: int) -> tuple[dict, set]:
        """The presets the OOB race compares, with the leaf-count size cap applied."""
        skip: set[str] = set()
        if profile.n_features > _EXTERNAL_PRESETS_MAX_P:
            skip = {"flaml", "autogluon"}
        candidates = [
            ("heuristic", heuristic_rf_params(profile)),
            ("default", default_rf_params()),
            ("flaml", flaml_rf_params(profile)),
            ("autogluon", autogluon_rf_params(profile)),
        ]
        presets = {}
        for name, params in candidates:
            if name in skip:
                continue
            params = self._resolve_params(params)
            cur = params.get("max_leaf_nodes")
            if cur is not None:
                # Only presets that already use max_leaf_nodes (flaml, autogluon)
                # get the size cap applied; heuristic and default rely on max_depth.
                cap = rf_leaf_cap(n, params["n_estimators"])
                presets[name] = dict(params, max_leaf_nodes=min(cur, cap))
            else:
                presets[name] = params
        return presets, skip

    def _fit_forest(self, X, y, presets: dict, skip: set) -> None:
        """OOB race between the presets (or the default alone below the minimum n)."""
        n = X.shape[0]
        t0 = _time.perf_counter()
        self.oob_scores_ = {}
        if n >= _PORTFOLIO_MIN_N:
            estimators = {}
            for name, params in presets.items():
                est = self._make_estimator(params, oob_score=True)
                est.fit(X, y)
                estimators[name] = est
                self.oob_scores_[name] = self._oob_score(est, y)

            default_score = self.oob_scores_.get("default", float("nan"))
            threshold = self._min_gain(y)
            winner = "default"
            best_gain = 0.0
            for name in ("heuristic", "flaml", "autogluon"):
                score = self.oob_scores_.get(name, float("nan"))
                if score != score:
                    continue
                gain = self._gain(score, default_score)
                if gain >= threshold and gain > best_gain:
                    best_gain = gain
                    winner = name

            self._estimator = estimators[winner]
            self.selected_preset_ = winner
            self.params_ = presets[winner]
            logger.portfolio_table(
                fast_scores=self.oob_scores_,
                params_map=presets,
                winner=winner,
                threshold=threshold,
                default_score=default_score,
                n_tr=n,
                n_splits=1,
                skipped=list(skip),
                model="rf",
            )
        else:
            # Too small for reliable OOB — use default directly
            params = presets["default"]
            self._estimator = self._make_estimator(params)
            self._estimator.fit(X, y)
            self.selected_preset_ = "default"
            self.params_ = params
            logger.info(
                f"RF portfolio skipped (n={n} < {_PORTFOLIO_MIN_N}), using default directly"
            )
        self.timing_ = {"fit": _time.perf_counter() - t0}

    def _log_fit(self, X, extra: str = "") -> None:
        p_ = self.params_
        logger.info(
            f"{type(self).__name__}.fit: "
            f"n={X.shape[0]:,} | p={X.shape[1]:,}"
            f" | n_estimators={p_['n_estimators']}"
            f" | max_depth={p_.get('max_depth', None)}"
            f" | max_leaf_nodes={p_.get('max_leaf_nodes', None)}"
            f" | min_samples_leaf={p_.get('min_samples_leaf', 1)}"
            f" | max_features={p_['max_features']}" + extra
        )

    def _oof(self, X, y, fold_splits, predict_fold) -> np.ndarray:
        def fit_predict(train_idx, val_idx):
            est = self._make_estimator(self.params_)
            est.fit(X[train_idx], y[train_idx])
            return predict_fold(est, X[val_idx])

        return self._oof_loop(fold_splits, fit_predict)

    def _onnx_bytes(self) -> bytes:
        return _forest_onnx(self)

    def save(self, directory: str) -> None:
        """Write ``randomforest.json`` (fit metadata) and ``randomforest.onnx``."""
        check_is_fitted(self, attributes=[self._fitted_attr])
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "randomforest.onnx"))
        params = getattr(self, "params_", {})
        metadata = {
            "task": self._task,
            "format": "onnx",
            "n_estimators": params.get("n_estimators", self.n_estimators),
            "selected_preset": getattr(self, "selected_preset_", "unknown"),
            "n_features_in": self.n_features_in_,
        }
        metadata.update(self._task_metadata())
        with open(os.path.join(directory, "randomforest.json"), "w") as f:
            json.dump(metadata, f, indent=2)


class BaseRFClassifier(_RFHead, ClassifierHead):
    """
    Binary classifier with automatically selected Random Forest hyperparameters.

    Parameters set to the sentinel ``_HEURISTIC`` are derived from the dataset
    profile at fit time; any explicit value overrides the heuristic.

    Parameters
    ----------
    n_estimators : int or _HEURISTIC
        Number of trees. Auto-selected from dataset size when ``_HEURISTIC``.
    max_depth : int, None, or _HEURISTIC
        Maximum tree depth. Auto-selected; None means unlimited.
    min_samples_leaf : int or _HEURISTIC
        Minimum samples per leaf. Auto-selected from n.
    max_features : str, int, float, or _HEURISTIC
        Features to consider at each split. Auto-selected from feature type.
    class_weight : str or dict
        ``"balanced"`` automatically switches to ``"balanced_subsample"``
        when imbalance ratio ≥ 3.0.
    n_jobs : int
        Parallelism for tree fitting. -1 uses all available cores.
    random_state : int or None
        Reproducibility seed.
    calibrated : bool
        If True (default), runs OOF calibration after fitting; if False,
        fits the raw forest only (no probability calibration).

    Attributes (after fit)
    ----------------------
    selected_preset_ : str
        Winning preset name (``"heuristic"``, ``"default"``, ``"flaml"``,
        or ``"autogluon"``).
    params_ : dict
        Hyperparameters of the winning preset.
    oof_probas_ : ndarray, shape (n,)
        Calibrated out-of-fold probabilities for class 1 (after calibrate()).
    oof_y_ : ndarray, shape (n,)
        Training labels in the same order as X.
    decision_cutoff_raw_ : float
        OOF-learned threshold that maximises balanced accuracy.
    classes_ : ndarray
        ``[0, 1]``
    """

    _task = "classification"

    def __init__(
        self,
        *,
        n_estimators: int | str = _HEURISTIC,
        max_depth: int | None | str = _HEURISTIC,
        min_samples_leaf: int | str = _HEURISTIC,
        max_features: str | int | float = _HEURISTIC,
        class_weight: str | dict | None = "balanced",
        n_jobs: int = -1,
        random_state: int | None = 42,
        calibrated: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.calibrated = calibrated

    @staticmethod
    def _imbalance_ratio(y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or counts.min() == 0:
            return 1.0
        return float(counts.max() / counts.min())

    def _resolve_class_weight(self, y: np.ndarray):
        if self.class_weight not in (None, "balanced", "balanced_subsample"):
            return self.class_weight
        ratio = self._imbalance_ratio(y)
        if self.class_weight == "balanced_subsample":
            return "balanced_subsample"
        if self.class_weight == "balanced":
            if ratio >= _IMBALANCE_BALANCED_SUBSAMPLE_RATIO:
                return "balanced_subsample"
            return "balanced"
        return None

    def fit(self, X, y) -> "BaseRFClassifier":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        if self._wants_calibration(y):
            return self.calibrate(X, y)
        return self._fit_raw(X, y)

    def _make_estimator(self, params: dict, oob_score: bool = False):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params["max_features"],
            max_leaf_nodes=params.get("max_leaf_nodes", None),
            criterion=params.get("criterion", "gini"),
            class_weight=self.class_weight_,
            oob_score=oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    @staticmethod
    def _oob_score(est, y) -> float:
        try:
            oob = est.oob_decision_function_[:, 1]
            mask = np.isfinite(oob)
            return roc_auc_score(y[mask], oob[mask])
        except ValueError:
            return float("nan")

    @staticmethod
    def _gain(score, default_score) -> float:
        return score - default_score

    @staticmethod
    def _min_gain(y) -> float:
        # AUC noise scales with the minority count.
        return min_gain_threshold(int(min(np.sum(y == 0), np.sum(y == 1))), len(y))

    def _fit_raw(self, X, y) -> "BaseRFClassifier":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        logger.rule("BaseRFClassifier")
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=int)
        self.n_features_in_ = X.shape[1]
        self._onnx_probe_ = probe_rows(X)
        self.imbalance_ratio_ = self._imbalance_ratio(y)
        self.class_weight_ = self._resolve_class_weight(y)

        presets, skip = self._candidate_presets(profile_rf_dataset(X, y), X.shape[0])
        self._fit_forest(X, y, presets, skip)
        self._bind_onnx()
        self.classes_ = np.array([0, 1])
        self._set_default_cutoffs()
        self._log_fit(
            X,
            f" | criterion={self.params_.get('criterion', 'gini')}"
            f" | imbalance={self.imbalance_ratio_:.1f}:1 | class_weight={self.class_weight_}",
        )
        logger.info(
            "decision cutoff: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_raw_source_}"
        )
        logger.rule("Done")
        return self

    def _raw_native(self, X) -> np.ndarray:
        """Raw P(y=1) from the scikit-learn forest itself."""
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict_proba(X)[:, 1]

    def score(self, X, y) -> float:
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    def calibrate(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseRFClassifier":
        """Full fit (OOB race runs once), stratified k-fold OOF, then the calibrator."""
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=int)
        k, fold_splits = make_stratified_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseRFClassifier.calibrate: full fit on n={len(y)} (forest fit runs once)"
        )
        self._fit_raw(X, y)
        logger.info(f"calibrate: {k}-fold OOF | fold_solver=RandomForestClassifier")
        oof_raw = self._oof(
            X, y, fold_splits, lambda est, Xv: est.predict_proba(Xv)[:, 1]
        )
        self._finish_calibration(y, oof_raw)
        return self


def _oob_predictions(forest, n: int) -> np.ndarray:
    """The forest's out-of-bag predictions, NaN for rows no tree left out.

    scikit-learn fills a row that was in every bootstrap sample with 0 (and warns), which
    would pass for a real prediction; those rows are marked NaN instead, so a caller can
    tell the OOB predictions are incomplete.
    """
    oob = np.asarray(forest.oob_prediction_, dtype=float).ravel().copy()
    left_out = np.zeros(n, dtype=bool)
    for sampled in forest.estimators_samples_:
        in_bag = np.zeros(n, dtype=bool)
        in_bag[sampled] = True
        left_out |= ~in_bag
    oob[~left_out] = np.nan
    return oob


def regression_rf_params(n: int) -> dict:
    """The regression forest: ``max_features=1/3``, ``min_samples_leaf=3``.

    Breiman's regression rule for the split candidates (p/3, as in R's randomForest),
    with a leaf floor of 3 rows. On LazyQSAR's regression benchmark (TDC ADMET, Biogen
    ADME, 30 MoleculeACE sets; random and scaffold splits; preprocessed Morgan
    fingerprints) it was within 1.6% of the per-task best RMSE on average, against 5.5%
    for the classification-derived heuristic it replaces, and its trees are ~3.5x
    smaller than unpruned ones -- which is what the ONNX graph and its load time scale
    with. The tree count follows the classification heuristic (more trees on more data).
    """
    if n < 500:
        n_estimators = 100
    elif n < 5_000:
        n_estimators = 200
    else:
        n_estimators = 300
    return {
        "n_estimators": n_estimators,
        "max_depth": None,
        "min_samples_leaf": 3,
        "max_features": 1.0 / 3.0,
    }


class BaseRFRegressor(_RFHead, RegressorHead):
    """
    Random-forest regressor with the benchmark-selected preset.

    One preset (:func:`regression_rf_params`), fitted once; quantile-stratified k-fold
    OOF predictions (or, with ``oof_source="oob"``, the forest's own out-of-bag
    predictions) feed the ensemble gate and the OOF ECDF ranker.

    Parameters
    ----------
    n_estimators, max_depth, min_samples_leaf, max_features
        Override the preset; ``_HEURISTIC`` keeps it.
    n_jobs : int
        Parallelism for tree fitting. -1 uses all available cores.
    random_state : int or None
        Reproducibility seed.
    oof : bool
        If True (default), collect OOF predictions after fitting.
    oof_source : {"auto", "folds", "oob"}
        Where the OOF predictions come from. ``"folds"`` refits the forest on k folds,
        like every other head; ``"oob"`` reads the final forest's out-of-bag
        predictions, which are also out-of-sample and cost no refit. ``"auto"`` (the
        default) uses the OOB above 5,000 rows, where the forest is the only head
        besides XGBoost and its fold refits were a third of the fit; on the benchmark
        the OOB cost 0.4% RMSE on average, mostly at small n where folds remain.

    Attributes (after fit)
    ----------------------
    selected_preset_ : str
    params_ : dict
    oob_preds_ : ndarray (n,)  — out-of-bag predictions of the final forest
    oof_preds_ : ndarray (n,)
    oof_rho_ : float or None
    train_rho_ : float or None
    timing_ : dict
    """

    _task = "regression"

    def __init__(
        self,
        *,
        n_estimators: int | str = _HEURISTIC,
        max_depth: int | None | str = _HEURISTIC,
        min_samples_leaf: int | str = _HEURISTIC,
        max_features: str | int | float = _HEURISTIC,
        n_jobs: int = -1,
        random_state: int | None = 42,
        oof: bool = True,
        oof_source: str = "auto",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.oof = oof
        self.oof_source = oof_source

    def fit(self, X, y) -> "BaseRFRegressor":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        if self.oof:
            return self.collect_oof(X, y)
        return self._fit_raw(X, y)

    def _make_estimator(self, params: dict, oob_score: bool = False):
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params["max_features"],
            max_leaf_nodes=params.get("max_leaf_nodes", None),
            oob_score=oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def _fit_raw(self, X, y) -> "BaseRFRegressor":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        logger.rule("BaseRFRegressor")
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float).ravel()
        self.n_features_in_ = X.shape[1]
        self._onnx_probe_ = probe_rows(X)

        t0 = _time.perf_counter()
        self.params_ = self._resolve_params(regression_rf_params(X.shape[0]))
        self.selected_preset_ = "p3_leaf3"
        # Out-of-bag scoring is nearly free and gives an out-of-sample prediction for
        # every row a tree left out (rows in every bootstrap sample are NaN).
        self._estimator = self._make_estimator(self.params_, oob_score=True)
        self._estimator.fit(X, y)
        self.oob_preds_ = _oob_predictions(self._estimator, X.shape[0])
        self.timing_ = {"fit": _time.perf_counter() - t0}
        self._bind_onnx()
        self.train_rho_ = spearman_score(y, self.predict(X))
        self._log_fit(X)
        logger.rule("Done")
        return self

    def collect_oof(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseRFRegressor":
        """Full fit, then OOF predictions from k quantile-stratified folds or the OOB."""
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float).ravel()
        k, fold_splits = make_quantile_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseRFRegressor.collect_oof: full fit on n={len(y)} (forest fit runs once)"
        )
        self._fit_raw(X, y)
        use_oob = self.oof_source == "oob" or (
            self.oof_source == "auto" and len(y) > _OOB_OOF_MIN_N
        )
        if use_oob and np.isfinite(self.oob_preds_).all():
            logger.info("collect_oof: out-of-bag predictions of the final forest")
            self.timing_["oof_folds"] = []
            self.timing_["oof_total"] = 0.0
            oof = self.oob_preds_.copy()
        else:
            logger.info(
                f"collect_oof: {k}-fold OOF | fold_solver=RandomForestRegressor"
            )
            oof = self._oof(X, y, fold_splits, lambda est, Xv: est.predict(Xv))
        self._finish_oof(y, oof)
        return self

    def _raw_native(self, X) -> np.ndarray:
        """Predictions from the scikit-learn forest itself."""
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict(X)


# Inference: one implementation per head, in lazyqsar.artifacts.
from lazyqsar.artifacts.rf import RandomForestArtifact as BaseRFArtifact  # noqa: E402,F401
