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
    DEFAULT_DECISION_CUTOFF as _DEFAULT_DECISION_CUTOFF,
    ClassifierHead,
    RegressorHead,
    require_fit_deps,
)
from lazyqsar.base._race import min_gain_threshold
from lazyqsar.utils.logging import logger
from lazyqsar.utils.onnx_check import probe_rows
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.ranking import binarize, prepare_knots, rank_from_knots
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
    with _rf_onnx_bool_attr_compat():
        onnx_model = convert_sklearn(model._estimator, initial_types=initial_types)
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


from lazyqsar.artifacts._common import (  # noqa: E402
    apply_calibrator as _apply_calibrator_artifact,
)


class _RFHead(BaseEstimator):
    """The forest-specific half of both RF heads: presets, OOB race, export, save.

    A subclass names its task and provides ``_make_estimator(params, oob_score)``,
    ``_oob_score(est, y)`` (higher is better), ``_gain(score, default_score)`` and
    ``_min_gain(y)``.
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
        task = self._task
        skip: set[str] = set()
        if profile.n_features > _EXTERNAL_PRESETS_MAX_P:
            skip = {"flaml", "autogluon"}
        if task == "regression":
            candidates = [
                ("heuristic", heuristic_rf_params(profile, task=task)),
                ("default", default_rf_params(task=task)),
                ("flaml", flaml_rf_params(profile, task=task)),
                ("autogluon", autogluon_rf_params(profile)),
            ]
        else:
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


class BaseRFRegressor(_RFHead, RegressorHead):
    """
    Regressor with automatically selected Random Forest hyperparameters.

    Mirrors :class:`BaseRFClassifier`: an OOB portfolio (heuristic / default /
    flaml / autogluon) picks the preset by out-of-bag RMSE, then
    quantile-stratified k-fold OOF predictions feed the ensemble gate and the
    OOF ECDF ranker. No calibration and no decision cutoff.

    Parameters
    ----------
    n_estimators, max_depth, min_samples_leaf, max_features
        As in :class:`BaseRFClassifier`; ``_HEURISTIC`` derives them from the
        dataset profile.
    n_jobs : int
        Parallelism for tree fitting. -1 uses all available cores.
    random_state : int or None
        Reproducibility seed.
    oof : bool
        If True (default), collect OOF predictions after fitting.

    Attributes (after fit)
    ----------------------
    selected_preset_ : str
    params_ : dict
    oob_scores_ : dict      — negative OOB RMSE per preset (portfolio path only)
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
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.oof = oof

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

    @staticmethod
    def _oob_score(est, y) -> float:
        # Higher is better: negative out-of-bag RMSE, the metric every regression
        # head's preset race uses.
        oob = np.asarray(est.oob_prediction_, dtype=float).ravel()
        ok = np.isfinite(oob)
        if not ok.any():
            return float("nan")
        return -float(np.sqrt(np.mean((y[ok] - oob[ok]) ** 2)))

    @staticmethod
    def _gain(score, default_score) -> float:
        # Relative RMSE improvement, so the margin means the same at any target scale.
        return (score - default_score) / max(abs(default_score), 1e-12)

    @staticmethod
    def _min_gain(y) -> float:
        # The relative OOB RMSE's sampling noise shrinks as ~1/sqrt(n).
        return min_gain_threshold(len(y), len(y))

    def _candidate_presets(self, profile, n: int) -> tuple[dict, set]:
        presets, skip = super()._candidate_presets(profile, n)
        # The sklearn default grows unpruned bagged trees (max_features=1.0,
        # min_samples_leaf=1); cap leaves so the ONNX export stays bounded.
        d = presets["default"]
        if d.get("max_leaf_nodes") is None and d.get("max_depth") is None:
            presets["default"] = dict(
                d, max_leaf_nodes=rf_leaf_cap(n, d["n_estimators"])
            )
        return presets, skip

    def _fit_raw(self, X, y) -> "BaseRFRegressor":
        require_fit_deps(_FIT_DEPS_AVAILABLE)
        logger.rule("BaseRFRegressor")
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float).ravel()
        self.n_features_in_ = X.shape[1]
        self._onnx_probe_ = probe_rows(X)

        profile = profile_rf_dataset(X, y, task="regression")
        presets, skip = self._candidate_presets(profile, X.shape[0])
        self._fit_forest(X, y, presets, skip)
        self._bind_onnx()
        self.train_rho_ = spearman_score(y, self.predict(X))
        self._log_fit(X)
        logger.rule("Done")
        return self

    def collect_oof(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseRFRegressor":
        """Full fit (portfolio runs once), then quantile-stratified k-fold OOF."""
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float).ravel()
        k, fold_splits = make_quantile_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )
        logger.info(
            f"BaseRFRegressor.collect_oof: full fit on n={len(y)} (forest fit runs once)"
        )
        self._fit_raw(X, y)
        logger.info(f"collect_oof: {k}-fold OOF | fold_solver=RandomForestRegressor")
        oof = self._oof(X, y, fold_splits, lambda est, Xv: est.predict(Xv))
        self._finish_oof(y, oof)
        return self

    def _raw_native(self, X) -> np.ndarray:
        """Predictions from the scikit-learn forest itself."""
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict(X)


class BaseRFArtifact:
    """
    Load a saved Random Forest model for forward inference.

    Reads the files written by ``BaseRFClassifier.save()``:
      - ``randomforest.onnx``  — model binary
      - ``randomforest.json``  — fit metadata (task, calibrator, ranker, …)

    Usage
    -----
    artifact = BaseRFArtifact.load("path/to/directory")
    proba = artifact.run(X)          # calibrated (N, 2)
    labels = artifact.predict(X)     # binary using learned cutoff
    """

    def __init__(self):
        self._session = None
        self.metadata = {}
        self.task = ""
        self._cal = None
        self.decision_cutoff_raw = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_raw_source = "default_0.5"
        self.decision_cutoff_proba = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_rank = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_logit = 0.0

    @classmethod
    def load(cls, directory: str) -> "BaseRFArtifact":
        """
        Load the model from *directory*.

        Parameters
        ----------
        directory : str
            Directory previously passed to ``BaseRFClassifier.save()``.

        Returns
        -------
        BaseRFArtifact
        """
        json_path = os.path.join(directory, "randomforest.json")
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

        onnx_path = os.path.join(directory, "randomforest.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"No ONNX model found at {onnx_path!r}")
        artifact._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        return artifact

    def run(self, X) -> np.ndarray:
        """
        Run inference on X.

        Returns
        -------
        Classification : ndarray, shape (n_samples, 2)
            Calibrated [P(class=0), P(class=1)].
        Regression : ndarray, shape (n_samples,)
            Predicted values.
        """
        X_f32 = np.asarray(X, dtype=np.float32)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: X_f32})
        if self.task == "regression":
            return np.asarray(outputs[0], dtype=np.float64).ravel()
        prob_raw = outputs[1]
        if isinstance(prob_raw, list):
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

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Return binary labels using the stored decision cutoff by default."""
        if self.task != "classification":
            raise RuntimeError(
                "predict() is only available for classification artifacts."
            )
        threshold = self.decision_cutoff_raw if cutoff is None else float(cutoff)
        # Threshold the same quantity the fit-time estimator does: raw score against
        # the raw cutoff. Comparing run() (calibrated) against decision_cutoff_raw
        # mixes two different scales.
        return binarize(self.predict_score(X)[:, 1], threshold)

    def predict_score(self, X) -> np.ndarray:
        """Return raw (pre-calibration) probabilities, shape (n_samples, 2)."""
        X_f32 = np.asarray(X, dtype=np.float32)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: X_f32})
        prob_raw = outputs[1]
        if isinstance(prob_raw, list):
            return np.array(
                [[d[k] for k in sorted(d)] for d in prob_raw], dtype=np.float64
            )
        proba = np.asarray(prob_raw, dtype=np.float64)
        if proba.ndim == 1:
            return np.column_stack([1 - proba, proba])
        return proba

    def predict_logit(self, X) -> np.ndarray:
        """Return logit of calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.run(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_rank(self, X) -> np.ndarray:
        """Map scores to [0, 1] ranks via OOF ECDF.

        Classification: shape (n_samples, 2). Regression: shape (n_samples,).
        """
        if "ranker" not in self.metadata:
            raise RuntimeError("No ranker stored in this artifact.")
        if getattr(self, "_ranker_prepared", None) is None:
            self._ranker_prepared = prepare_knots(self.metadata["ranker"]["knots"])
        if self.task == "regression":
            return rank_from_knots(self.run(X), prepared=self._ranker_prepared)
        rank_1 = rank_from_knots(
            self.predict_score(X)[:, 1], prepared=self._ranker_prepared
        )
        return np.column_stack([1 - rank_1, rank_1])
