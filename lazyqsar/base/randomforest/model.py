from __future__ import annotations

import json
import os
import time as _time
from contextlib import contextmanager

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.utils.validation import check_array, check_is_fitted

from lazyqsar.utils.logging import logger
from lazyqsar.utils.splits import make_stratified_oof_splits
from .params import (
    _HEURISTIC,
    profile_rf_dataset,
    heuristic_rf_params,
    default_rf_params,
    flaml_rf_params,
    autogluon_rf_params,
    rf_leaf_cap,
)


_DEFAULT_DECISION_CUTOFF = 0.5
_CALIBRATION_ISOTONIC_MIN_MINORITY = 500
_RANKER_MAX_KNOTS = 10_000
_IMBALANCE_BALANCED_SUBSAMPLE_RATIO = 3.0
_PORTFOLIO_MIN_N = (
    200  # below this skip OOB comparison; OOB is unreliable on tiny datasets
)
_PORTFOLIO_MIN_GAIN = (
    0.005  # base AUC margin; heuristic must beat default by at least this
)


def _min_gain_threshold(y: np.ndarray) -> float:
    """
    Adaptive minimum-gain threshold for RF portfolio selection.

    Mirrors the XGBoost portfolio logic: a non-default preset wins only if its
    OOB AUC exceeds the default's by at least this margin.  The threshold scales
    with 1/sqrt(n_minority) to account for OOB estimate noise on small or
    imbalanced datasets, with a higher coefficient for n < 2000 where OOB
    estimates are noisier.

    Formula: max(_PORTFOLIO_MIN_GAIN, coef / sqrt(n_minority))
    """
    n_minority = int(min(np.sum(y == 0), np.sum(y == 1)))
    coef = 0.3 if len(y) < 2_000 else 0.1
    noise_based = coef / max(1, n_minority) ** 0.5
    return max(_PORTFOLIO_MIN_GAIN, noise_based)


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


def _learn_balanced_accuracy_cutoff(
    y_true: np.ndarray, p1: np.ndarray
) -> tuple[float, str]:
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


def _apply_calibrator_artifact(proba: np.ndarray, cal: dict) -> np.ndarray:
    raw_p1 = np.asarray(proba[:, 1], dtype=np.float64)
    if cal["method"] == "isotonic":
        p1 = np.clip(
            np.interp(raw_p1, cal["X_thresholds"], cal["y_thresholds"]),
            0,
            1,
        )
    else:
        A, B = cal["coef"], cal["intercept"]
        p1 = 1.0 / (1.0 + np.exp(-(A * raw_p1 + B)))
    return np.column_stack([1 - p1, p1])


class BaseRFClassifier(BaseEstimator):
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

    def fit(self, X, y) -> "BaseRFClassifier":
        if self.calibrated:
            y_arr = np.asarray(y, dtype=int)
            if np.bincount(y_arr).min() >= 2:
                return self.calibrate(X, y)
        return self._fit_raw(X, y)

    def _make_estimator(
        self, class_weight, params: dict, oob_score: bool = False
    ) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params["max_features"],
            max_leaf_nodes=params.get("max_leaf_nodes", None),
            criterion=params.get("criterion", "gini"),
            class_weight=class_weight,
            oob_score=oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def _fit_raw(self, X, y) -> "BaseRFClassifier":
        logger.rule("BaseRFClassifier")
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=int)
        self.n_features_in_ = X.shape[1]
        self.imbalance_ratio_ = self._imbalance_ratio(y)
        self.class_weight_ = self._resolve_class_weight(y)

        profile = profile_rf_dataset(X, y)
        n = X.shape[0]

        # FLAML was calibrated on datasets with p ≤ ~158 (center=28, scale=130 from
        # binary.json); AutoGluon on OpenML datasets with typical p ≤ 150.  Both are
        # out-of-distribution beyond p≈200, regardless of sparsity.
        _skip: set[str] = set()
        if profile.n_features > 200:
            _skip = {"flaml", "autogluon"}

        all_presets = {}
        for _name, _p in [
            ("heuristic", self._resolve_params(heuristic_rf_params(profile))),
            ("default", self._resolve_params(default_rf_params())),
            ("flaml", self._resolve_params(flaml_rf_params(profile))),
            ("autogluon", self._resolve_params(autogluon_rf_params(profile))),
        ]:
            if _name in _skip:
                continue
            _cur = _p.get("max_leaf_nodes")
            if _cur is not None:
                # Only presets that already use max_leaf_nodes (flaml, autogluon)
                # get the size cap applied; heuristic and default rely on max_depth.
                _cap = rf_leaf_cap(n, _p["n_estimators"])
                all_presets[_name] = dict(_p, max_leaf_nodes=min(_cur, _cap))
            else:
                all_presets[_name] = _p

        _t_fit = _time.perf_counter()
        if n >= _PORTFOLIO_MIN_N:
            # OOB portfolio: train all presets, pick the best above threshold
            estimators = {}
            oob_scores = {}
            for name, params in all_presets.items():
                est = self._make_estimator(self.class_weight_, params, oob_score=True)
                est.fit(X, y)
                estimators[name] = est
                try:
                    oob = est.oob_decision_function_[:, 1]
                    mask = np.isfinite(oob)
                    oob_scores[name] = roc_auc_score(y[mask], oob[mask])
                except ValueError:
                    oob_scores[name] = float("nan")

            default_score = oob_scores.get("default", float("nan"))
            threshold = _min_gain_threshold(y)

            winner = "default"
            best_gain = 0.0
            for name in ("heuristic", "flaml", "autogluon"):
                score = oob_scores.get(name, float("nan"))
                if score != score:
                    continue
                gain = score - default_score
                if gain >= threshold and gain > best_gain:
                    best_gain = gain
                    winner = name

            self._estimator = estimators[winner]
            self.selected_preset_ = winner
            self.params_ = all_presets[winner]

            logger.portfolio_table(
                fast_scores=oob_scores,
                params_map=all_presets,
                winner=winner,
                threshold=threshold,
                default_score=default_score,
                n_tr=n,
                n_splits=1,
                skipped=list(_skip),
                model="rf",
            )
        else:
            # Too small for reliable OOB — use default directly
            d_params = all_presets["default"]
            self._estimator = self._make_estimator(self.class_weight_, d_params)
            self._estimator.fit(X, y)
            self.selected_preset_ = "default"
            self.params_ = d_params
            logger.info(
                f"RF portfolio skipped (n={n} < {_PORTFOLIO_MIN_N}), using default directly"
            )

        self.timing_ = {"fit": _time.perf_counter() - _t_fit}
        self.classes_ = np.array([0, 1])
        self.decision_cutoff_raw_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_raw_source_ = "default_0.5"
        self.decision_cutoff_proba_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_rank_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_logit_ = 0.0
        p_ = self.params_
        logger.info(
            "BaseRFClassifier.fit: "
            f"n={n:,} | p={X.shape[1]:,}"
            f" | n_estimators={p_['n_estimators']}"
            f" | max_depth={p_.get('max_depth', None)}"
            f" | max_leaf_nodes={p_.get('max_leaf_nodes', None)}"
            f" | min_samples_leaf={p_.get('min_samples_leaf', 1)}"
            f" | max_features={p_['max_features']}"
            f" | criterion={p_.get('criterion', 'gini')}"
            f" | imbalance={self.imbalance_ratio_:.1f}:1 | class_weight={self.class_weight_}"
        )
        logger.info(
            "decision cutoff: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_raw_source_}"
        )
        logger.rule("Done")
        return self

    def predict_proba(self, X):
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        proba = self._estimator.predict_proba(X)
        if hasattr(self, "calibrator_"):
            if self.calibrator_method_ == "isotonic":
                p1 = np.clip(self.calibrator_.predict(proba[:, 1]), 0, 1)
            else:
                p1 = self.calibrator_.predict_proba(proba[:, 1].reshape(-1, 1))[:, 1]
            proba = np.column_stack([1 - p1, p1])
        return proba

    def predict_score(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict_proba(X)

    def predict(self, X, cutoff: float | None = None):
        threshold = self.decision_cutoff_raw_ if cutoff is None else float(cutoff)
        return (self.predict_score(X)[:, 1] >= threshold).astype(int)

    def predict_logit(self, X) -> np.ndarray:
        p = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_rank(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["_ranker_knots"])
        scores = self.predict_score(X)[:, 1]
        n_k = len(self._ranker_knots)
        rank_1 = np.interp(scores, self._ranker_knots, np.linspace(0.0, 1.0, n_k))
        return np.column_stack([1 - rank_1, rank_1])

    def score(self, X, y) -> float:
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    def calibrate(
        self, X, y, n_splits: int | None = None, random_state: int = 42
    ) -> "BaseRFClassifier":
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=int)
        n = len(y)
        k, fold_splits = make_stratified_oof_splits(
            y, n_splits=n_splits, random_state=random_state
        )

        logger.info(
            f"BaseRFClassifier.calibrate: full fit on n={n} (forest fit runs once)"
        )
        self._fit_raw(X, y)

        oof_raw = np.full(n, np.nan, dtype=float)
        logger.info(f"calibrate: {k}-fold OOF | fold_solver=RandomForestClassifier")
        fold_times = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            logger.debug(
                f"  Fold {fold_idx + 1}/{k}: train={len(train_idx)}  val={len(val_idx)}"
            )
            _t_fold = _time.perf_counter()
            fold_est = self._make_estimator(self.class_weight_, self.params_)
            fold_est.fit(X[train_idx], y[train_idx])
            oof_raw[val_idx] = fold_est.predict_proba(X[val_idx])[:, 1]
            fold_times.append(_time.perf_counter() - _t_fold)

        self.timing_["calibration_folds"] = fold_times
        self.timing_["calibration_total"] = sum(fold_times)

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
            f"Calibrator fitted ({self.calibrator_method_}, minority={minority_count}) on OOF predictions."
        )
        logger.info(
            "calibration cutoff learned from OOF scores: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_raw_source_}"
        )
        return self

    def to_onnx(self, path: str) -> None:
        """
        Export the trained forest to an ONNX file.

        Accepts a float32 input named ``"float_input"`` with shape
        ``(n_samples, n_features_in_)`` and produces:
          - ``"output_label"``       int64  (n_samples,)   — predicted class
          - ``"output_probability"`` float32 (n_samples, 2) — [P(0), P(1)]

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model.onnx"``.
        """
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        check_is_fitted(self, attributes=["_estimator"])
        initial_types = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        with _rf_onnx_bool_attr_compat():
            onnx_model = convert_sklearn(self._estimator, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def save(self, directory: str, onnx: bool = True) -> None:
        """
        Save the trained model to a directory.

        Always writes ``randomforest.json`` (fit metadata). The model binary is
        written as either:
          - ``randomforest.onnx``   when ``onnx=True`` (default)
          - ``randomforest.joblib`` when ``onnx=False``

        Parameters
        ----------
        directory : str
            Destination directory (created if it does not exist).
        onnx : bool
            If True, export to ONNX format; otherwise use joblib.
        """
        check_is_fitted(self, attributes=["_estimator"])
        os.makedirs(directory, exist_ok=True)
        if onnx:
            self.to_onnx(os.path.join(directory, "randomforest.onnx"))
        else:
            joblib.dump(self._estimator, os.path.join(directory, "randomforest.joblib"))
        params = getattr(self, "params_", {})
        metadata = {
            "task": "classification",
            "format": "onnx" if onnx else "joblib",
            "n_estimators": params.get("n_estimators", self.n_estimators),
            "selected_preset": getattr(self, "selected_preset_", "unknown"),
            "n_features_in": self.n_features_in_,
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
            else:
                metadata["calibrator"] = {
                    "method": "platt",
                    "coef": float(self.calibrator_.coef_[0][0]),
                    "intercept": float(self.calibrator_.intercept_[0]),
                }
        if hasattr(self, "_ranker_knots"):
            metadata["ranker"] = {"knots": self._ranker_knots.tolist()}
        with open(os.path.join(directory, "randomforest.json"), "w") as f:
            json.dump(metadata, f, indent=2)


class BaseRFArtifact:
    """
    Load a saved Random Forest model for forward inference.

    Reads the files written by ``BaseRFClassifier.save()``:
      - ``randomforest.onnx``  or ``randomforest.joblib`` — model binary
      - ``randomforest.json``  — fit metadata (task, calibrator, ranker, …)

    Usage
    -----
    artifact = BaseRFArtifact.load("path/to/directory")
    proba = artifact.run(X)          # calibrated (N, 2)
    labels = artifact.predict(X)     # binary using learned cutoff
    """

    def __init__(self):
        self._session = None
        self._estimator = None
        self._format = ""
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

        Detects ONNX or joblib format from ``randomforest.json``.

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
        artifact._format = artifact.metadata.get("format", "onnx")
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

        if artifact._format == "onnx":
            import onnxruntime as rt

            onnx_path = os.path.join(directory, "randomforest.onnx")
            artifact._session = rt.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
        else:
            joblib_path = os.path.join(directory, "randomforest.joblib")
            artifact._estimator = joblib.load(joblib_path)
        return artifact

    def run(self, X) -> np.ndarray:
        """
        Run calibrated inference on X.

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Calibrated [P(class=0), P(class=1)].
        """
        X_f32 = np.asarray(X, dtype=np.float32)
        if self._format == "onnx":
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: X_f32})
            prob_raw = outputs[1]
            if isinstance(prob_raw, list):
                proba = np.array(
                    [[d[k] for k in sorted(d)] for d in prob_raw], dtype=np.float64
                )
            else:
                proba = np.asarray(prob_raw, dtype=np.float64)
                if proba.ndim == 1:
                    proba = np.column_stack([1 - proba, proba])
        else:
            proba = self._estimator.predict_proba(X_f32).astype(np.float64)
        if self._cal is not None:
            proba = _apply_calibrator_artifact(proba, self._cal)
        return proba

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Return binary labels using the stored decision cutoff by default."""
        threshold = self.decision_cutoff_raw if cutoff is None else float(cutoff)
        return (self.run(X)[:, 1] >= threshold).astype(int)

    def predict_score(self, X) -> np.ndarray:
        """Return raw (pre-calibration) probabilities, shape (n_samples, 2)."""
        X_f32 = np.asarray(X, dtype=np.float32)
        if self._format == "onnx":
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
        return self._estimator.predict_proba(X_f32).astype(np.float64)

    def predict_logit(self, X) -> np.ndarray:
        """Return logit of calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.run(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_rank(self, X) -> np.ndarray:
        """Map raw scores to [0, 1] ranks via OOF ECDF, shape (n_samples, 2)."""
        if "ranker" not in self.metadata:
            raise RuntimeError("No ranker stored in this artifact.")
        knots = np.asarray(self.metadata["ranker"]["knots"])
        rank_1 = np.interp(
            self.predict_score(X)[:, 1], knots, np.linspace(0.0, 1.0, len(knots))
        )
        return np.column_stack([1 - rank_1, rank_1])
