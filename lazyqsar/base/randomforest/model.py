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


_DEFAULT_DECISION_CUTOFF = 0.5
_CALIBRATION_ISOTONIC_MIN_MINORITY = 500
_RANKER_MAX_KNOTS = 10_000
_IMBALANCE_BALANCED_SUBSAMPLE_RATIO = 3.0


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


def _learn_balanced_accuracy_cutoff(y_true: np.ndarray, p1: np.ndarray) -> tuple[float, str]:
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
        np.array([
            np.nextafter(unique[0], -np.inf),
            _DEFAULT_DECISION_CUTOFF,
            np.nextafter(unique[-1], np.inf),
        ], dtype=float),
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


class BaseRFClassifier(BaseEstimator):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        class_weight: str | dict | None = "balanced",
        n_jobs: int = -1,
        random_state: int | None = 42,
        calibrated: bool = True,
    ):
        self.n_estimators = n_estimators
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
        if self.calibrated:
            y_arr = np.asarray(y, dtype=int)
            if np.bincount(y_arr).min() >= 2:
                return self.calibrate(X, y)
        return self._fit_raw(X, y)

    def _make_estimator(self, class_weight) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight=class_weight,
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

        _t_fit = _time.perf_counter()
        self._estimator = self._make_estimator(self.class_weight_)
        self._estimator.fit(X, y)
        self.timing_ = {"fit": _time.perf_counter() - _t_fit}

        self.classes_ = np.array([0, 1])
        self.decision_cutoff_ = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_source_ = "default_0.5"
        logger.info(
            "BaseRFClassifier.fit: "
            f"n={X.shape[0]:,} | p={X.shape[1]:,} | n_estimators={self.n_estimators}"
            f" | imbalance={self.imbalance_ratio_:.1f}:1 | class_weight={self.class_weight_}"
        )
        logger.info(
            "decision cutoff: "
            f"{self.decision_cutoff_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_source_}"
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
        threshold = self.decision_cutoff_ if cutoff is None else float(cutoff)
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

    def calibrate(self, X, y, n_splits: int | None = None, random_state: int = 42) -> "BaseRFClassifier":
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=int)
        n = len(y)
        k, fold_splits = make_stratified_oof_splits(y, n_splits=n_splits, random_state=random_state)

        logger.info(f"BaseRFClassifier.calibrate: full fit on n={n} (forest fit runs once)")
        self._fit_raw(X, y)

        oof_raw = np.full(n, np.nan, dtype=float)
        logger.info(f"calibrate: {k}-fold OOF | fold_solver=RandomForestClassifier")
        fold_times = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            logger.debug(f"  Fold {fold_idx + 1}/{k}: train={len(train_idx)}  val={len(val_idx)}")
            _t_fold = _time.perf_counter()
            fold_est = self._make_estimator(self.class_weight_)
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
        self.oof_y_ = y.copy()
        sorted_scores = np.sort(oof_raw)
        n_r = len(sorted_scores)
        if n_r > _RANKER_MAX_KNOTS:
            idx = np.round(np.linspace(0, n_r - 1, _RANKER_MAX_KNOTS)).astype(int)
            self._ranker_knots = sorted_scores[idx]
        else:
            self._ranker_knots = sorted_scores
        self.decision_cutoff_, self.decision_cutoff_source_ = _learn_balanced_accuracy_cutoff(y, oof_raw)
        logger.success(
            f"Calibrator fitted ({self.calibrator_method_}, minority={minority_count}) on OOF predictions."
        )
        logger.info(
            "calibration cutoff learned from OOF scores: "
            f"{self.decision_cutoff_:.4f} | metric=balanced_accuracy | source={self.decision_cutoff_source_}"
        )
        return self

    def to_onnx(self, path: str) -> None:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        check_is_fitted(self, attributes=["_estimator"])
        initial_types = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        with _rf_onnx_bool_attr_compat():
            onnx_model = convert_sklearn(self._estimator, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def save(self, directory: str, onnx: bool = True) -> None:
        check_is_fitted(self, attributes=["_estimator"])
        os.makedirs(directory, exist_ok=True)
        if onnx:
            self.to_onnx(os.path.join(directory, "randomforest.onnx"))
        else:
            joblib.dump(self._estimator, os.path.join(directory, "randomforest.joblib"))
        metadata = {
            "task": "classification",
            "format": "onnx" if onnx else "joblib",
            "n_estimators": self.n_estimators,
            "n_features_in": self.n_features_in_,
            "decision_cutoff": float(getattr(self, "decision_cutoff_", _DEFAULT_DECISION_CUTOFF)),
            "decision_cutoff_source": getattr(self, "decision_cutoff_source_", "default_0.5"),
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
    def __init__(self):
        self._session = None
        self._estimator = None
        self._format = ""
        self.metadata = {}
        self.task = ""
        self._cal = None
        self.decision_cutoff = _DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_source = "default_0.5"

    @classmethod
    def load(cls, directory: str) -> "BaseRFArtifact":
        json_path = os.path.join(directory, "randomforest.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"No metadata found at {json_path!r}")
        artifact = cls()
        with open(json_path) as f:
            artifact.metadata = json.load(f)
        artifact.task = artifact.metadata["task"]
        artifact._format = artifact.metadata.get("format", "onnx")
        artifact._cal = artifact.metadata.get("calibrator", None)
        artifact.decision_cutoff = float(artifact.metadata.get("decision_cutoff", _DEFAULT_DECISION_CUTOFF))
        artifact.decision_cutoff_source = artifact.metadata.get("decision_cutoff_source", "default_0.5")

        if artifact._format == "onnx":
            import onnxruntime as rt

            onnx_path = os.path.join(directory, "randomforest.onnx")
            artifact._session = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        else:
            joblib_path = os.path.join(directory, "randomforest.joblib")
            artifact._estimator = joblib.load(joblib_path)
        return artifact

    def run(self, X) -> np.ndarray:
        X_f32 = np.asarray(X, dtype=np.float32)
        if self._format == "onnx":
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: X_f32})
            prob_raw = outputs[1]
            if isinstance(prob_raw, list):
                proba = np.array([[d[k] for k in sorted(d)] for d in prob_raw], dtype=np.float64)
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
        threshold = self.decision_cutoff if cutoff is None else float(cutoff)
        return (self.run(X)[:, 1] >= threshold).astype(int)

    def predict_score(self, X) -> np.ndarray:
        X_f32 = np.asarray(X, dtype=np.float32)
        if self._format == "onnx":
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: X_f32})
            prob_raw = outputs[1]
            if isinstance(prob_raw, list):
                return np.array([[d[k] for k in sorted(d)] for d in prob_raw], dtype=np.float64)
            proba = np.asarray(prob_raw, dtype=np.float64)
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba
        return self._estimator.predict_proba(X_f32).astype(np.float64)

    def predict_logit(self, X) -> np.ndarray:
        p = np.clip(self.run(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_rank(self, X) -> np.ndarray:
        if "ranker" not in self.metadata:
            raise RuntimeError("No ranker stored in this artifact.")
        knots = np.asarray(self.metadata["ranker"]["knots"])
        rank_1 = np.interp(self.predict_score(X)[:, 1], knots, np.linspace(0.0, 1.0, len(knots)))
        return np.column_stack([1 - rank_1, rank_1])
