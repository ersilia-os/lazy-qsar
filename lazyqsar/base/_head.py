"""What every fitted head does the same way, whichever model family or task it is.

A head is one model family (XGBoost, linear, random forest, SVM) fitted for one task.
Each family has a classifier and a regressor, and all eight follow one template:

1. ``_fit_raw`` -- choose hyperparameters on the full data, fit the final model, bind it
   to its ONNX graph (:class:`~lazyqsar.base._onnx_binding.OnnxBoundHead`);
2. an out-of-fold pass with those hyperparameters frozen (:meth:`_HeadBase._oof_loop`);
3. task-specific post-processing of the OOF output: calibrator, decision cutoffs and
   ranker knots for a classifier (:meth:`ClassifierHead._finish_calibration`), OOF
   Spearman rho and ranker knots for a regressor (:meth:`RegressorHead._finish_oof`);
4. export (``to_onnx``) and ``save``.

The family modules keep what genuinely differs: the model, its hyperparameter search,
and how it is converted to ONNX. Everything else is here, so the eight heads cannot
drift apart.

A classifier family provides ``_score1(X)``: the pre-calibration class-1 score in
[0, 1] that calibration, cutoffs and ranks are defined on. For probability heads it is
the bound graph's P(y=1); the SVM classifier maps its decision function through a
sigmoid.
"""

from __future__ import annotations

import time as _time

import numpy as np

from lazyqsar.base._onnx_binding import OnnxBoundHead
from lazyqsar.utils.logging import logger
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.ranking import (
    binarize,
    prepare_knots,
    rank_from_knots,
    subsample_knots,
)

DEFAULT_DECISION_CUTOFF = 0.5
RANKER_MAX_KNOTS = 10_000
# Minority-class OOF samples needed for isotonic calibration; below it, Platt scaling
# (2 parameters) is used, since isotonic regression overfits small or very imbalanced
# OOF sets.
CALIBRATION_ISOTONIC_MIN_MINORITY = 500


def require_fit_deps(available: bool, what: str = "scikit-learn") -> None:
    """Raise the install hint when a fit-time dependency is missing."""
    if not available:
        raise ImportError(
            f"Training requires {what}. Install with: pip install 'lazyqsar[fit]'"
        )


def learn_balanced_accuracy_cutoff(y_true, p1) -> tuple[float, str]:
    """Learn a deterministic balanced-accuracy cutoff from OOF class-1 scores.

    Every distinct score is a candidate, plus 0.5 and a point just outside each end.
    Ties in balanced accuracy go to the cutoff nearest 0.5, then the smaller one.
    """
    from sklearn.metrics import balanced_accuracy_score

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(p1, dtype=float)
    mask = np.isfinite(p_arr)
    if mask.sum() == 0 or len(np.unique(y_arr[mask])) < 2:
        return DEFAULT_DECISION_CUTOFF, "default_0.5"

    probs = p_arr[mask]
    labels = y_arr[mask]
    unique = np.unique(probs)
    if unique.size == 0:
        return DEFAULT_DECISION_CUTOFF, "default_0.5"

    candidates = np.unique(
        np.concatenate(
            [
                unique,
                np.array(
                    [
                        np.nextafter(unique[0], -np.inf),
                        DEFAULT_DECISION_CUTOFF,
                        np.nextafter(unique[-1], np.inf),
                    ],
                    dtype=float,
                ),
            ]
        )
    )

    best_threshold = DEFAULT_DECISION_CUTOFF
    best_key = None
    for thr in candidates:
        score = balanced_accuracy_score(labels, (probs >= thr).astype(int))
        key = (-float(score), abs(float(thr) - DEFAULT_DECISION_CUTOFF), float(thr))
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(thr)
    return best_threshold, "oof_balanced_accuracy"


class _HeadBase(OnnxBoundHead):
    """Template shared by classifier and regressor heads."""

    # Name of the fitted-model attribute ``check_is_fitted`` looks for.
    _fitted_attr: str = ""
    # Prefix of the OOF timing keys: "calibration" (classifiers) or "oof" (regressors).
    _timing_prefix: str = "oof"

    def _oof_loop(self, fold_splits, fit_predict_fold, n_jobs: int = 1) -> np.ndarray:
        """Run ``fit_predict_fold(train_idx, val_idx) -> predictions`` over the folds.

        Returns the out-of-fold vector (NaN where a row was never held out) and records
        per-fold wall time in ``timing_``. ``n_jobs > 1`` runs the folds in threads -- for
        fold models that release the GIL and whose result does not depend on thread
        count (XGBoost's ``hist``: checked bit-identical across thread counts).
        """
        from concurrent.futures import ThreadPoolExecutor

        n = int(sum(len(val) for _, val in fold_splits))
        oof = np.full(n, np.nan, dtype=float)
        k = len(fold_splits)

        def timed(fold):
            fold_idx, (train_idx, val_idx) = fold
            logger.debug(
                f"  Fold {fold_idx + 1}/{k}: train={len(train_idx)}  val={len(val_idx)}"
            )
            t0 = _time.perf_counter()
            pred = fit_predict_fold(train_idx, val_idx)
            return val_idx, pred, _time.perf_counter() - t0

        folds = list(enumerate(fold_splits))
        if n_jobs > 1 and k > 1:
            with ThreadPoolExecutor(max_workers=min(n_jobs, k)) as pool:
                results = list(pool.map(timed, folds))
        else:
            results = [timed(fold) for fold in folds]
        fold_times = []
        for val_idx, pred, seconds in results:
            oof[val_idx] = pred
            fold_times.append(seconds)
        self.timing_[f"{self._timing_prefix}_folds"] = fold_times
        self.timing_[f"{self._timing_prefix}_total"] = sum(fold_times)
        return oof

    def _ranker_prepared(self):
        prepared = getattr(self, "_ranker_prepared_", None)
        if prepared is None:
            prepared = self._ranker_prepared_ = prepare_knots(self._ranker_knots)
        return prepared

    def _set_ranker_knots(self, scores) -> None:
        self._ranker_knots = subsample_knots(np.sort(scores), RANKER_MAX_KNOTS)
        self._ranker_prepared_ = prepare_knots(self._ranker_knots)

    def _ranker_metadata(self) -> dict:
        if hasattr(self, "_ranker_knots"):
            return {"ranker": {"knots": self._ranker_knots.tolist()}}
        return {}

    def _check_onnx_export(self, path: str) -> None:
        """Verify the written graph: byte-identical to the bound one, or probe-checked.

        A bound head was converted once and its graph checked against the native model on
        the fit-time probe (:meth:`OnnxBoundHead._bind_onnx`), and ``_write_onnx`` writes
        exactly those bytes. So the file is verified by comparing it with them -- stricter
        than re-running the probe, and without building a second onnxruntime session,
        which for a large forest costs seconds. An unbound head is probe-checked.
        """
        from lazyqsar.utils.onnx_check import (
            check_decision,
            check_probability,
            check_values,
        )

        if self._written_is_bound(path):
            return

        probe = getattr(self, "_onnx_probe_", None)
        if probe is None:
            return
        checker = {
            "probability": check_probability,
            "decision": check_decision,
            "values": check_values,
        }[self._onnx_kind]
        checker(path, probe, self._raw_native(probe), type(self).__name__)

    def to_onnx(self, path: str) -> None:
        """Write the graph the head is bound to and verify it against the native model."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, self._fitted_attr)
        self._write_onnx(path)
        self._check_onnx_export(path)


class ClassifierHead(_HeadBase):
    """Calibration, decision cutoffs, ranks and the ``predict_*`` family for classifiers."""

    _onnx_kind = "probability"
    _timing_prefix = "calibration"

    # ------------------------------------------------------------------ to implement

    def _score1(self, X) -> np.ndarray:
        """Pre-calibration class-1 score in [0, 1]; the bound graph's P(y=1) by default."""
        return self._raw(X)

    def _check_X(self, X):
        return X

    # ------------------------------------------------------------------ fit helpers

    def _wants_calibration(self, y) -> bool:
        """Calibrate when asked and when every class has two rows for the folds."""
        if not self.calibrated:
            return False
        return bool(np.bincount(np.asarray(y, dtype=int)).min() >= 2)

    def _set_default_cutoffs(self) -> None:
        self.decision_cutoff_raw_ = DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_raw_source_ = "default_0.5"
        self.decision_cutoff_proba_ = DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_rank_ = DEFAULT_DECISION_CUTOFF
        self.decision_cutoff_logit_ = 0.0

    def _finish_calibration(self, y, oof_score) -> int:
        """Calibrator, decision cutoffs and ranker knots from OOF class-1 scores.

        The calibrator is isotonic when the minority class has at least
        ``CALIBRATION_ISOTONIC_MIN_MINORITY`` rows, Platt otherwise. Returns the
        minority count for the caller's log line.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y, dtype=int)
        minority_count = int(np.bincount(y).min())
        if minority_count >= CALIBRATION_ISOTONIC_MIN_MINORITY:
            cal = IsotonicRegression(out_of_bounds="clip")
            self.oof_probas_ = cal.fit_transform(oof_score, y)
            self.calibrator_method_ = "isotonic"
        else:
            cal = LogisticRegression(C=1.0, solver="lbfgs")
            cal.fit(oof_score.reshape(-1, 1), y)
            self.oof_probas_ = cal.predict_proba(oof_score.reshape(-1, 1))[:, 1]
            self.calibrator_method_ = "platt"
        self.calibrator_ = cal
        self.decision_cutoff_raw_, self.decision_cutoff_raw_source_ = (
            learn_balanced_accuracy_cutoff(y, oof_score)
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
        # Uncalibrated [0,1] out-of-fold scores (the calibrator's input), kept alongside
        # the calibrated oof_probas_ so downstream can report the raw score too.
        self.oof_raw_ = np.asarray(oof_score).copy()
        self._set_ranker_knots(oof_score)
        self.decision_cutoff_rank_ = float(
            rank_from_knots(self.decision_cutoff_raw_, prepared=self._ranker_prepared_)
        )
        p = np.clip(self.decision_cutoff_proba_, 1e-7, 1.0 - 1e-7)
        self.decision_cutoff_logit_ = float(np.log(p / (1.0 - p)))
        logger.success(
            f"Calibrator fitted ({self.calibrator_method_}, minority={minority_count}) "
            f"on OOF predictions."
        )
        logger.info(
            "calibration cutoff learned from OOF scores: "
            f"{self.decision_cutoff_raw_:.4f} | metric=balanced_accuracy | "
            f"source={self.decision_cutoff_raw_source_}"
        )
        return minority_count

    # ------------------------------------------------------------------ prediction

    def _calibrated_p1(self, p_raw) -> np.ndarray:
        if not hasattr(self, "calibrator_"):
            return p_raw
        if self.calibrator_method_ == "isotonic":
            return np.clip(self.calibrator_.predict(p_raw), 0, 1)
        return self.calibrator_.predict_proba(p_raw.reshape(-1, 1))[:, 1]

    def predict_score(self, X) -> np.ndarray:
        """Return raw (pre-calibration) class scores, shape (n_samples, 2)."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, self._fitted_attr)
        p1 = self._score1(self._check_X(X))
        return np.column_stack([1 - p1, p1])

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated class probabilities, shape (n_samples, 2)."""
        p1 = self._calibrated_p1(self.predict_score(X)[:, 1])
        return np.column_stack([1 - p1, p1])

    def predict_logit(self, X) -> np.ndarray:
        """Return the logit of the calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Return 0/1 labels: raw score at or above the learned cutoff (or ``cutoff``)."""
        threshold = self.decision_cutoff_raw_ if cutoff is None else float(cutoff)
        return binarize(self.predict_score(X)[:, 1], threshold)

    def predict_rank(self, X) -> np.ndarray:
        """Map raw scores to [0, 1] ranks via the OOF ECDF, shape (n_samples, 2)."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "_ranker_knots")
        rank_1 = rank_from_knots(
            self.predict_score(X)[:, 1], prepared=self._ranker_prepared()
        )
        return np.column_stack([1 - rank_1, rank_1])

    # ------------------------------------------------------------------ save

    def _task_metadata(self) -> dict:
        """Cutoffs, calibrator and ranker knots, as every classifier head saves them."""
        meta = {
            "decision_cutoff_raw": float(
                getattr(self, "decision_cutoff_raw_", DEFAULT_DECISION_CUTOFF)
            ),
            "decision_cutoff_raw_source": getattr(
                self, "decision_cutoff_raw_source_", "default_0.5"
            ),
            "decision_cutoff_proba": float(
                getattr(self, "decision_cutoff_proba_", DEFAULT_DECISION_CUTOFF)
            ),
            "decision_cutoff_rank": float(
                getattr(self, "decision_cutoff_rank_", DEFAULT_DECISION_CUTOFF)
            ),
            "decision_cutoff_logit": float(
                getattr(self, "decision_cutoff_logit_", 0.0)
            ),
        }
        if hasattr(self, "calibrator_"):
            if self.calibrator_method_ == "isotonic":
                meta["calibrator"] = {
                    "method": "isotonic",
                    "X_thresholds": self.calibrator_.X_thresholds_.tolist(),
                    "y_thresholds": self.calibrator_.y_thresholds_.tolist(),
                }
            else:  # platt
                meta["calibrator"] = {
                    "method": "platt",
                    "coef": float(self.calibrator_.coef_[0][0]),
                    "intercept": float(self.calibrator_.intercept_[0]),
                }
        meta.update(self._ranker_metadata())
        return meta


class RegressorHead(_HeadBase):
    """OOF statistics, ranks and ``predict`` for regressors."""

    _onnx_kind = "values"
    _timing_prefix = "oof"

    def predict(self, X) -> np.ndarray:
        """Return continuous predictions, shape (n_samples,)."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, self._fitted_attr)
        return self._raw(X)

    def predict_rank(self, X) -> np.ndarray:
        """Map predictions to [0, 1] via the OOF ECDF, shape (n_samples,)."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "_ranker_knots")
        return rank_from_knots(self.predict(X), prepared=self._ranker_prepared())

    def _finish_oof(self, y, oof) -> None:
        """OOF predictions, their Spearman rho and the ranker knots."""
        self.oof_preds_ = oof
        self.oof_y_ = np.asarray(y, dtype=float).copy()
        self.oof_rho_ = spearman_score(y, oof)
        self._set_ranker_knots(oof)
        logger.success(
            f"OOF collected | rho_oof={self.oof_rho_} | "
            f"rho_train={getattr(self, 'train_rho_', None)}"
        )

    def _task_metadata(self) -> dict:
        meta = {
            "oof_rho": getattr(self, "oof_rho_", None),
            "train_rho": getattr(self, "train_rho_", None),
        }
        meta.update(self._ranker_metadata())
        return meta
