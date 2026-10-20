"""Pieces every head artifact shares, in one place so they cannot drift apart.

These used to be re-implemented per head -- the calibrator formula four times, the
decision-cutoff bookkeeping twice -- identical today, but nothing kept them identical.
numpy only: this module sits on the inference path.
"""

from __future__ import annotations

import numpy as np

from lazyqsar.utils.ranking import binarize

DEFAULT_DECISION_CUTOFF = 0.5


def apply_calibrator(proba: np.ndarray, cal: dict) -> np.ndarray:
    """Apply a saved calibrator (isotonic or Platt) to a ``(n, 2)`` probability array.

    The fit-time estimators calibrate with scikit-learn (``IsotonicRegression`` and a
    one-feature ``LogisticRegression``); this is the closed form of both, evaluated from
    the parameters the head saved.
    """
    raw_p1 = np.asarray(proba[:, 1], dtype=np.float64)
    if cal["method"] == "isotonic":
        p1 = np.clip(np.interp(raw_p1, cal["X_thresholds"], cal["y_thresholds"]), 0, 1)
    else:  # platt
        A, B = cal["coef"], cal["intercept"]
        p1 = 1.0 / (1.0 + np.exp(-(A * raw_p1 + B)))
    return np.column_stack([1 - p1, p1])


class DecisionCutoffMixin:
    """The per-head decision cutoffs a classifier head saved, and ``predict`` on them."""

    decision_cutoff_raw: float = DEFAULT_DECISION_CUTOFF
    decision_cutoff_raw_source: str = "default_0.5"
    decision_cutoff_proba: float = DEFAULT_DECISION_CUTOFF
    decision_cutoff_rank: float = DEFAULT_DECISION_CUTOFF
    decision_cutoff_logit: float = 0.0

    def _load_cutoffs(self, metadata: dict) -> None:
        self.decision_cutoff_raw = float(
            metadata.get("decision_cutoff_raw", DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_raw_source = metadata.get(
            "decision_cutoff_raw_source", "default_0.5"
        )
        self.decision_cutoff_proba = float(
            metadata.get("decision_cutoff_proba", DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_rank = float(
            metadata.get("decision_cutoff_rank", DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_logit = float(metadata.get("decision_cutoff_logit", 0.0))

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Return class labels using the stored decision cutoff by default.

        This artifact exposes calibrated probabilities, so it compares against the
        calibrated cutoff. ``decision_cutoff_proba`` is the calibrator's image of
        ``decision_cutoff_raw``, and calibration is monotone, so this matches the
        fit-time estimator's raw-versus-raw decision.
        """
        if self.task != "classification":
            raise RuntimeError(
                "predict() is only available for classification artifacts."
            )
        threshold = self.decision_cutoff_proba if cutoff is None else float(cutoff)
        classes = np.asarray(self.metadata.get("classes", [0, 1]))
        return classes[binarize(self.run(X)[:, 1], threshold)]
