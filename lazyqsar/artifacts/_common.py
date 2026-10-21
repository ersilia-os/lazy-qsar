"""Pieces every head artifact shares, in one place so they cannot drift apart.

These used to be re-implemented per head -- the calibrator formula four times, the
decision-cutoff bookkeeping twice -- identical today, but nothing kept them identical.
numpy only: this module sits on the inference path.
"""

from __future__ import annotations

import numpy as np


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
