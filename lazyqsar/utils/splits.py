from __future__ import annotations

import numpy as np

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    StratifiedKFold = None  # type: ignore[assignment,misc]


def auto_stratified_oof_n_splits(y: np.ndarray) -> int:
    """Auto-select k-fold count from minority class size (capped at 5, at least 2)."""
    # `minlength=2` so a single-class y reports a minority of 0 rather than the size of
    # the only class present: bincount([0,0,0]) is [3], whose min is 3, not 0.
    minority = int(np.bincount(np.asarray(y, dtype=int), minlength=2).min())
    k = min(5, max(3, minority // 10))
    return max(2, min(k, minority))


def check_trainable(y, where="fit"):
    """Raise unless *y* can actually train a binary classifier.

    Checked before featurizing rather than inside a head, because the heads discover it
    late: the calibration guard routes a one-minority dataset to the uncalibrated path,
    which then asks for a single fold and raises out of scikit-learn. On a slow-mode fit
    that happens *after* every descriptor has been computed, so the whole featurization
    pass is lost to a condition visible from the labels alone.

    Two classes with at least two members each is the real requirement -- two members
    because every downstream split is stratified, so a class of one cannot appear in both
    a train and a validation fold.
    """
    counts = np.bincount(np.asarray(y, dtype=int).ravel(), minlength=2)
    if len(counts) > 2:
        raise ValueError(
            f"{where}: labels must be binary 0/1, found {len(counts)} distinct values."
        )
    if counts.min() == 0:
        present = int(np.argmax(counts))
        raise ValueError(
            f"{where}: every label is {present}. Classification needs both classes; "
            "there is nothing to separate."
        )
    if counts.min() == 1:
        minority = int(np.argmin(counts))
        raise ValueError(
            f"{where}: class {minority} has a single member ({counts[minority]} of "
            f"{counts.sum()}). Stratified splitting cannot place one compound in both a "
            "training and a validation fold, so no fold count works. Add more examples "
            "of that class, or drop it and treat the task as something other than "
            "classification."
        )


def make_stratified_oof_splits(
    y: np.ndarray,
    n_splits: int | None = None,
    random_state: int = 42,
) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Build deterministic stratified OOF splits for binary classification.

    Returns the resolved fold count together with the exact train/validation
    index pairs so multiple models can consume identical calibration folds.
    """
    y_arr = np.asarray(y, dtype=int)
    k = n_splits if n_splits is not None else auto_stratified_oof_n_splits(y_arr)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    dummy_X = np.zeros(len(y_arr), dtype=np.int8)
    splits = [(train_idx, val_idx) for train_idx, val_idx in skf.split(dummy_X, y_arr)]
    return k, splits
