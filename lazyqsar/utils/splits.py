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


_REGRESSION_MIN_SAMPLES = 10


def check_trainable_regression(y, where="fit"):
    """Raise unless *y* can train a regressor: finite, enough samples, not constant."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(y_arr)):
        n_bad = int((~np.isfinite(y_arr)).sum())
        raise ValueError(
            f"{where}: {n_bad} target value(s) are NaN or infinite. Regression needs "
            "a finite numeric value for every compound."
        )
    if y_arr.size < _REGRESSION_MIN_SAMPLES:
        raise ValueError(
            f"{where}: only {y_arr.size} samples; regression needs at least "
            f"{_REGRESSION_MIN_SAMPLES}."
        )
    if np.ptp(y_arr) == 0.0:
        raise ValueError(
            f"{where}: every target value is {y_arr[0]}. Regression needs variation "
            "in y; there is nothing to fit."
        )


# Above this many rows regression OOF uses 3 folds instead of 5: every head refits once
# per fold, so the folds are most of a large fit's cost, and with thousands of rows three
# folds already give the stacker and the interval model plenty of held-out predictions.
_REGRESSION_FEW_FOLDS_N = 2_000


def auto_regression_oof_n_splits(y: np.ndarray) -> int:
    """Auto-select k-fold count for regression from sample size (between 3 and 5)."""
    n = len(np.asarray(y))
    if n > _REGRESSION_FEW_FOLDS_N:
        return 3
    return max(2, min(5, max(3, n // 50), n // 2))


def quantile_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to one of *n_bins* quantile bins (0..n_bins-1)."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if n_bins <= 1 or y_arr.size == 0:
        return np.zeros(y_arr.size, dtype=int)
    edges = np.quantile(y_arr, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, y_arr, side="right").astype(int)


def quantile_strata(y: np.ndarray, n_bins: int, min_count: int) -> np.ndarray:
    """Quantile bins of *y* with every bin holding at least *min_count* rows.

    Tied targets (a censored value repeated many times, say) leave some quantile bins
    nearly empty; bins are merged, from fewer bins down to one, until none is too small
    to stratify on.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    bins = quantile_bins(y_arr, n_bins)
    while n_bins > 1 and np.bincount(bins).min() < min_count:
        n_bins -= 1
        bins = quantile_bins(y_arr, n_bins)
    # quantile_bins can leave an empty label in the middle; relabel densely.
    return np.unique(bins, return_inverse=True)[1]


def make_quantile_oof_splits(
    y: np.ndarray,
    n_splits: int | None = None,
    random_state: int = 42,
) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Build deterministic OOF splits for regression, stratified on quantile bins of y.

    Stratifying on bins keeps every fold spanning the full target range, which
    plain KFold does not guarantee on small or sorted datasets.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    k = n_splits if n_splits is not None else auto_regression_oof_n_splits(y_arr)
    n_bins = max(1, min(5, len(y_arr) // max(1, 2 * k)))
    bins = quantile_bins(y_arr, n_bins)
    # Merge bins too small to appear in every fold into a neighbour.
    while n_bins > 1 and np.bincount(bins).min() < k:
        n_bins -= 1
        bins = quantile_bins(y_arr, n_bins)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    dummy_X = np.zeros(len(y_arr), dtype=np.int8)
    splits = [(train_idx, val_idx) for train_idx, val_idx in skf.split(dummy_X, bins)]
    return k, splits
