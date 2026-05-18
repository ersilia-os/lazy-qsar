from __future__ import annotations

import numpy as np

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    StratifiedKFold = None  # type: ignore[assignment,misc]


def auto_stratified_oof_n_splits(y: np.ndarray) -> int:
    """Auto-select k-fold count from minority class size (capped at 5, at least 2)."""
    minority = int(np.bincount(np.asarray(y, dtype=int)).min())
    k = min(5, max(3, minority // 10))
    return max(2, min(k, minority))


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
