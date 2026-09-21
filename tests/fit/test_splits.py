"""Stratified out-of-fold splits, shared by every model that needs calibration.

The point of building them once and handing them round is that several models calibrate on
*identical* folds -- otherwise their out-of-fold scores are not comparable, and the ensemble
weights derived from those scores are comparing different experiments.
"""

import numpy as np
import pytest

from lazyqsar.utils.splits import (
    auto_stratified_oof_n_splits,
    make_stratified_oof_splits,
)


def labels(n, n_minority):
    y = np.zeros(n, dtype=int)
    y[:n_minority] = 1
    return y


@pytest.mark.parametrize(
    "n_minority,expected",
    [(2, 2), (4, 3), (15, 3), (29, 3), (30, 3), (40, 4), (50, 5), (200, 5)],
)
def test_fold_count_follows_the_minority_class(n_minority, expected):
    assert auto_stratified_oof_n_splits(labels(500, n_minority)) == expected


@pytest.mark.parametrize("n_minority", [2, 3, 5, 11, 40, 120, 400])
def test_fold_count_is_always_usable(n_minority):
    """Never more folds than minority members, never fewer than two, never above five."""
    k = auto_stratified_oof_n_splits(labels(1000, n_minority))
    assert 2 <= k <= 5
    assert k <= n_minority, "a fold without a positive cannot be scored"


def test_the_same_labels_give_the_same_folds():
    """The property the whole design rests on: two callers must get identical folds."""
    y = labels(300, 40)
    k_a, splits_a = make_stratified_oof_splits(y)
    k_b, splits_b = make_stratified_oof_splits(y)

    assert k_a == k_b
    for (tr_a, va_a), (tr_b, va_b) in zip(splits_a, splits_b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


def test_validation_folds_partition_the_data():
    y = labels(300, 40)
    _, splits = make_stratified_oof_splits(y)
    covered = np.concatenate([val for _, val in splits])
    assert np.array_equal(np.sort(covered), np.arange(len(y))), (
        "every sample must be validated exactly once, or the OOF score is not an OOF score"
    )


def test_train_and_validation_never_overlap():
    y = labels(300, 40)
    _, splits = make_stratified_oof_splits(y)
    for train, val in splits:
        assert not set(train) & set(val), "a sample cannot be trained and validated on"
        assert len(train) + len(val) == len(y)


def test_every_fold_sees_both_classes():
    """A single-class validation fold makes AUC undefined."""
    y = labels(300, 40)
    _, splits = make_stratified_oof_splits(y)
    for _, val in splits:
        assert set(y[val]) == {0, 1}


def test_an_explicit_fold_count_overrides_the_automatic_one():
    y = labels(300, 200)
    assert auto_stratified_oof_n_splits(y) == 5, "the automatic choice here is 5"
    k, splits = make_stratified_oof_splits(y, n_splits=3)
    assert k == 3
    assert len(splits) == 3
