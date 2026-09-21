"""How training data is partitioned into batches, and when it is partitioned at all.

Severely imbalanced data is not fitted in one go: every batch gets all the positives plus a
disjoint slice of the negatives, so each model sees a workable class ratio and the ensemble
averages them. It is the only branch that reshapes the training set, and until now nothing
tested it — `reference_imbalanced.csv` is low-prevalence at 13% but only 6.7:1, well under
the 100:1 the branch needs, so it took the single-batch path exactly like the balanced set.

Pure function, so this costs nothing to run. The end-to-end consequences are in
``tests/chem/test_imbalanced_batching.py``, which does have to fit.
"""

import numpy as np
import pytest
from _helpers.smiles import load_severely_imbalanced_dataset

from lazyqsar.assemblers.classifier import _plan_batches

RATIO = 100  # _plan_batches' max_imbalance_ratio default


def _plan(n_pos, n_neg):
    y = np.array([1] * n_pos + [0] * n_neg)
    return _plan_batches(np.zeros((len(y), 4)), y), y


def test_the_committed_fixtures_do_not_reach_this_branch():
    """Recorded deliberately, because the data README used to claim they did.

    If a future fixture does trip the branch that is fine, but it should be a decision
    rather than a surprise -- the batching changes what every model in the ensemble is
    trained on.
    """
    from _helpers.smiles import load_imbalanced_dataset, load_reference_dataset

    for loader in (load_reference_dataset, load_imbalanced_dataset):
        _, y = loader()
        y = np.asarray(y)
        batches = _plan_batches(np.zeros((len(y), 4)), y)
        assert len(batches) == 1, (
            "a committed fixture now reaches the imbalance branch; the tests that assume "
            "one batch, and the data README, both need revisiting"
        )


@pytest.mark.parametrize("n_pos,n_neg", [(1, 101), (2, 201), (3, 301), (1, 550)])
def test_above_the_ratio_the_data_is_split(n_pos, n_neg):
    batches, _ = _plan(n_pos, n_neg)
    assert len(batches) > 1
    assert len(batches) == int(np.ceil(n_neg / (RATIO * n_pos)))


@pytest.mark.parametrize("n_pos,n_neg", [(10, 100), (56, 376), (5, 500), (1, 100)])
def test_at_or_below_the_ratio_it_is_not(n_pos, n_neg):
    batches, _ = _plan(n_pos, n_neg)
    assert len(batches) == 1
    assert len(batches[0]) == n_pos + n_neg


def test_every_batch_carries_all_the_positives():
    """The contract the averaging depends on: batches differ only in their negatives."""
    batches, y = _plan(3, 301)
    positives = set(np.flatnonzero(y == 1).tolist())
    for i, batch in enumerate(batches):
        assert positives <= set(batch.tolist()), f"batch {i} is missing positives"


def test_the_negatives_are_partitioned_not_resampled():
    """Disjoint slices covering every negative exactly once -- no molecule seen twice."""
    batches, y = _plan(3, 301)
    negatives = set(np.flatnonzero(y == 0).tolist())
    seen = [set(b.tolist()) & negatives for b in batches]
    union = set().union(*seen)
    assert union == negatives, "some negatives were dropped from every batch"
    assert sum(len(s) for s in seen) == len(negatives), (
        "a negative appears in two batches"
    )


def test_planning_is_deterministic():
    """The negative shuffle is seeded, so two fits of one dataset agree."""
    first, _ = _plan(2, 201)
    second, _ = _plan(2, 201)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_the_derived_fixture_trips_the_branch():
    """What ``load_severely_imbalanced_dataset`` exists for."""
    _, y = load_severely_imbalanced_dataset()
    y = np.asarray(y)
    batches = _plan_batches(np.zeros((len(y), 4)), y)
    assert len(batches) == 2, (
        "the derived fixture no longer plans multiple batches, so the end-to-end "
        "imbalance test is silently exercising the single-batch path"
    )


def test_a_single_class_is_not_split():
    """Degenerate input must fall out early rather than divide by zero."""
    for y in (np.ones(50, dtype=int), np.zeros(50, dtype=int)):
        batches = _plan_batches(np.zeros((len(y), 4)), y)
        assert len(batches) == 1
        assert len(batches[0]) == len(y)
