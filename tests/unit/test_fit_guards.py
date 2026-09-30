"""The guards added after the 3.6.0 audit, pinned.

Each of these replaced a defect that returned a plausible number or crashed somewhere
unhelpful. Fixing a defect without pinning it is how it comes back, and all four are
cheap to assert: pure arithmetic over labels and scores, no fitting, no ONNX.

Numpy only, so these run in the base tier -- which is where they belong, because three of
the four are about refusing to guess rather than about any model.
"""

import numpy as np
import pytest

from lazyqsar.qsar import _quality_weight
from lazyqsar.utils.metrics import bedroc_score
from lazyqsar.utils.ranking import (
    DECISION_HIT_RATE,
    DECISION_RANK,
    TAIL_ANCHORS,
    prepare_knots,
    proba_from_reference_rank,
)
from lazyqsar.utils.splits import auto_stratified_oof_n_splits, check_trainable


# ------------------------------------------------------------------ trainable labels


def _labels(n_neg, n_pos):
    return np.r_[np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)]


def test_a_single_minority_compound_is_refused_before_any_fitting():
    """The condition that used to crash inside StratifiedKFold after featurizing.

    A class of one cannot appear in both a training and a validation fold, so no fold
    count works -- and the calibration guard routed exactly this case to the uncalibrated
    path, which then asked for a single fold. On a slow-mode fit that surfaced after every
    descriptor had been computed, so the whole featurization pass was lost to something
    visible from the labels alone.
    """
    with pytest.raises(ValueError, match="single member"):
        check_trainable(_labels(119, 1))


def test_a_single_class_is_refused_and_says_which_one():
    """Naming the class present is the difference between a diagnosis and a complaint."""
    with pytest.raises(ValueError, match="every label is 0"):
        check_trainable(np.zeros(50, dtype=int))
    with pytest.raises(ValueError, match="every label is 1"):
        check_trainable(np.ones(50, dtype=int))


def test_more_than_two_classes_is_refused():
    with pytest.raises(ValueError, match="binary 0/1"):
        check_trainable(np.r_[_labels(10, 10), np.full(10, 2)])


@pytest.mark.parametrize("n_pos", [2, 5, 50])
def test_two_of_each_is_enough(n_pos):
    """Two per class is the real floor, and it must not be raised by accident."""
    assert check_trainable(_labels(50, n_pos)) is None


def test_the_guard_message_names_where_it_came_from():
    """Both fit entry points share this check, so the message has to say which one."""
    with pytest.raises(ValueError, match="LazyClassifier.fit"):
        check_trainable(np.zeros(10, dtype=int), where="LazyClassifier.fit")


def test_the_fold_count_reads_the_minority_not_the_only_class():
    """``np.bincount`` on a single-class y returns one bin, whose ``min`` is that class.

    So an all-negative vector reported a minority of *n* and got 5 folds of a single
    class, and every out-of-fold AUC computed from them was meaningless. ``minlength=2``
    makes the absent class count as zero, which is what it is.
    """
    assert auto_stratified_oof_n_splits(np.zeros(50, dtype=int)) == 2
    assert auto_stratified_oof_n_splits(np.ones(50, dtype=int)) == 2
    # Unchanged for real data: 10 actives still gives 3 folds.
    assert auto_stratified_oof_n_splits(_labels(40, 10)) == 3


# --------------------------------------------------------------------- quality weight


def test_quality_is_capped_at_one():
    """``2*oof - train`` is not bounded above, and the excess was rewarded.

    ``train < oof`` happens on small data, and a failed training-AUC computation used to
    stand in as 0.5 -- which on a strong descriptor gives ``2*0.9 - 0.5 = 1.3``, roughly
    double the weight of a genuinely good descriptor. The ensemble reads this as a skill
    score, so a value above 1 is not a large skill, it is a broken measurement.
    """
    assert _quality_weight(0.9, 0.5) == 1.0
    assert _quality_weight(0.9, 0.95) == pytest.approx(0.85)
    assert _quality_weight(0.5, 1.0) == 0.0  # clipped from below too


def test_an_unknown_training_auc_falls_back_to_the_out_of_fold_auc():
    """There is no honest gap to charge when one side is unknown.

    The point of returning ``None`` from the AUC helpers was to stop a failure looking
    like a measurement; weighting on the out-of-fold AUC alone keeps that honest rather
    than inventing a penalty.
    """
    assert _quality_weight(0.82, None) == pytest.approx(0.82)
    assert _quality_weight(None, 0.99) is None
    assert _quality_weight(None, None) is None


# ------------------------------------------------------------- the cutoff's own claim


def test_the_expected_hit_rate_is_derived_from_the_anchors():
    """``DECISION_HIT_RATE`` must not become a stale copy of 0.01.

    It is the share of the reference the cutoff is *defined* to admit, so it has to follow
    ``DECISION_RANK`` and ``TAIL_ANCHORS`` rather than being written down beside them.
    """
    percentile = next(q for q, rank in TAIL_ANCHORS if rank == DECISION_RANK)
    assert DECISION_HIT_RATE == pytest.approx(1.0 - percentile / 100.0)
    assert DECISION_HIT_RATE == pytest.approx(0.01)


def test_a_healthy_reference_admits_exactly_the_defined_share():
    """The construction the warning checks, in its working case."""
    rng = np.random.default_rng(0)
    reference = np.sort(rng.beta(1.2, 9.0, size=50_000))
    cutoff = proba_from_reference_rank(DECISION_RANK, prepared=prepare_knots(reference))
    assert float((reference >= cutoff).mean()) == pytest.approx(
        DECISION_HIT_RATE, abs=1e-3
    )


def test_a_degenerate_reference_breaks_the_construction_detectably():
    """Why the warning exists, and that its condition is real rather than theoretical.

    When the reference's probabilities are nearly all equal, the four anchors interpolate
    to the same value, ``_collapse_anchors`` keeps only the topmost, and inverting
    ``DECISION_RANK`` lands *below* the whole library -- so the cutoff admits everything
    instead of 1%. Unreachable on a healthy 50,000-molecule library; reachable on a small
    reference tier or a saturated model, and silent until it was checked.
    """
    reference = np.full(1000, 0.3)
    cutoff = proba_from_reference_rank(DECISION_RANK, prepared=prepare_knots(reference))
    admitted = float((reference >= cutoff).mean())
    assert admitted == 1.0, (
        "the degenerate case no longer collapses; if the anchor table changed, this test "
        "is the record of what the warning was guarding against"
    )
    assert abs(admitted - DECISION_HIT_RATE) > 10 * DECISION_HIT_RATE, (
        "the warning's threshold would not fire on the very case it exists for"
    )


# ---------------------------------------------------------------------------- bedroc


@pytest.mark.parametrize(
    ("y", "scores"),
    [
        ([1, 1, 1, 0], [0.9, 0.8, 0.7, 0.1]),
        ([1, 0, 0], [0.9, 0.5, 0.1]),
        ([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]),
    ],
    ids=["3-of-4-active", "1-of-3-active", "balanced"],
)
def test_a_perfect_ranking_scores_one_however_small_the_input(y, scores):
    """``bedroc_score`` lacked the zero-denominator guard its sibling has.

    On a handful of molecules the normalising range collapses, and the bare division
    returned ``-1.3e-18`` for a *perfect* ranking -- outside the documented [0, 1] and, far
    worse, indistinguishable from the worst possible answer.
    """
    assert bedroc_score(np.array(y), np.array(scores)) == pytest.approx(1.0)


def test_bedroc_stays_inside_its_documented_range():
    """The property the guard restores, over inputs small enough to break the maths."""
    rng = np.random.default_rng(1)
    for n in range(2, 12):
        for n_a in range(1, n):
            y = _labels(n - n_a, n_a)
            for _ in range(5):
                value = bedroc_score(y, rng.random(n))
                assert 0.0 <= value <= 1.0, f"n={n}, n_a={n_a} gave {value!r}"
