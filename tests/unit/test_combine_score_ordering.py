"""``score`` must order molecules exactly as ``proba`` does.

Tested here, on ``combine`` as a pure function, rather than end to end. Both realistic
end-to-end fixtures are blind to this: a fast-mode checkpoint has one descriptor, where the
weights collapse to a single column and the old fallback degenerates to ``score == proba``;
and a stub multi-descriptor checkpoint produces columns so correlated that linear and
logit-space pooling agree by accident. Neither can tell a working pooled score map from a
broken one. Hand-built columns that genuinely disagree can.

The defect this pins: pooling the raw per-descriptor scores independently is not a monotone
transform of the pooled probability, so the same model could rank two molecules one way by
``proba`` and the other way by ``score``. Deriving ``score`` from the pooled probability
through a stored monotone map removes that by construction.
"""

import numpy as np
import pytest

from lazyqsar.ensemble.combine import EnsembleSpec, combine


def flipped_pairs(a, b):
    """Pairs *a* and *b* strictly disagree about. Ties are not disagreement."""
    sign_a = np.sign(np.subtract.outer(a, a))
    sign_b = np.sign(np.subtract.outer(b, b))
    return int(((sign_a * sign_b) < 0).sum() // 2)


@pytest.fixture
def disagreeing():
    """Two descriptors whose raw and calibrated columns rank molecules differently.

    Descriptor 0 is confident and well spread; descriptor 1 is compressed toward the middle
    and, on the raw scale, disagrees with it. That is the shape a real ensemble takes when
    its heads carry calibrators of different slopes.
    """
    rng = np.random.default_rng(0)
    n = 60
    a = np.linspace(0.05, 0.95, n)
    b = 0.5 + 0.12 * np.sin(np.linspace(0, 9, n))
    Y = np.column_stack([a, b])
    # Raw scores that are monotone per column but compressed differently, so pooling them
    # linearly does not reproduce the pooled-probability order.
    S = np.column_stack([a**3, 0.5 + 0.45 * (b - 0.5) * 6])
    S = np.clip(S, 1e-6, 1 - 1e-6)
    R = np.column_stack([rng.uniform(size=n), rng.uniform(size=n)])
    return Y, R, S


def _spec(score_knots=None):
    return EnsembleSpec(
        descriptor_names=("morgan", "rdkit"),
        oof_aucs=(0.80, 0.75),
        population_prior=0.3,
        pooled_score_knots=score_knots,
    )


def test_without_a_map_score_can_contradict_proba(disagreeing):
    """The behaviour the map replaces. If this stops holding the fixture has gone stale."""
    Y, R, S = disagreeing
    out = combine(Y, R, S, None, spec=_spec(), outputs=("proba", "score"))
    assert flipped_pairs(out.values["proba"][:, 1], out.values["score"][:, 1]) > 0, (
        "the fixture no longer produces a disagreement, so the test below proves nothing"
    )


def test_with_a_map_score_never_contradicts_proba(disagreeing):
    """Zero inverted pairs, which is the whole point of the pooled score map."""
    Y, R, S = disagreeing
    reference = combine(Y, R, S, None, spec=_spec(), outputs=("proba", "score"))
    p1 = reference.values["proba"][:, 1]
    s1 = reference.values["score"][:, 1]

    from lazyqsar.ensemble.reference import build_pooled_score_knots

    knots = build_pooled_score_knots(Y, R, S, None, _spec())
    assert knots is not None

    out = combine(Y, R, S, None, spec=_spec(knots), outputs=("proba", "score"))
    mapped = out.values["score"][:, 1]

    assert flipped_pairs(out.values["proba"][:, 1], mapped) == 0
    np.testing.assert_allclose(out.values["proba"][:, 1], p1, rtol=0, atol=0)
    # Still on the raw scale rather than collapsed onto proba: the map is fitted to the
    # raw values, so it cannot leave their range. How *closely* it tracks them is a
    # property of real data, not of this fixture -- here the wiggle being flattened is
    # deliberately enormous, and moving those points is the map doing its job. The real
    # measurement lives in tests/chem/test_outputs_agree_on_order.py.
    assert s1.min() <= mapped.min() and mapped.max() <= s1.max(), (
        "the mapped score left the range of the raw scores it is fitted to"
    )


def test_the_map_is_monotone_and_ascending(disagreeing):
    """`np.interp` needs ascending x, and the ordering guarantee needs non-decreasing y."""
    from lazyqsar.ensemble.reference import build_pooled_score_knots

    Y, R, S = disagreeing
    x, y = build_pooled_score_knots(Y, R, S, None, _spec())
    assert np.all(np.diff(x) > 0)
    assert np.all(np.diff(y) >= 0)


def test_no_map_is_built_without_the_raw_channel(disagreeing):
    """`S is None` means no descriptor could supply raw scores; there is nothing to map."""
    from lazyqsar.ensemble.reference import build_pooled_score_knots

    Y, R, _ = disagreeing
    assert build_pooled_score_knots(Y, R, None, None, _spec()) is None
    assert build_pooled_score_knots(None, R, None, None, _spec()) is None
