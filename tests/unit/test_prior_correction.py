"""Shifting a batch's probabilities from its training prevalence to the population's.

When severely imbalanced data is split into batches, each batch is enriched with positives
relative to the library it came from, so a model fitted on it is calibrated to the wrong
base rate. ``_correct_prior`` rescales the odds to undo that before the batches are
averaged.

It has never run in the test suite. It returns its input unchanged when the two priors
agree, and they always do on a single-batch checkpoint, which is what every committed
fixture produces -- so the arithmetic below was dead code as far as the suite was
concerned. These are pure-numpy tests on the inference path, so they cost nothing.
"""

import numpy as np
import pytest

from lazyqsar.artifacts.classifier import _correct_prior


def test_equal_priors_change_nothing():
    """The single-batch case, which is every checkpoint the fixtures produce."""
    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    out = _correct_prior(p, 0.3, 0.3)
    np.testing.assert_array_equal(out, p)


def test_a_rarer_population_pushes_probabilities_down():
    """A batch enriched to 20% positives, scoring a library that is really 1%."""
    p = np.array([0.1, 0.5, 0.9])
    out = _correct_prior(p, 0.2, 0.01)
    assert np.all(out < p), f"probabilities should fall, got {out} from {p}"
    assert np.all((out > 0) & (out < 1))


def test_a_commoner_population_pushes_them_up():
    p = np.array([0.1, 0.5, 0.9])
    assert np.all(_correct_prior(p, 0.01, 0.2) > p)


def test_it_is_strictly_monotone():
    """The property the rest of the package leans on: the correction never reorders.

    ``predict_proba`` applies this per batch and averages; ``rank``, ``logit``, ``lift``
    and ``binary`` are all monotone views of that average. A correction that could invert
    two molecules would put every one of them at odds with the others.
    """
    p = np.linspace(0.001, 0.999, 500)
    out = _correct_prior(p, 0.25, 0.02)
    assert np.all(np.diff(out) > 0)


@pytest.mark.parametrize("train,pop", [(0.5, 0.001), (0.001, 0.5), (0.02, 0.019)])
def test_it_stays_in_range(train, pop):
    p = np.linspace(1e-6, 1 - 1e-6, 200)
    out = _correct_prior(p, train, pop)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


@pytest.mark.parametrize("train", [0.0, 1.0, -0.5, 1.5])
def test_a_degenerate_training_prior_is_declined(train):
    """No positives or no negatives in a batch means no odds ratio to form."""
    p = np.array([0.2, 0.6])
    np.testing.assert_array_equal(_correct_prior(p, train, 0.1), p)


def test_a_half_to_half_correction_is_the_identity():
    """The odds ratio is 1, so nothing should move -- a floating-point sanity check."""
    p = np.array([0.13, 0.42, 0.77])
    np.testing.assert_allclose(_correct_prior(p, 0.5, 0.5), p, rtol=0, atol=0)


def test_the_realistic_case_from_a_two_batch_fit():
    """The numbers a 2-positive/201-negative fit actually produces.

    Batch prior 0.0194 against a population prior of 0.0099: roughly a halving of the
    odds, which is the correction the averaging step has been applying to nothing.
    """
    p = np.array([0.05, 0.3, 0.7, 0.95])
    out = _correct_prior(p, 0.0194, 0.0099)
    assert np.all(out < p)
    assert np.all(np.diff(out) > 0)
    # Odds are scaled by a constant factor, so the ratio of odds ratios is flat.
    odds_in = p / (1 - p)
    odds_out = out / (1 - out)
    ratios = odds_out / odds_in
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
