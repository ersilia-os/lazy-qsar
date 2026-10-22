"""Split-conformal intervals binned by applicability domain (lazyqsar.ensemble.intervals)."""

import numpy as np
import pytest

from lazyqsar.ensemble.intervals import fit_intervals, interval_half_width


def _calibration(n=3000, seed=0):
    """Residual scale grows as the AD score falls: in-domain rows are predicted better."""
    rng = np.random.default_rng(seed)
    ad = rng.uniform(size=n)
    noise = rng.normal(size=n) * (0.2 + 1.0 * (1.0 - ad))
    y = rng.normal(size=n)
    return y, y + noise, ad


def test_coverage_holds_within_each_ad_bin():
    y, pred, ad = _calibration()
    model = fit_intervals(y, pred, ad)
    assert len(model["half_widths"]) == 3
    y2, pred2, ad2 = _calibration(seed=1)
    half = interval_half_width(model, len(y2), ad2, level=0.9)
    covered = np.abs(y2 - pred2) <= half
    bins = np.searchsorted(model["ad_edges"], ad2, side="right")
    for b in range(3):
        assert 0.86 <= covered[bins == b].mean() <= 0.94


def test_out_of_domain_rows_get_wider_intervals():
    y, pred, ad = _calibration()
    model = fit_intervals(y, pred, ad)
    low, high = interval_half_width(model, 2, np.array([0.01, 0.99]), level=0.9)
    assert low > high


def test_small_calibration_sets_use_one_bin_and_no_ad_uses_the_widest():
    y, pred, ad = _calibration(n=60)
    assert fit_intervals(y, pred, ad)["ad_edges"] == []
    y, pred, ad = _calibration()
    model = fit_intervals(y, pred, ad)
    widest = max(np.interp(0.9, model["levels"], w) for w in model["half_widths"])
    assert np.allclose(interval_half_width(model, 4, None, 0.9), widest)


def test_level_must_be_a_probability():
    y, pred, ad = _calibration(n=200)
    model = fit_intervals(y, pred, ad)
    with pytest.raises(ValueError):
        interval_half_width(model, 3, None, level=1.0)
