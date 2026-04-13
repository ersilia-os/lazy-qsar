import numpy as np
import pytest

from lazyqsar.poolers.classification.inner_pooler import _all_metrics
from lazyqsar.utils.metrics import (
    bedroc_random_baseline,
    bedroc_score,
    composite_metrics,
    composite_score,
)


def test_auroc_normalization_maps_random_and_perfect_to_unit_interval():
    y = np.array([0, 0, 1, 1], dtype=int)

    random_like = composite_metrics(y, np.array([0.1, 0.9, 0.2, 0.8], dtype=float))
    perfect = composite_metrics(y, np.array([0.1, 0.2, 0.8, 0.9], dtype=float))

    assert random_like["auroc"] == pytest.approx(0.5)
    assert random_like["auroc_excess"] == pytest.approx(0.0)
    assert perfect["auroc"] == pytest.approx(1.0)
    assert perfect["auroc_excess"] == pytest.approx(1.0)


def test_aupr_normalization_uses_prevalence_as_random_baseline():
    y = np.array([1, 0, 0, 0], dtype=int)

    flat = composite_metrics(y, np.array([0.5, 0.5, 0.5, 0.5], dtype=float))
    perfect = composite_metrics(y, np.array([0.9, 0.3, 0.2, 0.1], dtype=float))

    assert flat["aupr"] == pytest.approx(y.mean())
    assert flat["aupr_random"] == pytest.approx(y.mean())
    assert flat["aupr_excess"] == pytest.approx(0.0)
    assert perfect["aupr"] == pytest.approx(1.0)
    assert perfect["aupr_excess"] == pytest.approx(1.0)


def test_bedroc_normalization_uses_random_baseline_and_perfect_ranking_maps_near_one():
    y = np.array([1, 1, 0, 0, 0, 0], dtype=int)

    random_baseline = bedroc_random_baseline(y)
    perfect_scores = np.array([0.99, 0.98, 0.3, 0.2, 0.1, 0.0], dtype=float)
    perfect = composite_metrics(y, perfect_scores)

    assert 0.0 <= random_baseline <= 1.0
    assert bedroc_score(y, perfect_scores) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert perfect["bedroc_random"] == pytest.approx(random_baseline)
    assert perfect["bedroc_excess"] == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_composite_differs_from_raw_mean_on_imbalanced_problem():
    scores = np.linspace(1.0, 0.0, 100, dtype=float)
    y = np.zeros(100, dtype=int)
    y[[0, 10, 20, 30, 40]] = 1

    metrics = composite_metrics(y, scores)
    raw_mean = (metrics["auroc"] + metrics["aupr"] + metrics["bedroc"]) / 3.0

    assert metrics["aupr_random"] == pytest.approx(0.05)
    assert metrics["composite"] != pytest.approx(raw_mean)


def test_all_metrics_uses_shared_normalized_composite():
    y = np.array([0, 1, 0, 1, 0, 0, 0, 1], dtype=int)
    scores = np.array([0.1, 0.8, 0.4, 0.7, 0.2, 0.3, 0.6, 0.9], dtype=float)

    metrics = _all_metrics(y, scores)

    assert metrics["composite"] == pytest.approx(composite_score(y, scores))
    assert metrics["auroc"] == pytest.approx(composite_metrics(y, scores)["auroc"])
    assert metrics["aupr"] == pytest.approx(composite_metrics(y, scores)["aupr"])
    assert metrics["bedroc"] == pytest.approx(composite_metrics(y, scores)["bedroc"])


def test_composite_metrics_handles_single_class_without_division_errors():
    y_all_zero = np.zeros(8, dtype=int)
    y_all_one = np.ones(8, dtype=int)
    scores = np.linspace(0.1, 0.9, 8, dtype=float)

    zero_metrics = composite_metrics(y_all_zero, scores)
    one_metrics = composite_metrics(y_all_one, scores)

    for metrics in (zero_metrics, one_metrics):
        for value in metrics.values():
            assert np.isfinite(value)
        assert 0.0 <= metrics["composite"] <= 1.0
        assert metrics["auroc_excess"] == pytest.approx(0.0)
        assert metrics["aupr_excess"] == pytest.approx(0.0)
        assert metrics["bedroc_excess"] == pytest.approx(0.0)


def test_below_random_values_clip_to_zero():
    y = np.array([0, 0, 1, 1], dtype=int)
    anti_ranked = composite_metrics(y, np.array([0.9, 0.8, 0.2, 0.1], dtype=float))

    assert anti_ranked["auroc"] == pytest.approx(0.0)
    assert anti_ranked["auroc_excess"] == pytest.approx(0.0)
    assert anti_ranked["aupr_excess"] == pytest.approx(0.0)
    assert anti_ranked["bedroc_excess"] == pytest.approx(0.0)
