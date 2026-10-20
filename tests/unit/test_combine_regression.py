"""The regression branch of ``combine`` and its spec -- separate from classification.

A regression checkpoint carries ``"task": "regression"`` and Spearman ρ under ``*_rhos``
keys. Its ensemble output is ``value``: the weighted mean of per-descriptor predictions
under the same weight machinery as classification (skill, rank-error reliability, AD
veto), but with ρ in place of AUC and no logit pooling. Classification arithmetic is
pinned separately by the golden test; these tests pin the regression semantics. Numpy only.
"""

import numpy as np
import pytest

from lazyqsar.ensemble import (
    OUTPUT_NAMES,
    REGRESSION_OUTPUT_NAMES,
    EnsembleSpec,
    build_weight_matrix,
    combine,
    read_task,
)


def _spec(**kw):
    kw.setdefault("descriptor_names", ("a", "b", "c"))
    return EnsembleSpec(task="regression", **kw)


def _values(n=50, D=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=6.0, scale=1.5, size=(n, D))


# --- task resolution -----------------------------------------------------------------


def test_missing_task_key_is_classification():
    """Every pre-regression checkpoint has no task key and must load unchanged."""
    assert read_task({}) == "classification"
    assert read_task(None) == "classification"
    spec, _ = EnsembleSpec.from_metadata({"oof_aucs": {"a": 0.8}}, ["a"])
    assert spec.task == "classification"


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown task"):
        read_task({"task": "multiclass"})


def test_regression_spec_reads_rhos_and_ignores_classification_keys():
    meta = {
        "task": "regression",
        "quality_rhos": {"a": 0.6},
        "oof_rhos": {"a": 0.7, "b": 0.5},
        "proxy_rhos": {"b": 0.4},
        # Classification keys must not leak into a regression spec.
        "oof_aucs": {"a": 0.99, "b": 0.99, "c": 0.99},
        "population_prior": 0.1,
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b", "c"])
    assert spec.task == "regression"
    assert active == ["a", "b", "c"]
    assert spec.oof_aucs == (0.6, 0.5, None)  # quality first, then oof, else unknown
    assert spec.proxy_aucs == (None, 0.4, None)
    assert spec.pooled_rank_knots is None


# --- outputs ---------------------------------------------------------------------------


def test_regression_outputs_are_validated_per_task():
    Y = _values()
    with pytest.raises(ValueError, match="for a regression model"):
        combine(Y, spec=_spec(), outputs=("proba",))
    classification = EnsembleSpec(descriptor_names=("a", "b", "c"))
    with pytest.raises(ValueError, match="for a classification model"):
        combine(np.full((5, 3), 0.5), spec=classification, outputs=("value",))


def test_default_outputs_follow_the_task():
    res = combine(_values(), spec=_spec())
    assert set(res.values) == set(REGRESSION_OUTPUT_NAMES)
    # A classification spec without a reference refuses `rank`, so asking for every
    # output by default must fail on exactly that -- proving the default is the full
    # classification set, not the regression one.
    with pytest.raises(ValueError, match="reference"):
        combine(
            np.full((4, 3), 0.5), spec=EnsembleSpec(descriptor_names=("a", "b", "c"))
        )
    assert "rank" in OUTPUT_NAMES


def test_value_is_one_dimensional_weighted_mean():
    Y = _values()
    res = combine(Y, spec=_spec())
    v = res.values["value"]
    assert v.shape == (Y.shape[0],)
    np.testing.assert_allclose(res.weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(v, (res.weights * Y).sum(axis=1))


def test_value_stays_inside_the_descriptor_range():
    """A convex combination can never leave the span of its inputs."""
    Y = _values()
    v = combine(Y, A=np.ones_like(Y), spec=_spec(oof_aucs=(0.9, 0.2, 0.5))).values[
        "value"
    ]
    assert (v >= Y.min(axis=1) - 1e-12).all() and (v <= Y.max(axis=1) + 1e-12).all()


def test_single_descriptor_returns_its_predictions():
    Y = _values(D=1)
    res = combine(Y, spec=_spec(descriptor_names=("a",)))
    np.testing.assert_allclose(res.values["value"], Y[:, 0])


def test_constant_offset_shifts_value_by_the_offset():
    """Weights do not depend on Y's level, so the pooled value is translation-equivariant."""
    Y = _values()
    A = np.full_like(Y, 0.9)
    spec = _spec(oof_aucs=(0.8, 0.3, 0.6))
    v0 = combine(Y, A=A, spec=spec).values["value"]
    v1 = combine(Y + 100.0, A=A, spec=spec).values["value"]
    np.testing.assert_allclose(v1, v0 + 100.0)


# --- weights ---------------------------------------------------------------------------


def test_skill_term_is_half_the_positive_rho():
    Y = _values()
    _, base = build_weight_matrix(Y, None, None, _spec(oof_aucs=(0.8, -0.3, None)))
    np.testing.assert_allclose(base, [0.4, 0.0, 0.0])


def test_no_skill_information_is_uniform():
    Y = _values()
    _, base = build_weight_matrix(Y, None, None, _spec(oof_aucs=(None, None, None)))
    np.testing.assert_allclose(base, [1.0, 1.0, 1.0])


def test_with_ad_and_no_curves_weights_follow_skill_not_rank_extremity():
    """No `|R - 0.5| * 2` fallback for regression: an extreme value is not a surer one."""
    Y = _values()
    A = np.ones_like(Y)
    spec = _spec(oof_aucs=(0.8, 0.4, 0.4))
    R_mid = np.full_like(Y, 0.5)
    R_edge = np.full_like(Y, 0.99)
    W_mid, _ = build_weight_matrix(Y, R_mid, A, spec)
    W_edge, _ = build_weight_matrix(Y, R_edge, A, spec)
    np.testing.assert_allclose(W_mid, W_edge)
    np.testing.assert_allclose(W_mid[0], [0.5, 0.25, 0.25])


def test_rank_error_curves_lower_the_weight_where_error_is_high():
    Y = _values(n=2)
    A = np.ones_like(Y)
    flat = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    rising = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))  # error grows with rank
    spec = _spec(oof_aucs=(0.5, 0.5, 0.5), rank_error_curves=(flat, rising, flat))
    R = np.array([[0.5, 0.0, 0.5], [0.5, 0.95, 0.5]])
    W, _ = build_weight_matrix(Y, R, A, spec)
    assert W[1, 1] < W[0, 1]


def test_ad_veto_zeroes_a_descriptor():
    Y = _values()
    A = np.ones_like(Y)
    A[:10, 1] = 0.0
    spec = _spec(oof_aucs=(0.6, 0.6, 0.6), ad_hard_cutoffs=(0.5, 0.5, 0.5))
    res = combine(Y, A=A, spec=spec)
    assert (res.weights[:10, 1] == 0).all()
    assert (res.weights[10:, 1] > 0).all()
    np.testing.assert_allclose(res.weights.sum(axis=1), 1.0)


def test_vetoed_everywhere_falls_back_to_skill():
    Y = _values(n=3)
    A = np.zeros_like(Y)
    spec = _spec(oof_aucs=(0.8, 0.4, 0.4), ad_hard_cutoffs=(0.5, 0.5, 0.5))
    W, _ = build_weight_matrix(Y, None, A, spec)
    np.testing.assert_allclose(W, np.tile([0.5, 0.25, 0.25], (3, 1)))


def test_rows_are_independent():
    """Streaming chunks the query, so row i may depend only on row i."""
    Y = _values(n=40)
    A = np.random.default_rng(3).random(Y.shape)
    spec = _spec(oof_aucs=(0.7, 0.5, 0.3), ad_hard_cutoffs=(0.2, 0.2, 0.2))
    whole = combine(Y, A=A, spec=spec).values["value"]
    parts = np.concatenate(
        [
            combine(Y[i : i + 7], A=A[i : i + 7], spec=spec).values["value"]
            for i in range(0, 40, 7)
        ]
    )
    np.testing.assert_array_equal(whole, parts)


def test_diagnostics_use_rho_keys():
    Y = _values()
    res = combine(Y, A=np.ones_like(Y), spec=_spec(oof_aucs=(0.7, None, 0.2)))
    row = res.diagnostics[0]
    assert "oof_rho" in row and "oof_auc" not in row
    assert np.isnan(res.diagnostics[1]["oof_rho"])
