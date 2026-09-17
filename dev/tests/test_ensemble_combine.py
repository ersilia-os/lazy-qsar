"""``combine()`` must reproduce today's ensemble arithmetic bit for bit.

The reference implementations below are frozen verbatim copies of ``qsar.py`` as of
3.4.4 — ``_build_weight_matrix`` and the six ``predict_*`` bodies. They are duplicated
here on purpose: once ``qsar.py`` is routed through ``combine()`` there is nothing left
to compare against, so the old behaviour has to live somewhere that the refactor cannot
quietly change.

Assertions use ``array_equal``, not ``allclose``. The claim is that the arithmetic is
unchanged, and a tolerance would hide exactly the reordering mistakes this file exists to
catch.

Numpy only, so it runs on every install.
"""

import numpy as np
import pytest

from lazyqsar.ensemble import OUTPUT_NAMES, EnsembleSpec, combine

# ---------------------------------------------------------------------------
# Frozen reference: lazyqsar/qsar.py at 3.4.4
# ---------------------------------------------------------------------------


def _legacy_build_weight_matrix(
    Y, R, A, oof_aucs, proxy_aucs, rank_error_curves, active_indices, ad_hard_cutoffs
):
    B, D = Y.shape

    base_scores = []
    for i in active_indices:
        vals = []
        if oof_aucs and oof_aucs[i] is not None:
            vals.append(float(oof_aucs[i]))
        if proxy_aucs and proxy_aucs[i] is not None:
            vals.append(float(proxy_aucs[i]))
        base_scores.append(max(0.0, float(np.mean(vals)) - 0.5) if vals else 0.0)
    base = np.array(base_scores, dtype=np.float64)
    if base.sum() == 0:
        base = np.ones(D, dtype=np.float64)

    if A is not None:
        if R is not None:
            if rank_error_curves and all(
                rank_error_curves[i] is not None for i in active_indices
            ):
                reliability = np.zeros((B, D), dtype=np.float64)
                for j, i in enumerate(active_indices):
                    r_knots, e_knots = rank_error_curves[i]
                    reliability[:, j] = 1.0 - np.interp(R[:, j], r_knots, e_knots)
            else:
                reliability = np.abs(R - 0.5) * 2
            W = 0.5 * base[np.newaxis, :] + 0.5 * reliability
        else:
            W = np.tile(base, (B, 1))

        if ad_hard_cutoffs is not None:
            for j, i in enumerate(active_indices):
                W[A[:, j] < ad_hard_cutoffs[i], j] = 0.0

        all_ood = W.sum(axis=1) == 0
        if all_ood.any():
            W[all_ood] = base

        W /= W.sum(axis=1, keepdims=True)
    else:
        W = np.full((B, D), 1.0 / D, dtype=np.float64)

    return W, base


def _legacy_outputs(Y, R, S, A, spec_kwargs, cutoff=0.5):
    """The six predict_* bodies, verbatim, sharing one _compute_ensemble result."""
    B, D = Y.shape
    active_indices = list(range(D))
    if S is None:
        S = Y.copy()
    W, _ = _legacy_build_weight_matrix(
        Y,
        R,
        A,
        spec_kwargs["oof_aucs"],
        spec_kwargs["proxy_aucs"],
        spec_kwargs["rank_error_curves"],
        active_indices,
        spec_kwargs["ad_hard_cutoffs"],
    )
    if R is None:
        R = np.full((B, D), 0.5, dtype=np.float64)

    logits = np.log(np.clip(Y, 1e-7, 1 - 1e-7) / np.clip(1 - Y, 1e-7, 1 - 1e-7))
    p1 = 1.0 / (1.0 + np.exp(-(W * logits).sum(axis=1)))
    proba = np.vstack((1 - p1, p1)).T

    l1 = (W * logits).sum(axis=1)
    r1 = (W * R).sum(axis=1)
    s1 = (W * S).sum(axis=1)
    prior = spec_kwargs["population_prior"]

    return {
        "proba": proba,
        "logit": np.vstack((-l1, l1)).T,
        "rank": np.vstack((1 - r1, r1)).T,
        "score": np.vstack((1 - s1, s1)).T,
        "lift": np.column_stack(
            [proba[:, 0] / max(1 - prior, 1e-7), proba[:, 1] / max(prior, 1e-7)]
        ),
        "binary": (proba[:, 1] >= cutoff).astype(int),
    }, W


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------


def _curves(rng, D, how_many_none=0):
    out = []
    for j in range(D):
        if j < how_many_none:
            out.append(None)
        else:
            knots = np.linspace(0.0, 1.0, 20)
            out.append((knots, np.sort(rng.random(20))[::-1].copy()))
    return out


def _case(rng, B, D, *, with_ad, with_rank, with_score, curves, veto, no_skill):
    Y = rng.uniform(0.01, 0.99, size=(B, D))
    R = rng.uniform(0.0, 1.0, size=(B, D)) if with_rank else None
    S = rng.uniform(0.0, 1.0, size=(B, D)) if with_score else None
    A = rng.uniform(0.0, 1.0, size=(B, D)) if with_ad else None

    if no_skill:
        oof = [0.4] * D  # every value <= 0.5 -> base.sum() == 0 -> ones fallback
        proxy = [None] * D
    else:
        oof = [float(v) for v in rng.uniform(0.55, 0.99, size=D)]
        proxy = [
            None if j % 3 == 0 else float(v)
            for j, v in enumerate(rng.uniform(0.55, 0.99, size=D))
        ]

    if veto == "none":
        cutoffs = None
    elif veto == "partial":
        cutoffs = [0.5] * D
    else:  # "all" -- every descriptor vetoed for every sample
        cutoffs = [1.5] * D

    spec_kwargs = {
        "oof_aucs": oof,
        "proxy_aucs": proxy,
        "rank_error_curves": curves,
        "ad_hard_cutoffs": cutoffs,
        "population_prior": 0.23,
    }
    return Y, R, S, A, spec_kwargs


def _spec(D, spec_kwargs, cutoff=0.5):
    return EnsembleSpec(
        descriptor_names=tuple(f"d{j}" for j in range(D)),
        oof_aucs=tuple(spec_kwargs["oof_aucs"]),
        proxy_aucs=tuple(spec_kwargs["proxy_aucs"]),
        rank_error_curves=(
            tuple(spec_kwargs["rank_error_curves"])
            if spec_kwargs["rank_error_curves"]
            else None
        ),
        ad_hard_cutoffs=(
            tuple(spec_kwargs["ad_hard_cutoffs"])
            if spec_kwargs["ad_hard_cutoffs"] is not None
            else None
        ),
        population_prior=spec_kwargs["population_prior"],
        decision_cutoff=cutoff,
    )


# Each entry selects a different branch of the weighting logic.
BRANCHES = [
    (
        "no_ad_uniform",
        {"with_ad": False, "with_rank": True, "with_score": True, "veto": "none"},
    ),
    (
        "ad_no_rank",
        {"with_ad": True, "with_rank": False, "with_score": True, "veto": "none"},
    ),
    (
        "ad_rank_curves",
        {"with_ad": True, "with_rank": True, "with_score": True, "veto": "none"},
    ),
    (
        "ad_partial_veto",
        {"with_ad": True, "with_rank": True, "with_score": True, "veto": "partial"},
    ),
    (
        "ad_all_ood",
        {"with_ad": True, "with_rank": True, "with_score": True, "veto": "all"},
    ),
    (
        "no_score",
        {"with_ad": True, "with_rank": True, "with_score": False, "veto": "none"},
    ),
    (
        "no_rank_no_score",
        {"with_ad": True, "with_rank": False, "with_score": False, "veto": "none"},
    ),
]


@pytest.mark.parametrize("branch_name,opts", BRANCHES, ids=[b[0] for b in BRANCHES])
@pytest.mark.parametrize("D", [1, 2, 3, 5])
@pytest.mark.parametrize("B", [1, 7, 1000])
@pytest.mark.parametrize("curve_mode", ["all", "some_none", "absent"])
@pytest.mark.parametrize("no_skill", [False, True])
def test_combine_matches_legacy_bit_for_bit(
    branch_name, opts, D, B, curve_mode, no_skill
):
    rng = np.random.default_rng(
        abs(hash((branch_name, D, B, curve_mode, no_skill))) % 2**32
    )
    if curve_mode == "absent":
        curves = None
    elif curve_mode == "all":
        curves = _curves(rng, D)
    else:
        curves = _curves(rng, D, how_many_none=1)

    Y, R, S, A, spec_kwargs = _case(rng, B, D, curves=curves, no_skill=no_skill, **opts)
    expected, W_expected = _legacy_outputs(Y, R, S, A, spec_kwargs)

    result = combine(Y, R, S, A, spec=_spec(D, spec_kwargs), outputs=OUTPUT_NAMES)

    assert np.array_equal(result.weights, W_expected), "weight matrix diverged"
    for name in OUTPUT_NAMES:
        assert np.array_equal(result.values[name], expected[name]), (
            f"{name} diverged from the 3.4.4 implementation"
        )


def test_weights_sum_to_one():
    rng = np.random.default_rng(0)
    Y, R, S, A, kw = _case(
        rng,
        500,
        4,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=_curves(rng, 4),
        veto="partial",
        no_skill=False,
    )
    result = combine(Y, R, S, A, spec=_spec(4, kw), outputs=("proba",))
    assert np.allclose(result.weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_proba_columns_sum_to_one():
    rng = np.random.default_rng(1)
    Y, R, S, A, kw = _case(
        rng,
        200,
        3,
        with_ad=False,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    proba = combine(Y, R, S, A, spec=_spec(3, kw), outputs=("proba",)).values["proba"]
    assert np.allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_requesting_one_output_matches_requesting_all():
    """Output selection is an optimisation, not a behaviour change."""
    rng = np.random.default_rng(2)
    Y, R, S, A, kw = _case(
        rng,
        64,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=_curves(rng, 3),
        veto="partial",
        no_skill=False,
    )
    spec = _spec(3, kw)
    everything = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES).values
    for name in OUTPUT_NAMES:
        alone = combine(Y, R, S, A, spec=spec, outputs=(name,)).values[name]
        assert np.array_equal(alone, everything[name]), name


def test_float32_inputs_give_float64_results():
    """Channels may be accumulated in float32; the arithmetic still runs in float64."""
    rng = np.random.default_rng(3)
    Y, R, S, A, kw = _case(
        rng,
        50,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    spec = _spec(3, kw)
    wide = combine(Y, R, S, A, spec=spec, outputs=("proba",)).values["proba"]
    narrow = combine(
        Y.astype(np.float32),
        R.astype(np.float32),
        S.astype(np.float32),
        A.astype(np.float32),
        spec=spec,
        outputs=("proba",),
    ).values["proba"]
    assert wide.dtype == narrow.dtype == np.float64
    assert np.allclose(wide, narrow, rtol=0, atol=1e-6)


def test_rows_are_independent():
    """Splitting the input and combining the halves must match one call.

    This is the property the streaming prediction path depends on.
    """
    rng = np.random.default_rng(4)
    Y, R, S, A, kw = _case(
        rng,
        128,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=_curves(rng, 3),
        veto="partial",
        no_skill=False,
    )
    spec = _spec(3, kw)
    whole = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES).values
    halves = [
        combine(Y[s], R[s], S[s], A[s], spec=spec, outputs=OUTPUT_NAMES).values
        for s in (slice(0, 51), slice(51, 128))
    ]
    for name in OUTPUT_NAMES:
        stitched = np.concatenate([h[name] for h in halves], axis=0)
        assert np.array_equal(stitched, whole[name]), name


def test_diagnostics_only_when_ad_present():
    rng = np.random.default_rng(5)
    Y, R, S, A, kw = _case(
        rng,
        20,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    spec = _spec(3, kw)
    assert combine(Y, R, S, A, spec=spec, outputs=("proba",)).diagnostics is not None
    assert combine(Y, R, S, None, spec=spec, outputs=("proba",)).diagnostics is None

    rows = combine(Y, R, S, A, spec=spec, outputs=("proba",)).diagnostics
    assert [r["name"] for r in rows] == ["d0", "d1", "d2"]
    for key in ("ad_mean", "weight_mean", "vetoed", "pred_mean"):
        assert key in rows[0]


def test_unknown_output_rejected():
    rng = np.random.default_rng(6)
    Y, R, S, A, kw = _case(
        rng,
        8,
        2,
        with_ad=False,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    with pytest.raises(ValueError, match="Unknown output"):
        combine(Y, R, S, A, spec=_spec(2, kw), outputs=("probability",))


def test_spec_width_mismatch_rejected():
    rng = np.random.default_rng(7)
    Y, R, S, A, kw = _case(
        rng,
        8,
        3,
        with_ad=False,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    with pytest.raises(ValueError, match="spec covers"):
        combine(
            Y,
            R,
            S,
            A,
            spec=_spec(
                2,
                {
                    **kw,
                    "oof_aucs": kw["oof_aucs"][:2],
                    "proxy_aucs": kw["proxy_aucs"][:2],
                    "ad_hard_cutoffs": None,
                },
            ),
        )


# ---------------------------------------------------------------------------
# EnsembleSpec.from_metadata
# ---------------------------------------------------------------------------


def test_from_metadata_prefers_quality_auc():
    """quality_aucs is the deployed convention; oof_aucs is the fallback."""
    meta = {
        "quality_aucs": {"a": 0.8, "b": 0.7},
        "oof_aucs": {"a": 0.9, "b": 0.95, "c": 0.6},
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b", "c"])
    assert active == ["a", "b", "c"]
    assert spec.oof_aucs == (0.8, 0.7, 0.6)


def test_from_metadata_honours_active_descriptors():
    meta = {
        "oof_aucs": {"a": 0.9, "b": 0.8, "c": 0.7},
        "active_descriptors": {"a": True, "b": False, "c": True},
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b", "c"])
    assert active == ["a", "c"]
    assert spec.descriptor_names == ("a", "c")
    assert spec.oof_aucs == (0.9, 0.7)


def test_from_metadata_all_inactive_falls_back_to_all():
    """A mask that excludes everything would leave nothing to predict with."""
    meta = {"active_descriptors": {"a": False, "b": False}}
    _, active = EnsembleSpec.from_metadata(meta, ["a", "b"])
    assert active == ["a", "b"]


def test_from_metadata_empty_metadata_gives_working_defaults():
    spec, active = EnsembleSpec.from_metadata({}, ["a", "b"])
    assert active == ["a", "b"]
    assert spec.oof_aucs == (1.0, 1.0)
    assert spec.proxy_aucs == (None, None)
    assert spec.rank_error_curves is None
    assert spec.ad_hard_cutoffs is None
    assert spec.population_prior == 0.5
    assert spec.decision_cutoff == 0.5


def test_from_metadata_slices_curves_and_cutoffs_to_active():
    meta = {
        "active_descriptors": {"a": True, "b": False, "c": True},
        "ad_hard_cutoffs": {"a": 0.1, "b": 0.2, "c": 0.3},
        "rank_error_curves": {
            "a": [[0.0, 1.0], [0.5, 0.1]],
            "b": [[0.0, 1.0], [0.4, 0.2]],
            "c": [[0.0, 1.0], [0.3, 0.3]],
        },
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b", "c"])
    assert active == ["a", "c"]
    assert spec.ad_hard_cutoffs == (0.1, 0.3)
    assert len(spec.rank_error_curves) == 2
    assert np.array_equal(spec.rank_error_curves[1][1], np.array([0.3, 0.3]))
