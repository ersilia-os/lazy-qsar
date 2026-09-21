"""Input generation for the ``combine()`` tests, shared with the golden regenerator.

``combine()`` takes four optional arrays and a spec, and the weighting logic branches on
which of them are present, on whether the rank-error curves are usable, and on whether the
applicability domain vetoes a descriptor. The builders below turn a named scenario into a
concrete set of inputs, deterministically, so ``dev/tools/regenerate_combine_golden.py`` and
``tests/unit/test_combine_golden.py`` construct byte-identical cases from the same seed.
"""

import numpy as np

from lazyqsar.ensemble import EnsembleSpec


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

_BRANCH_OPTS = dict(BRANCHES)

# The golden scenarios.
#
# The suite this replaces took the cross-product of 7 branches x 4 values of D x 3 of B x 3
# curve modes x 2 no-skill settings, which is 504 cases over roughly 20 genuinely distinct
# code paths -- the D and B axes mostly re-tested numpy broadcasting. What follows is one
# case per distinct path, plus explicit degenerate-shape and large-batch guards.
SCENARIOS = []

# Every branch, against each way the rank-error curves can arrive. `curve_mode` only changes
# behaviour on branches that use rank, but running it across all seven costs nothing and pins
# that the others genuinely ignore it.
for _name, _opts in BRANCHES:
    for _curve_mode in ("all", "some_none", "absent"):
        SCENARIOS.append(
            (
                f"{_name}__curves_{_curve_mode}",
                {**_opts, "D": 3, "B": 7, "curve_mode": _curve_mode, "no_skill": False},
            )
        )

# base.sum() == 0 falls back to uniform ones. Worth pinning with and without an AD, since the
# two reach the fallback through different arms of build_weight_matrix.
SCENARIOS += [
    (
        "no_skill__no_ad",
        {
            **_BRANCH_OPTS["no_ad_uniform"],
            "D": 3,
            "B": 7,
            "curve_mode": "all",
            "no_skill": True,
        },
    ),
    (
        "no_skill__with_ad",
        {
            **_BRANCH_OPTS["ad_rank_curves"],
            "D": 3,
            "B": 7,
            "curve_mode": "all",
            "no_skill": True,
        },
    ),
]

# Degenerate shapes: a single descriptor and a single row each hit broadcasting edges that a
# D=3, B=7 case cannot.
SCENARIOS += [
    (
        "degenerate__one_descriptor",
        {
            **_BRANCH_OPTS["ad_rank_curves"],
            "D": 1,
            "B": 7,
            "curve_mode": "all",
            "no_skill": False,
        },
    ),
    (
        "degenerate__one_row",
        {
            **_BRANCH_OPTS["ad_rank_curves"],
            "D": 3,
            "B": 1,
            "curve_mode": "all",
            "no_skill": False,
        },
    ),
    (
        "degenerate__one_row_one_descriptor",
        {
            **_BRANCH_OPTS["no_ad_uniform"],
            "D": 1,
            "B": 1,
            "curve_mode": "absent",
            "no_skill": False,
        },
    ),
    (
        "wide__five_descriptors",
        {
            **_BRANCH_OPTS["ad_partial_veto"],
            "D": 5,
            "B": 7,
            "curve_mode": "all",
            "no_skill": False,
        },
    ),
    (
        "large_batch",
        {
            **_BRANCH_OPTS["ad_rank_curves"],
            "D": 3,
            "B": 1000,
            "curve_mode": "all",
            "no_skill": False,
        },
    ),
]

SCENARIO_IDS = [case_id for case_id, _ in SCENARIOS]
_SCENARIOS_BY_ID = dict(SCENARIOS)


def build_case(case_id):
    """Rebuild a named scenario's ``(Y, R, S, A, spec)`` deterministically.

    The seed is derived from the case id alone, so the generator and the test produce
    identical inputs without the inputs themselves having to round-trip through the file.
    """
    opts = dict(_SCENARIOS_BY_ID[case_id])
    D = opts.pop("D")
    B = opts.pop("B")
    curve_mode = opts.pop("curve_mode")
    no_skill = opts.pop("no_skill")

    rng = np.random.default_rng(
        int.from_bytes(case_id.encode(), "little", signed=False) % 2**32
    )
    if curve_mode == "absent":
        curves = None
    elif curve_mode == "all":
        curves = _curves(rng, D)
    else:
        curves = _curves(rng, D, how_many_none=1)

    Y, R, S, A, spec_kwargs = _case(rng, B, D, curves=curves, no_skill=no_skill, **opts)
    return Y, R, S, A, _spec(D, spec_kwargs)
