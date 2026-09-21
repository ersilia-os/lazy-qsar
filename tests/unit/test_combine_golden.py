"""``combine()`` must keep producing the numbers this package shipped.

The expected values in ``tests/data/golden/combine.npz`` are frozen output from the current
implementation, one case per distinct branch of the weighting logic (see
``tests/_helpers/combine_cases.py`` for the scenario list and what each one selects).

Only ``binary`` is asserted bit-identical. Everything else is held to ``_PORTABLE_ATOL``,
because none of it is bit-reproducible across CPU architectures.

``proba``, ``logit`` and ``lift`` go through ``np.log``/``np.exp``, which is the obvious
case: on arm64 they dispatch to the platform libm while an AVX-512 x86 runner uses numpy's
own SIMD kernel. The frozen file has to be generated somewhere, so demanding bit-identity
would make the suite fail on a different CPU than the one that wrote it.

``weights``, ``rank`` and ``score`` were held exact until the file was first checked on a
second architecture, on the reasoning that they are "multiplies, divides and sums, which
IEEE-754 pins exactly". Two holes in that. ``build_weight_matrix`` ends in
``W /= W.sum(axis=1, keepdims=True)``, and a sum is a reduction whose blocking numpy picks
per SIMD width; and since ``rank`` became the pooled reference evaluated at the pooled
probability it is computed *from* ``p1``, so it inherits ``np.exp`` -- as ``score`` does now
that it is read off the same probability through a stored map. 17 of the 31 scenarios failed
on x86 against a file frozen on arm64, printing identically to eight significant figures.

How far apart the two architectures can actually be, measured rather than assumed: the
arithmetic was reimplemented, checked to reproduce ``combine()`` bit-for-bit on all 31
scenarios, then re-run with every ``axis=1`` reduction replaced by ``math.fsum`` (exactly
rounded, so it brackets any SIMD width a numpy build can choose) and every ``np.log``/
``np.exp`` result nudged one ULP away from zero (the worst libm disagreement). Worst case
over all scenarios, both perturbations at once:

    weights 2.220e-16   proba 2.220e-16   logit 8.882e-16
    rank    7.772e-16   score 3.331e-16   lift  8.882e-16

So ``_PORTABLE_ATOL`` carries about three orders of magnitude of headroom over the real
divergence, while still being far tighter than the float32 channel rounding this pipeline
already accepts -- between ~3e4 and ~1e6 times tighter, reading the per-output figures in
``combine()``'s own docstring. A real change in the arithmetic still fails here; a
perturbation of 1e-11 to the weight blend fails 14 of the 31 scenarios. What it no longer
catches is a last-bit reordering, which is the price of a suite that runs on more than the
machine that wrote its fixtures.

``binary`` stays exact, and that is a measured claim rather than a categorical one: under
the same worst-case perturbation no ``p1`` in these fixtures sits close enough to the cutoff
to flip a label. It is the one output where a tolerance would hide a real bug, since a
threshold either moved or it did not.

To change the expected values deliberately, run ``dev/tools/regenerate_combine_golden.py``
and review the resulting diff. Numpy only, so this runs on a base install.
"""

import os

import numpy as np
import pytest

from _helpers.combine_cases import SCENARIO_IDS, build_case

from lazyqsar.ensemble import OUTPUT_NAMES, combine

GOLDEN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "golden",
    "combine.npz",
)


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN) as data:
        return {k: data[k] for k in data.files}


# Everything float-valued: none of it is bit-reproducible across CPU architectures, either
# through np.log/np.exp directly or through a reduction whose blocking is SIMD-dependent.
# `binary` is integer labels and is excluded deliberately -- see the module docstring.
_PORTABLE = frozenset(OUTPUT_NAMES) - {"binary"}
_PORTABLE_ATOL = 1e-12


@pytest.mark.parametrize("case_id", SCENARIO_IDS)
def test_combine_matches_the_frozen_output(case_id, golden):
    Y, R, S, A, spec = build_case(case_id)
    result = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES)

    # Every mismatch in the case is collected and reported together. Asserting per output
    # aborts at the first one, which is how a cross-architecture failure came to be
    # reported as "the weight matrix diverged" with no word on the four other outputs that
    # had also moved -- the rest had to be inferred from a second run.
    problems = []
    checks = [("weights", result.weights, golden[f"{case_id}/weights"])] + [
        (name, result.values[name], golden[f"{case_id}/{name}"])
        for name in OUTPUT_NAMES
    ]
    for name, got, want in checks:
        exact = name not in _PORTABLE and name != "weights"
        if exact:
            if not np.array_equal(got, want):
                problems.append(f"{name}: not bit-identical")
        elif not np.allclose(got, want, rtol=0, atol=_PORTABLE_ATOL):
            problems.append(
                f"{name}: max {np.max(np.abs(got - want)):.3e} > {_PORTABLE_ATOL:g}"
            )
    assert not problems, f"[{case_id}] diverged from the frozen output -- " + "; ".join(
        problems
    )


def test_golden_file_covers_every_scenario(golden):
    """A scenario added without regenerating the file would otherwise fail obscurely."""
    expected = {
        f"{case_id}/{key}"
        for case_id in SCENARIO_IDS
        for key in ("weights",) + tuple(OUTPUT_NAMES)
    }
    assert set(golden) == expected
