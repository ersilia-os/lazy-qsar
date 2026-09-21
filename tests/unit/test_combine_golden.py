"""``combine()`` must keep producing the numbers this package shipped.

The expected values in ``tests/data/golden/combine.npz`` are frozen output from the current
implementation, one case per distinct branch of the weighting logic (see
``tests/_helpers/combine_cases.py`` for the scenario list and what each one selects).

``weights``, ``rank``, ``score`` and ``binary`` are asserted bit-identical: they are
multiplies, divides and sums, which IEEE-754 pins exactly for a fixed shape and order, so a
tolerance there would hide the reordering and accumulation-order mistakes this file exists
to catch.

``proba``, ``logit`` and ``lift`` go through ``np.log``/``np.exp``, and those are the one
place numpy does not promise a bit-identical answer across machines: on this architecture
they dispatch to the platform libm, while an AVX-512 x86 runner uses numpy's own SIMD
kernel. The frozen file has to be generated somewhere, so demanding bit-identity of those
three would make the suite fail on a different CPU than the one that wrote it -- a cryptic
failure that says nothing about the ensemble arithmetic. They are held to ``_TRANSCENDENTAL_ATOL``
instead, which is ~1e10 times tighter than the float32 channel rounding the pipeline
already accepts, so a real change still fails here.

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


# Outputs derived from np.log / np.exp, which are not bit-reproducible across CPU
# architectures. Everything else is exact arithmetic and is held to bit-identity.
_TRANSCENDENTAL = frozenset({"proba", "logit", "lift"})
_TRANSCENDENTAL_ATOL = 1e-12


@pytest.mark.parametrize("case_id", SCENARIO_IDS)
def test_combine_matches_the_frozen_output(case_id, golden):
    Y, R, S, A, spec = build_case(case_id)
    result = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES)

    assert np.array_equal(result.weights, golden[f"{case_id}/weights"]), (
        f"[{case_id}] weight matrix diverged from the frozen output"
    )
    for name in OUTPUT_NAMES:
        got, want = result.values[name], golden[f"{case_id}/{name}"]
        if name in _TRANSCENDENTAL:
            assert np.allclose(got, want, rtol=0, atol=_TRANSCENDENTAL_ATOL), (
                f"[{case_id}] {name} diverged from the frozen output by more than "
                f"{_TRANSCENDENTAL_ATOL:g} (max {np.max(np.abs(got - want)):.3e})"
            )
        else:
            assert np.array_equal(got, want), (
                f"[{case_id}] {name} diverged from the frozen output"
            )


def test_golden_file_covers_every_scenario(golden):
    """A scenario added without regenerating the file would otherwise fail obscurely."""
    expected = {
        f"{case_id}/{key}"
        for case_id in SCENARIO_IDS
        for key in ("weights",) + tuple(OUTPUT_NAMES)
    }
    assert set(golden) == expected
