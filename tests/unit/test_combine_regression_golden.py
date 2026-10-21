"""``combine()`` on regression specs must keep producing the numbers this package shipped.

The regression twin of ``test_combine_golden``, frozen in its own file
(``tests/data/golden/combine_regression.npz``) so that a deliberate change to one task's
arithmetic never rewrites the other's. Scenarios are listed in
``tests/_helpers/combine_cases.py`` (``REGRESSION_SCENARIOS``): AD on/off, rank with
complete, incomplete or no rank-error curves, every veto state, and known, partly unknown,
unknown and negative skill.

Same tolerance policy as the classification file and for the same reason: ``weights`` ends
in a row-sum normalisation whose reduction order numpy picks per SIMD width, and ``value``
is a reduction over it. Neither touches ``log``/``exp``, so the divergence is smaller still.

To change the expected values deliberately, run ``dev/tools/regenerate_combine_golden.py``
and review the diff. Numpy only.
"""

import os

import numpy as np
import pytest

from _helpers.combine_cases import REGRESSION_SCENARIO_IDS, build_regression_case

from lazyqsar.ensemble import combine

# The frozen arithmetic is the point estimate; interval bounds need an interval model,
# which these hand-built specs do not carry, and have their own tests.
REGRESSION_OUTPUT_NAMES = ("value",)

GOLDEN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "golden",
    "combine_regression.npz",
)
_ATOL = 1e-12


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize("case_id", REGRESSION_SCENARIO_IDS)
def test_regression_combine_matches_the_frozen_output(case_id, golden):
    Y, R, A, spec = build_regression_case(case_id)
    result = combine(Y, R, None, A, spec=spec, outputs=REGRESSION_OUTPUT_NAMES)
    checks = [("weights", result.weights)] + [
        (name, result.values[name]) for name in REGRESSION_OUTPUT_NAMES
    ]
    problems = [
        f"{name}: max {np.max(np.abs(got - golden[f'{case_id}/{name}'])):.3e}"
        for name, got in checks
        if not np.allclose(got, golden[f"{case_id}/{name}"], rtol=0, atol=_ATOL)
    ]
    assert not problems, f"[{case_id}] diverged -- " + "; ".join(problems)


def test_golden_file_covers_every_regression_scenario(golden):
    expected = {
        f"{case_id}/{key}"
        for case_id in REGRESSION_SCENARIO_IDS
        for key in ("weights",) + tuple(REGRESSION_OUTPUT_NAMES)
    }
    assert set(golden) == expected
