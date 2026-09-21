"""Regenerate the frozen ``combine()`` outputs in ``tests/data/golden/``.

Run this only when the ensemble arithmetic is *deliberately* changed:

    python dev/tools/regenerate_combine_golden.py

The resulting diff is the change's blast radius, and reviewing it is the point. If the
numbers move and you did not mean them to, that is the regression the golden file exists to
catch -- do not regenerate to make a test pass.

The expected values are pinned to the behaviour this package ships. The suite this replaced
compared against an inlined copy of 3.4.4, which 3.5.0 deliberately departs from (`binary` is
now a pooled label; `proba`, `logit` and `lift` combine in logit space), so that reference
could only still hold on the branches where nothing had changed.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tests"))

from _helpers.combine_cases import SCENARIO_IDS, build_case  # noqa: E402

from lazyqsar.ensemble import OUTPUT_NAMES, combine  # noqa: E402

OUT = os.path.join(REPO, "tests", "data", "golden", "combine.npz")


def main():
    payload = {}
    for case_id in SCENARIO_IDS:
        Y, R, S, A, spec = build_case(case_id)
        result = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES)
        payload[f"{case_id}/weights"] = result.weights
        for name in OUTPUT_NAMES:
            payload[f"{case_id}/{name}"] = result.values[name]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **payload)
    size = os.path.getsize(OUT)
    print(f"wrote {OUT} ({len(SCENARIO_IDS)} cases, {size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
