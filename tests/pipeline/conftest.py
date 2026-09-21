"""Guard the shared checkpoint against mutation.

The session checkpoint is the suite's main runtime saving, and the classic failure mode of a
shared fixture is one test quietly writing into it and poisoning every later test. Rather
than trusting each test not to, snapshot the tree and assert it is unchanged at teardown.
Tests that legitimately need to write should request ``mutable_checkpoint`` instead.
"""

from _helpers.tiers import skip_directory_if_tier_unavailable

collect_ignore_glob, _missing_modules = skip_directory_if_tier_unavailable("fit")

if _missing_modules:
    print(
        f"\nSkipping tests/pipeline: needs the [fit] tier; missing "
        f"{', '.join(_missing_modules)}"
    )


import os

import pytest


@pytest.fixture(autouse=True)
def _checkpoint_is_readonly(request):
    if "checkpoint" not in request.fixturenames:
        yield
        return
    root = request.getfixturevalue("checkpoint").root

    def snapshot():
        out = {}
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                st = os.stat(path)
                out[path] = (st.st_size, st.st_mtime_ns)
        return out

    before = snapshot()
    yield
    after = snapshot()
    if after != before:
        changed = sorted(set(after) ^ set(before)) or [
            p for p in before if after.get(p) != before[p]
        ]
        raise AssertionError(
            f"test mutated the shared checkpoint under {root}: {changed[:5]}"
        )
