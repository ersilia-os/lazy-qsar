"""Guard the shared checkpoint against mutation.

The session checkpoint is the suite's main runtime saving, and the classic failure mode of a
shared fixture is one test quietly writing into it and poisoning every later test. Rather
than trusting each test not to, snapshot the tree and assert it is unchanged at teardown.
Tests that legitimately need to write should build their own checkpoint in ``tmp_path``.
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


# Every shared checkpoint fixture, by the attribute its value exposes its root under.
# `pruned_checkpoint` joined this list when it went module-scoped: a shared fixture only
# stays safe while the guard knows about it.
_SHARED_ROOTS = {
    "checkpoint": lambda v: v.root,
    "pruned_checkpoint": lambda v: v["root"],
}


@pytest.fixture(autouse=True)
def _checkpoint_is_readonly(request):
    shared = [n for n in _SHARED_ROOTS if n in request.fixturenames]
    if not shared:
        yield
        return
    roots = [_SHARED_ROOTS[n](request.getfixturevalue(n)) for n in shared]

    def snapshot():
        out = {}
        for root in roots:
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
            f"test mutated a shared checkpoint under {roots}: {changed[:5]}"
        )
