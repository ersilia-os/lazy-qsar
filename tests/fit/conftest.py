"""Tier gate for ``tests/fit``. See ``skip_directory_if_tier_unavailable`` in the root conftest."""

from _helpers.tiers import skip_directory_if_tier_unavailable

collect_ignore_glob, _missing_modules = skip_directory_if_tier_unavailable("fit")

if _missing_modules:
    print(
        f"\nSkipping tests/fit: needs the [fit] tier; missing "
        f"{', '.join(_missing_modules)}"
    )
