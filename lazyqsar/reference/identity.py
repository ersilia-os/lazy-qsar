"""Names, paths and environment overrides for the reference library.

Standard library only, on purpose -- the same discipline ``lazyqsar/registry.py`` keeps.
This module is read during ``lazyqsar setup``, before the descriptor stack is necessarily
installed, and it must never be the reason an import fails.

Nothing on the inference path imports anything under ``lazyqsar.reference``. A deployed
model carries its reference as knots in ``metadata.json``; the matrices are a fit-time
input and a container that only scores molecules has no use for them.
"""

from __future__ import annotations

import os
from pathlib import Path

# Bumped whenever the molecule set or any published matrix changes. Published bundles are
# immutable: clients cache by filename and cannot notice an object edited in place.
REFERENCE_ID = "lazyqsar_reference_v1"
DEFAULT_N = 50_000

PUBLIC_BASE_URL = "https://eosvc-public.s3.amazonaws.com/lazy-qsar/reference/"


def lazyqsar_home() -> Path:
    """Root for every cached artifact, checkpoints included.

    ``LAZYQSAR_HOME`` exists because ``lazyqsar setup --target-dir`` previously moved only
    the download, while the descriptors kept reading ``~/.lazyqsar`` -- so a redirected
    setup silently re-downloaded everything on first use.
    """
    return Path(os.environ.get("LAZYQSAR_HOME", Path.home() / ".lazyqsar")).expanduser()


def reference_dir() -> Path:
    """Where the bundle lives locally.

    ``LAZYQSAR_REFERENCE_DIR`` points straight at a directory -- a maintainer's staging
    area, or a shared read-only copy on a cluster -- and skips the cache entirely.
    """
    override = os.environ.get("LAZYQSAR_REFERENCE_DIR")
    if override:
        return Path(override).expanduser()
    return lazyqsar_home() / "reference" / REFERENCE_ID


def descriptor_filename(descriptor: str, n: int = DEFAULT_N) -> str:
    """Published name of one descriptor matrix.

    The tier is in the filename rather than a parent directory so tiers can nest: a later
    ``*_n100000.h5`` sits beside this one, and a cached file is never ambiguous about how
    many rows it holds.
    """
    return f"{descriptor}_n{n}.h5"


def smiles_filename(n: int = DEFAULT_N) -> str:
    return f"reference_smiles_n{n}.csv"


def descriptor_url(descriptor: str, n: int = DEFAULT_N) -> str:
    return f"{PUBLIC_BASE_URL}{REFERENCE_ID}/{descriptor_filename(descriptor, n)}"


def smiles_url(n: int = DEFAULT_N) -> str:
    return f"{PUBLIC_BASE_URL}{REFERENCE_ID}/{smiles_filename(n)}"
