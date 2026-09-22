"""Fetch the reference bundle from public S3, via ``eosvc``.

``eosvc`` is how Ersilia moves data in and out of S3, and the same tool that publishes this
bundle fetches it, so there is one path and one set of conventions rather than a publisher
and an unrelated reader that can disagree.

It resolves its S3 prefix from the repository it is invoked inside: it walks up for an
``access.json`` and takes the repo name from ``EVC_REPO_NAME`` or the directory. An
installed package is not inside a checkout, so this stages a minimal one -- an
``access.json`` saying ``data`` is public -- and runs ``eosvc`` there, then moves the files
into the cache. That keeps eosvc an implementation detail of *fetching*; the cache layout
owes it nothing.

Credentials are not needed. ``eosvc`` falls back to anonymous access, which is all a public
bucket read requires, so a user who has never configured AWS can still fetch the bundle.

Fit-time only, like everything else under ``lazyqsar.reference``. A deployed model carries
its knots in ``metadata.json`` and never needs any of this, which is why ``eosvc`` -- and
the boto3 stack behind it -- lives in the ``fit`` extra and is banned from the inference
path by ``tests/packaging/test_import_purity.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..utils.logging import logger
from .identity import (
    REFERENCE_ID,
    default_n,
    descriptor_filename,
    reference_dir,
    smiles_filename,
)

# The repo whose S3 prefix the bundle lives under. eosvc maps repo name -> prefix, so this
# is what puts the files at s3://eosvc-public/lazy-qsar/data/reference/<id>/.
EOSVC_REPO = "lazy-qsar"

# Where the bundle sits inside that repo. `data/` is the prefix eosvc treats as public.
EOSVC_ROOT = "data/reference"

MANIFEST_FILENAME = "manifest.json"


class ReferenceDownloadError(RuntimeError):
    """The bundle could not be fetched."""


def offline() -> bool:
    """Whether fetching is disabled.

    The suite sets this so a test can never reach for 267 MB; a user can set it on an
    air-gapped node to get a clear refusal instead of a hang.
    """
    return bool(os.environ.get("LAZYQSAR_REFERENCE_OFFLINE"))


def eosvc_available() -> bool:
    return shutil.which("eosvc") is not None


def _staging_repo() -> Path:
    """A minimal repo for eosvc to resolve its prefix from."""
    root = Path(tempfile.mkdtemp(prefix="lazyqsar-eosvc-"))
    (root / "access.json").write_text(json.dumps({"data": "public"}))
    return root


def download(filenames, n: int | None = None, force: bool = False) -> list[Path]:
    """Fetch *filenames* from the published bundle into the local cache.

    Returns the local paths. Files already present are skipped unless *force*.
    """
    n = n or default_n()
    target = reference_dir()
    target.mkdir(parents=True, exist_ok=True)

    wanted = [f for f in filenames if force or not (target / f).is_file()]
    if not wanted:
        return [target / f for f in filenames]

    if offline():
        raise ReferenceDownloadError(
            f"LAZYQSAR_REFERENCE_OFFLINE is set, so {', '.join(wanted)} cannot be "
            f"fetched. Unset it, or place the bundle at {target} yourself."
        )
    if not eosvc_available():
        raise ReferenceDownloadError(
            "The reference library is fetched with `eosvc`, which is not installed.\n"
            "  pip install 'lazyqsar[fit]'   (or: pip install eosvc)\n"
            f"Alternatively, place the bundle at {target} and point "
            "LAZYQSAR_REFERENCE_DIR at it."
        )

    root = _staging_repo()
    env = dict(os.environ, EVC_REPO_NAME=EOSVC_REPO)
    landed = []
    try:
        for name in wanted:
            remote = f"{EOSVC_ROOT}/{REFERENCE_ID}/{name}"
            logger.info(f"Fetching {name} from {EOSVC_REPO}:{remote}")
            result = subprocess.run(
                ["eosvc", "download", "--path", remote],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            staged = root / remote
            if result.returncode != 0 or not staged.is_file():
                raise ReferenceDownloadError(
                    f"eosvc could not fetch {remote}.\n"
                    f"{(result.stderr or result.stdout).strip()[:500]}"
                )
            # Move rather than copy: same filesystem in the common case, and it means a
            # half-written file is never visible at the cache path.
            shutil.move(str(staged), str(target / name))
            landed.append(target / name)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # Verified after the move, and only once the manifest itself is cached -- fetching it
    # is what the first pass through here usually does.
    if MANIFEST_FILENAME not in wanted:
        from .manifest import load, verify_file

        checked = load(fetch_if_missing=False)
        if checked:
            for path in landed:
                verify_file(path, checked)

    logger.success(f"Fetched {len(landed)} file(s) into {target}")
    return [target / f for f in filenames]


def ensure_descriptor(
    descriptor: str, n: int | None = None, force: bool = False
) -> Path:
    n = n or default_n()
    return download([descriptor_filename(descriptor, n)], n=n, force=force)[0]


def ensure_smiles(n: int | None = None, force: bool = False) -> Path:
    n = n or default_n()
    return download([smiles_filename(n)], n=n, force=force)[0]


def ensure_manifest(force: bool = False) -> Path | None:
    """Fetch the manifest, or ``None`` if the published bundle has none yet."""
    try:
        return download([MANIFEST_FILENAME], force=force)[0]
    except ReferenceDownloadError as exc:
        logger.debug(f"No manifest available: {exc}")
        return None
