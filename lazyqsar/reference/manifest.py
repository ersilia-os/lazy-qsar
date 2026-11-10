"""Read the bundle's manifest, and check this install would compute the same values.

Three questions, in increasing cost, and the order matters because the last one is the only
expensive one.

**Did the file arrive intact?** :func:`verify_file`, a sha256 against the manifest.

**Is this the bundle we expect?** Reference id, tier, dimensions, and the sha256 of the
molecule list every matrix was built from.

**Would this install compute the same descriptor values?** :func:`check_environment`. This
is the one that cannot be answered by version strings. The RDKit descriptor *names* are
identical between 2025.09.1 and 2026.03.1, so a name or version comparison passes across a
two-release gap while the values may not; and the CheMeleon ``.pt`` and the CLAMP and CDDD
``.onnx`` files can be re-uploaded in place with nothing recording it. So the check is:
pin the checkpoints by hash, and recompute a handful of molecules and compare the numbers.

A drifted reference is the quiet kind of wrong. Percentiles computed against it stay
monotone, stay in [0, 1], and are simply calibrated against a population this install cannot
reproduce -- nothing downstream can tell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.logging import logger
from .identity import reference_dir

MANIFEST_FILENAME = "manifest.json"

# Morgan and RDKit are deterministic given the toolkit: any difference is a definition
# change, and the matrices no longer describe what this install computes.
EXACT = {"morgan"}
RDKIT_RTOL, RDKIT_ATOL = 1e-4, 1e-6

# The neural three run through ONNX Runtime or torch, where the BLAS backend, the CPU
# architecture and the thread count move results slightly. That is not drift, and failing a
# fit over it would be wrong. Measured on the published canary rows (1 - min row cosine):
#   float16 storage, same code (cddd, chemeleon)     ~3e-8
#   Hub Docker image (Linux) vs macOS, cddd          ~1e-11 (max abs 2.5e-6)
#   CLAMP fingerprint-folding bug (eos3l5f#3)         2.9e-3
# The CLAMP bug is a code change with an unchanged checkpoint, and it is the smallest real
# drift seen so far. The old thresholds (0.9999 / 0.99) only warned about it. These sit
# about 300x above the noise and 300x below that bug.
COSINE_PASS, COSINE_WARN = 1 - 1e-6, 1 - 1e-5


class ReferenceDriftError(RuntimeError):
    """The installed descriptor code does not reproduce the published matrices."""


def manifest_path(n: int | None = None) -> Path:
    return reference_dir() / MANIFEST_FILENAME


def manifest_sha256() -> str | None:
    """The sha256 of the cached manifest, or ``None`` if there is none to hash.

    The manifest pins every file in the bundle by hash, so this one value names the exact
    reference a model was fitted against. The reference id alone cannot: it is a promise
    that the bundle never changes, and nothing enforces it.
    """
    from ..utils.fetch import sha256_file

    path = manifest_path()
    return sha256_file(path) if path.is_file() else None


def load(n: int | None = None, fetch_if_missing: bool = True) -> dict | None:
    """The manifest, or ``None`` if the published bundle carries none.

    ``None`` is a real answer, not a failure: a bundle published before manifests existed
    still works, it just cannot be verified. The caller decides how much that matters.
    """
    path = manifest_path(n)
    if not path.is_file() and fetch_if_missing:
        from .download import ensure_manifest

        if ensure_manifest() is None:
            return None
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - corrupt manifest
        logger.warning(f"Could not parse {path}: {exc}")
        return None


def verify_file(path, manifest: dict | None) -> None:
    """Check one downloaded file against the manifest. Silent when it cannot be checked."""
    from ..utils.fetch import sha256_file

    if not manifest:
        return
    entry = (manifest.get("files") or {}).get(Path(path).name)
    if not entry or not entry.get("sha256"):
        return
    got = sha256_file(path)
    if got != entry["sha256"]:
        raise ReferenceDriftError(
            f"{Path(path).name} does not match the manifest.\n"
            f"  expected sha256 {entry['sha256']}\n"
            f"  got            {got}\n"
            "The published bundle is immutable, so this is a corrupt download. The file "
            "has been removed from the cache; run again to fetch it afresh."
        )


# Which checkpoint a descriptor's values come from. Morgan and RDKit have none -- they are a
# pure function of the toolkit. The hash is reported alongside the drift result, but it
# never replaces the recompute: the code around a checkpoint can change its values too.
DESCRIPTOR_CHECKPOINT = {
    "chemeleon": "chemeleon_mp.pt",
    "clamp": "clamp_encoder.onnx",
    "cddd": "cddd_encoder.onnx",
}


def _checkpoint_unchanged(manifest: dict, descriptor: str) -> tuple[bool, str]:
    """Whether *descriptor*'s checkpoint is provably the one the matrices came out of.

    Only a recorded hash that matches a local file counts. A missing record, a missing
    file or a mismatch all mean "not proven". This is diagnostic only: it explains a
    failed comparison, but :func:`check_environment` recomputes either way.
    """
    from ..utils.checkpoints import checkpoint_dir
    from ..utils.fetch import sha256_file

    filename = DESCRIPTOR_CHECKPOINT.get(descriptor)
    if filename is None:
        return False, "no checkpoint; values come from the toolkit"
    entry = (manifest.get("checkpoints") or {}).get(filename) or {}
    expected = entry.get("sha256")
    if not expected:
        return False, "manifest records no checkpoint hash"
    path = checkpoint_dir() / filename
    if not path.is_file():
        return False, "checkpoint not present locally"
    if sha256_file(path) != expected:
        return False, "checkpoint differs from the one the reference was built with"
    return True, "checkpoint unchanged"


def check_environment(descriptors, manifest: dict | None = None) -> dict:
    """Compare freshly computed descriptor values against the published canary rows.

    Returns ``{descriptor: {"status": "pass"|"warn"|"skip", ...}}``. Raises
    :class:`ReferenceDriftError` on a difference that means the matrices no longer describe
    what this install computes.
    """
    import h5py

    from ..registry import get_descriptor_type

    manifest = manifest if manifest is not None else load()
    if not manifest or "canary" not in manifest:
        return {}

    path = reference_dir() / manifest["canary"]["file"]
    if not path.is_file():
        logger.debug("No canary file cached; skipping the drift check.")
        return {}

    smiles = list(manifest["canary"]["smiles"])
    results: dict[str, dict] = {}

    with h5py.File(path, "r") as handle:
        for name in descriptors:
            if name not in handle:
                continue
            # Always recompute. This used to be skipped when the checkpoint hash matched,
            # but an unchanged checkpoint does not prove unchanged values: the CLAMP
            # folding fix (eos3l5f#3) changed the code around an identical encoder, and
            # the stale matrices passed as "checkpoint unchanged". The canary is 16 rows,
            # so the recompute is cheap.
            _, reason = _checkpoint_unchanged(manifest, name)
            stored = np.asarray(handle[name], dtype=np.float64)
            fresh = np.asarray(
                get_descriptor_type(name)().transform(smiles), dtype=np.float64
            )
            try:
                results[name] = _compare(name, stored, fresh)
            except ReferenceDriftError as exc:
                raise ReferenceDriftError(f"{exc} (checkpoint: {reason})") from None
            if name in DESCRIPTOR_CHECKPOINT:
                results[name]["checkpoint"] = reason

    return results


def _compare(name: str, stored: np.ndarray, fresh: np.ndarray) -> dict:
    if stored.shape != fresh.shape:
        raise ReferenceDriftError(
            f"{name}: this install produces {fresh.shape[1]} features, the reference was "
            f"built with {stored.shape[1]}."
        )

    if name in EXACT:
        if not np.array_equal(stored, fresh):
            raise ReferenceDriftError(
                f"{name} is deterministic given the toolkit, and its values have changed. "
                "The published matrices no longer describe what this install computes; "
                "`rank` would be a percentile against a population it cannot reproduce."
            )
        return {"status": "pass"}

    if name == "rdkit":
        if not np.allclose(stored, fresh, rtol=RDKIT_RTOL, atol=RDKIT_ATOL):
            worst = int(np.argmax(np.abs(stored - fresh).max(axis=0)))
            raise ReferenceDriftError(
                f"rdkit values differ beyond tolerance (worst column {worst}). RDKit's "
                "descriptor names are identical across releases that change the values, "
                "so a version check cannot catch this -- only the numbers can."
            )
        return {"status": "pass"}

    # Neural embeddings: per-row cosine, which separates a backend difference from a
    # different checkpoint by orders of magnitude.
    num = (stored * fresh).sum(axis=1)
    den = np.linalg.norm(stored, axis=1) * np.linalg.norm(fresh, axis=1)
    ok = den > 0
    cosine = float(np.min(num[ok] / den[ok])) if ok.any() else 1.0
    if cosine < COSINE_WARN:
        raise ReferenceDriftError(
            f"{name}: embeddings differ from the published reference (min row cosine "
            f"{cosine:.8f}). That is the size of a changed checkpoint or changed "
            "descriptor code, not a numerical difference between backends."
        )
    if cosine < COSINE_PASS:
        logger.warning(
            f"{name}: embeddings differ slightly from the reference "
            f"(min row cosine {cosine:.8f}). Within the range a different BLAS or CPU "
            "produces, so this is recorded rather than fatal."
        )
        return {"status": "warn", "cosine": cosine}
    return {"status": "pass", "cosine": cosine}
