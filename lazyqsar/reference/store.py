"""Resolve and read the reference matrices. Fit-time only.

Uses ``h5py``, which is a core dependency, so this adds nothing to the install. It is still
never imported by the inference path -- ``lazyqsar/qsar.py`` reaches it through a
function-level import, the same way it already reaches ``validate_smiles``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .identity import (
    DEFAULT_N,
    REFERENCE_ID,
    descriptor_filename,
    descriptor_url,
    reference_dir,
    smiles_filename,
)


class ReferenceUnavailable(RuntimeError):
    """The bundle is not on disk. Recoverable by fetching it."""


class ReferenceIntegrityError(RuntimeError):
    """The bundle is present but does not describe what it claims to."""


def descriptor_path(descriptor: str, n: int = DEFAULT_N) -> Path:
    path = reference_dir() / descriptor_filename(descriptor, n)
    if not path.is_file():
        raise ReferenceUnavailable(
            f"No reference matrix for {descriptor!r} at {path}.\n"
            f"Fetch it with `lazyqsar setup --reference --only {descriptor}`, or point "
            f"LAZYQSAR_REFERENCE_DIR at a local copy.\n"
            f"Source: {descriptor_url(descriptor, n)}"
        )
    return path


def reference_smiles(n: int = DEFAULT_N) -> list[str]:
    """The molecule list, in the row order every matrix uses.

    This is all a bring-your-own-descriptor caller needs: featurize these, in this order,
    and hand the matrix back to ``LazyClassifier.fit(reference_X=...)``.
    """
    path = reference_dir() / smiles_filename(n)
    if not path.is_file():
        raise ReferenceUnavailable(f"No reference molecule list at {path}.")
    rows = path.read_text().splitlines()
    if rows and rows[0].strip().lower() == "smiles":
        rows = rows[1:]
    return [r.strip() for r in rows if r.strip()]


def _check(dset, descriptor: str, n: int, expected_dim: int | None) -> None:
    got = dset.attrs.get("descriptor")
    if got is not None and str(got) != descriptor:
        raise ReferenceIntegrityError(
            f"{descriptor}: file declares descriptor {got!r}."
        )
    if dset.shape[0] != n:
        raise ReferenceIntegrityError(
            f"{descriptor}: {dset.shape[0]} rows, expected {n}. Row i of every descriptor "
            "must be the same molecule, so a mismatched file cannot be used."
        )
    if expected_dim is not None and dset.shape[1] != expected_dim:
        raise ReferenceIntegrityError(
            f"{descriptor}: {dset.shape[1]} features, but the installed descriptor "
            f"produces {expected_dim}. The bundle does not match this version."
        )


def iter_chunks(
    descriptor: str,
    n: int = DEFAULT_N,
    chunk_size: int = 4096,
    expected_dim: int | None = None,
):
    """Yield float32 row blocks of one reference matrix.

    Deliberately not ``agnostic._load_h5``, which does ``f["X"][:].astype("float32")`` and
    so holds the stored array *and* its float32 copy at once -- 615 MB for chemeleon. h5py
    casts on assignment into an existing buffer, so peak allocation here is one chunk.

    The yielded array is reused between iterations. Callers that keep results must copy.
    """
    import h5py

    with h5py.File(descriptor_path(descriptor, n), "r") as f:
        dset = f["X"]
        _check(dset, descriptor, n, expected_dim)
        buf = np.empty((chunk_size, dset.shape[1]), dtype=np.float32)
        for start in range(0, dset.shape[0], chunk_size):
            end = min(start + chunk_size, dset.shape[0])
            view = buf[: end - start]
            view[:] = dset[start:end]
            yield view


def load(
    descriptor: str, n: int = DEFAULT_N, expected_dim: int | None = None
) -> np.ndarray:
    """The whole matrix as float32, allocated once."""
    import h5py

    with h5py.File(descriptor_path(descriptor, n), "r") as f:
        dset = f["X"]
        _check(dset, descriptor, n, expected_dim)
        out = np.empty(dset.shape, dtype=np.float32)
        step = max(1, 4096)
        for start in range(0, dset.shape[0], step):
            end = min(start + step, dset.shape[0])
            out[start:end] = dset[start:end]
    return out


def is_available(descriptor: str, n: int = DEFAULT_N) -> bool:
    return (reference_dir() / descriptor_filename(descriptor, n)).is_file()


def status(n: int = DEFAULT_N) -> dict:
    """What is cached locally, for ``lazyqsar reference status``."""
    root = reference_dir()
    out: dict = {"reference_id": REFERENCE_ID, "dir": str(root), "n": n, "files": {}}
    if root.is_dir():
        for path in sorted(root.glob("*")):
            out["files"][path.name] = path.stat().st_size
    return out
