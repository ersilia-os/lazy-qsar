"""Resolve and read the reference matrices. Fit-time only.

Uses ``h5py``, which is a core dependency, so this adds nothing to the install. It is still
never imported by the inference path -- ``lazyqsar/qsar.py`` reaches it through a
function-level import, the same way it already reaches ``validate_smiles``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .identity import (
    default_n,
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


def descriptor_path(descriptor: str, n: int | None = None) -> Path:
    """The local matrix for *descriptor*, fetching it if it is not cached.

    Fetching happens here rather than eagerly because the caller only knows which
    descriptors it needs once the portfolio has settled: a fast-mode model needs one 6.5 MB
    matrix, not the 267 MB bundle.
    """
    from .download import ReferenceDownloadError, download, offline

    n = n or default_n()
    path = reference_dir() / descriptor_filename(descriptor, n)
    if path.is_file():
        return path

    if not offline():
        try:
            return download([descriptor_filename(descriptor, n)], n=n)[0]
        except ReferenceDownloadError as exc:
            raise ReferenceUnavailable(str(exc)) from exc

    raise ReferenceUnavailable(
        f"No reference matrix for {descriptor!r} at {path}, and fetching is disabled by "
        "LAZYQSAR_REFERENCE_OFFLINE.\n"
        f"Fetch it with `lazyqsar setup --reference --only {descriptor}`, or point "
        f"LAZYQSAR_REFERENCE_DIR at a local copy.\n"
        f"Source: {descriptor_url(descriptor, n)}"
    )


def reference_smiles(n: int | None = None) -> list[str]:
    """The molecule list, in the row order every matrix uses.

    This is all a bring-your-own-descriptor caller needs: featurize these, in this order,
    and hand the matrix back to ``LazyClassifier.fit(reference_X=...)``.
    """
    from .download import ReferenceDownloadError, download, offline

    n = n or default_n()
    path = reference_dir() / smiles_filename(n)
    if not path.is_file() and not offline():
        try:
            path = download([smiles_filename(n)], n=n)[0]
        except ReferenceDownloadError as exc:
            raise ReferenceUnavailable(str(exc)) from exc
    if not path.is_file():
        raise ReferenceUnavailable(
            f"No reference molecule list at {path}. Fetch it with "
            "`lazyqsar reference fetch`, or write it out with "
            "`lazyqsar reference smiles --output ref.csv`."
        )
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
    n: int | None = None,
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

    n = n or default_n()
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
    descriptor: str, n: int | None = None, expected_dim: int | None = None
) -> np.ndarray:
    """The whole matrix as float32, allocated once."""
    import h5py

    n = n or default_n()
    with h5py.File(descriptor_path(descriptor, n), "r") as f:
        dset = f["X"]
        _check(dset, descriptor, n, expected_dim)
        out = np.empty(dset.shape, dtype=np.float32)
        step = max(1, 4096)
        for start in range(0, dset.shape[0], step):
            end = min(start + step, dset.shape[0])
            out[start:end] = dset[start:end]
    return out


def is_available(descriptor: str, n: int | None = None) -> bool:
    return (reference_dir() / descriptor_filename(descriptor, n)).is_file()


def status(n: int | None = None) -> dict:
    """What is cached locally, for ``lazyqsar reference status``."""
    n = n or default_n()
    root = reference_dir()
    out: dict = {"reference_id": REFERENCE_ID, "dir": str(root), "n": n, "files": {}}
    if root.is_dir():
        for path in sorted(root.glob("*")):
            out["files"][path.name] = path.stat().st_size
    return out


def verify(n: int | None = None) -> list[str]:
    """Check every cached reference file, returning a list of problems.

    Structural checks only -- the file opens, declares the descriptor it is named for, has
    the right number of rows, and its values are finite. Whether those values match what the
    *installed* descriptor code would compute is a different question, answered by the
    manifest's canary rows once the bundle carries them.
    """
    import h5py

    from ..registry import DESCRIPTOR_TYPES

    n = n or default_n()
    root = reference_dir()
    problems: list[str] = []

    smiles = root / smiles_filename(n)
    if not smiles.is_file():
        problems.append(f"missing molecule list: {smiles.name}")
    else:
        rows = [r for r in smiles.read_text().splitlines()[1:] if r.strip()]
        if len(rows) != n:
            problems.append(f"{smiles.name}: {len(rows)} molecules, expected {n}")

    for descriptor in sorted(DESCRIPTOR_TYPES):
        path = root / descriptor_filename(descriptor, n)
        if not path.is_file():
            continue  # not every descriptor has to be cached; fetching is per-descriptor
        try:
            with h5py.File(path, "r") as handle:
                dset = handle["X"]
                declared = dset.attrs.get("descriptor")
                if declared is not None and str(declared) != descriptor:
                    problems.append(f"{path.name}: declares descriptor {declared!r}")
                if dset.shape[0] != n:
                    problems.append(f"{path.name}: {dset.shape[0]} rows, expected {n}")
                block = np.asarray(dset[: min(1024, dset.shape[0])], dtype=np.float64)
                if not np.isfinite(block).all():
                    problems.append(f"{path.name}: non-finite values")
        except Exception as exc:  # pragma: no cover - corrupt file paths
            problems.append(f"{path.name}: unreadable ({exc})")
    return problems
