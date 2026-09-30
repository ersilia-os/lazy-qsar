"""Dtype policy and HDF5 writing for the published reference matrices.

Each descriptor gets the narrowest dtype that is *lossless enough*, and each choice is
asserted at write time rather than assumed:

``morgan`` is ``uint8``. The counts are clamped at 255 by the descriptor itself
(``morgan.py:50``) and only ~3.4% of entries are non-zero, so gzip compresses the matrix
about 15x -- which is what makes the ``fast``-mode reference a ~9 MB download instead of
102 MB.

``rdkit`` is ``float32`` and must never be ``float16``. ``rdkit_descriptors.py:47,51``
clips at +-1e5 while float16 saturates at 65504, and ``Ipc`` alone puts ~79% of rows over
that ceiling. float16 would turn them into ``inf``, which the preprocessor's imputer would
then propagate into every head.

``chemeleon``, ``clamp`` and ``cddd`` are ``float16``: all three are bounded well inside
the range (absmax 1.95, 3.2 and 1.0), and the round-trip cost is a per-row cosine of
0.99999976 -- three orders of magnitude clear of the 0.9999 the canary check calls drift.
"""

from __future__ import annotations

import numpy as np

# Cheapest first, so a failure surfaces in seconds rather than after CDDD's 27 minutes.
# Measured per molecule: morgan 0.07 ms, clamp 0.44, rdkit 3.65, chemeleon 5.26, cddd 31.4.
BUILD_ORDER = ("morgan", "clamp", "rdkit", "chemeleon", "cddd")

# descriptor -> (published dtype, gzip level, rows per chunk)
DTYPE_POLICY: dict[str, tuple[str, int, int]] = {
    "morgan": ("uint8", 4, 1024),
    "rdkit": ("float32", 4, 4096),
    "cddd": ("float16", 1, 2048),
    "clamp": ("float16", 1, 1024),
    "chemeleon": ("float16", 1, 512),
}

# A float16 cast is only safe well inside the format's range; past this the build promotes
# to float32 and records that it did, rather than shipping silently clipped values.
FLOAT16_MAX_ABS = 1.0e3
FLOAT16_MIN_COSINE = 0.9999


def check_and_cast(name: str, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Validate *X* against the policy for *name* and cast it. Returns (array, notes)."""
    dtype, _, _ = DTYPE_POLICY[name]
    notes: dict = {"requested_dtype": dtype}

    if not np.isfinite(X).all():
        raise ValueError(
            f"{name}: {int((~np.isfinite(X)).any(axis=1).sum())} rows are non-finite. "
            "The validity pass should have removed these before the cast."
        )

    if dtype == "uint8":
        if X.min() < 0 or X.max() > 255 or not np.array_equal(X, np.round(X)):
            raise ValueError(
                f"{name}: values outside the lossless uint8 range "
                f"[{X.min()}, {X.max()}], or not integral."
            )
        return X.astype(np.uint8), notes | {"dtype": "uint8"}

    if dtype == "float16":
        absmax = float(np.abs(X).max())
        notes["absmax"] = absmax
        cos = _roundtrip_cosine(X)
        notes["float16_roundtrip_min_cosine"] = cos
        if absmax >= FLOAT16_MAX_ABS or cos < FLOAT16_MIN_COSINE:
            # Promote rather than ship clipped values, and say so in the manifest.
            notes |= {"dtype": "float32", "promoted": True}
            return X.astype(np.float32), notes
        return X.astype(np.float16), notes | {"dtype": "float16"}

    absmax = float(np.abs(X).max())
    if absmax > 1.0e5:
        raise ValueError(
            f"{name}: absmax {absmax} exceeds the +-1e5 clip it should carry."
        )
    return X.astype(np.float32), notes | {"dtype": "float32", "absmax": absmax}


def _roundtrip_cosine(X: np.ndarray, sample: int = 2048) -> float:
    """Worst per-row cosine between *X* and its float16 round trip."""
    rows = X if len(X) <= sample else X[:: max(1, len(X) // sample)]
    a = rows.astype(np.float64)
    b = rows.astype(np.float16).astype(np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 0
    return float(np.min(num[ok] / den[ok])) if ok.any() else 1.0


def write_matrix(path, name: str, X: np.ndarray, smiles_sha256: str) -> dict:
    """Write *X* to *path* as dataset ``X``, with self-describing attributes.

    The dataset key is ``X`` to match ``_load_h5`` in ``lazyqsar/agnostic.py:11-21``, so a
    bring-your-own-descriptor caller can hand one of these files straight back to
    ``LazyClassifier.fit``.

    ``smiles_sha256`` is what makes row alignment checkable. The pooled probability of
    reference row *i* needs row *i* of every descriptor matrix to be the same molecule; if
    one matrix were built from a different molecule list, nothing downstream could tell.
    Every file records the hash of the exact list it consumed, and the manifest writer
    refuses to proceed unless all of them agree.
    """
    import h5py

    arr, notes = check_and_cast(name, X)
    _, gzip_level, chunk_rows = DTYPE_POLICY[name]
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            "X",
            data=arr,
            chunks=(min(chunk_rows, arr.shape[0]), arr.shape[1]),
            compression="gzip",
            compression_opts=gzip_level,
            shuffle=(notes["dtype"] == "uint8"),
        )
        dset.attrs["descriptor"] = name
        dset.attrs["n_rows"] = arr.shape[0]
        dset.attrs["n_features"] = arr.shape[1]
        dset.attrs["dtype"] = notes["dtype"]
        dset.attrs["smiles_sha256"] = smiles_sha256
    return notes | {"shape": list(arr.shape), "bytes": path.stat().st_size}
