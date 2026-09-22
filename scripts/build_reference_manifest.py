#!/usr/bin/env python
"""Describe the published bundle, so a client can tell it got what it expected.

    python scripts/build_reference_manifest.py

Writes ``manifest.json`` and ``canary_n<N>.h5`` into the bundle directory. The manifest is
the first thing a client fetches and the only thing it parses; every other file is checked
against it.

It answers three questions, in increasing cost:

**Did the file arrive intact?** A sha256 per published file. The downloader verifies it.

**Is this bundle the one this install expects?** Reference id, tier, row counts, descriptor
dimensions, and the sha256 of the molecule list each matrix was built from. Row *i* of every
matrix must be the same molecule, and that shared hash is what makes a mismatched pair
impossible to use silently.

**Would this install compute the same values?** The hard one, and the reason for the canary
rows. Descriptor versions alone cannot answer it: the RDKit descriptor *names* are identical
between 2025.09.1 and 2026.03.1, so a name or version comparison passes across a two-release
gap while the values may not. The checkpoints are worse -- the CheMeleon ``.pt`` and the
CLAMP and CDDD ``.onnx`` files can be re-uploaded in place with nothing recording it. So the
manifest stores the sha256 of every checkpoint *and* a handful of molecules' actual
descriptor values, recomputed at fit time and compared.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import config  # noqa: E402
from reference.descriptors import DTYPE_POLICY  # noqa: E402

SCHEMA_VERSION = 1

# Molecules whose descriptor values are stored for the drift check. Taken at a fixed stride
# rather than at random so the choice is reproducible from the tier size alone, and from
# across the whole set rather than the head, which is ordered.
N_CANARY = 16


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict:
    """What produced the matrices. Diagnostic, never a gate on its own -- see the docstring."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name, module in (
        ("rdkit", "rdkit"),
        ("numpy", "numpy"),
        ("onnxruntime", "onnxruntime"),
        ("torch", "torch"),
        ("chemprop", "chemprop"),
        ("chemeleon", "chemeleon"),
        ("h5py", "h5py"),
    ):
        # importlib.metadata first: `chemeleon` ships no `__version__`, so importing and
        # reading the attribute recorded None for a package that was installed and whose
        # version was knowable -- a silent hole in the drift record for one of the five
        # descriptors.
        try:
            out[name] = importlib.metadata.version(name)
        except Exception:
            try:
                out[name] = __import__(module).__version__
            except Exception:
                out[name] = None
    try:
        from rdkit.Chem import rdBase

        out["rdkit"] = rdBase.rdkitVersion
    except Exception:
        pass
    return out


def _checkpoint_hashes() -> dict:
    """The neural checkpoints the embeddings came out of.

    This is the layer that actually closes the hole. A changed CheMeleon or CLAMP or CDDD
    checkpoint moves every embedding by O(1), and nothing in the package records which one
    was used -- CLAMP's own metadata stores only the featurizer name.
    """
    from lazyqsar.utils.checkpoints import CHECKPOINT_SHA256, checkpoint_dir

    root = checkpoint_dir()
    out = {}
    for name, expected in CHECKPOINT_SHA256.items():
        path = root / name
        out[name] = {
            "sha256": sha256_file(path) if path.is_file() else expected,
            "present_locally": path.is_file(),
        }
    return out


def _descriptor_provenance(name: str) -> dict:
    """Per-descriptor parameters that change values without changing a version string."""
    from lazyqsar.registry import get_descriptor_type

    desc = get_descriptor_type(name)()
    out = {"n_dim": int(getattr(desc, "n_dim", 0) or 0)}
    if name == "morgan":
        out |= {
            "radius": int(desc.radius),
            "counts": True,
            "clamp": 255,
        }
    if name == "rdkit":
        # Stored in full: the list is RDKit-version dependent and is *sorted* here, where
        # RDKit's own order is not, so a permutation would silently re-label every column.
        out |= {
            "descriptor_names": list(desc.features),
            "clip": 1.0e5,
        }
    return out


def build_canary(bundle: Path, n: int, smiles: list[str]) -> tuple[Path, list[int]]:
    """Store a few molecules' values from every matrix, for the drift check."""
    import h5py

    positions = [int(round(i * (n - 1) / (N_CANARY - 1))) for i in range(N_CANARY)]
    path = bundle / f"canary_n{n}.h5"
    with h5py.File(path, "w") as out:
        for descriptor in sorted(DTYPE_POLICY):
            source = bundle / config.descriptor_filename(descriptor, n)
            if not source.is_file():
                continue
            with h5py.File(source, "r") as handle:
                rows = np.asarray(handle["X"][positions])
            out.create_dataset(descriptor, data=rows)
        out.attrs["positions"] = positions
        out.attrs["smiles"] = [smiles[i] for i in positions]
    return path, positions


def main() -> int:
    bundle = config.staging_dir()
    n = config.DEFAULT_N
    if not bundle.is_dir():
        print(f"No bundle at {bundle}", file=sys.stderr)
        return 1

    smiles_path = bundle / config.smiles_filename(n)
    smiles = [s.strip() for s in smiles_path.read_text().splitlines()[1:] if s.strip()]
    if len(smiles) != n:
        print(
            f"{smiles_path.name}: {len(smiles)} molecules, expected {n}",
            file=sys.stderr,
        )
        return 1

    canary_path, positions = build_canary(bundle, n, smiles)

    stage = json.loads((bundle / "stage_report.json").read_text())
    report = json.loads((bundle / "descriptor_report.json").read_text())

    descriptors = {}
    for descriptor in sorted(DTYPE_POLICY):
        path = bundle / config.descriptor_filename(descriptor, n)
        if not path.is_file():
            continue
        entry = dict(report["descriptors"].get(descriptor, {}))
        descriptors[descriptor] = {
            "file": path.name,
            "key": "X",
            "shape": entry.get("shape"),
            "dtype": entry.get("dtype"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "provenance": _descriptor_provenance(descriptor),
        }
        for extra in ("float16_roundtrip_min_cosine", "absmax", "promoted"):
            if extra in entry:
                descriptors[descriptor][extra] = entry[extra]

    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(bundle.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reference_id": config.REFERENCE_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": n,
        "source": {
            "library": config.SOURCE_LIBRARY_ID,
            "url": config.SOURCE_LIBRARY_URL,
            "n_rows": stage["stage_a"]["input"],
        },
        "selection": {
            "method": "bitbirch_morgan + minibatchkmeans_physchem",
            "note": (
                "BitBIRCH collapses near-duplicates in fingerprint space; k-means on "
                "physchem supplies the density strata; slots per stratum are proportional "
                "to size ** alpha. The set is CDDD-calculable by construction."
            ),
            **stage["params"],
            "stage_b1": stage["stage_b1"],
            "stage_b2": stage["stage_b2"],
            "stage_d": stage["stage_d"],
            "dropped_at_descriptor_stage": report["dropped"],
        },
        # Row i of every matrix is this molecule list's row i. A matrix built from a
        # different list is unusable, and this is what makes that detectable.
        "smiles": {
            "file": smiles_path.name,
            "sha256": report["smiles_sha256"],
            "n": n,
        },
        "descriptors": descriptors,
        "canary": {
            "file": canary_path.name,
            "sha256": sha256_file(canary_path),
            "n": N_CANARY,
            "positions": positions,
            "smiles": [smiles[i] for i in positions],
        },
        "checkpoints": _checkpoint_hashes(),
        "build_environment": _versions(),
        "files": files,
    }

    out = bundle / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out} ({out.stat().st_size / 1e3:.1f} KB)")
    print(f"Wrote {canary_path} ({canary_path.stat().st_size / 1e3:.1f} KB)")
    print(f"  {len(descriptors)} descriptor matrices, {len(files)} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
