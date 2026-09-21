#!/usr/bin/env python
"""Featurize the selected reference molecules, one HDF5 per descriptor.

    python scripts/compute_reference_descriptors.py --descriptor morgan
    python scripts/compute_reference_descriptors.py --all

Uses LazyQSAR's own descriptor classes through ``registry.get_descriptor_type``, never a
reimplementation: these matrices are compared against locally recomputed values at fit time,
so anything else guarantees drift.

**Validity headroom.** ``select_reference_subset.py`` emits ~2% more candidates than the
tier needs. This stage featurizes all of them, intersects the per-descriptor finite-row
masks, and keeps the first N survivors *in selection order*. A row that any descriptor
cannot produce is dropped from *all* of them -- the runtime median-imputes NaN
(``base/preprocessing/pipeline.py``), which would smear the ECDF with fabricated points.

Every descriptor must therefore be computed before any file is final, which is why
``--all`` rewrites the molecule list at the end. Running one descriptor at a time is
supported for parallelism, but the intersection still has to happen once they all exist.

**CDDD is sharded.** ``seq_to_emb`` wraps its entire batch loop in a single ``try/except``
(``cddd.py:403-423``), so one tokenizer failure nulls every remaining row rather than one.
Shards of 1000 bound the blast radius and make a failure nameable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import config  # noqa: E402
from reference.descriptors import BUILD_ORDER, DTYPE_POLICY, write_matrix  # noqa: E402

CDDD_SHARD = 1000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _featurize(name: str, smiles: list[str]) -> np.ndarray:
    from lazyqsar.registry import get_descriptor_type

    desc = get_descriptor_type(name)()
    if name != "cddd":
        return np.asarray(desc.transform(smiles), dtype=np.float64)

    out = []
    for start in range(0, len(smiles), CDDD_SHARD):
        shard = smiles[start : start + CDDD_SHARD]
        block = np.asarray(desc.transform(shard), dtype=np.float64)
        bad = int(np.isnan(block).any(axis=1).sum())
        if bad:
            log(f"  cddd shard {start}-{start + len(shard)}: {bad} NaN rows")
        out.append(block)
        log(f"  cddd {min(start + CDDD_SHARD, len(smiles)):,}/{len(smiles):,}")
    return np.vstack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--descriptor", choices=sorted(DTYPE_POLICY))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=config.DEFAULT_N)
    args = ap.parse_args()
    if not args.descriptor and not args.all:
        ap.error("pass --descriptor NAME or --all")

    out = config.staging_dir()
    work = config.work_dir()
    work.mkdir(parents=True, exist_ok=True)

    cand_path = out / "candidates.csv"
    if not cand_path.exists():
        log(f"ERROR: {cand_path} not found. Run select_reference_subset.py first.")
        return 1
    candidates = [line.strip() for line in cand_path.read_text().splitlines()[1:]]
    log(f"{len(candidates):,} candidates for a tier of {args.n:,}")

    names = list(BUILD_ORDER) if args.all else [args.descriptor]
    for name in names:
        raw = work / f"{name}_raw.npy"
        if raw.exists():
            log(f"{name}: cached at {raw.name}")
            continue
        log(f"{name}: featurizing {len(candidates):,} molecules")
        t = time.time()
        X = _featurize(name, candidates)
        np.save(raw, X)
        log(f"{name}: {X.shape} in {time.time() - t:.0f}s")

    # Intersect validity across every descriptor, then take the first N in selection order.
    present = [n for n in BUILD_ORDER if (work / f"{n}_raw.npy").exists()]
    if set(present) != set(DTYPE_POLICY):
        log(f"computed {present}; run the rest before the matrices can be finalised")
        return 0

    valid = np.ones(len(candidates), dtype=bool)
    for name in present:
        X = np.load(work / f"{name}_raw.npy", mmap_mode="r")
        ok = np.isfinite(np.asarray(X)).all(axis=1)
        log(f"{name}: {int((~ok).sum())} invalid rows")
        valid &= ok

    keep = np.flatnonzero(valid)[: args.n]
    dropped = len(candidates) - int(valid.sum())
    log(f"validity: {dropped} candidates dropped, {len(keep):,} kept")
    if dropped > 200:
        log(f"ERROR: {dropped} dropped is far beyond the expected ~0; not publishing.")
        return 1
    if len(keep) < args.n:
        log(f"ERROR: only {len(keep):,} valid rows for a tier of {args.n:,}")
        return 1

    smiles = [candidates[i] for i in keep]
    smiles_csv = "smiles\n" + "\n".join(smiles) + "\n"
    (out / config.smiles_filename(args.n)).write_text(smiles_csv)
    smiles_sha = hashlib.sha256(smiles_csv.encode()).hexdigest()
    log(f"molecule list rewritten, sha256 {smiles_sha[:16]}...")

    report = {
        "n": args.n,
        "smiles_sha256": smiles_sha,
        "dropped": dropped,
        "descriptors": {},
    }
    for name in present:
        X = np.load(work / f"{name}_raw.npy", mmap_mode="r")
        path = out / config.descriptor_filename(name, args.n)
        notes = write_matrix(path, name, np.asarray(X)[keep], smiles_sha)
        log(
            f"{name}: wrote {path.name} {notes['shape']} {notes['dtype']} "
            f"{notes['bytes'] / 1e6:.1f} MB"
        )
        report["descriptors"][name] = notes

    (out / "descriptor_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
