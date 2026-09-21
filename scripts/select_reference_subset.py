#!/usr/bin/env python
"""Choose the LazyQSAR reference library: 1.36M library molecules -> N representatives.

    python scripts/select_reference_subset.py --n 50000

Two clusterings, in two different spaces, because one space cannot do both jobs.

**BitBIRCH on Morgan fingerprints** collapses near-duplicates. Measured over the whole
library it yields 1,193,292 clusters from 1,355,109 molecules -- 90.5% singletons, largest
29 -- so it is *not* a source of density strata; this library was already diversity-curated
upstream. What it does buy is the guarantee that no two reference molecules are analogues
of each other, which is the thing that would otherwise let one chemotype occupy two slots.

**MiniBatchKMeans on physchem descriptors** supplies the strata. k-means Voronoi cells are
roughly equal in volume, so their populations track how crowded that region of chemical
space is -- which is the density the ``ALPHA`` exponent tempers.

Why temper rather than flatten: a rank is a percentile, so the reference set's density *is*
the calibration. ``ALPHA=0.5`` compresses crowded regions without erasing the fact that
they are where real screening compounds live. At ``ALPHA=1`` (proportional) roughly a third
of strata would get no representative at all; at ``ALPHA=0`` every stratum gets the same
number regardless of how much chemistry it covers.

The CDDD gate is applied at pick time, not to the library: a rejected representative is
replaced by another member of its own BitBIRCH cluster, so a chemotype is never lost to it.
Running the CDDD *encoder* over the library would take 11.8 hours; this costs ~20 seconds.

Outputs, under ``data/reference/<REFERENCE_ID>/``:

  reference_smiles_n<N>.csv  the selection, in order -- the published molecule list
  ordering.npy               every candidate in selection order, for nesting larger tiers
  stage_report.json          counts, cluster and stratum distributions, parameters
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from reference import config  # noqa: E402
from reference.eligibility import is_cddd_calculable, parse_and_dedup  # noqa: E402
from reference.selection import (  # noqa: E402
    allocate,
    cluster,
    cluster_medoids,
    morgan_bits,
    order_cluster,
    stratify,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=config.DEFAULT_N)
    ap.add_argument("--alpha", type=float, default=config.ALPHA)
    ap.add_argument("--strata", type=int, default=config.N_STRATA)
    ap.add_argument("--threshold", type=float, default=config.BITBIRCH_THRESHOLD)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    # Headroom: the descriptor stage may still drop a straggler, and re-running selection
    # to recover one row would be absurd.
    n_target = int(args.n * config.HEADROOM)

    out = config.staging_dir()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"params": vars(args) | {"n_target": n_target}}

    # --- stage A: parse and deduplicate -------------------------------------------
    src_csv = config.source_csv()
    if not src_csv.exists():
        log(f"ERROR: {src_csv} not found. Run `eosquality download` first.")
        return 1
    rows = [line.rstrip("\n") for line in src_csv.open()][1:]
    log(f"stage A: canonicalising {len(rows):,} molecules")
    canonical, source_index, counts = parse_and_dedup(rows)
    log(f"stage A: {counts}")
    report["stage_a"] = counts

    # Pin the order. BitBIRCH is a BIRCH variant, so its result depends on insertion
    # order, and this library is stored in a meaningful (non-random) order -- its first
    # rows are fatty acids and peptides. Sorting makes the run reproducible from the
    # content alone.
    order = np.argsort(np.asarray(canonical), kind="stable")
    canonical = [canonical[i] for i in order]
    source_index = np.asarray(source_index, dtype=np.int32)[order]

    # --- stage B1: collapse near-duplicates ---------------------------------------
    log("stage B1: Morgan fingerprints")
    fps = morgan_bits(canonical)
    log(f"stage B1: BitBIRCH at threshold {args.threshold}")
    tree = cluster(fps, args.threshold)
    clusters = tree.get_cluster_mol_ids()
    sizes = np.array([len(c) for c in clusters])
    log(
        f"stage B1: {len(clusters):,} clusters, mean {sizes.mean():.2f}, "
        f"max {sizes.max()}, singletons {100 * (sizes == 1).mean():.1f}%"
    )
    report["stage_b1"] = {
        "n_clusters": int(len(clusters)),
        "mean_size": float(sizes.mean()),
        "max_size": int(sizes.max()),
        "singleton_fraction": float((sizes == 1).mean()),
    }
    medoids = cluster_medoids(fps, clusters)

    # --- stage B2: density strata --------------------------------------------------
    log(f"stage B2: physchem strata over {len(medoids):,} medoids")
    physchem = np.load(config.source_index_dir() / "physchem_scaled.npy", mmap_mode="r")
    # Not np.sort(...): sorting the row indices would reorder X relative to `medoids`, and
    # every stratum label would then describe a different molecule than the one it is used
    # to pick.
    X = np.asarray(physchem[source_index[medoids]], dtype=np.float32)
    keep = X.std(axis=0) > 0
    X = np.clip(X[:, keep], -config.CLIP_SIGMA, config.CLIP_SIGMA)
    labels, seeds = stratify(X, args.strata, seed=args.seed, log=log)
    stratum_sizes = np.bincount(labels, minlength=args.strata)
    report["stage_b2"] = {
        "n_strata": int(args.strata),
        "dropped_zero_variance_columns": int((~keep).sum()),
        "mean_size": float(stratum_sizes.mean()),
        "max_size": int(stratum_sizes.max()),
        "empty": int((stratum_sizes == 0).sum()),
    }

    # --- stage C: apportion ---------------------------------------------------------
    alloc = allocate(stratum_sizes, n_target, args.alpha)
    log(f"stage C: {alloc.sum():,} slots over {int((alloc > 0).sum()):,} strata")
    report["stage_c"] = {
        "allocated": int(alloc.sum()),
        "strata_with_slots": int((alloc > 0).sum()),
    }

    # --- stage D: pick, with CDDD replacement ---------------------------------------
    log("stage D: picking representatives")
    picked, rejected = _pick(fps, medoids, labels, seeds, alloc, canonical, log)
    log(f"stage D: {len(picked):,} picked, {rejected:,} rejected by the CDDD gate")
    report["stage_d"] = {"picked": len(picked), "cddd_rejected": rejected}

    final = picked[: args.n]
    if len(final) < args.n:
        log(f"ERROR: only {len(final):,} of {args.n:,} could be selected")
        return 1

    np.save(out / "ordering.npy", np.asarray(picked, dtype=np.int64))
    smiles_path = out / config.smiles_filename(args.n)
    smiles_path.write_text("smiles\n" + "\n".join(canonical[i] for i in final) + "\n")
    (out / "stage_report.json").write_text(json.dumps(report, indent=2))
    log(f"wrote {smiles_path} ({args.n:,} molecules)")
    return 0


def _pick(fps, medoids, labels, seeds, alloc, canonical, log):
    """Walk each stratum's clusters, taking CDDD-calculable representatives.

    A representative that the CDDD gate rejects is replaced by the next candidate *from the
    same BitBIRCH cluster*, and only when a cluster is exhausted does the stratum move on.
    That is what keeps the gate from quietly deleting chemotypes.
    """
    picked: list[int] = []
    rejected = 0
    # Group rows by stratum with one sort rather than 1.2M dict appends.
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(len(alloc) + 1))

    for stratum, n_slots in enumerate(alloc):
        if n_slots <= 0:
            continue
        members = order[bounds[stratum] : bounds[stratum + 1]]
        if len(members) == 0:
            continue
        # Spread the picks across the stratum rather than clumping them at its centre,
        # starting from the member nearest the stratum centroid.
        seed_row = seeds[stratum]
        seed_mol = int(medoids[seed_row]) if seed_row >= 0 else int(medoids[members[0]])
        cand = order_cluster(fps, medoids[members], seed_mol, len(members))
        taken = 0
        for mol_id in cand:
            if taken >= n_slots:
                break
            if is_cddd_calculable(canonical[int(mol_id)]):
                picked.append(int(mol_id))
                taken += 1
            else:
                rejected += 1
    return picked, rejected


if __name__ == "__main__":
    raise SystemExit(main())
