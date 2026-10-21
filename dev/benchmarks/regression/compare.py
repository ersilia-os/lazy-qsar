"""Compare two ``run.py`` result files: accuracy per source and fit time.

    python dev/benchmarks/regression/compare.py baseline v2
    python dev/benchmarks/regression/compare.py path/to/baseline.jsonl v2 --markdown

Reads ``.cache/results/<label>.jsonl`` (or a path) for both runs, keeps the (task, split)
pairs present in both, and reports per data source (TDC, Biogen, MoleculeACE) and split
kind (random / structural): mean relative RMSE, wins, mean Spearman and R², and the
fit-time ratio to classification at the same rows (median and 90th percentile).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, ".cache", "results")


def _load(label_or_path):
    path = (
        label_or_path
        if os.path.isfile(label_or_path)
        else os.path.join(RESULTS, f"{label_or_path}.jsonl")
    )
    with open(path) as f:
        return {(r["task"], r["split"]): r for r in map(json.loads, f)}


def _source(task):
    return {"tdc": "TDC ADMET", "biogen": "Biogen ADME", "ace": "MoleculeACE"}[
        task.split("_")[0]
    ]


def compare(old, new, markdown=False):
    keys = sorted(set(old) & set(new))
    groups = defaultdict(list)
    for key in keys:
        task, split = key
        kind = "random" if split.startswith("random") else "structural"
        o, n = old[key], new[key]
        row = {
            "rel_rmse": n["test"]["rmse"] / o["test"]["rmse"] - 1,
            "rho_old": o["test"]["spearman"],
            "rho_new": n["test"]["spearman"],
            "r2_old": o["test"]["r2"],
            "r2_new": n["test"]["r2"],
            "ratio_old": o["fit_seconds"] / o["clf_fit_seconds"]
            if o.get("clf_fit_seconds")
            else np.nan,
            "ratio_new": n["fit_seconds"] / n["clf_fit_seconds"]
            if n.get("clf_fit_seconds")
            else np.nan,
        }
        groups[(_source(task), kind)].append(row)
        groups[("All", kind)].append(row)
        groups[("All", "all")].append(row)

    header = (
        "| Source | Split | Fits | RMSE vs old | Better | ρ old → new | R² old → new "
        "| Time ratio old → new (median / p90) |"
    )
    lines = [header, "|" + "---|" * 8] if markdown else []
    for (source, kind), rows in sorted(
        groups.items(), key=lambda kv: (kv[0][0] == "All", kv[0])
    ):
        a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
        better = int((a["rel_rmse"] < 0).sum())
        cells = [
            source,
            kind,
            str(len(rows)),
            f"{100 * a['rel_rmse'].mean():+.2f}%",
            f"{better}/{len(rows)}",
            f"{np.nanmean(a['rho_old']):.3f} → {np.nanmean(a['rho_new']):.3f}",
            f"{np.nanmean(a['r2_old']):.3f} → {np.nanmean(a['r2_new']):.3f}",
            f"{np.nanmedian(a['ratio_old']):.2f} / {np.nanpercentile(a['ratio_old'], 90):.2f}"
            f" → {np.nanmedian(a['ratio_new']):.2f} / {np.nanpercentile(a['ratio_new'], 90):.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |" if markdown else "  ".join(cells))
    print("\n".join(lines))
    print(f"\n{len(keys)} (task, split) pairs compared.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    compare(_load(args.old), _load(args.new), args.markdown)


if __name__ == "__main__":
    main()
