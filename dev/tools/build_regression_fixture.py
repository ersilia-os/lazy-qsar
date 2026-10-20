"""Build ``tests/data/reference_regression.csv``: 300 EGFR (CHEMBL203) compounds with pChEMBL.

Run from the repository root (needs network access and RDKit):

    python dev/tools/build_regression_fixture.py

What it does, deterministically given the same API responses:

1. Fetches three 1,000-record pages of binding-assay IC50 activities with an exact
   (``=``) relation and a pChEMBL value, at offsets 0, 6000 and 12000 of the ordered
   result set -- spread across the data rather than its first, oldest, documents.
2. Drops records with a ``data_validity_comment``, unparseable or multi-component SMILES,
   and molecules above 60 heavy atoms; canonicalises with RDKit.
3. Aggregates replicates per canonical SMILES by the median, and drops compounds whose
   replicates disagree by more than 1 log unit.
4. Picks 30 compounds from each pChEMBL decile (ordered by an MD5 of the ChEMBL id, so
   the pick is stable and not tied to document order), and writes them sorted by id.

ChEMBL's content changes between releases, so a later run may select different
compounds. The committed file is the fixture; this script documents how it was made.
"""

import csv
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "tests", "data", "reference_regression.csv")
URL = (
    "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL203"
    "&standard_type=IC50&pchembl_value__isnull=false&standard_relation=%3D"
    "&assay_type=B&only=molecule_chembl_id,canonical_smiles,pchembl_value,"
    "data_validity_comment&limit=1000&offset={offset}"
)
OFFSETS = (0, 6000, 12000)
PER_DECILE = 30
MAX_HEAVY_ATOMS = 60
MAX_REPLICATE_RANGE = 1.0


def fetch(offset):
    with urllib.request.urlopen(URL.format(offset=offset), timeout=300) as r:
        return json.load(r)["activities"]


def main():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    activities = [a for off in OFFSETS for a in fetch(off)]
    values, ids = defaultdict(list), {}
    for a in activities:
        smi = a.get("canonical_smiles")
        if a.get("data_validity_comment") or not smi or a.get("pchembl_value") is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or "." in smi or mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            continue
        can = Chem.MolToSmiles(mol)
        values[can].append(float(a["pchembl_value"]))
        ids[can] = a["molecule_chembl_id"]

    rows = [
        (s, float(np.median(v)), ids[s])
        for s, v in values.items()
        if max(v) - min(v) <= MAX_REPLICATE_RANGE
    ]
    rows.sort(key=lambda r: (r[1], r[2]))
    edges = np.quantile([r[1] for r in rows], np.linspace(0, 1, 11))
    picked = []
    for b in range(10):
        lo, hi = edges[b], edges[b + 1]
        members = [r for r in rows if lo <= r[1] < hi or (b == 9 and r[1] == hi)]
        members.sort(key=lambda r: hashlib.md5(r[2].encode()).hexdigest())
        picked += members[:PER_DECILE]
    picked.sort(key=lambda r: r[2])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "pchembl"])
        w.writerows([s, f"{v:.2f}"] for s, v, _ in picked)
    print(f"wrote {OUT}: {len(picked)} compounds from {len(activities)} activities")


if __name__ == "__main__":
    sys.exit(main())
