# Test data

Small, committed fixtures. Everything here is derived from **ChEMBL**; no other bioactivity
source is used.

## Why real molecules

`_helpers.smiles.make_smiles` is fine for the stubbed tiers, where all that matters is that a
string parses and is distinct from its neighbours. It is not fine for the `chem` tier: it
emits up to 90 alkyl ethers (`CO`, `COC`, `CCOCC`, …), a family with almost no structural
diversity, so Morgan fingerprints over it are nearly constant. A descriptor test, an
applicability-domain test or a portfolio test run against that set measures the generator
rather than the code.

## Files

| File | Rows | Positives | Purpose |
|---|---|---|---|
| `reference_binary.csv` | 233 | 54 (23%) | The general-purpose set: descriptors, end-to-end fit and predict, parity between entry points. |
| `reference_imbalanced.csv` | 432 | 56 (13%) | Low prevalence, to reach the imbalance-batching path in `assemblers/classifier.py` that a synthetic 50/50 label vector never exercises. |
| `invalid_smiles.txt` | 10 | — | Strings RDKit must reject. See the note in the file itself. |
| `golden/combine.npz` | 28 cases | — | Frozen `combine()` output. Regenerate with `dev/tools/regenerate_combine_golden.py`; see that script's docstring before you do. |

Both CSVs are `smiles,bin` with a header, which is exactly the shape `lazyqsar fit --input`
expects, so they can be used directly as CLI input.

## Provenance

Both CSVs are curated *Acinetobacter baumannii* bioactivity datasets from
[`ersilia-os/chembl-antimicrobial-models`](https://github.com/ersilia-os/chembl-antimicrobial-models),
which derives them from ChEMBL via
[`ersilia-os/chembl-antimicrobial-tasks`](https://github.com/ersilia-os/chembl-antimicrobial-tasks).

| This file | Source file in that repo |
|---|---|
| `reference_binary.csv` | `output/07_datasets/abaumannii/DR_0006.csv` |
| `reference_imbalanced.csv` | `output/07_datasets/abaumannii/SP_catchall.csv` |

The upstream files carry an `inchikey` column, dropped here since nothing in the suite uses
it. To regenerate, with that repository checked out and its data pulled via `eosvc`:

```bash
python - <<'PY'
import csv, os
SRC = "<path-to>/chembl-antimicrobial-models/output/07_datasets/abaumannii"
for src, dst in [("DR_0006.csv", "reference_binary.csv"),
                 ("SP_catchall.csv", "reference_imbalanced.csv")]:
    rows = list(csv.DictReader(open(os.path.join(SRC, src))))
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "bin"])
        w.writerows([r["smiles"], r["bin"]] for r in rows)
PY
```

## Licensing

ChEMBL data is released by EMBL-EBI under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Please cite ChEMBL if you
reuse these files.

These datasets are included solely as test fixtures. They are not a benchmark, and the
numbers a model gets on 233 compounds say nothing about its accuracy — the tests that use
them assert invariants and shapes, not performance.
