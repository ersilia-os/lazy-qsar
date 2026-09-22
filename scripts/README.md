# Reference-library build (maintainer only)

Builds the fixed reference library that `predict_rank` reports percentiles against, and
publishes it to public S3. **Nothing here ships in the wheel** — `pyproject.toml` packages
only `lazyqsar/`, and the installed package reads the published bundle over plain HTTPS.

`eosvc` and `boto3` are never imported. `publish_reference.py` shells out to the `eosvc`
binary, which is the strongest available guarantee that neither can leak into the package's
dependencies.

```bash
pip install "lazyqsar[all]" -r scripts/requirements-maintainer.txt
```

RDKit must be **2025.9.1**, the version `pyproject.toml` pins. The whole build is invalid
under a different one: the descriptors published here are recomputed at fit time by the
installed package, and they have to agree.

## Why the reference set is built this way

A rank is a percentile, so **the reference set's density is the calibration**. That single
fact drives every choice below.

### Two clusterings, in two spaces

Measured over the whole library, BitBIRCH on Morgan fingerprints gives:

| threshold | clusters | % of n | mean size | max | singletons |
|---|---|---|---|---|---|
| 0.65 | 1,193,292 | 88.1% | 1.14 | 29 | 90.5% |
| 0.55 | 1,078,515 | 79.6% | 1.26 | 64 | 84.9% |
| 0.45 | 951,089 | 70.2% | 1.42 | 160 | 79.4% |

There is no threshold at which this library forms large clusters — it was already
diversity-curated upstream, and the largest cluster at a chemically sensible threshold holds
29 molecules out of 1.36M. So BitBIRCH **cannot** supply density strata, and an early plan
to take one representative per cluster from ~10,000 clusters was simply unreachable.

What BitBIRCH does buy is analogue collapse: ~10% of molecules have a near-duplicate, and
collapsing them stops one chemotype occupying two reference slots.

The strata come from **MiniBatchKMeans on physchem descriptors** instead, over the BitBIRCH
medoids. k-means Voronoi cells are roughly equal in *volume*, so their populations track how
crowded that region of chemical space is — which is the density the `ALPHA` exponent
tempers. Equal-frequency binning would not work: it produces equal-sized strata by
construction, which makes `ALPHA` a no-op.

### Why `ALPHA = 0.5` and not one-per-cluster

Slots per stratum are proportional to `size ** ALPHA`, apportioned by largest remainder so
the total is exactly N. On a realistic size distribution:

| ALPHA | biggest stratum gets | strata with **zero** representation |
|---|---|---|
| 1.0 (proportional) | 1,898 | 3,092 |
| **0.5 (shipped)** | 152 | 0 |
| 0.0 (equal) | 6 | 0 |

Proportional allocation leaves roughly a third of chemistry unrepresented. Equal allocation
flattens the density that the percentile is quoted against — it answers "beats 99% of
chemotypes", not "beats 99% of drug-like compounds". 0.5 is the setting where every stratum
still earns a slot and crowded regions stay crowded.

### Why the CDDD gate runs at pick time

`ContinuousDataDrivenDescriptor.transform` substitutes *a different molecule's* embedding,
found by FPSim2 nearest-neighbour search, for any molecule it cannot preprocess
(`lazyqsar/descriptors/cddd.py:436-450`). A reference row carrying another molecule's
embedding is a silently wrong ECDF point, so those molecules are excluded rather than
substituted.

Measured reject rate on a random sample of the library: **1.88%** — above CDDD's own 0.1%
applicability gate (`cddd.py:463-470`), so a *random* 50k slice would make CDDD inapplicable
to the reference set.

The gate is applied to chosen representatives, not to the library, and a rejected
representative is replaced by another member of **its own BitBIRCH cluster**. Two
consequences: the final set is 100% CDDD-calculable by construction, and no chemotype is
lost merely because its most central molecule happens to be one CDDD cannot encode.

Cost matters here. The predicate is pure RDKit (it never touches the 182 MB FPSim2
database), so at pick time it costs ~20 s. Over the whole library it would cost 7.7 min.
Running the CDDD **encoder** over the library — which this build never does — would cost
**11.8 hours**.

### Determinism

- The library is stored in a meaningful order (its first rows are fatty acids and peptides),
  and BitBIRCH is a BIRCH variant, so its result depends on insertion order. The build sorts
  canonically before clustering.
- No RNG anywhere in the pick. k-means and PCA take a fixed `SEED`.
- Every entry script needs `if __name__ == "__main__":` — macOS and Windows spawn workers by
  re-importing the parent module, and without the guard a multiprocessing stage forks
  recursively instead of failing with anything that names the cause.

## Measured costs

| Stage | Molecules | Cost |
|---|---|---|
| A — parse + dedup (parallel) | 1,355,109 | 11 s |
| B1 — Morgan fingerprints | 1,355,109 | 104 s |
| B1 — BitBIRCH | 1,355,109 | 55 s |
| B1 — medoids | 1,193,292 clusters | 2 s |
| B2 — PCA to 50 dims (84.3% variance) | 1,193,292 | 3 s |
| B2 — MiniBatchKMeans k=10,000 | 1,193,292 | see `stage_report.json` |
| D — CDDD gate at pick time | ~51,000 | ~20 s |

The library is already fully canonical and deduplicated upstream: stage A finds **0
unparseable and 0 duplicates**, so it is an assertion rather than a filter.

## Order of operations

```bash
python scripts/select_reference_subset.py --n 50000
```

Writes `data/reference/<REFERENCE_ID>/` (gitignored):

| File | Purpose |
|---|---|
| `reference_smiles_n50000.csv` | the selection, in order — the published molecule list |
| `ordering.npy` | every candidate in selection order, so larger tiers nest |
| `stage_report.json` | counts, distributions and parameters for every stage |

Then describe it, and check what was published:

```bash
python scripts/compute_reference_descriptors.py --all   # the five matrices
python scripts/build_reference_manifest.py              # manifest.json + canary rows
eosvc upload --path data/reference/lazyqsar_reference_v1
python scripts/verify_reference_bundle.py               # the acceptance gate
```

`verify_reference_bundle.py` downloads from the **published prefix** into a throwaway
cache and never reads `data/`, so it cannot pass by finding something the build left
behind. It checks the manifest, every sha256, that all matrices were built from the same
molecule list, that this install reproduces the published canary values, and that a model
fits and ranks correctly against it. Run it on a second machine before anyone depends on
the bundle: passing here only says the prefix is good *from this environment*, and a
different RDKit or checkpoint is exactly what the canary exists to catch.

## Immutability

A published bundle is never overwritten. Any change to the molecule set or to any descriptor
matrix means a new `REFERENCE_ID`, because clients cache by filename and cannot notice an
object that changed underneath them.
