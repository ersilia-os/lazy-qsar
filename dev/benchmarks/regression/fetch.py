"""Download the regression benchmark sources into ``.cache/raw`` (gitignored).

Raw files only, straight from their publishers; nothing is installed. Each source is
written to its own file and only ever *read* as data by ``tasks.py``.

    python dev/benchmarks/regression/fetch.py

Sources
-------
- TDC ADMET regression group (Huang et al. 2021), Harvard Dataverse file ids from
  PyTDC 1.1.15 ``tdc/metadata.py`` (``name2id``).
- MoleculeACE (van Tilborg et al. 2022): 30 ChEMBL Ki/EC50 sets with their own split,
  github.com/molML/MoleculeACE, ``MoleculeACE/Data/benchmark_data``.
- Biogen ADME (Fang et al. 2023): ``ADME_public_set_3521.csv``,
  github.com/molecularinformatics/Computational-ADME.
"""

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, ".cache", "raw")

TDC_IDS = {
    "caco2_wang": 4259569,
    "lipophilicity_astrazeneca": 4259595,
    "solubility_aqsoldb": 4259610,
    "ppbr_az": 6413140,
    "clearance_microsome_az": 4266186,
    "clearance_hepatocyte_az": 4266187,
    "ld50_zhu": 4267146,
    "half_life_obach": 4266799,
    "vdss_lombardo": 4267387,
}
TDC_URL = "https://dataverse.harvard.edu/api/access/datafile/{id}"

MOLECULEACE = [
    "CHEMBL1862_Ki", "CHEMBL1871_Ki", "CHEMBL2034_Ki", "CHEMBL2047_EC50",
    "CHEMBL204_Ki", "CHEMBL2147_Ki", "CHEMBL214_Ki", "CHEMBL218_EC50",
    "CHEMBL219_Ki", "CHEMBL228_Ki", "CHEMBL231_Ki", "CHEMBL233_Ki",
    "CHEMBL234_Ki", "CHEMBL235_EC50", "CHEMBL236_Ki", "CHEMBL237_EC50",
    "CHEMBL237_Ki", "CHEMBL238_Ki", "CHEMBL239_EC50", "CHEMBL244_Ki",
    "CHEMBL262_Ki", "CHEMBL264_Ki", "CHEMBL2835_Ki", "CHEMBL287_Ki",
    "CHEMBL2971_Ki", "CHEMBL3979_EC50", "CHEMBL4005_Ki", "CHEMBL4203_Ki",
    "CHEMBL4616_EC50", "CHEMBL4792_Ki",
]  # fmt: skip
MOLECULEACE_URL = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/main/"
    "MoleculeACE/Data/benchmark_data/{name}.csv"
)
BIOGEN_URL = (
    "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/"
    "main/ADME_public_set_3521.csv"
)


_AGENT = "lazyqsar-regression-benchmark/1.0 (+https://github.com/ersilia-os/lazy-qsar)"


def _get(url, dest):
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return "cached"
    tmp = dest + ".part"
    # Dataverse rejects urllib's default agent; identify the benchmark honestly instead.
    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    with urllib.request.urlopen(request, timeout=300) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, dest)
    return "downloaded"


def main():
    os.makedirs(RAW, exist_ok=True)
    manifest = {}
    for name, fid in TDC_IDS.items():
        dest = os.path.join(RAW, f"tdc_{name}.tab")
        manifest[dest] = _get(TDC_URL.format(id=fid), dest)
    for name in MOLECULEACE:
        dest = os.path.join(RAW, f"moleculeace_{name}.csv")
        manifest[dest] = _get(MOLECULEACE_URL.format(name=name), dest)
    dest = os.path.join(RAW, "biogen_adme.csv")
    manifest[dest] = _get(BIOGEN_URL, dest)
    print(json.dumps({os.path.basename(k): v for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
