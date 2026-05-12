# dev/data/

Raw datasets used by smoke tests and benchmarks. These files are **gitignored** and must be obtained locally before running dataset-dependent scripts.

> **Reproducibility notice:** Results from scripts that depend on these datasets may not be reproducible across environments or package versions.

## Datasets

| File | Description | Source |
|------|-------------|--------|
| `ames.tab` | Ames mutagenicity dataset (SMILES + label) | TDC — `ames` task |
| `clintox.tab` | Clinical toxicity dataset (SMILES + label) | TDC — `clintox` task |

## How to download

Install the TDC Python package and run the following snippet:

```python
from tdc.single_pred import Tox
import os

os.makedirs("dev/data", exist_ok=True)

for name in ["ames", "clintox"]:
    data = Tox(name=name)
    df = data.get_data()          # columns: Drug_ID, Drug, Y
    df.to_csv(f"dev/data/{name}.tab", sep="\t", index=False)
```

The resulting `.tab` files are tab-separated with columns `Drug_ID`, `Drug` (SMILES), and `Y` (binary label).
