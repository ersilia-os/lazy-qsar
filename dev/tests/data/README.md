# dev/tests/data/

CSV and HDF5 train/test/validation splits used as fixtures by the pytest suite. These files are **gitignored** and must be regenerated locally.

> **Reproducibility notice:** Fixture generation depends on `dev/data/` raw datasets and the descriptor pipeline; results may differ across package versions.

## Contents

| Dataset | Files |
|---------|-------|
| Ames | `ames_train.csv`, `ames_test.csv`, `ames_train.h5`, `ames_test.h5` |
| ClinTox | `clintox_train.csv`, `clintox_test.csv`, `clintox_valid.csv`, `clintox_train.h5`, `clintox_test.h5`, `clintox_valid.h5` |
| Bioavailability MA | `bioavailability_ma_train.csv`, `bioavailability_ma_test.csv`, `bioavailability_ma_train.h5`, `bioavailability_ma_test.h5` |

## How to regenerate

The fixtures are pre-computed feature arrays (Morgan fingerprints or similar) derived from the raw datasets in `dev/data/`. Refer to the smoke scripts in `dev/smoke/` for the feature extraction pipeline used to produce them.
