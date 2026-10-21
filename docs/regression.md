# Regression in LazyQSAR: design and evidence

Every regression choice described here was made on a benchmark rather than carried over
from classification. This note records what was compared, what won, and by how much, so a
future change can be judged against the same evidence.

## The benchmark

`dev/benchmarks/regression/` downloads every dataset into a git-ignored cache; it needs
no package beyond LazyQSAR's own dependencies.

| Source | Tasks | Notes |
|---|---|---|
| TDC ADMET group | 9 | Caco2, lipophilicity, solubility, PPBR (human), LD50, VDss, half-life, hepatocyte and microsomal clearance; the last four log10-transformed |
| Biogen ADME (Fang et al. 2023) | 6 | HLM/RLM clearance, MDR1-MDCK efflux, solubility, human/rat PPB, in published log units |
| MoleculeACE (van Tilborg et al. 2022) | 30 | ChEMBL Ki/EC50 sets with an activity-cliff-aware split |

- **Splits:** random 80/20 (seeds 0–2), plus one structural split. That is a Bemis–Murcko scaffold split for TDC and Biogen, and MoleculeACE's own split.
- **Metrics:** RMSE, MAE, Spearman ρ and R² on the held-out rows, in the target's units.
- **Features:** fast mode (Morgan) unless stated otherwise.
- **Fit time:** measured against a `LazyClassifier` fitted on the same rows with the target binarised at its median.

| Script | What it does |
|---|---|
| `run.py` | Full LazyQSAR fits |
| `heads.py` | Each candidate preset of each head, fitted alone |
| `poolers.py` | Pooling rules and head subsets, scored offline on one set of head predictions |
| `tdc_official.py` | TDC's fixed `admet_group` split, with the leaderboard metrics |
| `ace_reference.py` | Reproduces MoleculeACE's best published method (SVR on ECFP with its per-dataset tuned hyperparameters: 0.671 RMSE / 0.751 on cliff compounds, as published) and runs LazyQSAR on the same splits |

## Heads: one preset each

Each preset was fitted alone on the preprocessed features of all 45 tasks, on `random-0`
and the structural split (90 task splits). In the table, "gap" is the mean test RMSE
above the best preset on each task and split.

| Head | Preset chosen | Gap | Best on | Alternatives (gap) |
|---|---|---|---|---|
| SVR | RBF, C=10, ε=0.05, γ='scale' | 0.29% | 50/86 | C=3 (0.49%), C=10 ε=0.1 (0.71%), Cherkassky–Ma analytic C/ε (1.86%), sklearn default C=1 (3.75%), LinearSVR (43.8%) |
| XGBoost | Probst, Boulesteix & Bischl (2019) tuned defaults: η 0.052, depth 11, min_child_weight 2, subsample 0.87, colsample_bytree 0.71, colsample_bylevel 0.64 | 1.08% | 41/90 | depth 7 (1.49%), η 0.05 + colsample 0.5 (1.95%), AutoGluon (4.44%), xgboost defaults (6.79%) |
| Random forest | max_features=1/3, min_samples_leaf=3 | 1.61% | 18/90 | sklearn default (1.46%, but trees 3.5× larger), Probst (1.67%), sqrt (2.62%), Breiman leaf 5 (4.18%), the 3.8 heuristic (5.48%) |
| Linear | RidgeCV with leave-one-out over α = n·logspace(−4, 2, 13) | 0.51% | 68/90 | ElasticNetCV (6.55%) |

**Why one preset and no race.** The 3.8 heads raced presets on holdout splits. The race
cost more fit time than it recovered, and for the SVR it sometimes picked the LinearSVR,
44% worse than the kernel. With one preset per head:
- XGBoost's number of rounds comes from early stopping on a quantile-stratified held-out
  split: 10% of the rows, 20% below 200 rows, and a fixed 100 rounds below 50 rows.
- The SVR's ONNX size guard can still halve C, or fall back to the LinearSVR, when the
  support vectors would make the graph larger than 5 MB.

**Above 2,000 rows**, XGBoost uses η 0.08 with depth 8. On the 32 benchmark splits of that
size this costs 0.8% RMSE on the head and 31% of its fit time. It was the cheapest of the
faster variants tried: η 0.1 cost 1.1% (0.58× time), depth 7 cost 0.9% (0.95×).

## Pooling: stacking, not a gate

Pooling rules were scored on the same heads' out-of-fold and test predictions, over 90
task splits. "RMSE vs gate" is the test RMSE relative to the 3.8 gate.

| Rule | RMSE vs gate | Better on | Mean ρ |
|---|---|---|---|
| Gaussian-likelihood Ridge gate (3.8) | — | — | 0.742 |
| Equal weights | −0.09% | 44/90 | 0.742 |
| **NNLS stacking on OOF predictions** | **−2.61%** | **76/90** | **0.756** |
| Best single head by OOF RMSE | −1.64% | 59/90 | 0.747 |
| Mean of the two best heads | −1.81% | 64/90 | 0.753 |

- **The gate added nothing.** Its per-sample weights all but collapse to uniform once
  softmaxed.
- **What replaces it:** non-negative least squares of the target on the heads' out-of-fold
  predictions, normalised to sum to one (Breiman 1996, "Stacked regressions").
  - Out-of-fold predictions are honest held-out predictions, so the weights are fitted on
    data no head trained on.
  - The checkpoint carries one weight per head.

## Which heads: by size

Every subset of the four heads was scored under the same stacking. The table shows RMSE
relative to all four heads.

| Rows | Best subset | Notes |
|---|---|---|
| ≤ 2,000 (62 splits) | lr + xgb + svr | +0.04%: the forest adds nothing once the kernel SVR is in |
| 2,000–5,000 (24 splits) | lr + xgb | −0.13% (better than all four): the SVR stops helping where its O(n²) kernel becomes the dominant cost |
| > 5,000 (4 splits) | xgb + rf | xgb alone is 0.9% worse; the SVR and the linear head are not fitted at this size |

No dataset-profile feature (sparsity, n/p, signal strength) changed the winning subset
once size was accounted for, so the regression selector has no profile scores.

## Out-of-fold predictions and speed

- **Folds:** out-of-fold predictions use 5 quantile-stratified folds up to 2,000 rows and
  3 folds above. Each head refits once per fold, so the folds were most of a large fit's
  cost.
- **Random forest above 5,000 rows:** its own out-of-bag predictions replace the fold
  refits. On the benchmark that cost 0.4% RMSE on average, mostly at small n, where folds
  remain.

Effect on the two large TDC tasks:
- solubility (7,984 rows): 163 s → 61 s, against about 55 s for classification;
- lipophilicity (3,360 rows): 46 s → 42 s, against 21 s for classification.

## Prediction intervals

The ensemble's pooled out-of-fold predictions give an honest residual for every training
row. Their split-conformal quantiles, with the finite-sample correction
(⌈(n+1)q⌉-th smallest), give an interval that covers a fraction q of molecules like the
training set.

- **Binning by applicability domain.** Residuals are split into tertiles of the
  molecule's mean applicability-domain score (a Mondrian conformal predictor), so coverage
  holds within each bin.
  - Molecules far from the training set take the widest bin's half-width.
  - With fewer than 90 training rows there is a single bin.
- **Storage.** Quantiles are stored at 101 levels and interpolated, so any coverage
  level can be requested at prediction time.
- **Feature level.** `LazyRegressor`, which has no applicability domain, uses a single
  bin.

Measured coverage of single-bin intervals on the 90 benchmark splits (stacked heads,
held-out test rows):

| Split | 0.80 | 0.90 | 0.95 |
|---|---|---|---|
| Random (45) | 0.806 | 0.903 | 0.952 |
| Scaffold / provided (45) | 0.774 | 0.879 | 0.935 |

On random splits coverage is nominal. On new scaffolds it falls 2–3 points short:
conformal coverage assumes test molecules resemble the training ones, and the
applicability-domain binning of the SMILES-level model is there to soften exactly that.

## Against published results

**MoleculeACE**, with the benchmark's own splits on all 30 datasets, in fast mode (Morgan only). Source: `ace_reference.py`.

| Method | RMSE | RMSE on cliff compounds |
|---|---|---|
| SVR on ECFP, the best published method (our reproduction) | 0.671 | 0.751 |
| GBM on ECFP (reproduction) | 0.698 | 0.787 |
| RF on ECFP (reproduction) | 0.700 | 0.785 |
| LazyQSAR 3.8 | 0.678 | 0.766 |
| **LazyQSAR now** | **0.664** | **0.752** |

**TDC ADMET group**, on the fixed test split with the leaderboard metrics (MAE ↓, or Spearman ↑ for the last four tasks). Fast mode, seed 0. Source: `tdc_official.py`.

| Task | 3.8 | Now |
|---|---|---|
| Caco2 (MAE) | 0.363 | **0.355** |
| Lipophilicity (MAE) | 0.644 | **0.634** |
| Solubility (MAE)¹ | — | 0.970 |
| PPBR (MAE) | 9.633 | **9.022** |
| LD50 (MAE) | 0.642 | **0.640** |
| VDss (ρ) | 0.616 | **0.642** |
| Half-life (ρ) | **0.550** | 0.516 |
| Hepatocyte clearance (ρ) | 0.412 | **0.419** |
| Microsomal clearance (ρ) | **0.574** | 0.547 |

¹ Two test SMILES do not parse in RDKit. LazyQSAR returns NaN for them, and the harness now scores them at the training mean; the 3.8 run predates that, so it has no score.

Slow mode (all five descriptors), measured with 3.8, reached 0.535 MAE on lipophilicity (MapLight: 0.539) and 0.297 on Caco2. It has not yet been rerun with the new portfolio.

## Results against 3.8

*To be filled from `run.py --label v2` (3 random + 1 structural split per task, with
classification fit time).*
