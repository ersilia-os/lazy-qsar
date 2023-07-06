## Benchmark
This folder contains the benchmarking of Lazy-QSAR for Morgan and Ersilia Compound Embeddings.

## Datasets
We have used the [ADMET dataset](https://tdcommons.ai/single_pred_tasks/adme/) from the Therapeutics Data Commons initiative.
It can be easily accessed by:

```python
from tdc.benchmark_group import admet_group
group = admet_group(path = 'data/')
```

## Lazy-QSAR
For this benchmark we have restricted FLAML to optimize the hyperparameters of a Random Forest classifier or regressor with a time cap of 10 minutes. Results of the 5-fold crossvalidation can be found in the /data subfolder.

## Results
We have used the automated evaluation provided by TDC. We recommend checking those values against the [TDC Leaderboards](https://tdcommons.ai/benchmark/overview/).

#### Classification tasks:
| Dataset    | Metric | Score Morgan | Score Eosce | 
| ----------- | ----------- | ----------- | ----------- |
| Bioavailability_Ma   | AUROC | 0.671 ± 0.017 | 0.695 ± 0.028 |
| HIA_Hou  | AUROC | 0.929 ± 0.011 | 0.896 ± 0.04 |
| Pgp_Broccatelli | AUROC | 0.886 ± 0.017 |  0.902 ± 0.02 |
| BBB_Martins   | AUROC | 0.864 ± 0.01| 0.845 ± 0.021|
| CYP2C9_Veith   | AUPRC | 0.766 ± 0.002 | 0.735 ± 0.032 |
| CYP2D6_Veith  | AUPRC | 0.662 ± 0.007 | 0.619 ± 0.048 |
| CYP3A4_Veith   | AUPRC | 0.842 ± 0.007 | 0.826 ± 0.018 |
| CYP2C9_Substrate_CarbonMangels   | AUPRC | 0.352 ± 0.053 | 0.378 ± 0.055 |
| CYP2D6_Substrate_CarbonMangels   | AUPRC | 0.636 ± 0.041 | 0.656 ± 0.036 |
| CYP3A4_Substrate_CarbonMangels   | AUPRC | 0.627 ± 0.024 | 0.62 ± 0.02 |
| hERG   | AUROC | 0.842 ± 0.016 | 0.823 ± 0.025 |
| AMES   | AUROC | 0.848 ± 0.005 | 0.816 ± 0.033 |
| DILI   | AUROC | 0.863 ± 0.033 | 0.859 ± 0.03 |

#### Regression tasks:
| Dataset    | Metric | Score Morgan | Score Eosce | 
| ----------- | ----------- | ----------- | ----------- |
| Caco2_Wang   | MAE | 0.35 ± 0.018 | 0.377 ± 0.031  
| Lipophilicity_Astrazeneca   | MAE | 0.733 ± 0.007 | 0.751 ± 0.019 |
| Solubility_Aqsoldb | MAE | 1.03 ± 0.006 | 1.067 ± 0.037 |
| PPBR_Az   | MAE | 9.546 ± 0.282 | 9.604 ± 0.256 |
| VDSS_Lombardo   | Spearman | 0.348 ± 0.091 | 0.28 ± 0.097 |
| Half_Life_Obach  | Spearman | 0.049 ± 0.124 | 0.069 ± 0.105 |
| Clearance_Microsome_Az   | Spearman | 0.417 ± 0.02 | 0.417 ± 0.015 |
| Clearance_Hepatocyte_Az   | Spearman | 0.316 ± 0.015 | 0.298 ± 0.021 |
| Ld50_zhu   | MAE | 0.643 ± 0.001 | 0.668 ± 0.025 |