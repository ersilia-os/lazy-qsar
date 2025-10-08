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
For this benchmark we have used the default pipeline in LazyQSAR comparing the two built-in descriptors, Chemeleon and Morgan.

## Results
We have used the automated evaluation provided by TDC. We recommend checking those values against the [TDC Leaderboards](https://tdcommons.ai/benchmark/overview/). We add here the top benchmark from July 2025 as a reference.


#### Classification tasks:
| Dataset    | Metric | Benchmark | Chemeleon | Morgan
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Bioavailability_Ma   | AUROC | 0.748 ± 0.033 | 0.670 | 0.647 |
| HIA_Hou  | AUROC | 0.989 ± 0.001 |  0.967 | 0.977 |
| Pgp_Broccatelli | AUROC | 0.938 ± 0.002 | 0.900 | 0.905 |
| BBB_Martins   | AUROC | 0.916 ± 0.001| 0.913 | 0.834 |
| CYP2C9_Veith   | AUPRC | 0.859 ± 0.001 | 0.772 | 0.732 |
| CYP2D6_Veith  | AUPRC | 0.790 ± 0.001 | 0.688 | 0.624 |
| CYP3A4_Veith   | AUPRC | 0.916 ± 0.000 | 0.853 | 0.831 |
| CYP2C9_Substrate_CarbonMangels   | AUPRC | 0.441 ± 0.033 | 0.406 | 0.440 |
| CYP2D6_Substrate_CarbonMangels   | AUPRC | 0.736 ± 0.024 | 0.717 | 0.682 |
| CYP3A4_Substrate_CarbonMangels   | AUPRC | 0.662 ± 0.031 | 0.628 | 0.620 |
| hERG   | AUROC | 0.880 ± 0.002 | 0.802 | 0.806 |
| AMES   | AUROC | 0.871 ± 0.002 | 0.846 | 0.814 |
| DILI   | AUROC | 0.925 ± 0.005 | 0.922 | 0.871 |