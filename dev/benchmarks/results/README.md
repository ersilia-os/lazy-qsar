# dev/benchmarks/results/

Generated benchmark outputs (PNGs, CSVs, text summaries). These files are **gitignored** and must be regenerated locally.

> **Reproducibility notice:** Benchmark outputs depend on external datasets, optional descriptor dependencies, and hardware. Results may vary across environments and package versions.

## How to regenerate

```bash
# From the repo root, with [fit] and [descriptors] extras installed:
python dev/benchmarks/run_benchmark.py --data-dir dev/data/

# Or open and run the notebook interactively:
jupyter notebook dev/benchmarks/benchmark_classifier.ipynb
```

Raw datasets (`dev/data/*.tab`) must be present — see [dev/data/README.md](../data/README.md).
