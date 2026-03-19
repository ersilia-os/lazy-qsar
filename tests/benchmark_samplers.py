"""Benchmark get_partition_indices across dataset sizes from 100 to 1M."""
import time
import numpy as np
from lazyqsar.utils.samplers import BinaryClassifierSamplingUtils

RNG = np.random.default_rng(42)

SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]


def make_dataset(n, n_features=64, pos_ratio=0.2):
    X = RNG.standard_normal((n, n_features)).astype(np.float32)
    y = (RNG.random(n) < pos_ratio).astype(int).tolist()
    return X, y


def run(n):
    X, y = make_dataset(n)
    su = BinaryClassifierSamplingUtils()
    t0 = time.perf_counter()
    partitions = list(su.get_partition_indices(
        X=X,
        y=y,
        h5_file=None,
        h5_idxs=None,
    ))
    elapsed = time.perf_counter() - t0
    part_size = len(partitions[0]) if partitions else 0
    return elapsed, len(partitions), part_size


if __name__ == "__main__":
    print(f"{'n':>10}  {'time (s)':>10}  {'partitions':>10}  {'part size':>10}")
    print("-" * 48)
    for n in SIZES:
        elapsed, n_parts, part_size = run(n)
        print(f"{n:>10,}  {elapsed:>10.3f}  {n_parts:>10}  {part_size:>10,}")
