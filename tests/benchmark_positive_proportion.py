"""Show how min/max_positive_proportion clamps the positive ratio in partitions."""
import numpy as np
from lazyqsar.utils.samplers import BinaryClassifierSamplingUtils

RNG = np.random.default_rng(42)
N = 500_000
N_FEATURES = 8

POS_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
MIN_POS_PROP = 0.10
MAX_POS_PROP = 0.50


def make_dataset(pos_ratio):
    X = RNG.standard_normal((N, N_FEATURES)).astype(np.float32)
    y = (RNG.random(N) < pos_ratio).astype(int).tolist()
    return X, y


def run(pos_ratio):
    X, y = make_dataset(pos_ratio)
    su = BinaryClassifierSamplingUtils(estimate_auc=False)
    partitions = list(su.get_partition_indices(X=X, y=y, h5_file=None, h5_idxs=None))
    actual_ratios = [sum(y[i] for i in p) / len(p) for p in partitions]
    return len(partitions), len(partitions[0]), np.mean(actual_ratios)


if __name__ == "__main__":
    print(f"N={N:,}  min_pos_prop={MIN_POS_PROP}  max_pos_prop={MAX_POS_PROP}")
    print()
    print(f"{'dataset pos%':>13}  {'partitions':>10}  {'part size':>10}  {'actual pos%':>12}")
    print("-" * 54)
    for r in POS_RATIOS:
        n_parts, part_size, actual = run(r)
        print(f"{r:>13.0%}  {n_parts:>10}  {part_size:>10,}  {actual:>12.1%}")
