"""Tests for the optimised get_partition_indices."""
import numpy as np
from lazyqsar.utils.samplers import BinaryClassifierSamplingUtils


RNG = np.random.default_rng(42)


def make_dataset(n=2000, n_features=64, pos_ratio=0.2):
    X = RNG.standard_normal((n, n_features)).astype(np.float32)
    y = (RNG.random(n) < pos_ratio).astype(int).tolist()
    return X, y


def collect_partitions(X, y, **kwargs):
    su = BinaryClassifierSamplingUtils()
    defaults = dict(
        h5_file=None,
        h5_idxs=None,
        max_num_partitions=5,
        max_samples=500,
    )
    defaults.update(kwargs)
    return list(su.get_partition_indices(X=X, y=y, **defaults))


def test_partitions_are_valid():
    """Each partition has the right size and valid indices."""
    X, y = make_dataset(n=2000)
    partitions = collect_partitions(X, y)
    assert len(partitions) > 0, "No partitions returned"
    for p in partitions:
        assert len(p) >= 50, f"Partition too small: {len(p)}"
        assert len(p) <= 500, f"Partition too large: {len(p)}"
        assert all(0 <= idx < len(y) for idx in p), "Out-of-range index"
        n_pos = sum(y[i] for i in p)
        assert n_pos >= 10, f"Too few positives: {n_pos}"
    print(f"Partitions valid: {len(partitions)} partitions, sizes {[len(p) for p in partitions]}")


def test_no_duplicate_partitions():
    """All returned partitions must be unique."""
    X, y = make_dataset(n=2000)
    partitions = collect_partitions(X, y)
    tuples = [tuple(sorted(p)) for p in partitions]
    assert len(tuples) == len(set(tuples)), "Duplicate partitions returned"
    print("No duplicate partitions")


def test_partition_count():
    """Number of partitions respects max_num_partitions and dataset size."""
    X, y = make_dataset(n=3000, n_features=128)
    partitions = collect_partitions(X, y, max_num_partitions=10, max_samples=800)
    assert 0 < len(partitions) <= 10
    print(f"{len(partitions)} partitions generated")


if __name__ == "__main__":
    test_partitions_are_valid()
    test_no_duplicate_partitions()
    test_partition_count()
    print("All sampler tests passed.")
