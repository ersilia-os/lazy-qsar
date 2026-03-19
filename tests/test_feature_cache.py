"""Verify that descriptor caching works: same SMILES list is not re-transformed."""
import hashlib
import numpy as np
import time

from lazyqsar.descriptors.morgan import MorganFingerprint
from lazyqsar.descriptors.rdkit_descriptors import RDKitDescriptor


# Minimal stand-in that exercises the same caching logic as LazyBinaryQSAR
class _CacheHost:
    def __init__(self, descriptors):
        self.descriptor_types = [type(d).__name__ for d in descriptors]
        self.descriptors = descriptors
        self._feature_cache = {}

    def _smiles_hash(self, smiles_list):
        h = hashlib.md5()
        for s in smiles_list:
            h.update(s.encode())
        return h.hexdigest()

    def _transform_cached(self, i, smiles_list):
        key = (i, self._smiles_hash(smiles_list))
        if key not in self._feature_cache:
            self._feature_cache[key] = self.descriptors[i].transform(smiles_list)
        return self._feature_cache[key]


SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC(=O)NC1=CC=C(C=C1)O", "CCCCCCCCCCCCCCCC(=O)O",
    "c1ccc2c(c1)ccc3cccc4cccc2c34",
]


def test_cache_is_hit():
    """Second transform call on the same SMILES should return cached array."""
    host = _CacheHost([MorganFingerprint(), RDKitDescriptor()])

    # First call — populates cache
    X0_first = host._transform_cached(0, SMILES)
    X1_first = host._transform_cached(1, SMILES)
    assert len(host._feature_cache) == 2

    # Second call — should return the exact same object (cache hit)
    X0_second = host._transform_cached(0, SMILES)
    X1_second = host._transform_cached(1, SMILES)
    assert X0_second is X0_first, "Cache miss: Morgan array was recomputed"
    assert X1_second is X1_first, "Cache miss: RDKit array was recomputed"
    print("Cache hit confirmed (same object returned)")


def test_different_smiles_not_cached():
    """Different SMILES should produce a different cache entry."""
    host = _CacheHost([MorganFingerprint()])

    X_a = host._transform_cached(0, SMILES[:5])
    X_b = host._transform_cached(0, SMILES[5:])
    assert len(host._feature_cache) == 2, "Expected two distinct cache entries"
    assert X_a.shape[0] == 5
    assert X_b.shape[0] == 5
    print("Distinct SMILES lists stored as separate cache entries")


def test_cache_speedup():
    """Cached call should be orders of magnitude faster than a fresh transform."""
    host = _CacheHost([RDKitDescriptor()])

    t0 = time.perf_counter()
    host._transform_cached(0, SMILES)
    t_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    host._transform_cached(0, SMILES)
    t_warm = time.perf_counter() - t0

    print(f"Cold: {t_cold:.3f}s  Warm (cached): {t_warm:.6f}s  Speedup: {t_cold/t_warm:.0f}x")
    assert t_warm < t_cold / 10, "Cached call should be at least 10x faster"


if __name__ == "__main__":
    test_cache_is_hit()
    test_different_smiles_not_cached()
    test_cache_speedup()
    print("All cache tests passed.")
