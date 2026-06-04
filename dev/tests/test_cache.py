"""
Tests for bounded cache behavior in LazyClassifierQSAR and ArtifactWrapper.

These tests use mode="fast" (Morgan fingerprints only) so they run without
DL model checkpoints. They verify that the caches do NOT grow with the
number of distinct SMILES batches seen.
"""

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

SMILES_TRAIN = [
    "CCO",
    "CCN",
    "CCC",
    "c1ccccc1",
    "CC(=O)O",
    "CC(=O)N",
    "c1ccncc1",
    "CC#N",
    "CCCO",
    "CCCN",
    "CC(C)O",
    "CC(C)N",
    "c1ccc(O)cc1",
    "c1ccc(N)cc1",
    "CC(=O)c1ccccc1",
    "CCOC(=O)C",
    "CCc1ccccc1",
    "c1ccc(CC)cc1",
    "CCCC",
    "CCCCO",
]
Y_TRAIN = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


def _distinct_batches(n: int = 10, size: int = 5):
    """Return n distinct SMILES batches of `size` molecules each."""
    base = ["CC" + "O" * (i + 1) for i in range(n * size)]
    return [base[i * size : (i + 1) * size] for i in range(n)]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained_qsar(tmp_path_factory):
    """Fit a LazyClassifierQSAR(mode='fast') on a tiny synthetic dataset."""
    from lazyqsar.qsar import LazyClassifierQSAR

    tmp = tmp_path_factory.mktemp("model")
    m = LazyClassifierQSAR(mode="fast")
    m.fit(smiles_list=SMILES_TRAIN, y=Y_TRAIN)
    m.save_onnx(str(tmp))
    return m, str(tmp)


@pytest.fixture(scope="module")
def artifact_wrapper(trained_qsar):
    """Load an ArtifactWrapper from the saved ONNX model."""
    from lazyqsar.qsar import LazyClassifierQSAR

    _, model_dir = trained_qsar
    return LazyClassifierQSAR.load_onnx(model_dir)


# ── ArtifactWrapper cache tests ───────────────────────────────────────────────


class TestArtifactWrapperCache:
    def test_cache_does_not_grow_unbounded(self, artifact_wrapper):
        """Calling predict_proba on N distinct batches must not accumulate N cache entries."""
        batches = _distinct_batches(n=20)
        for batch in batches:
            artifact_wrapper.predict_proba(batch)
        # single-entry cache: only the last key/value should be set
        assert artifact_wrapper._last_ensemble_key is not None
        assert artifact_wrapper._last_ensemble_value is not None
        # Crucially, there must be no unbounded dict holding old entries:
        assert not hasattr(artifact_wrapper, "_ensemble_cache"), (
            "_ensemble_cache dict must be removed; use _last_ensemble_key/value"
        )

    def test_same_batch_hits_cache(self, artifact_wrapper):
        """Two predict_* calls on identical SMILES must reuse the cached ensemble."""
        from unittest.mock import patch

        from lazyqsar import qsar as qsar_mod

        smi = ["CCO", "CCCO", "c1ccccc1"]
        artifact_wrapper.predict_proba(smi)  # populates cache

        hit_count = {"n": 0}
        original_build = qsar_mod._build_weight_matrix

        def counting_build(*args, **kwargs):
            hit_count["n"] += 1
            return original_build(*args, **kwargs)

        with patch.object(qsar_mod, "_build_weight_matrix", side_effect=counting_build):
            artifact_wrapper.predict_rank(smi)  # should NOT call _build_weight_matrix

        assert hit_count["n"] == 0, (
            "predict_rank on the same SMILES should reuse the cached ensemble, "
            "not recompute _build_weight_matrix"
        )

    def test_different_batch_evicts_cache(self, artifact_wrapper):
        """A distinct SMILES list must replace the cached entry, not append."""
        smi_a = ["CCO", "CCN"]
        smi_b = ["CCCO", "CCCN"]
        artifact_wrapper.predict_proba(smi_a)
        key_after_a = artifact_wrapper._last_ensemble_key
        artifact_wrapper.predict_proba(smi_b)
        key_after_b = artifact_wrapper._last_ensemble_key
        assert key_after_a != key_after_b

    def test_clear_cache_releases_memory(self, artifact_wrapper):
        """clear_cache() must reset both key and value to None."""
        artifact_wrapper.predict_proba(["CCO", "CCN"])
        assert artifact_wrapper._last_ensemble_key is not None
        artifact_wrapper.clear_cache()
        assert artifact_wrapper._last_ensemble_key is None
        assert artifact_wrapper._last_ensemble_value is None


# ── LazyClassifierQSAR cache tests ───────────────────────────────────────────


class TestLazyClassifierQSARCache:
    def test_feature_cache_bounded(self, trained_qsar):
        """_feature_cache must hold at most one entry per descriptor index."""
        m, _ = trained_qsar
        batches = _distinct_batches(n=30)
        for batch in batches:
            m.predict_proba(batch)
        n_descriptors = len(m.descriptor_types)
        assert len(m._feature_cache) <= n_descriptors, (
            f"_feature_cache must have at most {n_descriptors} entries (one per descriptor), "
            f"got {len(m._feature_cache)}"
        )

    def test_ensemble_cache_bounded(self, trained_qsar):
        """_last_ensemble_key/value must hold a single entry, not accumulate."""
        m, _ = trained_qsar
        batches = _distinct_batches(n=30)
        for batch in batches:
            m.predict_proba(batch)
        assert not hasattr(m, "_ensemble_cache"), (
            "_ensemble_cache dict must not exist; use _last_ensemble_key/value"
        )
        assert m._last_ensemble_key is not None

    def test_clear_cache(self, trained_qsar):
        m, _ = trained_qsar
        m.predict_proba(["CCO", "CCN"])
        m.clear_cache()
        assert m._last_ensemble_key is None
        assert m._last_ensemble_value is None
        assert len(m._feature_cache) == 0

    def test_fit_clears_caches(self, tmp_path):
        """Re-fitting must reset all cached state."""
        from lazyqsar.qsar import LazyClassifierQSAR

        m = LazyClassifierQSAR(mode="fast")
        m.fit(smiles_list=SMILES_TRAIN, y=Y_TRAIN)
        m.predict_proba(["CCO", "CCN"])
        assert m._last_ensemble_key is not None
        # fit again should reset
        m.fit(smiles_list=SMILES_TRAIN, y=Y_TRAIN)
        assert m._last_ensemble_key is None
        assert len(m._feature_cache) == 0
