"""
Tests for lazyqsar.base.preprocessing.

Covers:
  - PreprocessingProfile fields from inspect()
  - BaseClassifierPreprocessor / BaseRegressorPreprocessor fit + transform
  - Dimensionality reduction: n_features_out_ <= n_features_in_
  - kept_feature_indices_ validity
  - NaN imputation through the pipeline
  - ONNX save / load roundtrip via BasePreprocessorArtifact
  - FileNotFoundError on missing artifacts
"""

import os
import tempfile

import numpy as np
import pytest

from lazyqsar.base.preprocessing import (
    BaseClassifierPreprocessor,
    BaseRegressorPreprocessor,
    BasePreprocessorArtifact,
    PreprocessingProfile,
    inspect,
)


RNG = np.random.RandomState(7)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def make_clf_data(n=300, p=40, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.2 > 0).astype(int)
    return X, y


def make_reg_data(n=300, p=40, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float32)
    y = X[:, 0] * 2.0 + rng.randn(n) * 0.5
    return X, y


def make_fingerprint_data(n=500, p=512, density=0.05, seed=7):
    rng = np.random.RandomState(seed)
    X = (rng.random((n, p)) < density).astype(np.float32)
    y = (X.sum(axis=1) > p * density).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# PreprocessingProfile via inspect()
# ---------------------------------------------------------------------------


class TestPreprocessingProfile:
    def test_profile_fields_clf(self):
        X, y = make_clf_data(n=300, p=40)
        prof = inspect(X, y, task="classification")
        assert isinstance(prof, PreprocessingProfile)
        assert prof.task == "classification"
        assert prof.n_samples == 300
        assert prof.n_features == 40
        assert 0.0 <= prof.sparsity <= 1.0
        assert 0.0 <= prof.binary_feature_fraction <= 1.0

    def test_profile_fields_reg(self):
        X, y = make_reg_data(n=300, p=40)
        prof = inspect(X, y, task="regression")
        assert prof.task == "regression"
        assert prof.n_samples == 300
        assert prof.n_features == 40

    def test_binary_feature_fraction_for_fingerprints(self):
        X, y = make_fingerprint_data()
        prof = inspect(X, y, task="classification")
        assert prof.binary_feature_fraction > 0.9

    def test_binary_feature_fraction_for_continuous(self):
        X, y = make_clf_data()
        prof = inspect(X, y, task="classification")
        assert prof.binary_feature_fraction < 0.1

    def test_sparsity_high_for_fingerprints(self):
        X, y = make_fingerprint_data()
        prof = inspect(X, y, task="classification")
        assert prof.sparsity > 0.8


# ---------------------------------------------------------------------------
# BaseClassifierPreprocessor: fit + transform
# ---------------------------------------------------------------------------


class TestBaseClassifierPreprocessor:
    def test_fit_transform_shape(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        X_out = prep.fit_transform(X, y)
        assert X_out.ndim == 2
        assert X_out.shape[0] == 300
        assert X_out.shape[1] <= 40

    def test_n_features_out_le_in(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        assert prep.n_features_out_ <= prep.n_features_in_

    def test_n_features_in_correct(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        assert prep.n_features_in_ == 40

    def test_transform_shape_matches_fit(self):
        X_train, y = make_clf_data(n=200, p=40, seed=1)
        X_test, _ = make_clf_data(n=50, p=40, seed=2)
        prep = BaseClassifierPreprocessor()
        prep.fit(X_train, y)
        X_tr = prep.transform(X_test)
        assert X_tr.shape == (50, prep.n_features_out_)

    def test_profile_set_after_fit(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        assert isinstance(prep.profile_, PreprocessingProfile)
        assert prep.profile_.task == "classification"

    def test_scaler_name_set_after_fit(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        assert isinstance(prep.scaler_name_, str)
        assert len(prep.scaler_name_) > 0

    def test_reducer_name_set_after_fit(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        assert isinstance(prep.reducer_name_, str)
        assert len(prep.reducer_name_) > 0

    def test_kept_feature_indices_valid(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        indices = prep.kept_feature_indices_
        assert isinstance(indices, list)
        assert all(0 <= i < 40 for i in indices)

    def test_nan_imputation(self):
        X, y = make_clf_data(n=300, p=40)
        X[5, 3] = np.nan
        X[10, 10] = np.nan
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        X_out = prep.transform(X)
        assert not np.isnan(X_out).any()

    def test_constant_features_removed(self):
        X, y = make_clf_data(n=300, p=40)
        X[:, 7] = 0.0
        X[:, 15] = 1.0
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        assert prep.n_features_out_ < 40

    def test_transform_before_fit_raises(self):
        prep = BaseClassifierPreprocessor()
        X, _ = make_clf_data(n=10, p=5)
        with pytest.raises(Exception):
            prep.transform(X)


# ---------------------------------------------------------------------------
# BaseRegressorPreprocessor: fit + transform
# ---------------------------------------------------------------------------


class TestBaseRegressorPreprocessor:
    def test_fit_transform_shape(self):
        X, y = make_reg_data(n=300, p=40)
        prep = BaseRegressorPreprocessor()
        X_out = prep.fit_transform(X, y)
        assert X_out.ndim == 2
        assert X_out.shape[0] == 300
        assert X_out.shape[1] <= 40

    def test_profile_task_is_regression(self):
        X, y = make_reg_data(n=300, p=40)
        prep = BaseRegressorPreprocessor()
        prep.fit(X, y)
        assert prep.profile_.task == "regression"


# ---------------------------------------------------------------------------
# ONNX save / load via BasePreprocessorArtifact
# ---------------------------------------------------------------------------


class TestBasePreprocessorArtifact:
    def test_onnx_run_shape_matches_transform(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        X_sklearn = prep.transform(X[:10]).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            prep.save(d, onnx=True)
            artifact = BasePreprocessorArtifact.load(d)
            X_onnx = artifact.run(X[:10])
        assert X_onnx.shape == X_sklearn.shape

    def test_onnx_run_close_to_transform(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        X_sklearn = prep.transform(X).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            prep.save(d, onnx=True)
            artifact = BasePreprocessorArtifact.load(d)
            X_onnx = artifact.run(X)
        assert np.allclose(X_onnx, X_sklearn, atol=1e-4)

    def test_metadata_fields_in_json(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            prep.save(d, onnx=True)
            artifact = BasePreprocessorArtifact.load(d)
        assert artifact.task == "classification"
        assert artifact.n_features_in == 40
        assert artifact.n_features_out <= 40
        assert isinstance(artifact.kept_feature_indices, list)

    def test_load_missing_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                BasePreprocessorArtifact.load(d)

    def test_load_missing_model_file_raises(self):
        import json

        with tempfile.TemporaryDirectory() as d:
            meta = {
                "task": "classification",
                "scaler": "standard",
                "reducer": "variance_threshold",
                "n_features_in": 40,
                "n_features_out": 30,
                "kept_feature_indices": list(range(30)),
            }
            with open(os.path.join(d, "preprocessor.json"), "w") as f:
                json.dump(meta, f)
            with pytest.raises(FileNotFoundError):
                BasePreprocessorArtifact.load(d)

    def test_joblib_save_load_roundtrip(self):
        X, y = make_clf_data(n=300, p=40)
        prep = BaseClassifierPreprocessor()
        prep.fit(X, y)
        X_sklearn = prep.transform(X)
        with tempfile.TemporaryDirectory() as d:
            prep.save(d, onnx=False)
            artifact = BasePreprocessorArtifact.load(d)
            X_joblib = artifact.run(X)
        assert np.allclose(X_joblib, X_sklearn, atol=1e-6)
