"""
Tests for lazyqsar.base.xgboost.

Covers:
  - DatasetProfile fields and values
  - Parameter selection logic (rules, edge cases)
  - End-to-end fit / predict for classifier and regressor
  - Sparse matrix input
  - GPU path (param dict only, no actual GPU needed)
  - BaseXGBArtifact: ONNX + joblib save / load / run
"""

import json
import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

from lazyqsar.base.xgboost import BaseXGBClassifier, BaseXGBRegressor, BaseXGBArtifact
from lazyqsar.base.xgboost.inspector import DatasetProfile, inspect
from lazyqsar.base.xgboost.params import get_params

RNG = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def make_clf_data(n=500, p=20, imbalance=1.0, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    n_pos = int(n / (1 + imbalance))
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    return X, y


def make_reg_data(n=500, p=20, skew=0.0, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = rng.randn(n)
    if skew > 0:
        y = np.exp(y * skew)
    return X, y


def make_fingerprint_data(n=1000, p=1024, density=0.05, seed=0):
    """Sparse binary/count matrix mimicking Morgan fingerprints."""
    rng = np.random.RandomState(seed)
    X = (rng.random((n, p)) < density).astype(int)
    y = (X.sum(axis=1) > p * density).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Inspector: DatasetProfile
# ---------------------------------------------------------------------------


class TestInspector:
    def test_profile_fields_clf(self):
        X, y = make_clf_data(n=300, p=10)
        prof = inspect(X, y)
        assert prof.task == "classification"
        assert prof.n_samples == 300
        assert prof.n_features == 10
        assert pytest.approx(prof.n_p_ratio, abs=0.01) == 30.0
        assert 0.0 <= prof.sparsity <= 1.0
        assert 0.0 <= prof.binary_feature_fraction <= 1.0
        assert 0.0 <= prof.feature_signal_strength <= 1.0
        assert prof.imbalance_ratio >= 1.0

    def test_profile_fields_reg(self):
        X, y = make_reg_data(n=400, p=15)
        prof = inspect(X, y, task="regression")
        assert prof.task == "regression"
        assert isinstance(prof.y_skewness, float)
        assert isinstance(prof.y_all_positive, bool)

    def test_n_p_ratio(self):
        X, y = make_clf_data(n=200, p=100)
        prof = inspect(X, y)
        assert pytest.approx(prof.n_p_ratio, abs=0.01) == 2.0

    def test_imbalance_ratio(self):
        X, y = make_clf_data(n=1000, imbalance=9.0)
        prof = inspect(X, y)
        assert prof.imbalance_ratio > 5.0

    def test_sparse_counts_detected(self):
        X, y = make_fingerprint_data()
        prof = inspect(X, y)
        assert prof.is_sparse_counts is True
        assert prof.sparsity > 0.5

    def test_sparse_counts_not_detected_for_dense(self):
        X, y = make_clf_data(n=500, p=50)
        prof = inspect(X, y)
        assert prof.is_sparse_counts is False

    def test_binary_feature_fraction_all_binary(self):
        rng = np.random.RandomState(1)
        X = (rng.random((300, 50)) < 0.3).astype(float)
        y = (X.sum(axis=1) > 7).astype(int)
        prof = inspect(X, y)
        assert prof.binary_feature_fraction > 0.95

    def test_binary_feature_fraction_continuous(self):
        X, y = make_clf_data(n=400, p=40)
        prof = inspect(X, y)
        assert prof.binary_feature_fraction < 0.1

    def test_feature_signal_strength_range(self):
        X, y = make_clf_data(n=500, p=30)
        prof = inspect(X, y)
        assert 0.0 <= prof.feature_signal_strength <= 1.0
        assert 0.0 <= prof.feature_signal_p90 <= 1.0
        assert prof.feature_signal_p90 >= prof.feature_signal_strength

    def test_feature_signal_strength_noisy(self):
        rng = np.random.RandomState(7)
        X = rng.randn(500, 100)
        y = rng.randint(0, 2, size=500)
        prof = inspect(X, y)
        assert prof.feature_signal_strength < 0.15
        assert prof.feature_signal_p90 < 0.30

    def test_auto_task_detection_clf(self):
        X, y = make_clf_data()
        prof = inspect(X, y)
        assert prof.task == "classification"

    def test_auto_task_detection_reg(self):
        X, y = make_reg_data()
        prof = inspect(X, y)
        assert prof.task == "regression"

    def test_invalid_task_raises(self):
        X, y = make_clf_data()
        with pytest.raises(ValueError):
            inspect(X, y, task="multiclass")

    def test_sparse_matrix_input(self):
        X, y = make_fingerprint_data(n=500, p=512)
        X_sp = sp.csr_matrix(X)
        prof = inspect(X_sp, y)
        assert prof.n_samples == 500
        assert prof.n_features == 512
        assert prof.is_sparse_counts is True

    def test_repr_contains_key_fields(self):
        X, y = make_clf_data(n=300, p=10)
        prof = inspect(X, y)
        r = repr(prof)
        assert "n_p_ratio" in r
        assert "feature_signal_strength" in r
        assert "feature_signal_p90" in r
        assert "binary_feature_fraction" in r


# ---------------------------------------------------------------------------
# Parameter selection rules
# ---------------------------------------------------------------------------


class TestParams:
    def _prof(self, n, p, **kwargs):
        X = RNG.randn(n, p)
        y = (X[:, 0] > 0).astype(int)
        return inspect(X, y, **kwargs)

    # --- learning rate ---

    def test_lr_small_dataset(self):
        params = get_params(self._prof(500, 10))
        assert params["learning_rate"] == 0.1

    def test_lr_medium_dataset(self):
        params = get_params(self._prof(50_000, 10))
        assert params["learning_rate"] == 0.05

    def test_lr_large_dataset(self):
        params = get_params(self._prof(200_000, 10))
        assert params["learning_rate"] == 0.02

    # --- early stopping scales with lr ---

    def test_early_stopping_scales_with_lr(self):
        p_small = get_params(self._prof(500, 10))
        p_large = get_params(self._prof(200_000, 10))
        assert p_small["early_stopping_rounds"] < p_large["early_stopping_rounds"]
        assert p_large["early_stopping_rounds"] == pytest.approx(250, abs=1)

    def test_early_stopping_minimum(self):
        params = get_params(self._prof(500, 10))
        assert params["early_stopping_rounds"] >= 20

    # --- max_depth ---

    def test_max_depth_increases_with_n(self):
        d_small = get_params(self._prof(500, 5))["max_depth"]
        d_large = get_params(self._prof(200_000, 5))["max_depth"]
        assert d_large >= d_small

    def test_max_depth_capped_low_n_p_ratio(self):
        X = RNG.randn(100, 200)
        y = (X[:, 0] > 0).astype(int)
        params = get_params(inspect(X, y))
        assert params["max_depth"] <= 3

    def test_max_depth_capped_moderate_n_p_ratio(self):
        X = RNG.randn(300, 100)
        y = (X[:, 0] > 0).astype(int)
        params = get_params(inspect(X, y))
        assert params["max_depth"] <= 4

    def test_max_depth_reduced_for_binary_features(self):
        rng = np.random.RandomState(3)
        X_bin = (rng.random((5000, 100)) < 0.7).astype(float)
        X_cont = rng.randn(5000, 100)
        y = (X_bin.sum(axis=1) > 40).astype(int)
        prof_bin = inspect(X_bin, y)
        assert prof_bin.binary_feature_fraction > 0.8
        assert prof_bin.is_sparse_counts is False
        d_bin = get_params(prof_bin)["max_depth"]
        d_cont = get_params(inspect(X_cont, y))["max_depth"]
        assert d_bin <= d_cont

    # --- regularization ---

    def test_reg_lambda_higher_for_low_n_p_ratio(self):
        p_low = get_params(
            inspect(RNG.randn(100, 200), (RNG.randn(100) > 0).astype(int))
        )
        p_high = get_params(
            inspect(RNG.randn(5000, 20), (RNG.randn(5000) > 0).astype(int))
        )
        assert p_low["reg_lambda"] > p_high["reg_lambda"]

    def test_reg_alpha_nonzero_for_underdetermined(self):
        X = RNG.randn(100, 300)
        y = (X[:, 0] > 0).astype(int)
        params = get_params(inspect(X, y))
        assert params["reg_alpha"] > 0.0

    def test_reg_alpha_for_sparse_counts(self):
        X, y = make_fingerprint_data()
        params = get_params(inspect(X, y))
        assert params["reg_alpha"] > 0.0

    def test_reg_alpha_present_for_underdetermined_fingerprints(self):
        X, y = make_fingerprint_data(n=1000, p=1024, density=0.05)
        params = get_params(inspect(X, y))
        assert params["reg_alpha"] > 0.0

    def test_max_depth_not_penalized_for_ecfp(self):
        rng = np.random.RandomState(5)
        X_ecfp, y_ecfp = make_fingerprint_data(n=5000, p=512, density=0.05)
        X_dense_bin = (rng.random((5000, 512)) < 0.7).astype(float)
        y_dense = (X_dense_bin.sum(axis=1) > 180).astype(int)
        p_ecfp = inspect(X_ecfp, y_ecfp)
        p_dense = inspect(X_dense_bin, y_dense)
        assert p_ecfp.is_sparse_counts is True
        assert p_dense.is_sparse_counts is False
        params_ecfp = get_params(p_ecfp)
        d_dense = get_params(p_dense)["max_depth"]
        if params_ecfp.get("grow_policy") == "lossguide":
            assert params_ecfp["max_depth"] == 0
            assert "max_leaves" in params_ecfp
        else:
            assert params_ecfp["max_depth"] > d_dense

    def test_colsample_bytree_floor_for_sparse_counts(self):
        X, y = make_fingerprint_data(n=1000, p=2048, density=0.05)
        params = get_params(inspect(X, y))
        assert params["colsample_bytree"] >= 0.6

    def test_colsample_bynode_set_for_ecfp(self):
        X, y = make_fingerprint_data(n=1000, p=1024, density=0.05)
        params = get_params(inspect(X, y))
        assert "colsample_bynode" in params
        assert 0.05 <= params["colsample_bynode"] <= 0.3

    def test_colsample_bynode_not_set_for_low_dim(self):
        params = get_params(self._prof(1000, 100))
        assert "colsample_bynode" not in params

    # --- gamma ---

    def test_gamma_set_for_low_n_p_ratio(self):
        X = RNG.randn(200, 500)
        y = (X[:, 0] > 0).astype(int)
        params = get_params(inspect(X, y))
        assert params.get("gamma", 0) > 0

    def test_no_gamma_for_well_determined(self):
        X = RNG.randn(5000, 20)
        y = (X[:, 0] > 0).astype(int)
        params = get_params(inspect(X, y))
        assert params.get("gamma", 0) == 0

    # --- subsample ---

    def test_subsample_one_for_tiny_dataset(self):
        X = RNG.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        params = get_params(inspect(X, y))
        assert params["subsample"] == 1.0

    def test_subsample_normal_for_medium_dataset(self):
        params = get_params(self._prof(1000, 20))
        assert params["subsample"] == 0.8

    # --- num_parallel_tree ---

    def test_num_parallel_tree_for_small_n(self):
        params = get_params(self._prof(500, 20))
        assert params.get("num_parallel_tree", 1) == 3

    def test_num_parallel_tree_extended_to_n2000(self):
        params = get_params(self._prof(1500, 20))
        assert params.get("num_parallel_tree", 1) == 3

    def test_no_parallel_tree_for_large_n(self):
        params = get_params(self._prof(5000, 20))
        assert params.get("num_parallel_tree", 1) == 1

    def test_no_parallel_tree_for_tiny_n(self):
        params = get_params(self._prof(100, 10))
        assert params.get("num_parallel_tree", 1) == 1

    # --- max_bin ---

    def test_max_bin_64_for_sparse_counts(self):
        X, y = make_fingerprint_data()
        params = get_params(inspect(X, y))
        assert params["max_bin"] == 64

    def test_max_bin_256_for_normal(self):
        params = get_params(self._prof(1000, 50))
        assert params["max_bin"] == 256

    def test_max_bin_128_for_high_dim(self):
        params = get_params(self._prof(1000, 600))
        assert params["max_bin"] == 128

    # --- scale_pos_weight ---

    def test_scale_pos_weight_always_set(self):
        X, y = make_clf_data(n=500, p=20, imbalance=1.0)
        params = get_params(inspect(X, y))
        assert "scale_pos_weight" in params

    def test_scale_pos_weight_imbalanced(self):
        X, y = make_clf_data(n=1000, p=20, imbalance=9.0)
        params = get_params(inspect(X, y))
        assert params["scale_pos_weight"] > 5.0

    # --- eval_metric ---

    def test_eval_metric_aucpr_for_high_imbalance(self):
        X, y = make_clf_data(n=2000, p=20, imbalance=20.0)
        params = get_params(inspect(X, y))
        assert params["eval_metric"] == "aucpr"

    def test_eval_metric_auc_for_balanced(self):
        X, y = make_clf_data(n=500, p=20, imbalance=1.0)
        params = get_params(inspect(X, y))
        assert params["eval_metric"] == "auc"

    # --- regression objectives ---

    def test_regression_symmetric_uses_squarederror(self):
        X, y = make_reg_data(n=500, p=20, skew=0.0)
        params = get_params(inspect(X, y, task="regression"))
        assert params["objective"] == "reg:squarederror"

    def test_regression_skewed_positive_uses_tweedie(self):
        X, y = make_reg_data(n=500, p=20, skew=2.5)
        params = get_params(inspect(X, y, task="regression"))
        assert params["objective"] == "reg:tweedie"

    # --- GPU ---

    def test_gpu_sets_cuda_device(self):
        params = get_params(self._prof(500, 20), device="gpu")
        assert params["device"] == "cuda"

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError):
            get_params(self._prof(500, 20), device="tpu")


# ---------------------------------------------------------------------------
# End-to-end: fit / predict
# ---------------------------------------------------------------------------


class TestFitPredict:
    def test_classifier_fit_predict(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})
        assert preds.shape == (600,)

    def test_classifier_non_calibrated_fit_uses_default_decision_cutoff(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier(calibrated=False)
        clf.fit(X, y)
        assert clf.decision_cutoff_raw_ == 0.5
        assert clf.decision_cutoff_raw_source_ == "default_0.5"

    def test_classifier_calibrated_fit_learns_decision_cutoff(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier(calibrated=True)
        clf.fit(X, y)
        assert hasattr(clf, "decision_cutoff_raw_")
        assert clf.decision_cutoff_raw_source_ == "oof_balanced_accuracy"

    def test_classifier_predict_explicit_cutoff_overrides_default(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier(calibrated=True)
        clf.fit(X, y)
        preds_default = clf.predict(X)
        preds_loose = clf.predict(X, cutoff=0.0)
        assert preds_default.shape == preds_loose.shape
        assert np.all(preds_loose == 1)

    def test_classifier_predict_proba(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (600, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_regressor_fit_predict(self):
        X, y = make_reg_data(n=600, p=20)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (600,)
        assert np.isfinite(preds).all()

    def test_classifier_exposes_profile_and_params(self):
        X, y = make_clf_data(n=400, p=15)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        assert isinstance(clf.profile_, DatasetProfile)
        assert isinstance(clf.params_, dict)
        assert clf.best_iteration_ >= 0

    def test_regressor_exposes_profile_and_params(self):
        X, y = make_reg_data(n=400, p=15)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        assert isinstance(reg.profile_, DatasetProfile)
        assert isinstance(reg.params_, dict)
        assert reg.best_iteration_ >= 0

    def test_classifier_sparse_input(self):
        X, y = make_fingerprint_data(n=400, p=512)
        X_sp = sp.csr_matrix(X)
        clf = BaseXGBClassifier()
        clf.fit(X_sp, y)
        proba = clf.predict_proba(X_sp.toarray())
        assert proba.shape == (400, 2)

    def test_regressor_skewed_target(self):
        X, y = make_reg_data(n=500, p=20, skew=2.5)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        preds = reg.predict(X)
        assert np.isfinite(preds).all()

    def test_classifier_imbalanced(self):
        X, y = make_clf_data(n=1000, p=20, imbalance=9.0)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        assert clf.profile_.imbalance_ratio > 5.0
        assert clf.params_["scale_pos_weight"] > 5.0

    def test_classifier_tiny_dataset(self):
        X, y = make_clf_data(n=150, p=10)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        assert clf.params_["subsample"] == 1.0

    def test_high_dimensional_underdetermined(self):
        X = RNG.randn(200, 500)
        y = (X[:, 0] > 0).astype(int)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        assert clf.preset_name_ in {"heuristic", "default", "flaml", "autogluon"}
        assert clf.profile_.n_p_ratio < 1.0
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_preset_name_attribute(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        assert clf.preset_name_ in {"heuristic", "default", "flaml", "autogluon"}

    def test_preset_name_tiny_dataset_is_heuristic(self):
        X, y = make_clf_data(n=150, p=10)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        assert clf.preset_name_ == "heuristic"

    def test_predict_before_fit_raises(self):
        clf = BaseXGBClassifier()
        with pytest.raises(Exception):
            clf.predict(RNG.randn(10, 5))


# ---------------------------------------------------------------------------
# BaseXGBArtifact: save / load / run
# ---------------------------------------------------------------------------


class TestBaseXGBArtifact:
    # --- classifier ONNX ---

    def test_classifier_run_shape(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert out.shape == (600, 2)

    def test_classifier_run_probabilities_valid(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert (out >= 0).all() and (out <= 1).all()
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_classifier_run_matches_predict_proba(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        expected = clf.predict_proba(X)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert np.allclose(out, expected, atol=1e-4)

    def test_classifier_task_attribute(self):
        X, y = make_clf_data(n=400, p=15)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d)
            artifact = BaseXGBArtifact.load(d)
        assert artifact.task == "classification"

    def test_classifier_artifact_predict_uses_saved_cutoff(self):
        X, y = make_clf_data(n=400, p=15)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d)
            artifact = BaseXGBArtifact.load(d)
            preds = artifact.predict(X)
        assert preds.shape == (400,)
        assert set(preds).issubset({0, 1})

    def test_classifier_metadata_contents(self):
        X, y = make_clf_data(n=400, p=15)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d)
            artifact = BaseXGBArtifact.load(d)
        assert artifact.metadata["task"] == "classification"
        assert "preset_name" in artifact.metadata
        assert "best_iteration" in artifact.metadata
        assert "params" in artifact.metadata
        assert "profile" in artifact.metadata
        assert "portfolio_scores" in artifact.metadata

    # --- regressor ONNX ---

    def test_regressor_run_shape(self):
        X, y = make_reg_data(n=600, p=20)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            reg.save(d)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert out.shape == (600,)

    def test_regressor_run_finite(self):
        X, y = make_reg_data(n=600, p=20)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            reg.save(d)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert np.isfinite(out).all()

    def test_regressor_run_matches_predict(self):
        X, y = make_reg_data(n=600, p=20)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        expected = reg.predict(X)
        with tempfile.TemporaryDirectory() as d:
            reg.save(d)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert np.allclose(out, expected, atol=1e-4)

    def test_regressor_task_attribute(self):
        X, y = make_reg_data(n=400, p=15)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            reg.save(d)
            artifact = BaseXGBArtifact.load(d)
        assert artifact.task == "regression"

    # --- joblib path ---

    def test_classifier_joblib_run_shape(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d, onnx=False)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert out.shape == (600, 2)

    def test_classifier_joblib_run_matches_predict_proba(self):
        X, y = make_clf_data(n=600, p=20)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        expected = clf.predict_proba(X)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d, onnx=False)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert np.allclose(out, expected, atol=1e-6)

    def test_regressor_joblib_run_shape(self):
        X, y = make_reg_data(n=600, p=20)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            reg.save(d, onnx=False)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert out.shape == (600,)

    def test_regressor_joblib_run_matches_predict(self):
        X, y = make_reg_data(n=600, p=20)
        reg = BaseXGBRegressor()
        reg.fit(X, y)
        expected = reg.predict(X)
        with tempfile.TemporaryDirectory() as d:
            reg.save(d, onnx=False)
            artifact = BaseXGBArtifact.load(d)
            out = artifact.run(X)
        assert np.allclose(out, expected, atol=1e-6)

    def test_joblib_format_field_in_metadata(self):
        X, y = make_clf_data(n=400, p=15)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d, onnx=False)
            artifact = BaseXGBArtifact.load(d)
        assert artifact.metadata["format"] == "joblib"
        assert artifact._format == "joblib"

    def test_onnx_format_field_in_metadata(self):
        X, y = make_clf_data(n=400, p=15)
        clf = BaseXGBClassifier()
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d, onnx=True)
            artifact = BaseXGBArtifact.load(d)
        assert artifact.metadata["format"] == "onnx"
        assert artifact._format == "onnx"

    # --- error handling ---

    def test_load_missing_onnx_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "xgboost.json"), "w") as f:
                json.dump({"task": "classification", "format": "onnx"}, f)
            with pytest.raises(FileNotFoundError):
                BaseXGBArtifact.load(d)

    def test_load_missing_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "xgboost.onnx"), "wb") as f:
                f.write(b"")
            with pytest.raises(FileNotFoundError):
                BaseXGBArtifact.load(d)

    def test_artifact_predict_defaults_cutoff_when_missing_from_metadata(self):
        X, y = make_clf_data(n=300, p=12)
        clf = BaseXGBClassifier(calibrated=False)
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d, onnx=False)
            meta_path = os.path.join(d, "xgboost.json")
            with open(meta_path) as f:
                meta = json.load(f)
            meta.pop("decision_cutoff_raw", None)
            meta.pop("decision_cutoff_raw_source", None)
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            artifact = BaseXGBArtifact.load(d)
            preds = artifact.predict(X)
        assert artifact.decision_cutoff_raw == 0.5
        assert preds.shape == (300,)
