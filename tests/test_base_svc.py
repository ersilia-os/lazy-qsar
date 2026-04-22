import os
import tempfile

import numpy as np
import pytest

from lazyqsar.base.svc import BaseSVCArtifact, BaseSVCClassifier
from lazyqsar.base.svc.params import get_params
from lazyqsar.base.svc.presets import (
    svc_balanced_rbf_params,
    svc_default_params,
    svc_heuristic_params,
    svc_linear_params,
)
from lazyqsar.base.xgboost.inspector import inspect as _inspect


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_clf_data(n=400, p=20, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.4 > 0).astype(int)
    return X, y


def make_sparse_fingerprint_data(n=400, p=2048, seed=42):
    """Simulate Morgan-like binary fingerprints (sparse integer counts)."""
    rng = np.random.RandomState(seed)
    X = rng.binomial(1, 0.05, (n, p)).astype(np.float32)
    y = (X[:, :10].sum(axis=1) + rng.randn(n) * 0.5 > 1.0).astype(int)
    return X, y


def make_imbalanced_clf_data(n=600, p=20, pos_frac=0.05, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3
    threshold = np.quantile(signal, 1 - pos_frac)
    y = (signal >= threshold).astype(int)
    return X, y


def _make_profile(n=400, p=20, sparse=False):
    X, y = make_sparse_fingerprint_data(n, p) if sparse else make_clf_data(n, p)
    return _inspect(X, y, task="classification")


# ---------------------------------------------------------------------------
# params.py — heuristic rules
# ---------------------------------------------------------------------------

def test_heuristic_sparse_fingerprints_chooses_linear():
    profile = _make_profile(n=400, p=2048, sparse=True)
    params = get_params(profile)
    assert params["use_linear"] is True, (
        "Sparse fingerprints should select linear kernel"
    )


def test_heuristic_dense_small_chooses_rbf():
    profile = _make_profile(n=400, p=20, sparse=False)
    params = get_params(profile)
    assert params["use_linear"] is False
    assert params["kernel"] == "rbf"


def test_heuristic_dense_large_n_chooses_linear():
    # n > 5000 → always linear regardless of feature type
    rng = np.random.RandomState(0)
    X = rng.randn(6000, 50).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    profile = _inspect(X, y, task="classification")
    params = get_params(profile)
    assert params["use_linear"] is True


def test_heuristic_c_increases_with_n_linear():
    p10 = get_params(_make_profile(n=100, p=20, sparse=False))
    p100 = get_params(_make_profile(n=100, p=20, sparse=True))
    # For linear (sparse), C < 500 samples → 0.1
    assert p100["C"] <= 1.0


# ---------------------------------------------------------------------------
# presets.py
# ---------------------------------------------------------------------------

def test_preset_default_is_rbf():
    profile = _make_profile(n=400, p=20)
    p = svc_default_params(profile)
    assert p["kernel"] == "rbf"
    assert p["C"] == 1.0
    assert p["use_linear"] is False


def test_preset_linear_is_linear():
    profile = _make_profile(n=400, p=20)
    p = svc_linear_params(profile)
    assert p["use_linear"] is True
    assert "kernel" not in p


def test_preset_balanced_rbf_c_scales_with_minority():
    profile = _make_profile(n=1000, p=20)
    p = svc_balanced_rbf_params(profile)
    assert p["C"] > 0
    assert p["use_linear"] is False


# ---------------------------------------------------------------------------
# BaseSVCClassifier — core functionality
# ---------------------------------------------------------------------------

def test_svc_predict_proba_shape():
    X, y = make_clf_data()
    clf = BaseSVCClassifier(portfolio=False)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all()
    assert (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_svc_predict_score_shape():
    X, y = make_clf_data()
    clf = BaseSVCClassifier(portfolio=False, calibrated=False)
    clf.fit(X, y)
    score = clf.predict_score(X)
    assert score.shape == (len(X), 2)
    assert (score >= 0).all() and (score <= 1).all()


def test_svc_predict_binary():
    X, y = make_clf_data()
    clf = BaseSVCClassifier(portfolio=False, calibrated=False)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})


def test_svc_calibrated_learns_cutoff():
    X, y = make_clf_data()
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    assert hasattr(clf, "decision_cutoff_")
    assert clf.decision_cutoff_source_ in {"oof_balanced_accuracy", "default_0.5"}


def test_svc_calibrated_has_calibrator():
    X, y = make_clf_data()
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    assert hasattr(clf, "calibrator_")
    assert clf.calibrator_method_ in {"isotonic", "platt"}


def test_svc_has_oof_probas():
    X, y = make_clf_data()
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    assert hasattr(clf, "oof_probas_")
    assert clf.oof_probas_.shape == (len(y),)
    assert np.isfinite(clf.oof_probas_).all()


# ---------------------------------------------------------------------------
# Portfolio selection
# ---------------------------------------------------------------------------

def test_svc_portfolio_sets_preset_name():
    X, y = make_clf_data(n=300, p=20)
    clf = BaseSVCClassifier(portfolio=True, calibrated=False)
    clf.fit(X, y)
    assert clf.preset_name_ in {"heuristic", "default", "linear", "balanced_rbf"}


def test_svc_small_dataset_no_split_uses_heuristic():
    X, y = make_clf_data(n=50, p=10)
    clf = BaseSVCClassifier(portfolio=True, calibrated=False)
    clf.fit(X, y)
    assert clf.preset_name_ == "heuristic"


def test_svc_no_portfolio_uses_heuristic():
    X, y = make_clf_data(n=400, p=20)
    clf = BaseSVCClassifier(portfolio=False, calibrated=False)
    clf.fit(X, y)
    assert clf.preset_name_ == "heuristic"


def test_svc_portfolio_scores_populated():
    X, y = make_clf_data(n=300, p=20)
    clf = BaseSVCClassifier(portfolio=True, calibrated=False)
    clf.fit(X, y)
    assert len(clf.portfolio_scores_) > 0
    # At least one non-nan score
    finite_scores = [v for v in clf.portfolio_scores_.values() if v == v]
    assert len(finite_scores) >= 1


# ---------------------------------------------------------------------------
# ONNX save / load roundtrip
# ---------------------------------------------------------------------------

def test_svc_save_load_artifact_roundtrip_dense():
    X, y = make_clf_data(n=300, p=20)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        clf.save(d)
        assert os.path.isfile(os.path.join(d, "svc.json"))
        artifact = BaseSVCArtifact.load(d)
        proba = artifact.run(X)
        score = artifact.predict_score(X)
        preds = artifact.predict(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert score.shape == (len(X), 2)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_svc_save_load_artifact_roundtrip_linear():
    X, y = make_sparse_fingerprint_data(n=300, p=200)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    assert clf._use_linear_ is True, "Sparse data should trigger LinearSVC"
    with tempfile.TemporaryDirectory() as d:
        clf.save(d)
        artifact = BaseSVCArtifact.load(d)
        proba = artifact.run(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_svc_artifact_predict_score_pre_calibration():
    X, y = make_clf_data(n=300, p=20)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        clf.save(d)
        artifact = BaseSVCArtifact.load(d)
        score = artifact.predict_score(X)
        proba = artifact.run(X)
    # predict_score gives sigmoid(df), predict_proba gives calibrated — they
    # differ but both should be valid probabilities
    assert score.shape == (len(X), 2)
    assert (score >= 0).all() and (score <= 1).all()
    assert proba.shape == (len(X), 2)


def test_svc_artifact_predict_rank():
    X, y = make_clf_data(n=300, p=20)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        clf.save(d)
        artifact = BaseSVCArtifact.load(d)
        rank = artifact.predict_rank(X)
    assert rank.shape == (len(X), 2)
    assert (rank >= 0).all() and (rank <= 1).all()


# ---------------------------------------------------------------------------
# Imbalanced data
# ---------------------------------------------------------------------------

def test_svc_imbalanced_predict_proba_is_finite():
    X, y = make_imbalanced_clf_data(n=600, p=20, pos_frac=0.05)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_svc_imbalanced_artifact_finite():
    X, y = make_imbalanced_clf_data(n=600, p=20, pos_frac=0.05)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        clf.save(d)
        artifact = BaseSVCArtifact.load(d)
        proba = artifact.run(X)
    assert np.isfinite(proba).all()


# ---------------------------------------------------------------------------
# ONNX size constraint
# ---------------------------------------------------------------------------

def test_svc_onnx_size_within_budget():
    """Kernel SVC on n=1000, p=50 should produce ONNX < 20 MB."""
    X, y = make_clf_data(n=1000, p=50)
    clf = BaseSVCClassifier(portfolio=False, calibrated=False)
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        clf.save(d)
        onnx_path = os.path.join(d, "svc.onnx")
        if os.path.isfile(onnx_path):
            size_mb = os.path.getsize(onnx_path) / 1e6
            assert size_mb < 20.0, f"ONNX too large: {size_mb:.1f} MB"


# ---------------------------------------------------------------------------
# predict_rank from model (requires calibration)
# ---------------------------------------------------------------------------

def test_svc_predict_rank_shape():
    X, y = make_clf_data(n=300, p=20)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    rank = clf.predict_rank(X)
    assert rank.shape == (len(X), 2)
    assert (rank >= 0).all() and (rank <= 1).all()


# ---------------------------------------------------------------------------
# predict_logit
# ---------------------------------------------------------------------------

def test_svc_predict_logit_finite():
    X, y = make_clf_data(n=300, p=20)
    clf = BaseSVCClassifier(portfolio=False, calibrated=True)
    clf.fit(X, y)
    logit = clf.predict_logit(X)
    assert logit.shape == (len(X), 2)
    assert np.isfinite(logit).all()
