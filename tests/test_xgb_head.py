import os
import tempfile
import numpy as np
import pytest

from lazyqsar.heads.binary_classification import xgb
from lazyqsar.assemblers.eclectic_binary_classifier import (
    derive_shape_policy,
    BaseEclecticBinaryClassifier,
    ALL_HEADS,
    HEAD_FAMILY,
)


def make_data(n_samples=300, n_features=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = (rng.normal(size=n_samples) > 0).astype(int)
    return X, y


def test_xgb_available_in_all_heads():
    assert "xgb" in ALL_HEADS


def test_xgb_family_is_boosting():
    assert HEAD_FAMILY.get("xgb") == "boosting"


def test_xgb_in_small_dense_profile():
    X, y = make_data(500, 40)
    policy = derive_shape_policy(X, y)
    assert policy.profile == "small"
    assert policy.is_sparse is False
    assert "xgb" in policy.candidate_heads


def test_xgb_in_medium_dense_profile():
    X, y = make_data(3000, 40)
    y[:1500] = 1
    y[1500:] = 0
    policy = derive_shape_policy(X, y)
    assert policy.profile == "medium"
    assert "xgb" in policy.candidate_heads


def test_xgb_not_in_sparse_profile():
    X = np.zeros((500, 2048), dtype=np.float32)
    X[:, :50] = 1.0
    y = np.zeros(500, dtype=int)
    y[:120] = 1
    policy = derive_shape_policy(X, y)
    assert policy.is_sparse is True
    assert "xgb" not in policy.candidate_heads


def test_find_params_returns_expected_keys():
    X, y = make_data(300, 20)
    params = xgb.find_params(X, y)
    assert "portfolio" in params
    assert "cv_score" in params
    assert params["cv_score"] is None  # no CV — ZSX is self-tuning


def test_head_fit_predict():
    X, y = make_data(300, 20)
    params = xgb.find_params(X, y)
    head = xgb.Head(**params)
    head.fit(X, y)
    proba = head.predict_proba(X)
    assert proba.shape == (300, 2)
    preds = head.predict(X)
    assert set(preds).issubset({0, 1})


def test_head_save_load_roundtrip():
    X, y = make_data(300, 20)
    params = xgb.find_params(X, y)
    head = xgb.Head(**params)
    head.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        head.save("xgb", tmpdir)
        loaded = xgb.Head.load("xgb", tmpdir)

    np.testing.assert_allclose(
        head.predict_proba(X),
        loaded.predict_proba(X),
        rtol=1e-5,
    )
    assert loaded.input_dim == X.shape[1]
    assert loaded.score == head.score


def test_convert_to_onnx():
    X, y = make_data(300, 20)
    params = xgb.find_params(X, y)
    head = xgb.Head(**params)
    head.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        head.save("xgb", tmpdir)
        onnx_path = xgb.convert_to_onnx("xgb", tmpdir)
        assert os.path.exists(onnx_path)

        import onnxruntime as rt
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: X})[0]
        assert out.shape == (300,)
        assert np.all((out >= 0) & (out <= 1))

        sklearn_proba = head.predict_proba(X)[:, 1]
        np.testing.assert_allclose(out, sklearn_proba, rtol=1e-4, atol=1e-4)


def test_full_pipeline_includes_xgb():
    """End-to-end: fit BaseEclecticBinaryClassifier on small dense data, xgb should appear."""
    X, y = make_data(600, 40)
    y[:300] = 1
    y[300:] = 0

    model = BaseEclecticBinaryClassifier({"max_heads": 4})
    model.find_params(X, y)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (600, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # xgb should have been a candidate (small dense profile)
    assert "xgb" in list(model.shape_policy["candidate_heads"])
