import numpy as np

from lazyqsar.assemblers.eclectic_binary_classifier import derive_shape_policy


def make_dense(n_samples, n_features):
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_samples, n_features)).astype(np.float32)


def make_sparse_like(n_samples, n_features):
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    X[:, : max(1, n_features // 20)] = 1.0
    return X


def make_y(n_samples, positives):
    y = np.zeros(n_samples, dtype=int)
    y[:positives] = 1
    return y


def test_shape_policy_routes_tiny_profile():
    X = make_dense(120, 64)
    y = make_y(120, 20)
    policy = derive_shape_policy(X, y)
    assert policy.profile == "tiny"
    assert list(policy.candidate_heads) == ["lr"]
    assert policy.max_heads == 1


def test_shape_policy_routes_small_sparse_profile():
    X = make_sparse_like(500, 2048)
    y = make_y(500, 120)
    policy = derive_shape_policy(X, y, is_sparse=True)
    assert policy.profile == "small"
    assert policy.is_sparse is True
    assert list(policy.candidate_heads) == ["lr", "svc"]


def test_shape_policy_routes_medium_dense_profile():
    X = make_dense(4000, 128)
    y = make_y(4000, 1000)
    policy = derive_shape_policy(X, y)
    assert policy.profile == "medium"
    assert policy.is_sparse is False
    assert "et" in policy.candidate_heads
    assert "xgb" in policy.candidate_heads


def test_shape_policy_routes_large_profile():
    X = make_dense(25000, 128)
    y = make_y(25000, 8000)
    policy = derive_shape_policy(X, y)
    assert policy.profile == "large"
    assert "et" in policy.candidate_heads
    assert "xgb" in policy.candidate_heads
