import json
import tempfile
from types import SimpleNamespace

import numpy as np

from lazyqsar.portfolios.classification.portfolio import Portfolio
from lazyqsar.assemblers import classifier as classifier_module


def _profile(**overrides):
    base = dict(
        n_samples=1000,
        n_features=100,
        n_p_ratio=10.0,
        sparsity=0.0,
        is_sparse_counts=False,
        binary_feature_fraction=0.2,
        feature_signal_strength=0.06,
        feature_signal_p90=0.18,
        task="classification",
        imbalance_ratio=5.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_portfolio_tiny_dataset_adds_lr_but_keeps_xgb(monkeypatch):
    monkeypatch.setattr(
        "lazyqsar.portfolios.classification.portfolio.inspect_dataset",
        lambda X, y, task="classification": _profile(n_samples=200),
    )
    p = Portfolio()
    p.fit(np.zeros((200, 10)), np.array([0, 1] * 100))
    assert p.get() == ["lr", "xgb", "rf", "svc"]
    assert any("hard guard" in r for r in p.selector_reasons_)


def test_portfolio_high_dim_adds_lr_but_keeps_xgb(monkeypatch):
    monkeypatch.setattr(
        "lazyqsar.portfolios.classification.portfolio.inspect_dataset",
        lambda X, y, task="classification": _profile(
            n_samples=1000, n_features=2500, n_p_ratio=0.4
        ),
    )
    p = Portfolio()
    p.fit(np.zeros((1000, 10)), np.array([0, 1] * 500))
    assert p.get() == ["lr", "xgb", "rf", "svc"]


def test_portfolio_large_dense_signal_selects_xgb(monkeypatch):
    monkeypatch.setattr(
        "lazyqsar.portfolios.classification.portfolio.inspect_dataset",
        lambda X, y, task="classification": _profile(
            n_samples=10000,
            n_features=200,
            n_p_ratio=50.0,
            binary_feature_fraction=0.3,
            feature_signal_strength=0.08,
            feature_signal_p90=0.22,
        ),
    )
    p = Portfolio()
    p.fit(np.zeros((100, 10)), np.array([0, 1] * 50))
    assert p.get() == ["xgb", "rf"]
    assert any("skip lr" in r for r in p.selector_reasons_)


def test_portfolio_over_5000_samples_skips_lr_even_if_wide(monkeypatch):
    monkeypatch.setattr(
        "lazyqsar.portfolios.classification.portfolio.inspect_dataset",
        lambda X, y, task="classification": _profile(
            n_samples=6000,
            n_features=4000,
            n_p_ratio=1.1,
            sparsity=0.95,
            is_sparse_counts=True,
            binary_feature_fraction=0.98,
            feature_signal_strength=0.02,
            feature_signal_p90=0.09,
            imbalance_ratio=40.0,
        ),
    )
    p = Portfolio()
    p.fit(np.zeros((100, 10)), np.array([0, 1] * 50))
    assert p.get() == ["xgb", "rf"]
    assert any("skip lr" in r for r in p.selector_reasons_)


def test_portfolio_lr_win_still_keeps_xgb(monkeypatch):
    monkeypatch.setattr(
        "lazyqsar.portfolios.classification.portfolio.inspect_dataset",
        lambda X, y, task="classification": _profile(
            n_samples=1500,
            n_features=3000,
            n_p_ratio=1.2,
            sparsity=0.92,
            is_sparse_counts=True,
            binary_feature_fraction=0.95,
            feature_signal_strength=0.01,
            feature_signal_p90=0.08,
            imbalance_ratio=30.0,
        ),
    )
    p = Portfolio()
    p.fit(np.zeros((100, 10)), np.array([0, 1] * 50))
    assert p.get() == ["lr", "xgb", "rf", "svc"]
    assert any("hard guard" in r for r in p.selector_reasons_)


def test_portfolio_borderline_case_keeps_both(monkeypatch):
    monkeypatch.setattr(
        "lazyqsar.portfolios.classification.portfolio.inspect_dataset",
        lambda X, y, task="classification": _profile(
            n_samples=3000,
            n_features=2200,
            n_p_ratio=4.0,
            binary_feature_fraction=0.78,
            feature_signal_strength=0.04,
            feature_signal_p90=0.16,
            imbalance_ratio=25.0,
        ),
    )
    p = Portfolio()
    p.fit(np.zeros((100, 10)), np.array([0, 1] * 50))
    assert p.get() == ["lr", "xgb", "rf", "svc"]
    assert any("hard guard" in r for r in p.selector_reasons_)


def test_portfolio_save_load_roundtrip_structured(monkeypatch):
    monkeypatch.setattr(
        "lazyqsar.portfolios.classification.portfolio.inspect_dataset",
        lambda X, y, task="classification": _profile(),
    )
    p = Portfolio()
    p.fit(np.zeros((100, 10)), np.array([0, 1] * 50))

    with tempfile.TemporaryDirectory() as d:
        p.save(d)
        loaded = Portfolio.load(d)
        with open(f"{d}/portfolio.json", "r") as fh:
            payload = json.load(fh)

    assert isinstance(payload, dict)
    assert loaded.get() == p.get()
    assert loaded.selector_scores_ == p.selector_scores_
    assert loaded.selector_reasons_ == p.selector_reasons_


def test_portfolio_load_legacy_list_payload():
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/portfolio.json", "w") as fh:
            json.dump(["lr", "xgb"], fh)
        loaded = Portfolio.load(d)

    assert loaded.get() == ["lr", "xgb"]
    assert loaded.selector_version_ == "legacy"


def test_portfolio_defaults_include_xgb_and_rf():
    p = Portfolio()
    assert p.get() == ["xgb", "rf"]


def test_assembler_consumes_portfolio_get_unchanged(monkeypatch):
    class DummyPreprocessor:
        def fit(self, X, y):
            return None

        def transform(self, X):
            return X

        def save(self, directory):
            return None

    class DummyHead:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(
                decision_cutoff_=0.5,
                timing_={},
            )

        def fit(self, X, y):
            return None

        def predict_proba(self, X):
            p = np.full(X.shape[0], 0.4)
            return np.column_stack((1 - p, p))

        def predict_score(self, X):
            return self.predict_proba(X)

        def predict_rank(self, X):
            p = np.full(X.shape[0], 0.5)
            return np.column_stack((1 - p, p))

        def save(self, directory):
            return None

    class DummyInnerPooler:
        def __init__(self, portfolio):
            self.portfolio = portfolio

        def fit(self, S, y, X_prep=None):
            return None

        def get_weights(self, X_prep):
            return np.ones((len(X_prep), 1))

        def predict_proba(self, R, X_prep=None):
            return np.column_stack((1 - R[:, 0], R[:, 0]))

        def save(self, directory):
            return None

    class StubPortfolio:
        def fit(self, X, y):
            self._portfolio = ["lr"]

        def get(self):
            return self._portfolio

    monkeypatch.setattr(classifier_module, "Portfolio", StubPortfolio)
    monkeypatch.setattr(classifier_module, "Preprocessor", DummyPreprocessor)
    monkeypatch.setattr(classifier_module, "LRHead", DummyHead)
    monkeypatch.setattr(classifier_module, "RFHead", DummyHead)
    monkeypatch.setattr(classifier_module, "InnerPooler", DummyInnerPooler)

    model = classifier_module.LazyClassifier()
    X = np.zeros((20, 3), dtype=float)
    y = np.array([0, 1] * 10, dtype=int)
    model.fit(X, y)

    assert model.portfolio == ["lr"]
    assert len(model.models) == 1
    assert len(model.models[0].heads) == 1
