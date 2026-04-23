import numpy as np
import pytest

from lazyqsar.assemblers import classifier as classifier_module


def test_batch_lazy_classifier_rejects_unknown_head():
    with pytest.raises(ValueError, match="Unknown head svm"):
        classifier_module._BatchLazyClassifier(portfolio=["svm"])


def test_batch_lazy_classifier_weighted_proba_and_predict(monkeypatch):
    class DummyPreprocessor:
        def __init__(self):
            pass

        def fit(self, X, y):
            return None

        def transform(self, X):
            return X

        def save(self, directory):
            return None

    class DummyLRHead:
        class _model:
            decision_cutoff_ = 0.5

        def __init__(self, **kwargs):
            self.model = DummyLRHead._model()

        def fit(self, X, y):
            return None

        def predict_proba(self, X):
            p = np.array([0.1, 0.8], dtype=float)
            return np.column_stack((1 - p, p))

        def predict_score(self, X):
            return self.predict_proba(X)

        def predict_rank(self, X):
            r = np.full(X.shape[0], 0.5)
            return np.column_stack((1 - r, r))

        def save(self, directory):
            return None

    class DummyXGBHead:
        class _model:
            decision_cutoff_ = 0.5

        def __init__(self, **kwargs):
            self.model = DummyXGBHead._model()

        def fit(self, X, y):
            return None

        def predict_proba(self, X):
            p = np.array([0.3, 0.4], dtype=float)
            return np.column_stack((1 - p, p))

        def predict_score(self, X):
            return self.predict_proba(X)

        def predict_rank(self, X):
            r = np.full(X.shape[0], 0.5)
            return np.column_stack((1 - r, r))

        def save(self, directory):
            return None

    class DummyInnerPooler:
        def __init__(self, portfolio):
            self.portfolio = portfolio
            self.weights = [0.5, 0.5]

        def fit(self, S, y, X_prep=None):
            return None

        def get_weights(self, X_prep):
            # return equal weights
            n = len(X_prep)
            return np.full((n, 2), 0.5)

        def predict_proba(self, R, X_prep=None):
            # fixed 25/75 blend for sample 0, 60/40 for sample 1
            p = np.array(
                [0.25 * R[0, 0] + 0.75 * R[0, 1], 0.60 * R[1, 0] + 0.40 * R[1, 1]]
            )
            return np.column_stack([1 - p, p])

        def save(self, directory):
            return None

    monkeypatch.setattr(classifier_module, "Preprocessor", DummyPreprocessor)
    monkeypatch.setattr(classifier_module, "LRHead", DummyLRHead)
    monkeypatch.setattr(classifier_module, "XGBHead", DummyXGBHead)
    monkeypatch.setattr(classifier_module, "InnerPooler", DummyInnerPooler)

    model = classifier_module._BatchLazyClassifier(portfolio=["lr", "xgb"])
    X = np.zeros((2, 3), dtype=float)
    y = np.array([0, 1], dtype=int)
    model.fit(X, y)

    proba = model.predict_proba(X)
    expected = np.array([[0.75, 0.25], [0.36, 0.64]], dtype=float)

    assert np.allclose(proba, expected)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(model.predict(X), np.array([0, 1], dtype=int))


def test_lazy_classifier_batches_and_averages_predictions(monkeypatch):
    class DummyPortfolio:
        def fit(self, X, y):
            self._fitted = (X.copy(), y.copy())

        def get(self):
            return ["lr"]

    class DummyBatchLazyClassifier:
        counter = 0

        def __init__(self, portfolio, **kwargs):
            self.portfolio = portfolio
            self.idx = DummyBatchLazyClassifier.counter
            DummyBatchLazyClassifier.counter += 1
            self.fit_X = None
            self.fit_y = None

        def fit(self, X, y):
            self.fit_X = X.copy()
            self.fit_y = y.copy()
            self.train_prior_ = float(np.mean(y == 1))
            self.decision_cutoff_ = 0.5
            self.decision_cutoff_proba_ = 0.5
            self.decision_cutoff_rank_ = 0.5
            self.decision_cutoff_logit_ = 0.0

        def predict_proba(self, X):
            positive = np.full(X.shape[0], 0.2 + 0.2 * self.idx, dtype=float)
            return np.column_stack((1 - positive, positive))

        def predict_score(self, X):
            return self.predict_proba(X)

        def predict_rank(self, X):
            r = np.full(X.shape[0], 0.5)
            return np.column_stack((1 - r, r))

    monkeypatch.setattr(classifier_module, "Portfolio", DummyPortfolio)
    monkeypatch.setattr(
        classifier_module, "_BatchLazyClassifier", DummyBatchLazyClassifier
    )

    X = np.arange(15, dtype=float).reshape(5, 3)
    y = np.array([0, 1, 0, 1, 1], dtype=int)
    model = classifier_module.LazyClassifier(max_batch_size=2)
    model.fit(X, y)

    assert len(model.models) == 3
    assert [m.fit_X.shape[0] for m in model.models] == [2, 2, 1]
    assert np.array_equal(model.models[0].fit_X, X[:2])
    assert np.array_equal(model.models[1].fit_X, X[2:4])
    assert np.array_equal(model.models[2].fit_X, X[4:5])

    X_test = np.zeros((4, 3), dtype=float)
    proba = model.predict_proba(X_test)

    # Prior correction is applied: population_prior=0.6, batch 0&1 have train_prior=0.5,
    # batch 2 has train_prior=1.0 (skipped). Compute expected corrected values explicitly.
    pop, tp = 0.6, 0.5
    ratio = (pop / tp) / ((1 - pop) / (1 - tp))
    p0_corr = (ratio * 0.25) / (1 + ratio * 0.25)  # raw 0.2 from batch idx=0
    p1_corr = (ratio * (2 / 3)) / (1 + ratio * (2 / 3))  # raw 0.4 from batch idx=1
    p2_corr = 0.6  # batch idx=2 train_prior=1.0, no correction
    expected = (p0_corr + p1_corr + p2_corr) / 3
    assert np.allclose(proba[:, 1], expected)
    assert np.allclose(proba[:, 0], 1 - expected)
    assert np.array_equal(model.predict(X_test, cutoff=0.5), np.zeros(4, dtype=int))
    assert np.array_equal(model.predict(X_test, cutoff=0.3), np.ones(4, dtype=int))
