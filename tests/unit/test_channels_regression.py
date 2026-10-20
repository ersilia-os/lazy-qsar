"""Channel extraction from a regression artifact: 1-D values, rank from the value.

A regression artifact exposes ``predict`` (values, not ``[neg, pos]`` pairs) and
``_oof_percentile_from_value``. The channel reader must take ``y`` from ``predict``, derive
``r`` from ``y`` without re-running the graphs, and never ask for a raw ``score``.
"""

import numpy as np

from lazyqsar.ensemble.channels import _score_chunks


class _RegressionArtifact:
    task = "regression"

    def __init__(self):
        self.calls = {"predict": 0, "from_value": 0}

    def predict(self, X):
        self.calls["predict"] += 1
        return X.sum(axis=1)

    def _oof_percentile_from_value(self, y):
        self.calls["from_value"] += 1
        return np.clip((y + 10) / 20, 0, 1)

    def predict_score(self, X):  # pragma: no cover - must never be called
        raise AssertionError("regression has no score channel")


class _AD:
    def score(self, X):
        return np.ones(len(X))


def test_regression_channels_are_one_dimensional_and_rank_comes_from_value():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(5, 3)) for _ in range(3)]
    art = _RegressionArtifact()
    ch = _score_chunks(iter(chunks), art, _AD(), {"y", "r", "s", "a"})
    expected = np.concatenate([c.sum(axis=1) for c in chunks])
    np.testing.assert_allclose(ch.y, expected, rtol=1e-6)
    assert ch.y.ndim == 1 and ch.r.shape == ch.y.shape
    assert ch.s is None
    assert art.calls == {"predict": 3, "from_value": 3}
