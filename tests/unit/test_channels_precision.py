"""Regression values keep float64 through the chunked channel reader."""

import numpy as np

from lazyqsar.ensemble.channels import _score_chunks


class _RegressionArtifact:
    task = "regression"

    def predict(self, X):
        # A target with a large offset: float32 would keep ~3 decimals of it.
        return 123456.789 + X[:, 0].astype(np.float64)

    def _oof_percentile_from_value(self, values):
        return np.full(len(values), 0.5)


def test_regression_value_is_not_rounded_to_float32():
    X = np.linspace(0, 1, 10, dtype=np.float32)[:, None]
    ch = _score_chunks([X[:5], X[5:]], _RegressionArtifact(), None, {"y", "r"})
    assert ch.y.dtype == np.float64
    np.testing.assert_array_equal(ch.y, 123456.789 + X[:, 0].astype(np.float64))
    assert ch.r.dtype == np.float32
