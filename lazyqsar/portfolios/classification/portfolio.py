"""Rule-based head selection for classification (see :mod:`lazyqsar.portfolios._base`)."""

import numpy as np

from lazyqsar.base.xgboost import inspect as inspect_dataset

from .._base import _SELECTORS, HeadSelector, PortfolioArtifact  # noqa: F401


class Portfolio(HeadSelector):
    task = "classification"
    svm = "svc"
    selector_version = "rule_v1"

    def _profile(self, X, y):
        return inspect_dataset(X, y, task="classification")

    def _minority_count(self, y, profile) -> int:
        _, counts = np.unique(y, return_counts=True)
        return int(counts.min()) if len(counts) else 0


_SELECTORS["classification"] = Portfolio
