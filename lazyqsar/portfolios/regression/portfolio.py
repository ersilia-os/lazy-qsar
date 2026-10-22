"""Rule-based head selection for regression (see :mod:`lazyqsar.portfolios._base`).

The classification rules with ``svr`` in place of ``svc`` and no minority class: every
sample informs a regression fit, so the "minority" guard reduces to the sample-size one.
The profile's imbalance ratio is 1.0, so the imbalance terms never fire.
"""

import numpy as np

from lazyqsar.base.xgboost import inspect as inspect_dataset

from .._base import _SELECTORS, HeadSelector


class Portfolio(HeadSelector):
    task = "regression"
    svm = "svr"
    selector_version = "rule_v1_regression"

    def _profile(self, X, y):
        return inspect_dataset(X, np.asarray(y, dtype=float), task="regression")

    def _minority_count(self, y, profile) -> int:
        return profile.n_samples


_SELECTORS["regression"] = Portfolio
