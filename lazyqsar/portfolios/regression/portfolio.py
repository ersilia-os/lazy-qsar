"""Head selection for regression, by dataset size.

The heads are fitted with one preset each and blended by NNLS stacking, so a head earns
its place only if the stack is better with it. On LazyQSAR's regression benchmark (TDC
ADMET, Biogen ADME, 30 MoleculeACE sets; random and scaffold splits; 90 task splits),
measuring every subset of the four heads under the same stacking:

- up to 2,000 rows, ``lr + xgb + svr`` matches all four heads (+0.04% RMSE; the forest
  adds nothing once the kernel SVR is in) at a fraction of the cost;
- from 2,000 to 5,000 rows, ``lr + xgb`` is the best subset (-0.13% against all four):
  the SVR stops adding signal exactly where its O(n^2) kernel becomes the dominant
  cost;
- above 5,000 rows, where the SVR and the linear head were never fitted, ``xgb + rf``
  beats ``xgb`` alone by 0.9%.

Unlike the classification selector there are no dataset-profile scores: on this
benchmark no profile feature (sparsity, n/p, signal strength) changed which subset
won once size was accounted for.
"""

import numpy as np

from lazyqsar.base.xgboost import inspect as inspect_dataset

from .._base import _SELECTORS, HeadSelector

_SMALL_N = 2_000
_MEDIUM_N = 5_000


class Portfolio(HeadSelector):
    task = "regression"
    svm = "svr"
    selector_version = "size_v2_regression"

    def _profile(self, X, y):
        return inspect_dataset(X, np.asarray(y, dtype=float), task="regression")

    def _minority_count(self, y, profile) -> int:
        return profile.n_samples

    def fit(self, X, y):
        profile = self._profile(X, y)
        self.profile_ = profile
        self.selector_version_ = self.selector_version
        self.selector_scores_ = self._zero_scores()
        n = profile.n_samples
        if n <= _SMALL_N:
            self.portfolio = ["lr", "xgb", "svr"]
            reason = f"n={n:,} <= {_SMALL_N:,}: lr + xgb + svr"
        elif n <= _MEDIUM_N:
            self.portfolio = ["lr", "xgb"]
            reason = f"{_SMALL_N:,} < n={n:,} <= {_MEDIUM_N:,}: lr + xgb"
        else:
            self.portfolio = ["xgb", "rf"]
            reason = f"n={n:,} > {_MEDIUM_N:,}: xgb + rf"
        self.selector_reasons_ = [reason]
        self._log(profile)


_SELECTORS["regression"] = Portfolio
