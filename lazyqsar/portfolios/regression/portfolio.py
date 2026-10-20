"""Rule-based head selection for regression.

The same dataset rules as :class:`lazyqsar.portfolios.classification.Portfolio` -- sample
size, n/p ratio, sparsity, feature signal -- with the class-balance terms removed (there
is no minority class) and ``svr`` in place of ``svc``. Reuses the classification scorer
rather than copying it, so a rule tuned there carries over.
"""

import numpy as np

from lazyqsar.base.xgboost import inspect as inspect_dataset
from lazyqsar.utils.logging import logger

from ..classification.portfolio import Portfolio as _ClassificationPortfolio

_SELECTOR_VERSION = "rule_v1_regression"


def _rename(names):
    return ["svr" if h == "svc" else h for h in names]


class Portfolio(_ClassificationPortfolio):
    def __init__(self):
        super().__init__()
        self.selector_version_ = _SELECTOR_VERSION
        self.selector_scores_ = {"lr": 0, "xgb": 0, "rf": 0, "svr": 0}

    def fit(self, X, y):
        profile = inspect_dataset(X, np.asarray(y, dtype=float), task="regression")
        self.profile_ = profile
        self.selector_version_ = _SELECTOR_VERSION

        # Every sample informs a regression fit, so the "minority" guard reduces to the
        # sample-size one: pass n as the minority count.
        portfolio, hard_reasons = self._apply_hard_guards(profile, profile.n_samples)
        if portfolio is not None:
            self.selector_scores_ = {"lr": 0, "xgb": 0, "rf": 0, "svr": 0}
            self.selector_reasons_ = [r.replace("svc", "svr") for r in hard_reasons]
            self.portfolio = _rename(portfolio)
            self._log(profile)
            return

        # profile.imbalance_ratio is 1.0 for regression, so the imbalance terms of the
        # shared scorer never fire.
        scores, reasons = self._score_profile(profile)
        scores = {("svr" if k == "svc" else k): v for k, v in scores.items()}
        reasons = [r.replace("svc", "svr") for r in reasons]
        self.selector_scores_ = scores

        gap = abs(scores["lr"] - scores["xgb"])
        include_lr = scores["lr"] >= scores["xgb"] or gap < 2
        reasons.append(
            "decision: keep lr alongside mandatory xgb and rf"
            if include_lr
            else f"decision: xgb leads by {gap} -> keep xgb and mandatory rf"
        )
        include_svr = scores["svr"] >= 2
        reasons.append(
            f"decision: svr score={scores['svr']} "
            + (">= 2 -> include svr" if include_svr else "< 2 -> skip svr")
        )

        portfolio = ["xgb", "rf"]
        if include_lr:
            portfolio = ["lr"] + portfolio
        if include_svr:
            portfolio = portfolio + ["svr"]
        self.selector_reasons_ = reasons
        self.portfolio = portfolio
        self._log(profile)

    def _log(self, profile):
        logger.selector_table(
            portfolio=self.portfolio,
            profile=profile,
            scores=self.selector_scores_,
            reasons=self.selector_reasons_,
            selector_version=self.selector_version_,
        )
