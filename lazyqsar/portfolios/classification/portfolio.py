import json
from dataclasses import asdict, is_dataclass

import numpy as np

from lazyqsar.base.xgboost import inspect as inspect_dataset
from lazyqsar.utils.logging import logger


_SELECTOR_VERSION = "rule_v1"


class Portfolio(object):
    def __init__(self):
        self.portfolio = ["xgb", "rf"]
        self.selector_version_ = _SELECTOR_VERSION
        self.selector_scores_ = {"lr": 0, "xgb": 0, "rf": 0}
        self.selector_reasons_ = []
        self.profile_ = None

    @staticmethod
    def _minority_count(y):
        _, counts = np.unique(y, return_counts=True)
        return int(counts.min()) if len(counts) else 0

    def _apply_hard_guards(self, profile, minority_count):
        reasons = []
        if profile.n_samples > 5_000:
            reasons.append(
                f"hard guard -> skip lr (n_samples={profile.n_samples:,} > 5,000)"
            )
            return ["xgb", "rf"], reasons
        if profile.n_samples < 300 or minority_count < 25:
            reasons.append(
                f"hard guard -> include lr (n_samples={profile.n_samples}, minority={minority_count})"
            )
            return ["lr", "xgb", "rf"], reasons
        if profile.n_p_ratio < 1.0:
            reasons.append(
                f"hard guard -> include lr (n/p={profile.n_p_ratio:.2f} < 1.0)"
            )
            return ["lr", "xgb", "rf"], reasons
        if profile.n_features >= 2000 and profile.n_p_ratio < 5.0:
            reasons.append(
                f"hard guard -> include lr (p={profile.n_features}, n/p={profile.n_p_ratio:.2f})"
            )
            return ["lr", "xgb", "rf"], reasons
        return None, reasons

    def _score_profile(self, profile):
        scores = {"lr": 0, "xgb": 0}
        reasons = []

        if profile.is_sparse_counts and profile.sparsity >= 0.85:
            scores["xgb"] -= 2
            reasons.append(
                f"xgb disfavored: sparse counts with sparsity={profile.sparsity:.2f}"
            )
        if (
            profile.n_samples >= 20_000
            and profile.n_p_ratio >= 10
            and not profile.is_sparse_counts
            and profile.binary_feature_fraction < 0.8
        ):
            scores["lr"] -= 2
            reasons.append(
                f"lr disfavored: large well-determined dense data (n={profile.n_samples:,}, n/p={profile.n_p_ratio:.1f})"
            )

        if profile.n_p_ratio < 1.5:
            scores["lr"] += 3
            reasons.append("lr +3: low n/p ratio")
        if profile.n_features >= 2000:
            scores["lr"] += 2
            reasons.append("lr +2: very wide feature space")
        if profile.binary_feature_fraction >= 0.8 or profile.is_sparse_counts:
            scores["lr"] += 2
            reasons.append("lr +2: strongly binary/sparse-count features")
        if profile.n_samples < 1000:
            scores["lr"] += 1
            reasons.append("lr +1: small sample size")
        if profile.imbalance_ratio >= 20:
            scores["lr"] += 1
            reasons.append(f"lr +1: imbalance={profile.imbalance_ratio:.1f}:1")

        if profile.n_samples >= 5000:
            scores["xgb"] += 3
            reasons.append("xgb +3: enough samples for trees")
        if profile.feature_signal_p90 >= 0.15:
            scores["xgb"] += 2
            reasons.append(
                f"xgb +2: strong feature signal p90={profile.feature_signal_p90:.2f}"
            )
        if 0.1 <= profile.binary_feature_fraction < 0.8 and profile.sparsity < 0.95:
            scores["xgb"] += 2
            reasons.append("xgb +2: mixed feature types without extreme sparsity")
        if profile.n_p_ratio >= 5:
            scores["xgb"] += 1
            reasons.append(f"xgb +1: favorable n/p={profile.n_p_ratio:.1f}")
        if profile.feature_signal_strength >= 0.05:
            scores["xgb"] += 1
            reasons.append(
                f"xgb +1: mean feature signal={profile.feature_signal_strength:.2f}"
            )

        return scores, reasons

    def fit(self, X, y):
        profile = inspect_dataset(X, y, task="classification")
        minority_count = self._minority_count(y)
        self.profile_ = profile
        self.selector_version_ = _SELECTOR_VERSION

        portfolio, hard_reasons = self._apply_hard_guards(profile, minority_count)
        if portfolio is not None:
            self.selector_scores_ = {"lr": 0, "xgb": 0, "rf": 0}
            self.selector_reasons_ = hard_reasons
            self.portfolio = portfolio
            logger.selector_table(
                portfolio=self.portfolio,
                profile=profile,
                scores=self.selector_scores_,
                reasons=self.selector_reasons_,
                selector_version=self.selector_version_,
            )
            return

        scores, reasons = self._score_profile(profile)
        self.selector_scores_ = scores

        gap = abs(scores["lr"] - scores["xgb"])
        if scores["lr"] >= scores["xgb"] or gap < 2:
            portfolio = ["lr", "xgb", "rf"]
            if gap < 2:
                reasons = reasons + [
                    f"decision: score gap {gap} < 2 -> keep lr, while xgb and rf stay mandatory"
                ]
            else:
                reasons = reasons + [
                    "decision: lr signal strong, while xgb and rf stay mandatory"
                ]
        else:
            portfolio = ["xgb", "rf"]
            reasons = reasons + [
                f"decision: xgb leads by {gap} -> keep xgb and mandatory rf"
            ]

        self.selector_reasons_ = reasons
        self.portfolio = portfolio
        logger.selector_table(
            portfolio=self.portfolio,
            profile=profile,
            scores=self.selector_scores_,
            reasons=self.selector_reasons_,
            selector_version=self.selector_version_,
        )

    def get(self):
        return self.portfolio

    def save(self, directory):
        profile = getattr(self, "profile_", None)
        if profile is None:
            profile_payload = None
        elif is_dataclass(profile):
            profile_payload = asdict(profile)
        else:
            profile_payload = dict(vars(profile))
        payload = {
            "portfolio": self.portfolio,
            "selector_version": getattr(self, "selector_version_", _SELECTOR_VERSION),
            "scores": getattr(self, "selector_scores_", {"lr": 0, "xgb": 0, "rf": 0}),
            "reasons": getattr(self, "selector_reasons_", []),
            "profile": profile_payload,
        }
        with open(f"{directory}/portfolio.json", "w") as f:
            json.dump(payload, f, indent=4)
        logger.debug(f"Portfolio saved to {directory}/portfolio.json")

    @classmethod
    def load(cls, directory):
        with open(f"{directory}/portfolio.json", "r") as f:
            payload = json.load(f)
        instance = cls()
        if isinstance(payload, list):
            instance.portfolio = payload
            instance.selector_version_ = "legacy"
            instance.selector_scores_ = {"lr": 0, "xgb": 0, "rf": 0}
            instance.selector_reasons_ = []
            instance.profile_ = None
        else:
            instance.portfolio = payload.get("portfolio", ["xgb", "rf"])
            instance.selector_version_ = payload.get(
                "selector_version", _SELECTOR_VERSION
            )
            instance.selector_scores_ = payload.get(
                "scores", {"lr": 0, "xgb": 0, "rf": 0}
            )
            instance.selector_reasons_ = payload.get("reasons", [])
            instance.profile_ = payload.get("profile")
        logger.debug(f"Portfolio loaded from {directory}: {instance.portfolio}")
        return instance


class PortfolioArtifact(object):
    def __init__(self, portfolio):
        self.portfolio = portfolio

    @classmethod
    def load(cls, directory):
        return Portfolio.load(directory)
