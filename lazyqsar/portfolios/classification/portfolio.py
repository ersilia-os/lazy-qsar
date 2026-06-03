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
        self.selector_scores_ = {"lr": 0, "xgb": 0, "rf": 0, "svc": 0}
        self.selector_reasons_ = []
        self.profile_ = None

    @staticmethod
    def _minority_count(y):
        _, counts = np.unique(y, return_counts=True)
        return int(counts.min()) if len(counts) else 0

    def _apply_hard_guards(self, profile, minority_count):
        reasons = []
        if profile.n_samples > 5_000:
            # SVC excluded for large n: kernel cost is O(n^2); LinearSVC would
            # duplicate LR capability; XGB and RF are already strong here.
            reasons.append(
                f"hard guard -> skip lr, svc (n_samples={profile.n_samples:,} > 5,000)"
            )
            return ["xgb", "rf"], reasons
        if profile.n_samples < 300 or minority_count < 25:
            reasons.append(
                f"hard guard -> include lr, svc (n_samples={profile.n_samples}, minority={minority_count})"
            )
            return ["lr", "xgb", "rf", "svc"], reasons
        if profile.n_p_ratio < 1.0:
            reasons.append(
                f"hard guard -> include lr, svc (n/p={profile.n_p_ratio:.2f} < 1.0)"
            )
            return ["lr", "xgb", "rf", "svc"], reasons
        if profile.n_features >= 2000 and profile.n_p_ratio < 5.0:
            reasons.append(
                f"hard guard -> include lr, svc (p={profile.n_features}, n/p={profile.n_p_ratio:.2f})"
            )
            return ["lr", "xgb", "rf", "svc"], reasons
        return None, reasons

    def _score_profile(self, profile):
        scores = {"lr": 0, "xgb": 0, "svc": 0}
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

        # SVC scoring (only reaches here for 300 <= n <= 5000)
        # Bonuses: small n where margin-based generalisation is strongest
        # (Burbidge 2001; Heikamp & Bajorath 2014)
        if profile.n_samples < 500:
            scores["svc"] += 3
            reasons.append(f"svc +3: small n={profile.n_samples} favours SVM margin")
        elif profile.n_samples < 2000:
            scores["svc"] += 2
            reasons.append(f"svc +2: small-medium n={profile.n_samples}")
        else:
            scores["svc"] += 1
            reasons.append(f"svc +1: moderate n={profile.n_samples}")
        if not profile.is_sparse_counts and profile.binary_feature_fraction < 0.6:
            scores["svc"] += 2
            reasons.append("svc +2: dense continuous features suit RBF kernel")
        if profile.feature_signal_p90 >= 0.1:
            scores["svc"] += 1
            reasons.append(f"svc +1: signal p90={profile.feature_signal_p90:.2f}")
        if profile.n_p_ratio < 3:
            scores["svc"] += 1
            reasons.append(
                f"svc +1: low n/p={profile.n_p_ratio:.1f} suits SVM regularisation"
            )
        # Penalties
        if profile.is_sparse_counts and profile.sparsity >= 0.85:
            scores["svc"] -= 2
            reasons.append("svc -2: sparse fingerprints — LinearSVC duplicates LR")
        if profile.imbalance_ratio > 50:
            scores["svc"] -= 1
            reasons.append(f"svc -1: extreme imbalance={profile.imbalance_ratio:.0f}:1")

        return scores, reasons

    def fit(self, X, y):
        profile = inspect_dataset(X, y, task="classification")
        minority_count = self._minority_count(y)
        self.profile_ = profile
        self.selector_version_ = _SELECTOR_VERSION

        portfolio, hard_reasons = self._apply_hard_guards(profile, minority_count)
        if portfolio is not None:
            self.selector_scores_ = {"lr": 0, "xgb": 0, "rf": 0, "svc": 0}
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

        # LR decision: include when score gap vs XGB is small or LR leads
        gap = abs(scores["lr"] - scores["xgb"])
        if scores["lr"] >= scores["xgb"] or gap < 2:
            include_lr = True
            if gap < 2:
                reasons = reasons + [
                    f"decision: score gap {gap} < 2 -> keep lr, while xgb and rf stay mandatory"
                ]
            else:
                reasons = reasons + [
                    "decision: lr signal strong, while xgb and rf stay mandatory"
                ]
        else:
            include_lr = False
            reasons = reasons + [
                f"decision: xgb leads by {gap} -> keep xgb and mandatory rf"
            ]

        # SVC decision: include when score >= 2 (at least one bonus firing)
        include_svc = scores["svc"] >= 2
        if include_svc:
            reasons = reasons + [
                f"decision: svc score={scores['svc']} >= 2 -> include svc"
            ]
        else:
            reasons = reasons + [f"decision: svc score={scores['svc']} < 2 -> skip svc"]

        portfolio = ["xgb", "rf"]
        if include_lr:
            portfolio = ["lr"] + portfolio
        if include_svc:
            portfolio = portfolio + ["svc"]

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
            "scores": getattr(
                self, "selector_scores_", {"lr": 0, "xgb": 0, "rf": 0, "svc": 0}
            ),
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
            # Backward-compatible: older saves may lack "svc" key
            scores = payload.get("scores", {"lr": 0, "xgb": 0, "rf": 0, "svc": 0})
            if "svc" not in scores:
                scores["svc"] = 0
            instance.selector_scores_ = scores
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
