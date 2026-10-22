"""Rule-based head selection, shared by both tasks.

Which heads an assembler fits is decided from a profile of the training matrix -- sample
size, n/p ratio, sparsity, binary-feature fraction, feature signal -- by hard guards
first, then by a score per optional head. ``xgb`` and ``rf`` are always fitted; ``lr``
and the SVM head (``svc`` for classification, ``svr`` for regression) are optional.

The task subclasses say what differs: the SVM head's name, the effective minority count
(regression has none, so it is the sample count), and how the profile is computed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass

from lazyqsar.utils.logging import logger


class HeadSelector:
    """Choose the heads for one dataset; ``fit`` then ``get``."""

    task = ""
    svm = ""  # the SVM head's name for this task
    selector_version = ""

    def __init__(self):
        self.portfolio = ["xgb", "rf"]
        self.selector_version_ = self.selector_version
        self.selector_scores_ = self._zero_scores()
        self.selector_reasons_ = []
        self.profile_ = None

    def _zero_scores(self) -> dict:
        return {"lr": 0, "xgb": 0, "rf": 0, self.svm: 0}

    # ------------------------------------------------------------------ hooks

    def _profile(self, X, y):  # pragma: no cover - abstract
        raise NotImplementedError

    def _minority_count(self, y, profile) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ rules

    def _apply_hard_guards(self, profile, minority_count):
        svm = self.svm
        reasons = []
        if profile.n_samples > 5_000:
            # The SVM head is excluded for large n: kernel cost is O(n^2); a linear SVM
            # would duplicate LR; XGB and RF are already strong here.
            reasons.append(
                f"hard guard -> skip lr, {svm} (n_samples={profile.n_samples:,} > 5,000)"
            )
            return ["xgb", "rf"], reasons
        if profile.n_samples < 300 or minority_count < 25:
            reasons.append(
                f"hard guard -> include lr, {svm} (n_samples={profile.n_samples}, "
                f"minority={minority_count})"
            )
            return ["lr", "xgb", "rf", svm], reasons
        if profile.n_p_ratio < 1.0:
            reasons.append(
                f"hard guard -> include lr, {svm} (n/p={profile.n_p_ratio:.2f} < 1.0)"
            )
            return ["lr", "xgb", "rf", svm], reasons
        if profile.n_features >= 2000 and profile.n_p_ratio < 5.0:
            reasons.append(
                f"hard guard -> include lr, {svm} "
                f"(p={profile.n_features}, n/p={profile.n_p_ratio:.2f})"
            )
            return ["lr", "xgb", "rf", svm], reasons
        return None, reasons

    def _score_profile(self, profile):
        svm = self.svm
        scores = {"lr": 0, "xgb": 0, svm: 0}
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
                f"lr disfavored: large well-determined dense data "
                f"(n={profile.n_samples:,}, n/p={profile.n_p_ratio:.1f})"
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

        # SVM scoring (only reaches here for 300 <= n <= 5000). Bonuses: small n, where
        # margin-based generalisation is strongest (Burbidge 2001; Heikamp & Bajorath 2014).
        if profile.n_samples < 500:
            scores[svm] += 3
            reasons.append(f"{svm} +3: small n={profile.n_samples} favours SVM margin")
        elif profile.n_samples < 2000:
            scores[svm] += 2
            reasons.append(f"{svm} +2: small-medium n={profile.n_samples}")
        else:
            scores[svm] += 1
            reasons.append(f"{svm} +1: moderate n={profile.n_samples}")
        if not profile.is_sparse_counts and profile.binary_feature_fraction < 0.6:
            scores[svm] += 2
            reasons.append(f"{svm} +2: dense continuous features suit RBF kernel")
        if profile.feature_signal_p90 >= 0.1:
            scores[svm] += 1
            reasons.append(f"{svm} +1: signal p90={profile.feature_signal_p90:.2f}")
        if profile.n_p_ratio < 3:
            scores[svm] += 1
            reasons.append(
                f"{svm} +1: low n/p={profile.n_p_ratio:.1f} suits SVM regularisation"
            )
        # Penalties
        if profile.is_sparse_counts and profile.sparsity >= 0.85:
            scores[svm] -= 2
            reasons.append(
                f"{svm} -2: sparse fingerprints — a linear SVM duplicates LR"
            )
        if profile.imbalance_ratio > 50:
            scores[svm] -= 1
            reasons.append(
                f"{svm} -1: extreme imbalance={profile.imbalance_ratio:.0f}:1"
            )

        return scores, reasons

    # ------------------------------------------------------------------ fit

    def fit(self, X, y):
        profile = self._profile(X, y)
        self.profile_ = profile
        self.selector_version_ = self.selector_version

        portfolio, hard_reasons = self._apply_hard_guards(
            profile, self._minority_count(y, profile)
        )
        if portfolio is not None:
            self.selector_scores_ = self._zero_scores()
            self.selector_reasons_ = hard_reasons
            self.portfolio = portfolio
            self._log(profile)
            return

        scores, reasons = self._score_profile(profile)
        self.selector_scores_ = scores

        # LR: include when it leads XGB or trails it by less than 2.
        gap = abs(scores["lr"] - scores["xgb"])
        include_lr = scores["lr"] >= scores["xgb"] or gap < 2
        if include_lr and gap < 2:
            reasons.append(
                f"decision: score gap {gap} < 2 -> keep lr, while xgb and rf stay mandatory"
            )
        elif include_lr:
            reasons.append(
                "decision: lr signal strong, while xgb and rf stay mandatory"
            )
        else:
            reasons.append(f"decision: xgb leads by {gap} -> keep xgb and mandatory rf")

        # SVM: include when at least one bonus net of penalties reaches 2.
        include_svm = scores[self.svm] >= 2
        reasons.append(
            f"decision: {self.svm} score={scores[self.svm]} "
            + (
                f">= 2 -> include {self.svm}"
                if include_svm
                else f"< 2 -> skip {self.svm}"
            )
        )

        portfolio = ["xgb", "rf"]
        if include_lr:
            portfolio = ["lr"] + portfolio
        if include_svm:
            portfolio = portfolio + [self.svm]
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

    def get(self):
        return self.portfolio

    # ------------------------------------------------------------------ save / load

    def save(self, directory):
        profile = getattr(self, "profile_", None)
        if profile is None:
            profile_payload = None
        elif is_dataclass(profile):
            profile_payload = asdict(profile)
        else:
            profile_payload = dict(vars(profile))
        payload = {
            "task": self.task,
            "portfolio": self.portfolio,
            "selector_version": getattr(
                self, "selector_version_", self.selector_version
            ),
            "scores": getattr(self, "selector_scores_", self._zero_scores()),
            "reasons": getattr(self, "selector_reasons_", []),
            "profile": profile_payload,
        }
        with open(f"{directory}/portfolio.json", "w") as f:
            json.dump(payload, f, indent=4)
        logger.debug(f"Portfolio saved to {directory}/portfolio.json")

    @classmethod
    def load(cls, directory):
        """Load ``portfolio.json``, as the selector class of the task it records."""
        with open(f"{directory}/portfolio.json", "r") as f:
            payload = json.load(f)
        if (
            isinstance(payload, dict)
            and payload.get("task")
            and cls.task != payload["task"]
        ):
            cls = _SELECTORS[payload["task"]]
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
                "selector_version", instance.selector_version
            )
            # Older saves may lack the SVM head's key.
            scores = payload.get("scores") or instance._zero_scores()
            scores.setdefault(instance.svm, 0)
            instance.selector_scores_ = scores
            instance.selector_reasons_ = payload.get("reasons", [])
            instance.profile_ = payload.get("profile")
        logger.debug(f"Portfolio loaded from {directory}: {instance.portfolio}")
        return instance


# task -> selector class; filled in by the task modules.
_SELECTORS: dict = {}


class PortfolioArtifact(object):
    def __init__(self, portfolio):
        self.portfolio = portfolio

    @classmethod
    def load(cls, directory):
        # A file that records no task predates regression: classification.
        return _SELECTORS["classification"].load(directory)
