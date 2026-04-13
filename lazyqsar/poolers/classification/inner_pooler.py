import json
import os

import numpy as np
from sklearn.linear_model import Ridge

from lazyqsar.utils.logging import logger
from lazyqsar.utils.metrics import composite_metrics, composite_score


def _all_metrics(y, scores) -> dict:
    """Return dict with auroc, aupr, bedroc, and composite."""
    metrics = composite_metrics(y, scores)
    return {
        "auroc": metrics["auroc"],
        "aupr": metrics["aupr"],
        "bedroc": metrics["bedroc"],
        "composite": metrics["composite"],
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class InnerClassifierPooler(object):

    def __init__(self, portfolio: list):
        self.portfolio = portfolio
        self._mode = "equal"

    def fit(self, S, y, X_prep=None):
        """
        Fit the gating network from OOF predictions and features.

        S       : (n, n_heads) calibrated OOF probabilities for class 1, or None.
        y       : (n,) binary labels.
        X_prep  : (n, p) preprocessed features for learning per-sample head weights.
                  If None or n_heads==1, falls back to equal / passthrough mode.
        """
        n_heads = S.shape[1] if S is not None else len(self.portfolio)

        if S is None or n_heads == 1:
            self._mode = "passthrough" if n_heads == 1 else "equal"
            oof_scores = (
                [composite_score(y, S[:, 0])] if (S is not None and n_heads == 1) else None
            )
            logger.inner_pooler_table(
                portfolio=self.portfolio,
                mode=self._mode,
                n_samples=len(y),
                oof_aucs=oof_scores,
            )
            return

        # Per-head composite scores (normalized excess over random)
        oof_scores = [composite_score(y, S[:, i]) for i in range(n_heads)]

        if X_prep is None:
            # No features available: equal-weight passthrough
            self._mode = "equal"
            self._score = float(np.mean(oof_scores))
            logger.inner_pooler_table(
                portfolio=self.portfolio,
                mode=self._mode,
                n_samples=len(y),
                oof_aucs=oof_scores,
            )
            return

        # -------------------------------------------------------------------
        # Gating network: per-sample weights via Ridge regression
        # -------------------------------------------------------------------
        # Oracle target: for each sample, softmax of log P(y_i | head_j)
        eps = 1e-7
        log_scores = np.where(
            y[:, None] == 1,
            np.log(np.clip(S, eps, 1 - eps)),
            np.log(np.clip(1.0 - S, eps, 1 - eps)),
        )  # (n, n_heads)
        target_w = _softmax(log_scores)  # (n, n_heads)

        # Fit one Ridge regressor per head
        self._gate_coef = np.zeros((n_heads, X_prep.shape[1]), dtype=float)
        self._gate_intercept = np.zeros(n_heads, dtype=float)
        for j in range(n_heads):
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(X_prep, target_w[:, j])
            self._gate_coef[j] = ridge.coef_
            self._gate_intercept[j] = float(ridge.intercept_)

        self._mode = "gating"

        # Score the gated consensus on OOF; compute mean per-head weights
        W_oof = self.get_weights(X_prep)          # (n, n_heads)
        gated_p1 = (W_oof * S).sum(axis=1)        # (n,)
        self._score = composite_score(y, gated_p1)
        mean_w = W_oof.mean(axis=0).tolist()

        logger.inner_pooler_table(
            portfolio=self.portfolio,
            mode=self._mode,
            n_samples=len(y),
            oof_aucs=oof_scores,
            meta_auc=self._score,
            mean_weights=mean_w,
        )

    def get_weights(self, X_prep: np.ndarray) -> np.ndarray:
        """
        Return per-sample gating weights, shape (n_samples, n_heads).

        In equal / passthrough mode returns uniform weights.
        """
        if self._mode == "gating":
            raw = X_prep @ self._gate_coef.T + self._gate_intercept  # (n, n_heads)
            return _softmax(raw)
        n = len(X_prep)
        n_h = len(self.portfolio)
        return np.full((n, n_h), 1.0 / n_h)

    def predict_proba(self, R, X_prep=None):
        """
        R       : (n, n_heads) head predictions for class 1.
        X_prep  : (n, p) preprocessed features (required in gating mode).
        """
        if self._mode == "passthrough":
            return np.column_stack([1 - R[:, 0], R[:, 0]])
        W = self.get_weights(X_prep) if X_prep is not None else np.full(R.shape, 1.0 / R.shape[1])
        p1 = (W * R).sum(axis=1)
        return np.column_stack([1 - p1, p1])

    @property
    def weights(self):
        """Global mean weights for logging / display."""
        if self._mode == "gating":
            # Mean of absolute gate coefficients as a rough importance proxy
            return (np.abs(self._gate_coef).mean(axis=1) + 1e-9).tolist()
        return [1.0 / len(self.portfolio)] * len(self.portfolio)

    def save(self, directory):
        data = {"portfolio": self.portfolio, "mode": self._mode}
        if self._mode == "gating":
            data["gating_coef"] = self._gate_coef.tolist()
            data["gating_intercept"] = self._gate_intercept.tolist()
            data["score"] = self._score
            data["score_type"] = "composite"
        elif hasattr(self, "_score"):
            data["score"] = self._score
            data["score_type"] = "composite"
        with open(f"{directory}/pooler.json", "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, directory):
        with open(f"{directory}/pooler.json") as f:
            data = json.load(f)
        inst = cls(portfolio=data["portfolio"])
        inst._mode = data.get("mode", "equal")
        if inst._mode == "gating":
            inst._gate_coef = np.array(data["gating_coef"])
            inst._gate_intercept = np.array(data["gating_intercept"])
            inst._score = data.get("score")
        # backward compat: meta_lr saved files load as equal-weight
        elif inst._mode == "meta_lr":
            inst._mode = "equal"
        return inst


class InnerPoolerArtifact(object):
    """Pure-numpy inference artifact for the gating-network pooler."""

    def __init__(self, data: dict):
        self._data = data

    def get_weights(self, X_prep: np.ndarray) -> np.ndarray:
        """Return per-sample gating weights, shape (n_samples, n_heads)."""
        mode = self._data.get("mode", "equal")
        if mode == "gating":
            coef = np.array(self._data["gating_coef"])        # (n_heads, p)
            intercept = np.array(self._data["gating_intercept"])  # (n_heads,)
            raw = X_prep @ coef.T + intercept
            e = np.exp(raw - raw.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        n_h = len(self._data["portfolio"])
        return np.full((len(X_prep), n_h), 1.0 / n_h)

    def predict_proba(self, R, X_prep=None):
        """
        R       : (n, n_heads) head predictions for class 1.
        X_prep  : (n, p) preprocessed features.
        """
        mode = self._data.get("mode", "equal")
        if mode == "passthrough":
            return np.column_stack([1 - R[:, 0], R[:, 0]])
        if X_prep is not None:
            W = self.get_weights(X_prep)
        else:
            W = np.full(R.shape, 1.0 / R.shape[1])
        # backward compat: meta_lr stored in old files
        if mode == "meta_lr":
            coef = np.array(self._data["meta_coef"])
            intercept = self._data["meta_intercept"]
            p1 = 1.0 / (1.0 + np.exp(-(R @ coef + intercept)))
            return np.column_stack([1 - p1, p1])
        p1 = (W * R).sum(axis=1)
        return np.column_stack([1 - p1, p1])

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "pooler.json")) as f:
            data = json.load(f)
        return cls(data)
