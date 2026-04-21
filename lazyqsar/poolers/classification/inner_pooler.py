import json
import os

import numpy as np
from sklearn.linear_model import Ridge

from lazyqsar.utils.logging import logger
from lazyqsar.utils.metrics import composite_score


def _heuristic_alpha(X: np.ndarray) -> float:
    """alpha = mean_feature_variance * p / n, floored at 0.01."""
    return max(0.01, float(np.mean(np.var(X, axis=0)) * X.shape[1] / X.shape[0]))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class InnerClassifierPooler(object):

    def __init__(self, portfolio: list):
        self.portfolio = portfolio
        self._n_heads = len(portfolio)

    def fit(self, S, y, X_prep=None):
        y = np.asarray(y, dtype=int)
        oof_scores = [composite_score(y, S[:, i]) for i in range(self._n_heads)]

        if self._n_heads == 1 or X_prep is None:
            self._gate_coef = None
            self._gate_intercept = None
            logger.inner_pooler_table(
                portfolio=self.portfolio,
                n_samples=len(y),
                oof_aucs=oof_scores,
            )
            return

        eps = 1e-7
        log_scores = np.where(
            y[:, None] == 1,
            np.log(np.clip(S, eps, 1 - eps)),
            np.log(np.clip(1.0 - S, eps, 1 - eps)),
        )
        target_w = _softmax(log_scores)  # (n, n_heads)

        alpha = _heuristic_alpha(X_prep)
        self._gate_coef = np.zeros((self._n_heads, X_prep.shape[1]), dtype=float)
        self._gate_intercept = np.zeros(self._n_heads, dtype=float)
        for j in range(self._n_heads):
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_prep, target_w[:, j])
            self._gate_coef[j] = ridge.coef_
            self._gate_intercept[j] = float(ridge.intercept_)

        W_oof = self.get_weights(X_prep)
        self._score = composite_score(y, (W_oof * S).sum(axis=1))
        logger.inner_pooler_table(
            portfolio=self.portfolio,
            n_samples=len(y),
            oof_aucs=oof_scores,
            meta_auc=self._score,
            mean_weights=W_oof.mean(axis=0).tolist(),
            std_weights=W_oof.std(axis=0).tolist(),
        )

    def get_weights(self, X_prep: np.ndarray) -> np.ndarray:
        """Return per-sample weights (n_samples, n_heads)."""
        if self._gate_coef is None:
            return np.full((len(X_prep), self._n_heads), 1.0 / self._n_heads)
        return _softmax(X_prep @ self._gate_coef.T + self._gate_intercept)

    def predict_proba(self, R, X_prep=None):
        if self._n_heads == 1:
            return np.column_stack([1 - R[:, 0], R[:, 0]])
        W = self.get_weights(X_prep) if X_prep is not None else np.full(R.shape, 1.0 / self._n_heads)
        p1 = (W * R).sum(axis=1)
        return np.column_stack([1 - p1, p1])

    def save(self, directory):
        data = {"portfolio": self.portfolio}
        if self._gate_coef is not None:
            data["gate_coef"] = self._gate_coef.tolist()
            data["gate_intercept"] = self._gate_intercept.tolist()
        if hasattr(self, "_score"):
            data["score"] = self._score
        with open(f"{directory}/pooler.json", "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, directory):
        with open(f"{directory}/pooler.json") as f:
            data = json.load(f)
        inst = cls(portfolio=data["portfolio"])
        if "gate_coef" in data:
            inst._gate_coef = np.array(data["gate_coef"])
            inst._gate_intercept = np.array(data["gate_intercept"])
        else:
            inst._gate_coef = None
            inst._gate_intercept = None
        if "score" in data:
            inst._score = data["score"]
        return inst


class InnerPoolerArtifact(object):

    def __init__(self, data: dict):
        self._n_heads = len(data["portfolio"])
        if "gate_coef" in data:
            self._gate_coef = np.array(data["gate_coef"])
            self._gate_intercept = np.array(data["gate_intercept"])
        else:
            self._gate_coef = None

    def get_weights(self, X_prep: np.ndarray) -> np.ndarray:
        if self._gate_coef is None:
            return np.full((len(X_prep), self._n_heads), 1.0 / self._n_heads)
        raw = X_prep @ self._gate_coef.T + self._gate_intercept
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, R, X_prep=None):
        if self._n_heads == 1:
            return np.column_stack([1 - R[:, 0], R[:, 0]])
        W = self.get_weights(X_prep) if X_prep is not None else np.full(R.shape, 1.0 / self._n_heads)
        p1 = (W * R).sum(axis=1)
        return np.column_stack([1 - p1, p1])

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "pooler.json")) as f:
            data = json.load(f)
        return cls(data)
