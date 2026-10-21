"""Ridge gating network shared by the classification and regression inner poolers.

The gate maps preprocessed features to per-sample head weights: one Ridge regressor per
head predicts that head's share of an oracle target (softmax of per-sample
log-likelihoods), and a softmax across heads normalises the projection at inference.
Only the oracle differs by task -- Bernoulli likelihood for classification, Gaussian for
regression -- so everything else lives here.

The weight computation is numpy-only; fitting needs scikit-learn.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.linear_model import Ridge

    _FIT_DEPS_AVAILABLE = True
except ImportError:
    Ridge = None  # type: ignore[assignment,misc]
    _FIT_DEPS_AVAILABLE = False


def _heuristic_alpha(X: np.ndarray) -> float:
    """alpha = mean_feature_variance * p / n, floored at 0.01."""
    return max(0.01, float(np.mean(np.var(X, axis=0)) * X.shape[1] / X.shape[0]))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_gate(X_prep: np.ndarray, target_w: np.ndarray):
    """Fit one Ridge per head on the oracle weights.

    Parameters
    ----------
    X_prep : ndarray, shape (n, p)
    target_w : ndarray, shape (n, n_heads)
        Oracle per-sample head weights, rows summing to 1.

    Returns
    -------
    coef : ndarray, shape (n_heads, p)
    intercept : ndarray, shape (n_heads,)
    """
    n_heads = target_w.shape[1]
    alpha = _heuristic_alpha(X_prep)
    coef = np.zeros((n_heads, X_prep.shape[1]), dtype=float)
    intercept = np.zeros(n_heads, dtype=float)
    for j in range(n_heads):
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_prep, target_w[:, j])
        coef[j] = ridge.coef_
        intercept[j] = float(ridge.intercept_)
    return coef, intercept


def gate_weights(X_prep: np.ndarray, coef, intercept, n_heads: int) -> np.ndarray:
    """Per-sample head weights ``(n_samples, n_heads)``; uniform when no gate was fitted."""
    if coef is None:
        return np.full((len(X_prep), n_heads), 1.0 / n_heads)
    return _softmax(X_prep @ coef.T + intercept)


class GatedPooler:
    """Per-sample head weights from the Ridge gate, fitted on out-of-fold outputs.

    The template both tasks share. A subclass names its task and supplies the oracle
    (``_log_likelihood(P, y)``: how well each head explains each training row), the
    skill score shown per head (``_skill(y, p)``), and the label of that score.
    Falls back to equal weights with one head, or without OOF outputs or features.

    Parameters
    ----------
    portfolio : list of str
        Head names in the ensemble, e.g. ``["xgb", "rf"]``.
    """

    task = ""
    _metric_label = "AUC"

    def __init__(self, portfolio: list):
        self.portfolio = portfolio
        self._n_heads = len(portfolio)

    def _prepare_y(self, y) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _log_likelihood(self, P, y) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _skill(self, y, p):  # pragma: no cover - abstract
        raise NotImplementedError

    def _head_skills(self, y, P) -> list:
        """Each head's OOF skill for the table, NaN where it is undefined."""
        out = []
        for i in range(self._n_heads):
            value = self._skill(y, P[:, i])
            out.append(float("nan") if value is None else value)
        return out

    def fit(self, P, y, X_prep=None):
        """Fit the gate on OOF outputs ``P`` ``(n, n_heads)``; ``None`` forces equal weights."""
        from lazyqsar.utils.logging import logger

        y = self._prepare_y(y)
        self._gate_coef = None
        self._gate_intercept = None
        metric = {} if self._metric_label == "AUC" else {"metric": self._metric_label}

        # Scoring the heads needs `P`, so the equal-weight branch is taken before it is
        # touched: `P=None` is the documented route when a head kept no OOF outputs.
        if P is None or self._n_heads == 1 or X_prep is None:
            logger.inner_pooler_table(
                portfolio=self.portfolio,
                n_samples=len(y),
                oof_aucs=None if P is None else self._head_skills(y, P),
                **metric,
            )
            return

        oof_scores = self._head_skills(y, P)
        target_w = _softmax(self._log_likelihood(P, y))
        self._gate_coef, self._gate_intercept = fit_gate(X_prep, target_w)

        W_oof = self.get_weights(X_prep)
        self._score = self._skill(y, (W_oof * P).sum(axis=1))
        logger.inner_pooler_table(
            portfolio=self.portfolio,
            n_samples=len(y),
            oof_aucs=oof_scores,
            meta_auc=self._score,
            mean_weights=W_oof.mean(axis=0).tolist(),
            std_weights=W_oof.std(axis=0).tolist(),
            **metric,
        )

    def get_weights(self, X_prep: np.ndarray) -> np.ndarray:
        """Return per-sample weights (n_samples, n_heads).

        A pooler with fixed per-head weights (``_head_weights``, the regression
        stacker) returns them for every row; otherwise the gate's softmax.
        """
        head_weights = getattr(self, "_head_weights", None)
        if head_weights is not None:
            return np.tile(np.asarray(head_weights, dtype=float), (len(X_prep), 1))
        return gate_weights(
            X_prep, self._gate_coef, self._gate_intercept, self._n_heads
        )

    def _pool(self, P, X_prep=None) -> np.ndarray:
        """Gate-weighted mean of the per-head columns, shape (n,)."""
        if self._n_heads == 1:
            return np.asarray(P[:, 0], dtype=float)
        if getattr(self, "_head_weights", None) is not None:
            # Fixed per-head weights (the regression stacker) need no features.
            W = np.tile(np.asarray(self._head_weights, dtype=float), (len(P), 1))
        elif X_prep is not None:
            W = self.get_weights(X_prep)
        else:
            W = np.full(P.shape, 1.0 / self._n_heads)
        return (W * P).sum(axis=1)

    def _save_extra(self) -> dict:
        return {}

    def save(self, directory):
        """Write ``pooler.json`` containing gate weights and portfolio list."""
        import json
        import os

        data = {"task": self.task, "portfolio": self.portfolio}
        head_weights = getattr(self, "_head_weights", None)
        if head_weights is not None:
            data["head_weights"] = [float(w) for w in head_weights]
        if self._gate_coef is not None:
            data["gate_coef"] = self._gate_coef.tolist()
            data["gate_intercept"] = self._gate_intercept.tolist()
            data.update(self._save_extra())
        if getattr(self, "_score", None) is not None:
            data["score"] = self._score
        with open(os.path.join(directory, "pooler.json"), "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, directory):
        """Load a saved pooler from *directory* (reads ``pooler.json``)."""
        data = _read_pooler_json(directory)
        inst = cls(portfolio=data["portfolio"])
        inst._gate_coef, inst._gate_intercept = _gate_from_json(data)
        inst._head_weights = data.get("head_weights")
        if "score" in data:
            inst._score = data["score"]
        return inst


def _read_pooler_json(directory) -> dict:
    import json
    import os

    with open(os.path.join(directory, "pooler.json")) as f:
        return json.load(f)


def _gate_from_json(data: dict):
    if "gate_coef" in data:
        return np.array(data["gate_coef"]), np.array(data["gate_intercept"])
    return None, None


class GatedPoolerArtifact:
    """Inference-only pooler loaded from ``pooler.json``, for either task. numpy only."""

    def __init__(self, data: dict):
        self._n_heads = len(data["portfolio"])
        self._gate_coef, self._gate_intercept = _gate_from_json(data)
        self._head_weights = data.get("head_weights")

    get_weights = GatedPooler.get_weights
    _pool = GatedPooler._pool

    def predict(self, P, X_prep=None) -> np.ndarray:
        """Gate-weighted mean of per-head predictions, shape (n,)."""
        return self._pool(P, X_prep)

    def predict_proba(self, R, X_prep=None) -> np.ndarray:
        """Gate-weighted class-1 probability, as ``(n, 2)``."""
        p1 = self._pool(R, X_prep)
        return np.column_stack([1 - p1, p1])

    @classmethod
    def load(cls, directory):
        """Load pooler artifact from ``pooler.json`` in *directory*."""
        return cls(_read_pooler_json(directory))
