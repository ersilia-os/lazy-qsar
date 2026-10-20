"""Gating network for regression heads.

The regression twin of :class:`lazyqsar.poolers.classification.InnerClassifierPooler`.
The Ridge gate itself is shared (:mod:`lazyqsar.poolers._gate`); only the oracle changes.

Oracle: for head *j* on sample *i*, the Gaussian log-likelihood of the target under that
head's out-of-fold prediction,

    log L_ij = -(y_i - p_ij)^2 / (2 sigma_j^2) - log sigma_j,

with ``sigma_j = 1.4826 * MAD`` of head *j*'s OOF residuals -- a robust scale, so one
outlying compound does not flatten every head's likelihood. The softmax across heads is
the per-sample target the gate learns to predict from the preprocessed features.

Inputs are in whatever units the caller trains in; the assembler passes standardised
targets, which keeps ``sigma`` comparable across datasets.
"""

import json
import os

import numpy as np

from lazyqsar.poolers._gate import _softmax, fit_gate, gate_weights
from lazyqsar.utils.logging import logger
from lazyqsar.utils.metrics import spearman_score

_MAD_TO_SIGMA = 1.4826
_SIGMA_FLOOR = 1e-3


def _robust_sigma(residuals: np.ndarray) -> np.ndarray:
    """Per-column robust scale of residuals, ``(n_heads,)``, floored."""
    med = np.median(residuals, axis=0)
    mad = np.median(np.abs(residuals - med), axis=0)
    return np.maximum(_MAD_TO_SIGMA * mad, _SIGMA_FLOOR)


def _skill(value):
    return float("nan") if value is None else value


class InnerRegressorPooler(object):
    """
    Gating network that learns per-sample head weights from OOF predictions.

    Parameters
    ----------
    portfolio : list of str
        Head names in the ensemble, e.g. ``["xgb", "rf"]``.
    """

    def __init__(self, portfolio: list):
        self.portfolio = portfolio
        self._n_heads = len(portfolio)

    def fit(self, P, y, X_prep=None):
        """
        Fit the gating network on OOF predictions.

        Parameters
        ----------
        P : ndarray, shape (n, n_heads), or None
            OOF predictions, one column per head. ``None`` forces equal weights.
        y : array-like, shape (n,)
            Training targets, same units as *P*.
        X_prep : ndarray, shape (n, p) or None
            Preprocessed features. ``None`` forces equal weights.
        """
        y = np.asarray(y, dtype=float).ravel()
        self._gate_coef = None
        self._gate_intercept = None

        if P is None or self._n_heads == 1 or X_prep is None:
            logger.inner_pooler_table(
                portfolio=self.portfolio,
                n_samples=len(y),
                oof_aucs=(
                    None
                    if P is None
                    else [
                        _skill(spearman_score(y, P[:, i])) for i in range(self._n_heads)
                    ]
                ),
                metric="ρ",
            )
            return

        oof_scores = [_skill(spearman_score(y, P[:, i])) for i in range(self._n_heads)]
        resid = y[:, None] - P
        sigma = _robust_sigma(resid)
        self.sigma_ = sigma
        log_lik = -(resid**2) / (2.0 * sigma[None, :] ** 2) - np.log(sigma)[None, :]
        target_w = _softmax(log_lik)
        self._gate_coef, self._gate_intercept = fit_gate(X_prep, target_w)

        W_oof = self.get_weights(X_prep)
        self._score = spearman_score(y, (W_oof * P).sum(axis=1))
        logger.inner_pooler_table(
            portfolio=self.portfolio,
            n_samples=len(y),
            oof_aucs=oof_scores,
            meta_auc=self._score,
            mean_weights=W_oof.mean(axis=0).tolist(),
            std_weights=W_oof.std(axis=0).tolist(),
            metric="ρ",
        )

    def get_weights(self, X_prep: np.ndarray) -> np.ndarray:
        """Return per-sample weights (n_samples, n_heads)."""
        return gate_weights(
            X_prep, self._gate_coef, self._gate_intercept, self._n_heads
        )

    def predict(self, P, X_prep=None) -> np.ndarray:
        """Gate-weighted mean of per-head predictions, shape (n,)."""
        if self._n_heads == 1:
            return np.asarray(P[:, 0], dtype=float)
        W = (
            self.get_weights(X_prep)
            if X_prep is not None
            else np.full(P.shape, 1.0 / self._n_heads)
        )
        return (W * P).sum(axis=1)

    def save(self, directory):
        """Write ``pooler.json`` containing gate weights and portfolio list."""
        data = {"task": "regression", "portfolio": self.portfolio}
        if self._gate_coef is not None:
            data["gate_coef"] = self._gate_coef.tolist()
            data["gate_intercept"] = self._gate_intercept.tolist()
            data["sigma"] = self.sigma_.tolist()
        if getattr(self, "_score", None) is not None:
            data["score"] = self._score
        with open(os.path.join(directory, "pooler.json"), "w") as f:
            json.dump(data, f, indent=4)


class InnerRegressorPoolerArtifact(object):
    """Inference-only regression pooler loaded from ``pooler.json``. numpy only."""

    def __init__(self, data: dict):
        self._n_heads = len(data["portfolio"])
        if "gate_coef" in data:
            self._gate_coef = np.array(data["gate_coef"])
            self._gate_intercept = np.array(data["gate_intercept"])
        else:
            self._gate_coef = None
            self._gate_intercept = None

    def get_weights(self, X_prep: np.ndarray) -> np.ndarray:
        return gate_weights(
            X_prep, self._gate_coef, self._gate_intercept, self._n_heads
        )

    def predict(self, P, X_prep=None) -> np.ndarray:
        if self._n_heads == 1:
            return np.asarray(P[:, 0], dtype=float)
        W = (
            self.get_weights(X_prep)
            if X_prep is not None
            else np.full(P.shape, 1.0 / self._n_heads)
        )
        return (W * P).sum(axis=1)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "pooler.json")) as f:
            data = json.load(f)
        return cls(data)
