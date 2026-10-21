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

import numpy as np

from lazyqsar.poolers._gate import GatedPooler, GatedPoolerArtifact
from lazyqsar.utils.metrics import spearman_score

_MAD_TO_SIGMA = 1.4826
_SIGMA_FLOOR = 1e-3


def _robust_sigma(residuals: np.ndarray) -> np.ndarray:
    """Per-column robust scale of residuals, ``(n_heads,)``, floored."""
    med = np.median(residuals, axis=0)
    mad = np.median(np.abs(residuals - med), axis=0)
    return np.maximum(_MAD_TO_SIGMA * mad, _SIGMA_FLOOR)


class InnerRegressorPooler(GatedPooler):
    """Gating network that learns per-sample head weights from OOF predictions."""

    task = "regression"
    _metric_label = "ρ"

    def _prepare_y(self, y) -> np.ndarray:
        return np.asarray(y, dtype=float).ravel()

    def _log_likelihood(self, P, y) -> np.ndarray:
        resid = y[:, None] - P
        sigma = _robust_sigma(resid)
        self.sigma_ = sigma
        return -(resid**2) / (2.0 * sigma[None, :] ** 2) - np.log(sigma)[None, :]

    def _skill(self, y, p):
        return spearman_score(y, p)

    def _save_extra(self) -> dict:
        return {"sigma": self.sigma_.tolist()}

    def predict(self, P, X_prep=None) -> np.ndarray:
        """Gate-weighted mean of per-head predictions, shape (n,)."""
        return self._pool(P, X_prep)


# Inference: one pooler artifact for both tasks.
InnerRegressorPoolerArtifact = GatedPoolerArtifact
