"""Stacking pooler for regression heads: non-negative least squares on OOF predictions.

Each head contributes one fixed weight, fitted by non-negative least squares of the
target on the heads' out-of-fold predictions and normalised to sum to one (Breiman
1996, "Stacked regressions"; the non-negativity is his constraint, and what keeps the
blend a weighted average). Every head's OOF predictions are honest held-out
predictions, so the weights are fitted on data no head trained on.

Why not the per-sample Ridge gate the classifier uses: on LazyQSAR's regression
benchmark (TDC ADMET, Biogen ADME, 30 MoleculeACE sets; random and scaffold splits) the
Gaussian-likelihood gate did no better than equal weights (-0.1% RMSE), while stacking
lowered test RMSE by 2.6% on average and won on 76 of 90 task splits. The gate's
per-sample weights all but collapse to uniform once softmaxed; a fixed weight per head
is both more accurate and one number per head in the ONNX checkpoint.

Inputs are in whatever units the caller trains in; the assembler passes standardised
targets.
"""

import numpy as np

from lazyqsar.poolers._gate import GatedPooler, GatedPoolerArtifact
from lazyqsar.utils.metrics import spearman_score


def stacking_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least-squares weights of *y* on the columns of *P*, summing to one.

    Rows with a non-finite prediction or target are left out of the fit. Falls back to
    equal weights when every coefficient is zero (no head predicts anything useful) or
    no row is usable.
    """
    from scipy.optimize import nnls

    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(P).all(axis=1) & np.isfinite(y)
    if not ok.any():
        return np.full(P.shape[1], 1.0 / P.shape[1])
    w, _ = nnls(P[ok], y[ok])
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.full(P.shape[1], 1.0 / P.shape[1])
    return w / w.sum()


class InnerRegressorPooler(GatedPooler):
    """Fixed per-head weights from NNLS stacking of the heads' OOF predictions."""

    task = "regression"
    _metric_label = "ρ"

    def _prepare_y(self, y) -> np.ndarray:
        return np.asarray(y, dtype=float).ravel()

    def _skill(self, y, p):
        return spearman_score(y, p)

    def fit(self, P, y, X_prep=None):
        """Fit the stacking weights on OOF predictions ``P`` ``(n, n_heads)``.

        ``P=None`` (a head kept no OOF predictions) and a single head give equal
        weights.
        """
        from lazyqsar.utils.logging import logger

        y = self._prepare_y(y)
        self._gate_coef = None
        self._gate_intercept = None
        if P is None or self._n_heads == 1:
            self._head_weights = [1.0 / self._n_heads] * self._n_heads
        else:
            self._head_weights = stacking_weights(P, y).tolist()
            self._score = self._skill(y, self._pool(P))
        logger.inner_pooler_table(
            portfolio=self.portfolio,
            n_samples=len(y),
            oof_aucs=None if P is None else self._head_skills(y, P),
            meta_auc=getattr(self, "_score", None),
            mean_weights=list(self._head_weights),
            std_weights=[0.0] * self._n_heads,
            metric=self._metric_label,
        )

    def predict(self, P, X_prep=None) -> np.ndarray:
        """Weighted mean of per-head predictions, shape (n,).

        The same arithmetic as the inference artifact (:meth:`GatedPooler._pool`), so a
        fitted model and its checkpoint agree to the last bit.
        """
        return self._pool(np.asarray(P, dtype=float), X_prep)


# Inference: one pooler artifact for both tasks.
InnerRegressorPoolerArtifact = GatedPoolerArtifact
