"""Regression assembler: preprocessor + heads + gating pooler, batched by size.

The regression twin of :mod:`lazyqsar.assemblers.classifier`. It shares the building
blocks (base preprocessing, the four head families, the Ridge gate) but none of the
classification calibration: no prior correction, no decision cutoff, no reference rank.

Targets are standardised once, here, before anything is fitted. Every head and the
pooler see z-scores; :meth:`LazyRegressor.predict` and the ONNX artifact map back with the
stored ``y_mean`` / ``y_std``. That keeps the pooler's Gaussian oracle and every head's
hyperparameter rules on the same scale whatever units the assay uses.

Batching is by size only: below ``max_batch_size`` one batch; above it, quantile-stratified
partitions so every batch spans the full target range, averaged at inference.
"""

import numpy as np

from lazyqsar.utils.logging import logger
from lazyqsar.utils.metrics import spearman_score
from lazyqsar.utils.ranking import rank_from_knots
from lazyqsar.utils.splits import check_trainable_regression

from ..poolers.regression import InnerPooler
from ..portfolios.regression import Portfolio
from ._base import _AssemblerBase, _BatchBase


def _plan_regression_batches(y, max_batch_size=100_000, random_state=42):
    """Index arrays, one per batch, each spanning the full range of *y*.

    Rows are sorted by target and dealt round-robin (with a random order inside each
    group of ``k``) into ``k`` batches, so every batch receives one row from each
    consecutive block of ``k`` -- a quantile-stratified partition.
    """
    n = len(y)
    if n <= max_batch_size:
        return [np.arange(n)]
    k = int(np.ceil(n / max_batch_size))
    rng = np.random.default_rng(random_state)
    order = np.argsort(y, kind="stable")
    assignment = np.empty(n, dtype=int)
    for start in range(0, n, k):
        block = order[start : start + k]
        assignment[block] = rng.permutation(k)[: len(block)]
    return [np.sort(np.where(assignment == b)[0]) for b in range(k)]


class _BatchLazyRegressor(_BatchBase):
    """Single-batch ensemble on standardised targets: preprocessor + heads + gate."""

    task = "regression"
    _pooler_cls = InnerPooler
    _oof_attr = "oof_preds_"

    def fit(self, X, z):
        """Fit preprocessor, heads and pooler on (X, z) with z standardised."""
        self._fit_batch(X, z)

    def predict(self, X):
        """Gated prediction in standardised units, shape (n,)."""
        X_prep = self.prep.transform(X)
        P = np.column_stack([head.predict(X_prep) for head in self.heads])
        return self.pooler.predict(P, X_prep)


class LazyRegressor(_AssemblerBase):
    """
    Ensemble regressor for pre-computed feature arrays.

    Parameters
    ----------
    max_batch_size : int
        Maximum samples per batch; larger datasets are split into quantile-stratified
        batches whose predictions are averaged.
    max_rounds : int or None
        XGBoost round cap passed to each batch.

    Attributes (after fit)
    ----------------------
    portfolio : list of str
    models : list of _BatchLazyRegressor
    y_mean_, y_std_ : float
        Target standardisation; predictions are returned in original units.
    oof_rho_ : float or None
        Spearman ρ of the pooled out-of-fold predictions.
    train_rho_ : float or None
        Spearman ρ on the training set (optimistic).
    """

    task = "regression"

    def __init__(self, max_batch_size=100_000, max_rounds=None):
        self.max_batch_size = max_batch_size
        self.max_rounds = max_rounds

    def fit(self, X, y):
        logger.rule("LazyRegressor — fit")
        y = np.asarray(y, dtype=float).ravel()
        check_trainable_regression(y, where="LazyRegressor.fit")
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        z = (y - self.y_mean_) / self.y_std_

        p = Portfolio()
        p.fit(X, y)
        self.portfolio = p.get()
        logger.dataset_table(X.shape, y=y, portfolio=self.portfolio, task="regression")

        batch_indices = _plan_regression_batches(y, self.max_batch_size)
        logger.batch_table([len(idx) for idx in batch_indices], strategy="sequential")

        self.models = []
        for b, idx in enumerate(batch_indices):
            logger.info(
                f"Batch {b + 1}/{len(batch_indices)} — n={len(idx):,}  "
                f"portfolio={self.portfolio}"
            )
            model = _BatchLazyRegressor(
                portfolio=self.portfolio, max_rounds=self.max_rounds
            )
            model.fit(X[idx], z[idx])
            self.models.append(model)
        self.batch_indices_ = batch_indices

        self._build_oof_percentile(X, y)
        self.train_rho_ = spearman_score(y, self.predict(X))

        def _fmt(value):
            return "unknown" if value is None else f"{value:.4f}"

        logger.success(
            f"LazyRegressor fitted — {len(self.models)} batch(es)  "
            f"portfolio={self.portfolio}  OOF ρ={_fmt(self.oof_rho_)}  "
            f"train ρ={_fmt(self.train_rho_)}"
        )

    # ------------------------------------------------------------------
    # Out-of-fold reference
    # ------------------------------------------------------------------

    def _oof_accumulate(self, X):
        """Pooled out-of-fold prediction per training row, original units, or None.

        Each batch's heads are pooled with that batch's gating weights exactly as at
        inference, then destandardised. Batches partition the rows, so every row has
        exactly one OOF value. ``None`` when any head kept no OOF predictions.
        """
        indices = getattr(self, "batch_indices_", None)
        if not indices or not getattr(self, "models", None):
            return None
        out = np.full(X.shape[0], np.nan, dtype=np.float64)
        for batch, idx in zip(self.models, indices):
            heads = [getattr(h, "model", None) for h in batch.heads]
            if not all(m is not None and hasattr(m, "oof_preds_") for m in heads):
                return None
            P = np.column_stack([m.oof_preds_ for m in heads])
            if P.shape[0] != len(idx):
                return None
            # Fixed stacking weights need no features, so the training rows are not
            # pushed through the preprocessor again just to be counted.
            pooled = (
                batch.pooler._pool(P)
                if getattr(batch.pooler, "_head_weights", None) is not None
                else (batch.pooler.get_weights(batch.prep.transform(X[idx])) * P).sum(
                    axis=1
                )
            )
            out[idx] = pooled * self.y_std_ + self.y_mean_
        if not np.isfinite(out).all():
            return None
        return out

    def _build_oof_percentile(self, X, y):
        """OOF ECDF knots (the reliability rank) and the out-of-fold ρ."""
        self.oof_rho_ = None
        pooled = self._oof_accumulate(X)
        self._set_oof_reference(pooled)
        if pooled is None:
            logger.warning(
                "No out-of-fold predictions were kept, so the out-of-fold ρ is unknown "
                "for this model."
            )
            return
        self.oof_preds_ = pooled
        self.oof_rho_ = spearman_score(y, pooled)

    def oof_channels(self, X):
        """Out-of-fold ``(value, rank)`` per training row, or None. Fit-time only."""
        prepared = getattr(self, "_oof_percentile_prepared_", None)
        pooled = self._oof_accumulate(X)
        if pooled is None or prepared is None:
            return None
        return pooled, rank_from_knots(pooled, prepared=prepared)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X):
        """Predicted values in original target units, shape (n,)."""
        logger.debug(f"predict: X={X.shape}  batches={len(self.models)}")
        Z = np.array([m.predict(X) for m in self.models])
        return Z.mean(axis=0) * self.y_std_ + self.y_mean_

    def _oof_percentile_from_value(self, values):
        prepared = getattr(self, "_oof_percentile_prepared_", None)
        if prepared is None:
            return None
        return rank_from_knots(np.asarray(values, dtype=np.float64), prepared=prepared)

    def _oof_percentile(self, X):
        """Percentile of the prediction among the OOF predictions, shape (n,).

        The reliability signal ``LazyRegressorQSAR`` weights descriptors by. Not a
        public output: it is relative to this model's own training set.
        """
        rank = self._oof_percentile_from_value(self.predict(X))
        if rank is None:
            raise RuntimeError("This model kept no out-of-fold reference.")
        return rank

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _metadata(self) -> dict:
        return {
            "portfolio": self.portfolio,
            "max_batch_size": self.max_batch_size,
            "num_batches": len(self.models),
            "y_mean": self.y_mean_,
            "y_std": self.y_std_,
            "oof_rho": self.oof_rho_,
            "train_rho": self.train_rho_,
        }
