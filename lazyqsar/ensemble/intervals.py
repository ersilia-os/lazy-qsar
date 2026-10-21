"""Prediction intervals for regression: split-conformal, binned by applicability domain.

The ensemble's pooled out-of-fold predictions are honest held-out predictions for every
training row, so their absolute residuals are a calibration set (split-conformal
prediction; Vovk et al. 2005, Lei et al. 2018) that costs no extra fit. A residual
quantile at level ``q`` gives an interval ``value +- half_width`` that covers a new
molecule's true value with probability ``q``, *if* the new molecule resembles the
training ones.

Molecules far from the training set are predicted worse, so one residual distribution
for everybody would be too wide in the domain and too narrow outside it. The residuals
are therefore split into bins by the molecule's applicability-domain score (a Mondrian
conformal predictor: coverage holds within each bin), and a query takes the half-width
of the bin its own AD score falls in. A query below every training bin uses the lowest
one, the bin of the least typical training molecules.

numpy only; the interval model travels in ``metadata.json`` as plain lists.
"""

from __future__ import annotations

import numpy as np

# AD bins: tertiles of the training rows' AD score. Fewer than this many rows per bin and
# the bins are merged into one: a quantile of a handful of residuals is mostly noise.
_N_BINS = 3
_MIN_ROWS_PER_BIN = 30
# Residual quantiles are stored on this grid of coverage levels and interpolated.
LEVEL_GRID = tuple(np.round(np.linspace(0.0, 1.0, 101), 2).tolist())
DEFAULT_LEVEL = 0.9


def _conformal_quantiles(residuals: np.ndarray) -> list:
    """Half-width at each level of :data:`LEVEL_GRID`, with the finite-sample correction.

    Split-conformal coverage ``>= q`` needs the ``ceil((n + 1) q)``-th smallest residual,
    not the plain ``q`` quantile; above the largest residual it saturates there.
    """
    r = np.sort(np.abs(residuals))
    n = len(r)
    out = []
    for q in LEVEL_GRID:
        k = int(np.ceil((n + 1) * q)) - 1
        out.append(float(r[min(max(k, 0), n - 1)]))
    return out


def fit_intervals(y, oof_pred, ad_summary=None) -> dict | None:
    """The interval model from out-of-fold predictions and (optionally) AD scores.

    Returns ``{"levels", "ad_edges", "half_widths"}``: ``ad_edges`` are the inner bin
    edges on the AD score (empty for a single bin), ``half_widths[b]`` the residual
    quantiles of bin ``b`` on ``levels``. ``None`` when there are no usable residuals.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(oof_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ad_summary is not None:
        ad = np.asarray(ad_summary, dtype=float)
        ok &= np.isfinite(ad)
    if ok.sum() < 2:
        return None
    residuals = (y - pred)[ok]
    n_bins = _N_BINS if ok.sum() >= _N_BINS * _MIN_ROWS_PER_BIN else 1
    if ad_summary is None or n_bins == 1:
        return {
            "levels": list(LEVEL_GRID),
            "ad_edges": [],
            "half_widths": [_conformal_quantiles(residuals)],
        }
    ad = ad[ok]
    edges = np.quantile(ad, np.linspace(0, 1, n_bins + 1)[1:-1]).tolist()
    bins = np.searchsorted(edges, ad, side="right")
    widths = [_conformal_quantiles(residuals[bins == b]) for b in range(n_bins)]
    return {"levels": list(LEVEL_GRID), "ad_edges": edges, "half_widths": widths}


def interval_half_width(model: dict, n: int, ad_summary=None, level=DEFAULT_LEVEL):
    """Per-row half-width ``(n,)`` at coverage *level* in ``(0, 1)``."""
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"Interval level must be in (0, 1); got {level!r}.")
    levels = np.asarray(model["levels"], dtype=float)
    per_bin = np.array(
        [
            np.interp(float(level), levels, np.asarray(w, dtype=float))
            for w in model["half_widths"]
        ]
    )
    edges = model.get("ad_edges") or []
    if not edges or ad_summary is None:
        # One bin, or no AD at inference: the widest bin, so the interval errs wide.
        return np.full(n, per_bin.max(), dtype=float)
    ad = np.asarray(ad_summary, dtype=float)
    bins = np.searchsorted(np.asarray(edges, dtype=float), ad, side="right")
    width = per_bin[np.clip(bins, 0, len(per_bin) - 1)]
    # A row with no AD score gets the widest bin rather than a guess.
    return np.where(np.isfinite(ad), width, per_bin.max())


def ad_summary(A):
    """One applicability-domain score per row: the mean over the active descriptors."""
    if A is None:
        return None
    A = np.asarray(A, dtype=float)
    return A.mean(axis=1) if A.ndim == 2 else A
