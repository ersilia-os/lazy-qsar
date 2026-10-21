"""Rules shared by the heads' preset races.

Each head compares a few hyperparameter presets on held-out data (a validation split,
or out-of-bag rows for the random forest) and keeps a challenger only when it beats the
baseline preset by more than the comparison's own noise.
"""

from __future__ import annotations

# Floor on the margin a challenger preset must win by.
PORTFOLIO_MIN_GAIN = 0.005
# Below this many training rows the held-out score is noisier, so the margin is wider.
SMALL_N = 2_000


def min_gain_threshold(n_eff: int, n: int) -> float:
    """Margin a challenger preset must beat the baseline by: ``max(0.005, c / sqrt(n_eff))``.

    The noise of a held-out score shrinks as ``1/sqrt(n_eff)``: ``n_eff`` is the
    minority-class count for AUC, the row count for (relative) RMSE. ``c`` is 0.3 when
    the training set has fewer than ``SMALL_N`` rows, where even averaged splits leave
    spurious gains of ~0.02, and 0.1 above it.

    Examples: n_eff=100 at small n gives 0.030; n_eff=200 at large n gives 0.007.
    """
    coef = 0.3 if n < SMALL_N else 0.1
    return max(PORTFOLIO_MIN_GAIN, coef / max(1, n_eff) ** 0.5)
