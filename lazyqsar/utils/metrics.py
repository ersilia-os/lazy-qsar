"""
Evaluation metrics for imbalance-aware classifier assessment.

bedroc_score  : Boltzmann-Enhanced Discrimination of ROC (Truchon & Bayly 2007)
aupr_score    : Area under precision-recall curve (average precision)
composite_score: Mean normalized excess over random for AUROC, AUPR, BEDROC
"""

import numpy as np


_AUROC_RANDOM_BASELINE = 0.5


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_excess(value: float, baseline: float, upper: float = 1.0) -> float:
    """Normalize a metric to its useful range above baseline, clipped to [0, 1]."""
    if not np.isfinite(value) or not np.isfinite(baseline) or not np.isfinite(upper):
        return 0.0
    denom = upper - baseline
    if denom <= 0.0:
        return 0.0
    return _clip01((value - baseline) / denom)


def _bedroc_rie_components(n: int, n_a: int, alpha: float) -> tuple[float, float]:
    """Return theoretical minimum and maximum RIE for the given label prevalence."""
    Ra = n_a / n
    denom = Ra * (
        np.exp(-alpha / n) * (1.0 - np.exp(-alpha)) / (1.0 - np.exp(-alpha / n))
    )

    top_geo = np.exp(-alpha / n) * (1.0 - np.exp(-alpha * Ra)) / (1.0 - np.exp(-alpha / n))
    rie_max = top_geo / denom

    start = n - n_a + 1
    bot_geo = np.exp(-alpha * start / n) * (1.0 - np.exp(-alpha * Ra)) / (1.0 - np.exp(-alpha / n))
    rie_min = bot_geo / denom
    return float(rie_min), float(rie_max)


def bedroc_random_baseline(y_true: np.ndarray, alpha: float = 20.0) -> float:
    """
    Expected BEDROC value under random ranking for the observed class prevalence.

    Since E[RIE_random] = 1, the random BEDROC baseline is obtained by mapping
    that RIE value back to the normalized BEDROC scale.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    n_a = int(y_true.sum())
    if n == 0 or n_a == 0 or n_a == n:
        return 0.0

    rie_min, rie_max = _bedroc_rie_components(n=n, n_a=n_a, alpha=alpha)
    denom = rie_max - rie_min
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return _clip01((1.0 - rie_min) / denom)


def bedroc_score(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    BEDROC score (Truchon & Bayly 2007, J. Chem. Inf. Model.).

    Returns a value in [0, 1]:
      1.0  when all actives are ranked at the very top
      0.0  when all actives are ranked at the very bottom
      ~0.5 at random

    alpha=20 is the standard cheminformatics setting: roughly 80% of the
    score weight falls on the top 5% of the ranked list.

    Parameters
    ----------
    y_true  : binary array (0/1)
    y_score : continuous score (higher = more likely positive)
    alpha   : enrichment weight parameter (default 20)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    n_a = int(y_true.sum())
    if n_a == 0:
        return 0.0
    if n_a == n:
        return 1.0

    Ra = n_a / n

    # Ranks (1-indexed) of true positives in descending score order
    order = np.argsort(-y_score)
    ranks = np.where(y_true[order] == 1)[0] + 1  # shape (n_a,)

    # Exponential sum over active ranks
    ri_sum = np.exp(-alpha * ranks / n).sum()

    # Expected exponential sum under random ordering (geometric series)
    # = Ra * sum_{i=1}^{n} exp(-alpha*i/n) / n
    #   where sum_{i=1}^{n} exp(-alpha*i/n) = exp(-alpha/n) * (1-exp(-alpha)) / (1-exp(-alpha/n))
    geo = np.exp(-alpha / n) * (1.0 - np.exp(-alpha)) / (1.0 - np.exp(-alpha / n))
    denom = Ra * geo

    rie = ri_sum / denom

    rie_min, rie_max = _bedroc_rie_components(n=n, n_a=n_a, alpha=alpha)

    return float((rie - rie_min) / (rie_max - rie_min))


def aupr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area under the precision-recall curve (average precision score).

    Equivalent to sklearn's average_precision_score — reimported here
    so callers only need to import from this module.
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y_true), np.asarray(y_score)))


def composite_metrics(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> dict:
    """
    Return raw metrics, random baselines, normalized excess components, and composite.

    The normalized components map each metric's useful range above random
    performance to [0, 1] before averaging.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.size == 0:
        prevalence = 0.0
        auroc = _AUROC_RANDOM_BASELINE
        aupr = 0.0
        bedroc = 0.0
    else:
        prevalence = float(np.mean(y_true))
        unique = np.unique(y_true)
        auroc = _AUROC_RANDOM_BASELINE if unique.size < 2 else float(roc_auc_score(y_true, y_score))
        aupr = float(prevalence) if unique.size < 2 else aupr_score(y_true, y_score)
        bedroc = bedroc_score(y_true, y_score, alpha=alpha)

    auroc_random = _AUROC_RANDOM_BASELINE
    aupr_random = prevalence
    bedroc_random = bedroc_random_baseline(y_true, alpha=alpha)

    auroc_excess = _safe_excess(auroc, auroc_random)
    aupr_excess = _safe_excess(aupr, aupr_random)
    bedroc_excess = _safe_excess(bedroc, bedroc_random)
    composite = float(np.mean([auroc_excess, aupr_excess, bedroc_excess]))

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "bedroc": float(bedroc),
        "auroc_random": float(auroc_random),
        "aupr_random": float(aupr_random),
        "bedroc_random": float(bedroc_random),
        "auroc_excess": float(auroc_excess),
        "aupr_excess": float(aupr_excess),
        "bedroc_excess": float(bedroc_excess),
        "composite": composite,
    }


def composite_score(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    Composite classification score = mean normalized excess over random.

    Each component is normalized to its useful range above a random baseline:
      - AUROC vs 0.5
      - AUPR vs class prevalence
      - BEDROC vs its expected random-ranking baseline

    Parameters
    ----------
    y_true  : binary array (0/1)
    y_score : continuous score (higher = more likely positive)
    alpha   : BEDROC alpha parameter (default 20)
    """
    return composite_metrics(y_true, y_score, alpha=alpha)["composite"]
