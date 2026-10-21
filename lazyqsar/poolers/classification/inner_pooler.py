import numpy as np

from lazyqsar.poolers._gate import (  # noqa: F401 - re-exported for callers
    _FIT_DEPS_AVAILABLE,
    _heuristic_alpha,
    _softmax,
    fit_gate,
    gate_weights,
    GatedPooler,
    GatedPoolerArtifact,
)
from lazyqsar.utils.metrics import composite_score


class InnerClassifierPooler(GatedPooler):
    """
    Gating network that learns per-sample head weights from OOF probabilities.

    The Ridge target for head j on sample i is the softmax of Bernoulli
    log-likelihoods across heads -- how well each head explains each training sample.
    At inference, weights are a linear projection of the preprocessed features through
    the Ridge coefficients, softmax-normalized across heads.
    """

    task = "classification"
    _metric_label = "AUC"

    def _prepare_y(self, y) -> np.ndarray:
        return np.asarray(y, dtype=int)

    def _log_likelihood(self, S, y) -> np.ndarray:
        eps = 1e-7
        return np.where(
            y[:, None] == 1,
            np.log(np.clip(S, eps, 1 - eps)),
            np.log(np.clip(1.0 - S, eps, 1 - eps)),
        )

    def _skill(self, y, p):
        return composite_score(y, p)

    def predict_proba(self, R, X_prep=None):
        """Gated ensemble probabilities, shape (n, 2), from per-head P(y=1) ``R``."""
        p1 = self._pool(R, X_prep)
        return np.column_stack([1 - p1, p1])


# Inference: one pooler artifact for both tasks.
InnerPoolerArtifact = GatedPoolerArtifact
