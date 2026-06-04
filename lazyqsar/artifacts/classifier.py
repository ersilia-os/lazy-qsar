"""
Classifier-level inference artifacts.

LazyClassifierArtifact — loads the batch structure written by LazyClassifier.save().

Only requires numpy, onnxruntime, json, os — no sklearn or xgboost.
"""

import json
import os

import numpy as np


def _correct_prior(p1, train_prior, population_prior):
    """Adjust P(y=1) trained at train_prior to reflect population_prior."""
    if abs(train_prior - population_prior) < 1e-9:
        return p1
    if train_prior <= 0.0 or train_prior >= 1.0:
        return p1
    ratio = (population_prior / train_prior) / (
        (1.0 - population_prior) / (1.0 - train_prior)
    )
    odds = p1 / np.clip(1.0 - p1, 1e-15, None)
    corrected_odds = ratio * odds
    return corrected_odds / (1.0 + corrected_odds)


from .preprocessor import PreprocessorArtifact  # noqa: E402
from .xgboost import XGBoostArtifact  # noqa: E402
from .linear import LinearArtifact  # noqa: E402
from .rf import RandomForestArtifact  # noqa: E402
from .svc import SVCArtifact  # noqa: E402
from ..poolers.classification.inner_pooler import InnerPoolerArtifact  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers used by LazyClassifierArtifact
# ---------------------------------------------------------------------------


def _load_head(directory: str, head_name: str):
    if head_name == "xgb":
        return XGBoostArtifact.load(directory)
    if head_name == "lr":
        return LinearArtifact.load(directory)
    if head_name == "rf":
        return RandomForestArtifact.load(directory)
    if head_name == "svc":
        return SVCArtifact.load(directory)
    raise ValueError(f"Unknown head {head_name!r}")


class _BatchArtifact:
    """Inference over one batch: preprocessor → heads → weighted average."""

    @classmethod
    def load(cls, directory: str, portfolio: list) -> "_BatchArtifact":
        self = cls.__new__(cls)
        self.preprocessor = PreprocessorArtifact.load(directory)
        self.heads = [_load_head(directory, name) for name in portfolio]
        self.pooler = InnerPoolerArtifact.load(directory)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_t = self.preprocessor.run(X)
        R = np.column_stack([h.run(X_t)[:, 1] for h in self.heads])  # (n, n_heads)
        return self.pooler.predict_proba(R, X_t)

    def predict_score(self, X) -> np.ndarray:
        X_t = self.preprocessor.run(X)
        R = np.column_stack([h.predict_score(X_t)[:, 1] for h in self.heads])
        W = self.pooler.get_weights(X_t)
        score_1 = (W * R).sum(axis=1)
        return np.column_stack([1 - score_1, score_1])

    def predict_rank(self, X) -> np.ndarray:
        X_t = self.preprocessor.run(X)
        R = np.column_stack([h.predict_rank(X_t)[:, 1] for h in self.heads])
        W = self.pooler.get_weights(X_t)
        rank_1 = (W * R).sum(axis=1)
        return np.column_stack([1 - rank_1, rank_1])


# ---------------------------------------------------------------------------
# Top-level artifact: matches LazyClassifier.save() output
# ---------------------------------------------------------------------------


class LazyClassifierArtifact:
    """
    Inference-only loader for a model saved by LazyClassifier.save().

    Directory structure expected:
        model_dir/
            metadata.json
            batch_0/
                preprocessor.onnx / preprocessor.json
                xgboost.onnx / xgboost.json   (xgb head)
                linear.onnx  / linear.json    (lr head)
                pooler.json
            batch_1/ ...
    """

    @classmethod
    def load(cls, directory: str) -> "LazyClassifierArtifact":
        meta_path = os.path.join(directory, "metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"metadata.json not found in {directory!r}")
        self = cls.__new__(cls)
        with open(meta_path) as f:
            metadata = json.load(f)
        portfolio = metadata["portfolio"]
        num_batches = metadata["num_batches"]
        self._batches = [
            _BatchArtifact.load(os.path.join(directory, f"batch_{i}"), portfolio)
            for i in range(num_batches)
        ]
        self._population_prior = metadata.get("population_prior", None)
        self._batch_priors = metadata.get("batch_priors", None)
        if "decision_cutoff_raw" in metadata:
            self._decision_cutoff_raw = float(metadata["decision_cutoff_raw"])
        else:
            all_cutoffs = [
                h.metadata.get("decision_cutoff_raw", 0.5)
                for b in self._batches
                for h in b.heads
            ]
            self._decision_cutoff_raw = (
                float(np.mean(all_cutoffs)) if all_cutoffs else 0.5
            )
        self._decision_cutoff_proba = float(metadata.get("decision_cutoff_proba", 0.5))
        self._decision_cutoff_rank = float(metadata.get("decision_cutoff_rank", 0.5))
        self._decision_cutoff_logit = float(metadata.get("decision_cutoff_logit", 0.0))
        raw_lift = metadata.get("decision_cutoff_lift")
        self._decision_cutoff_lift = float(raw_lift) if raw_lift is not None else None
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities, shape (n_samples, 2)."""
        if self._population_prior is not None and self._batch_priors is not None:
            R = np.array(
                [
                    _correct_prior(b.predict_proba(X)[:, 1], tp, self._population_prior)
                    for b, tp in zip(self._batches, self._batch_priors)
                ]
            )
        else:
            R = np.array([b.predict_proba(X)[:, 1] for b in self._batches])
        proba = R.mean(axis=0)
        return np.column_stack([1 - proba, proba])

    @property
    def decision_cutoff_proba(self) -> float:
        """Threshold to apply against predict_proba() output for binary predictions."""
        return self._decision_cutoff_proba

    @property
    def decision_cutoff_rank(self) -> float:
        """Threshold to apply against predict_rank() output for binary predictions."""
        return self._decision_cutoff_rank

    @property
    def decision_cutoff_logit(self) -> float:
        """Threshold to apply against predict_logit() output for binary predictions."""
        return self._decision_cutoff_logit

    @property
    def decision_cutoff_lift(self):
        """Threshold to apply against predict_lift() output; None if population_prior unavailable."""
        return self._decision_cutoff_lift

    def predict(self, X, cutoff: float = None) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        threshold = self._decision_cutoff_raw if cutoff is None else cutoff
        return (self.predict_score(X)[:, 1] >= threshold).astype(int)

    def predict_lift(self, X) -> np.ndarray:
        """Return lift over population prior, shape (n_samples, 2)."""
        if self._population_prior is None:
            raise RuntimeError("No population_prior stored in this artifact.")
        proba = self.predict_proba(X)
        return np.column_stack(
            [
                proba[:, 0] / (1.0 - self._population_prior),
                proba[:, 1] / self._population_prior,
            ]
        )

    def predict_logit(self, X) -> np.ndarray:
        """Return logit of calibrated probabilities averaged across batches, shape (n_samples, 2)."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_score(self, X) -> np.ndarray:
        """Return raw (pre-calibration) probabilities averaged across batches, shape (n_samples, 2)."""
        R = np.array([b.predict_score(X)[:, 1] for b in self._batches])
        proba = R.mean(axis=0)
        return np.column_stack([1 - proba, proba])

    def predict_rank(self, X) -> np.ndarray:
        """Return [0, 1] ranks averaged across batches, shape (n_samples, 2)."""
        R = np.array([b.predict_rank(X)[:, 1] for b in self._batches])
        rank_1 = R.mean(axis=0)
        return np.column_stack([1 - rank_1, rank_1])
