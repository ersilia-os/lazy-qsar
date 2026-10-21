"""
Regressor-level inference artifact.

LazyRegressorArtifact — loads the batch structure written by
``lazyqsar.assemblers.regressor.LazyRegressor.save()``.

Only requires numpy, onnxruntime, json, os — no sklearn or xgboost.
"""

import json
import os

import numpy as np

from lazyqsar.ensemble.intervals import DEFAULT_LEVEL, interval_half_width
from lazyqsar.utils.ranking import prepare_knots, rank_from_knots

from ..poolers.regression.inner_pooler import InnerRegressorPoolerArtifact
from .linear import LinearArtifact
from .preprocessor import PreprocessorArtifact
from .rf import RandomForestArtifact
from .svc import SVRArtifact
from .xgboost import XGBoostArtifact


def _load_head(directory: str, head_name: str):
    if head_name == "xgb":
        return XGBoostArtifact.load(directory)
    if head_name == "lr":
        return LinearArtifact.load(directory)
    if head_name == "rf":
        return RandomForestArtifact.load(directory)
    if head_name == "svr":
        return SVRArtifact.load(directory)
    raise ValueError(f"Unknown regression head {head_name!r}")


class _BatchArtifact:
    """Inference over one batch: preprocessor → heads → gated mean (standardised units)."""

    @classmethod
    def load(cls, directory: str, portfolio: list) -> "_BatchArtifact":
        self = cls.__new__(cls)
        self.preprocessor = PreprocessorArtifact.load(directory)
        self.heads = [_load_head(directory, name) for name in portfolio]
        self.pooler = InnerRegressorPoolerArtifact.load(directory)
        return self

    def predict(self, X) -> np.ndarray:
        X_t = self.preprocessor.run(X)
        P = np.column_stack([h.run(X_t) for h in self.heads])  # (n, n_heads)
        return self.pooler.predict(P, X_t)


class LazyRegressorArtifact:
    """
    Inference-only loader for a model saved by ``LazyRegressor.save()``.

    Directory structure expected:
        model_dir/
            metadata.json            ("task": "regression")
            batch_0/
                preprocessor.onnx / preprocessor.json
                linear.* / xgboost.* / randomforest.* / svr.*   (per head)
                pooler.json
            batch_1/ ...
    """

    task = "regression"

    @classmethod
    def load(cls, directory: str) -> "LazyRegressorArtifact":
        meta_path = os.path.join(directory, "metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"metadata.json not found in {directory!r}")
        with open(meta_path) as f:
            metadata = json.load(f)
        if metadata.get("task") != "regression":
            raise ValueError(
                f"{directory!r} is not a regression checkpoint "
                f"(task={metadata.get('task', 'classification')!r})."
            )
        self = cls.__new__(cls)
        self.metadata = metadata
        portfolio = metadata["portfolio"]
        self._batches = [
            _BatchArtifact.load(os.path.join(directory, f"batch_{i}"), portfolio)
            for i in range(metadata["num_batches"])
        ]
        self._y_mean = float(metadata["y_mean"])
        self._y_std = float(metadata["y_std"])
        knots = (metadata.get("oof_percentile") or {}).get("knots")
        self._oof_percentile_prepared = (
            prepare_knots(np.asarray(knots, dtype=np.float64))
            if knots is not None and len(knots)
            else None
        )
        return self

    @property
    def oof_rho_(self):
        return self.metadata.get("oof_rho")

    @property
    def train_rho_(self):
        return self.metadata.get("train_rho")

    @property
    def oof_metrics_(self):
        return self.metadata.get("oof_metrics")

    def predict(self, X) -> np.ndarray:
        """Predicted values in original target units, shape (n_samples,)."""
        Z = np.array([b.predict(X) for b in self._batches])
        return Z.mean(axis=0) * self._y_std + self._y_mean

    def predict_interval(self, X, level: float = DEFAULT_LEVEL):
        """Prediction interval ``(lower, upper)`` at coverage *level*, each ``(n,)``.

        Split-conformal on the pooled out-of-fold residuals (one bin: this level has no
        applicability domain). See :mod:`lazyqsar.ensemble.intervals`.
        """
        model = self._interval_model()
        if not model:
            raise ValueError(
                "This model carries no prediction-interval model (fitted before "
                "intervals existed); refit it to get intervals."
            )
        value = np.asarray(self.predict(X), dtype=np.float64)
        half = interval_half_width(model, len(value), None, level)
        return value - half, value + half

    def _interval_model(self):
        return self.metadata.get("intervals")

    def _oof_percentile_from_value(self, values):
        """OOF-ECDF rank of predictions the caller already has; None if no reference."""
        if self._oof_percentile_prepared is None:
            return None
        return rank_from_knots(
            np.asarray(values, dtype=np.float64), prepared=self._oof_percentile_prepared
        )

    def _oof_percentile(self, X) -> np.ndarray:
        """Percentile of the prediction among the training OOF predictions, shape (n,)."""
        rank = self._oof_percentile_from_value(self.predict(X))
        if rank is None:
            raise RuntimeError("This checkpoint carries no out-of-fold reference.")
        return rank
