"""
Inference-only artifact for BaseLinearClassifier / BaseLinearRegressor.

Loads a linear.onnx written by BaseLinearClassifier.save() or BaseLinearRegressor.save().
Only requires numpy and onnxruntime — no sklearn.
"""

import numpy as np

from lazyqsar.artifacts._head import HeadArtifact, proba_from_skl2onnx


class LinearArtifact(HeadArtifact):
    """Load and run a saved linear model ONNX pipeline.

    The one inference implementation for linear heads; ``BaseLinearArtifact`` is an
    alias of this class.
    """

    _stem = "linear"

    def _score_from_outputs(self, outputs) -> np.ndarray:
        # skl2onnx pipeline: outputs[0]=labels, outputs[1]=proba (array or list of dicts)
        return proba_from_skl2onnx(outputs[1])
