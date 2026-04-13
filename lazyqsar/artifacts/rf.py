"""
Inference-only artifact for BaseRFClassifier.

Loads a randomforest.onnx written by BaseRFClassifier.save().
Only requires numpy and onnxruntime for ONNX inference.
"""

from lazyqsar.base.randomforest import BaseRFArtifact as RandomForestArtifact

