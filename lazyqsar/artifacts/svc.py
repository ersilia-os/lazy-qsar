"""
Inference-only artifact for BaseSVCClassifier.

Loads an svc.onnx written by BaseSVCClassifier.save().
Only requires numpy and onnxruntime — no sklearn or SVC.
"""

from lazyqsar.base.svc import BaseSVCArtifact as SVCArtifact  # noqa: F401
