"""
Inference-only artifacts for BaseSVCClassifier and BaseSVR.

Loads an svc.onnx written by BaseSVCClassifier.save(), or an svr.onnx written
by BaseSVR.save(). Only requires numpy and onnxruntime — no sklearn or SVC.
"""

from lazyqsar.base.svc import BaseSVCArtifact as SVCArtifact  # noqa: F401
from lazyqsar.base.svc import BaseSVRArtifact as SVRArtifact  # noqa: F401
