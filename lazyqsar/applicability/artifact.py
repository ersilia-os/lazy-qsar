"""Inference-only applicability domain artifact.

Split out of :mod:`lazyqsar.applicability.domain` so it can be imported on the base
install. ``domain.py`` imports scikit-learn at module scope for the *fit-time*
estimator, and the package ``__init__`` used to import both classes from it, so loading
the inference-only class dragged scikit-learn in with it.

That was harmless only for as long as nothing on the inference path touched the
applicability domain. It stops being harmless the moment the shared prediction runner
loads AD artifacts, because Ersilia Model Hub templates install neither scikit-learn nor
SciPy — the failure would surface in a deployed container, not in CI.

This module must stay importable under numpy + onnxruntime alone.
``dev/tests/test_inference_purity.py`` enforces that.
"""

from __future__ import annotations

import json
import os

import numpy as np


class ApplicabilityDomainArtifact:
    """
    Inference-only applicability domain loaded from a saved ONNX model.

    Requires only onnxruntime and numpy — no sklearn, no scipy.
    """

    def __init__(self) -> None:
        self._session = None
        self.metadata: dict = {}

    @classmethod
    def load(cls, directory: str) -> "ApplicabilityDomainArtifact":
        inst = cls()
        json_path = os.path.join(directory, "applicability_domain.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"No AD metadata found at {json_path!r}")
        with open(json_path) as fh:
            inst.metadata = json.load(fh)

        import onnxruntime as rt

        onnx_path = os.path.join(directory, "applicability_domain.onnx")
        inst._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        return inst

    def score(self, X) -> np.ndarray:
        """
        Return AD scores in [0, 1] for each row of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix — same featurizer as used during fit.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,), dtype float32
            1.0 = fully in-domain, 0.0 = fully out-of-domain.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_f32 = np.asarray(X, dtype=np.float32)
        input_name = self._session.get_inputs()[0].name
        return self._session.run(None, {input_name: X_f32})[0]
