"""One inference implementation for every saved head, whatever its family or task.

A head saves ``<stem>.onnx`` and ``<stem>.json``. Loading and running it differ by
family only in how the class-1 score (classifiers) or the value (regressors) is read
from the graph's outputs; everything else -- metadata, calibration, decision cutoffs,
ranks -- is the same and lives here. numpy + onnxruntime only: this is the inference
path.
"""

from __future__ import annotations

import json
import os

import numpy as np

from lazyqsar.artifacts._common import (
    DEFAULT_DECISION_CUTOFF,
    apply_calibrator,
)
from lazyqsar.utils.ranking import binarize, prepare_knots, rank_from_knots


def proba_from_skl2onnx(prob_raw) -> np.ndarray:
    """skl2onnx's probability output -- an array or a list of per-row dicts -- as (n, 2)."""
    if isinstance(prob_raw, list):
        return np.array([[d[k] for k in sorted(d)] for d in prob_raw], dtype=np.float64)
    proba = np.asarray(prob_raw, dtype=np.float64)
    if proba.ndim == 1:
        return np.column_stack([1 - proba, proba])
    return proba


class HeadArtifact:
    """A saved head: ``load`` it, then ``run`` / ``predict_*`` exactly as the fitted head.

    Subclasses set ``_stem`` and implement ``_score_from_outputs(outputs)`` (classifier
    families: the pre-calibration ``(n, 2)`` score) and, if the value is not the first
    output as-is, ``_values_from_outputs(outputs)`` (regressor families).
    """

    _stem = ""

    task: str = ""
    decision_cutoff_raw: float = DEFAULT_DECISION_CUTOFF
    decision_cutoff_raw_source: str = "default_0.5"
    decision_cutoff_proba: float = DEFAULT_DECISION_CUTOFF
    decision_cutoff_rank: float = DEFAULT_DECISION_CUTOFF
    decision_cutoff_logit: float = 0.0

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, directory: str):
        json_path = os.path.join(directory, f"{cls._stem}.json")
        onnx_path = os.path.join(directory, f"{cls._stem}.onnx")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"{cls._stem}.json not found in {directory!r}")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"{cls._stem}.onnx not found in {directory!r}")
        self = cls.__new__(cls)
        with open(json_path) as f:
            self.metadata = json.load(f)
        fmt = self.metadata.get("format", "onnx")
        if fmt != "onnx":
            raise ValueError(
                f"Unsupported {cls._stem} artifact format {fmt!r} in {directory!r}. "
                "Only 'onnx' is supported."
            )
        self.task = self.metadata.get("task", "classification")
        self._session = self._build_session(onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        self._cal = self.metadata.get("calibrator", None)
        self._ranker = self.metadata.get("ranker", None)
        self._ranker_prepared = None
        self._load_cutoffs(self.metadata)
        self._after_load()
        return self

    @staticmethod
    def _build_session(onnx_path: str):
        import onnxruntime as rt

        return rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _after_load(self) -> None:
        """Family-specific set-up once the session and metadata are in place."""

    def _load_cutoffs(self, metadata: dict) -> None:
        self.decision_cutoff_raw = float(
            metadata.get("decision_cutoff_raw", DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_raw_source = metadata.get(
            "decision_cutoff_raw_source", "default_0.5"
        )
        self.decision_cutoff_proba = float(
            metadata.get("decision_cutoff_proba", DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_rank = float(
            metadata.get("decision_cutoff_rank", DEFAULT_DECISION_CUTOFF)
        )
        self.decision_cutoff_logit = float(metadata.get("decision_cutoff_logit", 0.0))

    # ------------------------------------------------------------------ outputs

    def _outputs(self, X) -> list:
        X_f32 = np.asarray(X, dtype=np.float32)
        return self._session.run(None, {self._input_name: X_f32})

    def _score_from_outputs(self, outputs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _values_from_outputs(self, outputs) -> np.ndarray:
        return np.asarray(outputs[0], dtype=np.float64).ravel()

    # ------------------------------------------------------------------ predict

    def predict_score(self, X) -> np.ndarray:
        """Raw (pre-calibration) class scores, shape (n_samples, 2)."""
        return self._score_from_outputs(self._outputs(X))

    def run(self, X) -> np.ndarray:
        """Calibrated ``(n, 2)`` probabilities (classification) or ``(n,)`` values."""
        if self.task == "regression":
            return self._values_from_outputs(self._outputs(X))
        proba = self.predict_score(X)
        if self._cal is not None:
            proba = apply_calibrator(proba, self._cal)
        return proba

    def predict_logit(self, X) -> np.ndarray:
        """Logit of the calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.run(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict(self, X, cutoff: float | None = None) -> np.ndarray:
        """Class labels: the raw score at or above the raw decision cutoff.

        The same comparison the fitted head makes. (Comparing calibrated probabilities
        with the calibrated cutoff can disagree on rows an isotonic calibrator maps onto
        the cutoff's own plateau.)
        """
        if self.task != "classification":
            raise RuntimeError(
                "predict() is only available for classification artifacts."
            )
        threshold = self.decision_cutoff_raw if cutoff is None else float(cutoff)
        classes = np.asarray(self.metadata.get("classes", [0, 1]))
        return classes[binarize(self.predict_score(X)[:, 1], threshold)]

    def predict_rank(self, X) -> np.ndarray:
        """Map scores to [0, 1] via the OOF ECDF.

        Classification: shape (n_samples, 2). Regression: shape (n_samples,), the ECDF
        of the prediction over the OOF predictions.
        """
        if self._ranker is None:
            raise RuntimeError("No ranker stored in this artifact.")
        if getattr(self, "_ranker_prepared", None) is None:
            self._ranker_prepared = prepare_knots(self._ranker["knots"])
        if self.task == "regression":
            return rank_from_knots(self.run(X), prepared=self._ranker_prepared)
        rank_1 = rank_from_knots(
            self.predict_score(X)[:, 1], prepared=self._ranker_prepared
        )
        return np.column_stack([1 - rank_1, rank_1])
