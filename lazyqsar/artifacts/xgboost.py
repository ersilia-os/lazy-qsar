"""
Inference-only artifact for BaseXGBClassifier / BaseXGBRegressor.

Loads an xgboost.onnx written by BaseXGBClassifier.save() or BaseXGBRegressor.save().
Only requires numpy and onnxruntime — no xgboost or sklearn.
"""

import numpy as np
import onnxruntime as rt

from lazyqsar.artifacts._head import HeadArtifact


def _build_xgb_session(onnx_path: str) -> rt.InferenceSession:
    """Build an ORT session for an XGBoost classifier ONNX, suppressing the benign
    MergeShapeInfo warning emitted on legacy models (saved before the export-time fix).

    The warning comes from onnxmltools annotating the 'probabilities' output with
    dim_value=2 while ORT infers {N,1} from the missing n_targets attribute. When the
    `onnx` package is available we clear the fixed dim so the model becomes well-formed;
    otherwise we silence warnings on the session as a fallback.
    """
    try:
        import onnx  # type: ignore

        model = onnx.load(onnx_path)
        changed = False
        for _out in model.graph.output:
            if _out.name == "probabilities":
                _dims = _out.type.tensor_type.shape.dim
                if len(_dims) >= 2 and _dims[1].dim_value != 0:
                    _dims[1].ClearField("dim_value")
                    changed = True
                break
        if changed:
            return rt.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
    except Exception:
        pass

    so = rt.SessionOptions()
    so.log_severity_level = 3
    return rt.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


class XGBoostArtifact(HeadArtifact):
    """Load and run a saved XGBoost ONNX model."""

    _stem = "xgboost"
    _build_session = staticmethod(_build_xgb_session)

    def _score_from_outputs(self, outputs) -> np.ndarray:
        """The ``probabilities`` output as (n, 2), whatever shape the graph emits.

        A legacy export -- onnxmltools annotating ``probabilities`` with dim_value=2
        while ORT infers {N,1} from a missing n_targets, the case
        :func:`_build_xgb_session` exists to repair -- emits one column. Normalising it
        here, for ``run`` and ``predict_score`` alike, keeps ``predict_type="score"``
        from silently falling back to the calibrated probabilities. Every head shipped
        today emits ``(n, 2)``.
        """
        prob_output = next(
            o
            for o, meta in zip(outputs, self._session.get_outputs())
            if meta.name == "probabilities"
        )
        proba = np.asarray(prob_output, dtype=np.float64)
        if proba.ndim == 1 or proba.shape[1] == 1:
            proba = np.column_stack([1 - proba.ravel(), proba.ravel()])
        return proba
