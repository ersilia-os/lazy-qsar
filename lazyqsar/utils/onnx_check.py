"""Export self-check: a written ONNX graph must reproduce the model it was exported from.

Every head's ``to_onnx`` ends by running a small probe -- real rows from its own training
matrix -- through both the fitted model and the graph it just wrote, and raises
:class:`OnnxExportMismatch` if they disagree. The comparison is on the *raw* model output
(before calibration), because that is what the graph contains.

The tolerance -- 1e-5 relative plus 1e-5 times the output's scale -- catches structural
conversion errors, down to a single dropped tree of a slow-learning booster (~1e-3),
while never firing on the float32-versus-float64 rounding every export carries (measured
at ~1e-6 relative by the parity tests). Precision itself is pinned separately
by the parity tests; this check exists so that a wrong graph can never be *saved*.

Fit-time only: it needs onnxruntime (always installed) and the model's native predict.
"""

from __future__ import annotations

import numpy as np

PROBE_ROWS = 64
_RTOL = 1e-5
_ATOL_SCALE = 1e-5


class OnnxExportMismatch(RuntimeError):
    """The exported ONNX graph does not reproduce the model it came from."""


def probe_rows(X, n: int = PROBE_ROWS, seed: int = 0) -> np.ndarray:
    """A deterministic sample of up to *n* training rows, dense float32."""
    n_rows = X.shape[0]
    idx = np.sort(
        np.random.default_rng(seed).choice(n_rows, size=min(n, n_rows), replace=False)
    )
    rows = X[idx]
    if hasattr(rows, "toarray"):
        rows = rows.toarray()
    return np.ascontiguousarray(rows, dtype=np.float32)


def _run(path: str, probe: np.ndarray) -> list:
    import onnxruntime as rt

    session = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: probe}), session


def _positive_column(raw) -> np.ndarray:
    """Class-1 column of a probability output: (n, 2), (n, 1), (n,) or zipmap dicts."""
    if isinstance(raw, list):
        return np.array([d[max(d)] for d in raw], dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return arr[:, -1]


def _assert_close(got, expected, what: str, atol: float) -> None:
    got = np.asarray(got, dtype=np.float64).ravel()
    expected = np.asarray(expected, dtype=np.float64).ravel()
    if got.shape != expected.shape:
        raise OnnxExportMismatch(
            f"{what}: ONNX returned {got.size} values for {expected.size} probe rows."
        )
    if not np.allclose(got, expected, rtol=_RTOL, atol=atol, equal_nan=True):
        worst = float(np.nanmax(np.abs(got - expected)))
        raise OnnxExportMismatch(
            f"{what}: the exported graph does not reproduce the fitted model "
            f"(max |difference| {worst:.3g} on {expected.size} probe rows). "
            "The model was not saved as-is; this is a conversion bug, not noise."
        )


def _atol(expected) -> float:
    scale = float(np.nanmax(np.abs(expected))) if np.size(expected) else 1.0
    return _ATOL_SCALE * max(1.0, scale)


def check_values(
    path: str,
    probe,
    expected,
    what: str,
    atol: float | None = None,
    output_index: int = 0,
) -> None:
    """Regression-style graph: one output (the first by default) against continuous
    values. *atol* overrides the default, scale-relative tolerance."""
    if probe is None:
        return
    outputs, _ = _run(path, probe)
    _assert_close(
        outputs[output_index],
        expected,
        what,
        _atol(expected) if atol is None else atol,
    )


def check_probability(path: str, probe, expected_p1, what: str) -> None:
    """Classifier graph: its probability output's class-1 column against P(y=1)."""
    if probe is None:
        return
    outputs, session = _run(path, probe)
    names = [o.name for o in session.get_outputs()]
    index = next((i for i, n in enumerate(names) if "prob" in n), 1)
    _assert_close(_positive_column(outputs[index]), expected_p1, what, _ATOL_SCALE)


def check_decision(path: str, probe, expected_decision, what: str) -> None:
    """SVC graph: one of the two (mutually negated) score columns is the decision value.

    Which column depends on the estimator type (see ``base.svc.model._svc_onnx_bytes``), so
    either may match; neither matching is the failure.
    """
    if probe is None:
        return
    outputs, _ = _run(path, probe)
    scores = np.asarray(outputs[1], dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    atol = _atol(expected_decision)
    errors = []
    for c in range(scores.shape[1]):
        try:
            _assert_close(scores[:, c], expected_decision, what, atol)
            return
        except OnnxExportMismatch as exc:
            errors.append(str(exc))
    raise OnnxExportMismatch(errors[0])
