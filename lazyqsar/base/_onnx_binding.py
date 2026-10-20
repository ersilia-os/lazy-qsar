"""Make a fitted head predict through the ONNX graph it will be shipped as.

The preprocessor has done this since v3.5 (``BasePreprocessor._bind_onnx_runtime``): the
fitted pipeline runs its own exported graph, so fit-time and serving-time features are
bit-identical. This mixin does the same for the heads. After the final model is fitted,
it is converted to ONNX in memory, the graph is checked against the native model on a
probe of training rows (:mod:`lazyqsar.utils.onnx_check`), and from then on every
prediction the fitted head makes -- probabilities, scores, ranks, in-sample metrics, the
reference-library scan -- comes from that graph. A model in memory and the checkpoint it
writes therefore compute the same function by construction, not by tolerance.

What stays native, deliberately:

- **Out-of-fold fold models.** They are fitted and discarded inside ``calibrate`` /
  ``collect_oof``; exporting each would multiply the conversion cost. Their predictions
  only feed statistics (calibrators, cutoffs, gating, rank-error curves), and the
  native-vs-ONNX gap on them is bounded by a test.
- **The self-check itself**, which must compare against the native model or it would be
  comparing the graph with itself.

What "the same function" means, precisely: for linear, SVM, preprocessing and AD graphs
the fitted model and a loaded checkpoint are bit-identical. For tree ensembles (XGBoost,
random forest) onnxruntime sums the trees' contributions per thread and merges them in an
order that depends on the session's thread count, so a single-threaded fit-time session
and a multi-threaded serving session can differ in the last float32 bits (~1e-6
measured). Serving stays multi-threaded by choice; the parity tests hold that gap to
1e-5.

Each head implements ``_onnx_bytes()`` (the serialised graph) and ``_raw_native(X)`` (the
native raw output, 1-D), and declares ``_onnx_kind``: ``"values"`` for regressors,
``"probability"`` for classifiers whose graph emits class probabilities, or
``"decision"`` for the SVC, whose graph emits two mutually negated score columns.
"""

from __future__ import annotations

import numpy as np

from lazyqsar.utils.onnx_check import OnnxExportMismatch, _assert_close, _atol


def _session_options(rt):
    """Same memory-safe options as the bound preprocessor; see ``_fit_session_options``.

    A fit can hold one bound head per head type per batch per descriptor -- hundreds on a
    heavily imbalanced dataset -- so per-session arenas and thread pools are switched off
    for the same reason they are on the preprocessor.
    """
    options = rt.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def _positive_column(raw) -> np.ndarray:
    if isinstance(raw, list):
        return np.array([d[max(d)] for d in raw], dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    return arr if arr.ndim == 1 else arr[:, -1]


class OnnxBoundHead:
    """Mixin: route a fitted head's raw predictions through its own ONNX graph."""

    _onnx_kind = "values"
    # Override for outputs that move in discrete steps (the applicability domain).
    _onnx_bind_atol: float | None = None

    # ------------------------------------------------------------------ to implement

    def _onnx_bytes(self) -> bytes:  # pragma: no cover - abstract
        raise NotImplementedError

    def _raw_native(self, X) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ binding

    def _bind_onnx(self) -> None:
        """Convert the fitted model, verify the graph, and route predictions through it.

        Raises :class:`OnnxExportMismatch` if the graph does not reproduce the native
        model on the probe -- the fit fails rather than producing a model whose
        checkpoint would compute something else.
        """
        import onnxruntime as rt

        self._onnx_session_ = None
        data = self._onnx_bytes()
        session = rt.InferenceSession(
            data, sess_options=_session_options(rt), providers=["CPUExecutionProvider"]
        )
        self._onnx_bytes_ = data
        self._onnx_input_ = session.get_inputs()[0].name
        self._onnx_output_index_ = self._resolve_output(session)
        self._onnx_session_ = session

        probe = getattr(self, "_onnx_probe_", None)
        if probe is not None:
            native = np.asarray(self._raw_native(probe), dtype=np.float64)
            if self._onnx_kind == "decision":
                self._onnx_column_ = self._resolve_decision_column(probe, native)
            if self._onnx_bind_atol is not None:
                atol = self._onnx_bind_atol
            elif self._onnx_kind == "probability":
                atol = 1e-4
            else:
                atol = _atol(native)
            _assert_close(
                self._raw_onnx(probe), native, type(self).__name__ + " (bound)", atol
            )

    def _resolve_output(self, session) -> int:
        if self._onnx_kind == "values":
            return 0
        names = [o.name for o in session.get_outputs()]
        if self._onnx_kind == "probability":
            return next((i for i, n in enumerate(names) if "prob" in n), 1)
        return 1  # decision: the two-column score output

    def _resolve_decision_column(self, probe, native) -> int:
        """Which of the SVC graph's two negated score columns is ``decision_function``."""
        out = self._session_run(probe)
        scores = np.asarray(out[self._onnx_output_index_], dtype=np.float64)
        if scores.ndim == 1:
            return -1
        errors = [np.abs(scores[:, c] - native).max() for c in range(scores.shape[1])]
        best = int(np.argmin(errors))
        if errors[best] > _atol(native) + 1e-3 * np.abs(native).max():
            raise OnnxExportMismatch(
                f"{type(self).__name__} (bound): no ONNX score column reproduces the "
                f"decision function (best max |difference| {errors[best]:.3g})."
            )
        return best

    def _session_run(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._onnx_session_.run(None, {self._onnx_input_: X})

    def _raw_onnx(self, X) -> np.ndarray:
        raw = self._session_run(X)[self._onnx_output_index_]
        if self._onnx_kind == "probability":
            return _positive_column(raw)
        arr = np.asarray(raw, dtype=np.float64)
        if self._onnx_kind == "decision" and arr.ndim == 2:
            return arr[:, getattr(self, "_onnx_column_", 0)]
        return arr.ravel()

    def _raw(self, X) -> np.ndarray:
        """The raw output every prediction method builds on: ONNX when bound."""
        if getattr(self, "_onnx_session_", None) is not None:
            return self._raw_onnx(X)
        return np.asarray(self._raw_native(X), dtype=np.float64)

    def _write_onnx(self, path: str) -> None:
        """Write the graph the head is bound to, or a fresh conversion if unbound."""
        data = getattr(self, "_onnx_bytes_", None) or self._onnx_bytes()
        with open(path, "wb") as fh:
            fh.write(data)

    # ------------------------------------------------------------------ pickling

    def __getstate__(self):
        # An InferenceSession cannot be pickled; the bytes can, and rebinding from them
        # reproduces it exactly. Defer to scikit-learn's own hook for the rest.
        parent = getattr(super(), "__getstate__", None)
        state = parent() if parent is not None else None
        state = dict(self.__dict__ if state is None else state)
        state.pop("_onnx_session_", None)
        return state

    def __setstate__(self, state):
        parent = getattr(super(), "__setstate__", None)
        if parent is not None:
            parent(state)
        else:
            self.__dict__.update(state)
        data = state.get("_onnx_bytes_")
        if data is not None:
            import onnxruntime as rt

            self._onnx_session_ = rt.InferenceSession(
                data,
                sess_options=_session_options(rt),
                providers=["CPUExecutionProvider"],
            )
