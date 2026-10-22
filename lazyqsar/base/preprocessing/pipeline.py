"""
BasePreprocessor — sklearn-compatible transformer that automatically
selects imputation, scaling, and dimensionality reduction strategies based
on dataset characteristics.
"""

import json
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .inspector import PreprocessingProfile, inspect
from .reducer import build_reducer, select_reducer
from .scaler import build_scaler, select_scaler
from lazyqsar.utils.logging import logger


def _fit_session_options(rt):
    """onnxruntime options for the session a fitted preprocessor keeps for the whole fit.

    Left at its defaults, a session keeps what each run allocated in a per-session memory
    arena so the next run can reuse it, and sizes an intra-op thread pool to every core.
    Both are the right trade for one long-lived serving session and the wrong one here:
    fitting scores the full training matrix through every batch's preprocessor, and
    ``batches x rows x width`` of retained buffers is quadratic in the dataset, because the
    batch count of an imbalanced dataset grows with its size. Measured with this change
    absent, one 121,000-compound dataset needed over 128 GB where v3.4.2 needed 15.6 GB.

    - ``enable_cpu_mem_arena=False`` hands every buffer back when the run ends.
    - ``enable_mem_pattern=False`` drops the plan onnxruntime derives from a run's shapes to
      reuse memory across runs; with the arena off there is nothing to plan, and the row
      count differs between the calls a fit makes.
    - ``intra_op_num_threads=1`` keeps hundreds of sessions from starting hundreds of
      cores' worth of threads each. The graph is an imputer, a scaler and a projection,
      cheap next to the heads that follow it.

    None of this changes a computed value: the graph and the arithmetic are untouched, so
    the output is bit-identical to a default session's (tested), which is what the session
    exists to guarantee.
    """
    options = rt.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.intra_op_num_threads = 1
    return options


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Automatically selects and fits a preprocessing pipeline for
    classification and regression tasks.
    """

    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X, y) -> "BasePreprocessor":
        """
        Fit the preprocessing pipeline to (X, y).

        Profiles the dataset, selects scaler and reducer, builds and fits the
        sklearn Pipeline. If ``PowerTransformer`` fails, falls back to
        ``RobustScaler`` automatically.

        Sets ``pipeline_``, ``scaler_name_``, ``reducer_name_``,
        ``n_features_in_``, ``n_features_out_``, and ``kept_feature_indices_``.
        """
        logger.rule("BasePreprocessor")

        y = np.asarray(y).ravel()

        self.profile_: PreprocessingProfile = inspect(X, y, task=self.task)
        logger.profile_summary(self.profile_)

        self.scaler_name_: str = select_scaler(self.profile_)
        self.reducer_name_: str = select_reducer(self.profile_)
        logger.info(f"scaler={self.scaler_name_} | reducer={self.reducer_name_}")

        scaler = build_scaler(self.scaler_name_)
        reducer = build_reducer(self.reducer_name_, self.profile_)

        self.pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("vt0", VarianceThreshold(threshold=1e-6)),
                ("scaler", scaler),
                ("reducer", reducer),
            ]
        )

        self.n_features_in_: int = X.shape[1]

        try:
            self.pipeline_.fit(X, y)
        except Exception as exc:
            if self.scaler_name_ == "power":
                logger.warning(
                    f"PowerTransformer fit failed ({exc!r}); "
                    "falling back to RobustScaler."
                )
                self.scaler_name_ = "robust"
                scaler = build_scaler("robust")
                self.pipeline_ = Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("vt0", VarianceThreshold(threshold=1e-6)),
                        ("scaler", scaler),
                        ("reducer", reducer),
                    ]
                )
                self.pipeline_.fit(X, y)
            else:
                raise

        if self.scaler_name_ == "robust":
            self._rescale_zero_iqr_columns(X, y)

        self._n_features_out: int = self.pipeline_.transform(
            np.zeros((1, self.n_features_in_))
        ).shape[1]

        self.kept_feature_indices_: list = self._compute_kept_indices()
        self._bind_onnx_runtime(X)

        logger.success(
            f"scaler={self.scaler_name_} | reducer={self.reducer_name_} | "
            f"{self.n_features_in_} → {self._n_features_out} features"
        )
        return self

    def _rescale_zero_iqr_columns(self, X, y) -> None:
        """Scale zero-IQR columns by their standard deviation where it exceeds one.

        ``RobustScaler`` divides by the interquartile range and falls back to 1 where it
        is zero, so a column that is constant on most rows keeps its raw units. RDKit
        descriptors include such columns (a sentinel of 1e5 on 97% of rows), which then
        reach the heads at magnitude 1e5: a kernel's ``gamma="scale"`` collapses to
        ~1e-8 and its float32 graph loses precision to cancellation (a 3e-3 bind
        failure). The reducer is refitted on the corrected scale.
        """
        steps = self.pipeline_.named_steps
        scaler = steps["scaler"]
        Xs = self.pipeline_[:2].transform(X)
        q25, q75 = np.percentile(Xs, [25, 75], axis=0)
        zero_iqr = (q75 - q25) == 0
        if not zero_iqr.any():
            return
        # Only shrink: a rare count column (std < 1) keeps scikit-learn's scale of one, so
        # it is not amplified; a sentinel column (std ~1e4) comes down to unit spread.
        scale = np.maximum(Xs[:, zero_iqr].std(axis=0), 1.0)
        if np.all(scale == 1.0):
            return
        scaler.scale_[zero_iqr] = scale
        logger.debug(f"robust scaler: {int((scale > 1).sum())} zero-IQR columns rescaled by std")
        steps["reducer"].fit(scaler.transform(Xs), y)

    def _bind_onnx_runtime(self, X=None) -> None:
        """Make the fitted preprocessor compute the way the exported one will.

        The shipped checkpoint runs this pipeline as a float32 ONNX graph, while
        scikit-learn runs it in float64 and rounds afterwards. The two agree to about a
        float32 ULP -- which sounds harmless and is not, because the heads downstream are
        piecewise constant. A value that lands a hair either side of a learned split falls
        into a different leaf, so an input difference of ~1e-07 can move a score by ~0.4.
        That is the whole of the historical fit-versus-export gap.

        Rather than chase the rounding, remove the difference: run the exported graph here
        too, so the matrix the heads are *fitted* on is bit-identical to the matrix they
        will be *served*. Split thresholds, calibrators, out-of-fold scores and ranker
        knots are then all learned on exactly the values inference produces.

        The session lives for the whole fit, and a ``LazyClassifier`` holds one per batch per
        descriptor -- 65 batches for a dataset of 19 actives among 121,000 compounds -- each
        of which is later handed the full training set. That is why the session is built
        with :func:`_fit_session_options` rather than onnxruntime's defaults: see there.

        The graph is checked against the scikit-learn pipeline on a probe of training rows
        before it is used, and the bytes are kept so that ``save`` writes exactly the graph
        the heads were fitted against. A failure raises :class:`OnnxExportMismatch`: it
        used to fall back to the scikit-learn path with a warning, which produced a model
        whose checkpoint could compute something else.
        """
        import onnxruntime as rt

        from lazyqsar.utils.onnx_check import OnnxExportMismatch, probe_rows

        self._onnx_session_ = None
        self._onnx_input_ = None
        try:
            data = self._to_onnx_bytes()
            session = rt.InferenceSession(
                data,
                sess_options=_fit_session_options(rt),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise OnnxExportMismatch(
                f"The preprocessor ({self.scaler_name_} + {self.reducer_name_}) could "
                f"not be exported to ONNX ({type(exc).__name__}: {exc})."
            ) from exc
        input_name = session.get_inputs()[0].name
        if X is not None:
            probe = probe_rows(X)
            if hasattr(probe, "toarray"):
                probe = probe.toarray()
            probe = np.asarray(probe, dtype=np.float32)
            got = session.run(None, {input_name: probe})[0]
            want = self.pipeline_.transform(probe)
            scale = max(1.0, float(np.nanmax(np.abs(want))) if want.size else 1.0)
            if got.shape != want.shape or not np.allclose(
                got, want, rtol=1e-3, atol=1e-4 * scale, equal_nan=True
            ):
                raise OnnxExportMismatch(
                    f"The preprocessor's ONNX graph ({self.scaler_name_} + "
                    f"{self.reducer_name_}) does not reproduce the scikit-learn pipeline "
                    "on the training probe; this is a conversion bug."
                )
        self._onnx_bytes_ = data
        self._onnx_session_ = session
        self._onnx_input_ = input_name

    def transform(self, X) -> np.ndarray:
        """Apply the fitted pipeline to X, returning the preprocessed array.

        Runs the ONNX form of the pipeline when one is bound, so that fit-time and
        inference-time features are bit-identical. See :meth:`_bind_onnx_runtime`.
        """
        check_is_fitted(self, "pipeline_")
        session = getattr(self, "_onnx_session_", None)
        if session is None:
            return self.pipeline_.transform(X)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return session.run(None, {self._onnx_input_: np.asarray(X, dtype=np.float32)})[
            0
        ]

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _compute_kept_indices(self) -> list:
        vt0_mask = self.pipeline_.named_steps["vt0"].get_support()
        vt0_indices = np.where(vt0_mask)[0]
        reducer = self.pipeline_.named_steps["reducer"]
        if self.reducer_name_ == "variance_threshold":
            reducer_mask = reducer.get_support()
        else:
            vt_mask = reducer.named_steps["vt"].get_support()
            cf_mask = reducer.named_steps["select"].mask_
            vt_indices = np.where(vt_mask)[0]
            cf_indices = np.where(cf_mask)[0]
            reducer_mask = np.zeros(len(vt0_indices), dtype=bool)
            reducer_mask[vt_indices[cf_indices]] = True
        kept = vt0_indices[reducer_mask]
        return kept.tolist()

    @property
    def n_features_out_(self) -> int:
        check_is_fitted(self, "_n_features_out")
        return self._n_features_out

    def _metadata_dict(self) -> dict:
        check_is_fitted(self, "pipeline_")
        return {
            "task": self.task,
            "scaler": self.scaler_name_,
            "reducer": self.reducer_name_,
            "n_features_in": self.n_features_in_,
            "n_features_out": self.n_features_out_,
            "kept_feature_indices": self.kept_feature_indices_,
        }

    def save(self, directory: str) -> None:
        """
        Save the fitted pipeline to *directory*.

        Writes ``preprocessor.json`` (metadata) and ``preprocessor.onnx``.
        """
        check_is_fitted(self, "pipeline_")
        os.makedirs(directory, exist_ok=True)
        base = os.path.join(directory, "preprocessor")
        self.to_onnx(base + ".onnx")
        with open(base + ".json", "w") as f:
            json.dump(self._metadata_dict(), f, indent=2)

    def _to_onnx_bytes(self) -> bytes:
        """Serialise the fitted pipeline to an ONNX graph (opset 15).

        One definition, used both by :meth:`save` and by :meth:`_bind_onnx_runtime`: the
        graph fitting runs against has to be the same graph that gets written, or binding
        it buys nothing.
        """
        check_is_fitted(self, "pipeline_")
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from .reducer import _register_correlation_filter_onnx_converter

        _register_correlation_filter_onnx_converter()
        initial_type = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        return convert_sklearn(
            self.pipeline_, initial_types=initial_type, target_opset=15
        ).SerializeToString()

    def to_onnx(self, path: str) -> None:
        """Write the pipeline's ONNX graph (opset 15) to *path*.

        The bytes the fitted preprocessor was bound to and checked against, when bound:
        re-converting could only ever produce a graph nobody verified.
        """
        data = getattr(self, "_onnx_bytes_", None) or self._to_onnx_bytes()
        with open(path, "wb") as f:
            f.write(data)


class BasePreprocessorArtifact:
    """
    Inference-only preprocessor loaded from a saved directory.

    Reads ``preprocessor.json`` and ``preprocessor.onnx``.
    Only ``onnxruntime`` is required at inference time.
    """

    @classmethod
    def load(cls, directory: str) -> "BasePreprocessorArtifact":
        """Load the preprocessor from *directory*."""
        self = cls.__new__(cls)
        json_path = os.path.join(directory, "preprocessor.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No preprocessor.json found in {directory!r}")
        with open(json_path) as f:
            meta = json.load(f)
        self.task: str = meta["task"]
        self.scaler: str = meta["scaler"]
        self.reducer: str = meta["reducer"]
        self.n_features_in: int = meta["n_features_in"]
        self.n_features_out: int = meta["n_features_out"]
        self.kept_feature_indices: list = meta["kept_feature_indices"]
        onnx_path = os.path.join(directory, "preprocessor.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"No preprocessor.onnx found in {directory!r}")
        import onnxruntime as rt

        self._session = rt.InferenceSession(onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        return self

    def run(self, X) -> np.ndarray:
        """Apply the preprocessor to X, returning float32 array."""
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self._session.run(
            None, {self._input_name: np.asarray(X, dtype=np.float32)}
        )[0]


class BaseClassifierPreprocessor(BasePreprocessor):
    """BasePreprocessor fixed to classification task."""

    def __init__(self):
        super().__init__(task="classification")


class BaseRegressorPreprocessor(BasePreprocessor):
    """BasePreprocessor fixed to regression task."""

    def __init__(self):
        super().__init__(task="regression")
