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

        self._n_features_out: int = self.pipeline_.transform(
            np.zeros((1, self.n_features_in_))
        ).shape[1]

        self.kept_feature_indices_: list = self._compute_kept_indices()
        self._bind_onnx_runtime()

        logger.success(
            f"scaler={self.scaler_name_} | reducer={self.reducer_name_} | "
            f"{self.n_features_in_} → {self._n_features_out} features"
        )
        return self

    def _bind_onnx_runtime(self) -> None:
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

        Falls back to the scikit-learn pipeline, with a warning, if the export or the
        session fails -- a preprocessor that works is worth more than one that matches.
        """
        self._onnx_session_ = None
        self._onnx_input_ = None
        try:
            import onnxruntime as rt

            session = rt.InferenceSession(
                self._to_onnx_bytes(), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - see docstring; degrade, don't raise
            logger.warning(
                f"Preprocessor could not be bound to its ONNX form "
                f"({type(exc).__name__}: {exc}); fitting against the scikit-learn path "
                "instead. The exported checkpoint may not reproduce this model exactly."
            )
            return
        self._onnx_session_ = session
        self._onnx_input_ = session.get_inputs()[0].name

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
        """Export the pipeline to ONNX (opset 15) at *path*."""
        with open(path, "wb") as f:
            f.write(self._to_onnx_bytes())


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
