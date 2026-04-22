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

        logger.success(
            f"scaler={self.scaler_name_} | reducer={self.reducer_name_} | "
            f"{self.n_features_in_} → {self._n_features_out} features"
        )
        return self

    def transform(self, X) -> np.ndarray:
        """Apply the fitted pipeline to X, returning the preprocessed array."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.transform(X)

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

    def save(self, directory: str, onnx: bool = True) -> None:
        """
        Save the fitted pipeline to *directory*.

        Writes ``preprocessor.json`` (metadata) and either
        ``preprocessor.onnx`` (default) or ``preprocessor.joblib``.
        """
        check_is_fitted(self, "pipeline_")
        os.makedirs(directory, exist_ok=True)
        base = os.path.join(directory, "preprocessor")
        if onnx:
            self.to_onnx(base + ".onnx")
        else:
            import joblib

            joblib.dump(self.pipeline_, base + ".joblib")
        with open(base + ".json", "w") as f:
            json.dump(self._metadata_dict(), f, indent=2)

    def to_onnx(self, path: str) -> None:
        """Export the pipeline to ONNX (opset 15) at *path*."""
        check_is_fitted(self, "pipeline_")
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from .reducer import _register_correlation_filter_onnx_converter

        _register_correlation_filter_onnx_converter()
        initial_type = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        onnx_model = convert_sklearn(
            self.pipeline_, initial_types=initial_type, target_opset=15
        )
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())


class BasePreprocessorArtifact:
    """
    Inference-only preprocessor loaded from a saved directory.

    Reads ``preprocessor.json`` and either ``preprocessor.onnx`` or
    ``preprocessor.joblib``. No sklearn fit dependencies are required
    when using the ONNX backend.
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
        joblib_path = os.path.join(directory, "preprocessor.joblib")
        if os.path.exists(onnx_path):
            import onnxruntime as rt

            self._session = rt.InferenceSession(onnx_path)
            self._input_name = self._session.get_inputs()[0].name
            self._backend = "onnx"
        elif os.path.exists(joblib_path):
            import joblib

            self._pipeline = joblib.load(joblib_path)
            self._backend = "joblib"
        else:
            raise FileNotFoundError(
                f"No preprocessor.onnx or preprocessor.joblib found in {directory!r}"
            )
        return self

    def run(self, X) -> np.ndarray:
        """Apply the preprocessor to X, returning float32 array."""
        if self._backend == "onnx":
            if hasattr(X, "toarray"):
                X = X.toarray()
            return self._session.run(
                None, {self._input_name: np.asarray(X, dtype=np.float32)}
            )[0]
        else:
            return self._pipeline.transform(X)


class BaseClassifierPreprocessor(BasePreprocessor):
    """BasePreprocessor fixed to classification task."""

    def __init__(self):
        super().__init__(task="classification")


class BaseRegressorPreprocessor(BasePreprocessor):
    """BasePreprocessor fixed to regression task."""

    def __init__(self):
        super().__init__(task="regression")
