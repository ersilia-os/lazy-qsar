"""The feature preprocessor an assembler fits first: scaler, reducer, ONNX-bound."""

from __future__ import annotations

from lazyqsar.utils.logging import logger


class Preprocessor:
    """``BaseClassifierPreprocessor`` or ``BaseRegressorPreprocessor``, by task."""

    def __init__(self, task: str = "classification"):
        from lazyqsar.base.preprocessing import (
            BaseClassifierPreprocessor,
            BaseRegressorPreprocessor,
        )

        cls = (
            BaseRegressorPreprocessor
            if task == "regression"
            else BaseClassifierPreprocessor
        )
        self.preprocessor = cls()

    def fit(self, X, y):
        logger.debug(f"Fitting preprocessor | X={X.shape}")
        self.preprocessor.fit(X, y)
        logger.debug(
            f"Preprocessor fitted | "
            f"scaler={self.preprocessor.scaler_name_} "
            f"reducer={self.preprocessor.reducer_name_} "
            f"{self.preprocessor.n_features_in_}→{self.preprocessor.n_features_out_} features"
        )

    def transform(self, X):
        return self.preprocessor.transform(X)

    def save(self, directory):
        self.preprocessor.save(directory)
        logger.debug(f"Preprocessor saved to {directory}")
