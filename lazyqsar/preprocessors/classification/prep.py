from lazyqsar.base.preprocessing import (
    BaseClassifierPreprocessor as ClassifierPreprocessor,
)
from lazyqsar.utils.logging import logger


class Preprocessor(object):
    def __init__(self):
        self.preprocessor = ClassifierPreprocessor()

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
        self.preprocessor.save(directory, onnx=True)
        logger.debug(f"Preprocessor saved to {directory}")
