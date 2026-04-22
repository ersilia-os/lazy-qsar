from lazyqsar.base.svc import BaseSVCClassifier as SVCClassifier
from lazyqsar.utils.logging import logger


class Head(object):

    def __init__(self, calibrated=True):
        self.model = SVCClassifier(calibrated=calibrated)

    def fit(self, X, y):
        logger.debug(f"Fitting SVC head | X={X.shape}")
        self.model.fit(X, y)
        logger.debug(
            f"SVC head fitted | preset={self.model.preset_name_} "
            f"| use_linear={self.model._use_linear_}"
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_score(self, X):
        return self.model.predict_score(X)

    def predict_rank(self, X):
        return self.model.predict_rank(X)

    def predict(self, X, cutoff=None):
        if cutoff is None:
            return self.model.predict(X)
        proba = self.predict_proba(X)[:, 1]
        return (proba >= cutoff).astype(int)

    def save(self, directory):
        self.model.save(directory, onnx=True)
        logger.debug(f"SVC head saved to {directory}")
