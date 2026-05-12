from lazyqsar.base.xgboost import BaseXGBClassifier as XGBClassifier
from lazyqsar.utils.logging import logger


class Head(object):
    def __init__(self, calibrated=True, max_rounds=None):
        self.model = XGBClassifier(calibrated=calibrated, max_rounds=max_rounds)

    def fit(self, X, y):
        logger.debug(f"Fitting XGB head | X={X.shape}")
        self.model.fit(X, y)
        logger.debug(
            f"XGB head fitted | preset={self.model.preset_name_} "
            f"best_iter={self.model.best_iteration_}"
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
        self.model.save(directory)
        logger.debug(f"XGB head saved to {directory}")
