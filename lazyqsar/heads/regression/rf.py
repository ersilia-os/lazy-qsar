from lazyqsar.base.randomforest import BaseRFRegressor as RFRegressor
from lazyqsar.utils.logging import logger


class Head(object):
    def __init__(self):
        self.model = RFRegressor()

    def fit(self, X, y):
        logger.debug(f"Fitting RF head | X={X.shape}")
        self.model.fit(X, y)
        logger.debug(f"RF head fitted | rho_oof={self.model.oof_rho_}")

    def predict(self, X):
        return self.model.predict(X)

    def _oof_percentile(self, X):
        return self.model.predict_rank(X)

    def save(self, directory):
        self.model.save(directory)
        logger.debug(f"RF head saved to {directory}")
