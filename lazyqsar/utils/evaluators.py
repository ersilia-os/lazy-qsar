import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class QuickAUCEstimator(object):
    def __init__(self):
        pass

    @staticmethod
    def is_integer_matrix(X):
        X_ = X[:10]
        if np.issubdtype(X_.dtype, np.integer):
            return True
        if np.issubdtype(X_.dtype, int):
            return True
        return np.all(np.equal(X_, np.floor(X_)))

    def estimate(self, X, y):
        if self.is_integer_matrix(X):
            model = BernoulliNB()
        else:
            model = GaussianNB()
        if np.sum(y) == 0:
            raise Exception("No positive samples in the data!!!")
        X = np.array(X)
        y = np.array(y, dtype=int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            scores.append(roc_auc_score(y_test, preds))
        return float(np.mean(scores))