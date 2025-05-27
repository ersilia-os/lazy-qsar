from flaml import AutoML
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

import numpy as np
import joblib

from ..descriptors.descriptors import ErsiliaEmbedding

class ErsiliaRegressor(object):

    def __init__(self, automl=True, reduced=False, time_budget_sec=20, estimator_list=["rf"]):
        self.time_budget_sec=time_budget_sec
        self.estimator_list=estimator_list
        self.model = None
        self.reducer = None
        self._automl = automl
        self._reduced = reduced
        self.descriptor = ErsiliaEmbedding()

    def fit_automl(self, smiles, y):
        model = AutoML(task="regression", time_budget=self.time_budget_sec)
        X = np.array(self.descriptor.transform(smiles))
        y = np.array(y)
        if self._reduced:
            self.reducer = PCA(n_components=100)
            self.reducer.fit(X)
            X = self.reducer.transform(X)
        model.fit(X, y, time_budget=self.time_budget_sec, estimator_list=self.estimator_list)
        self._n_pos = int(np.sum(y))
        self._n_neg = len(y) - self._n_pos
        self._r2_score = 1-model.best_loss
        self.meta = {
            "n_pos": self._n_pos,
            "n_neg": self._n_neg,
            "r2_score": self._r2_score
        }
        self.model = model.model.estimator
        self.model.fit(X, y)

    def fit_default(self, smiles, y):
        model = RandomForestRegressor()
        X = np.array(self.descriptor.transform(smiles))
        y = np.array(y)
        if self._reduced:
            self.reducer = PCA(n_components=100)
            self.reducer.fit(X)
            X = self.reducer.transform(X)
        model.fit(X, y)
        self.model = model

    def fit(self, smiles, y):
        if self._automl:
            self.fit_automl(smiles, y)
        else:
            self.fit_default(smiles, y)

    def predict(self, smiles):
        X = np.array(self.descriptor.transform(smiles))
        if self._reduced:
            X = self.reducer.transform(X)
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    def load(self, path):
        return joblib.load(path)