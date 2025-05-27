from flaml import AutoML
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import joblib

import shap

from ..descriptors.descriptors import RdkitDescriptor


class RdkitBinaryClassifier(object):

    def __init__(self, automl=True, time_budget_sec=20, estimator_list=None):
        self.time_budget_sec=time_budget_sec
        self.estimator_list=estimator_list
        self.model = None
        self.explainer = None
        self._automl = automl
        self.descriptor = RdkitDescriptor()
        
    def fit_automl(self, smiles, y):
        model = AutoML(task="classification", time_budget=self.time_budget_sec)
        X = self.descriptor.fit(smiles)
        y = np.array(y)
        model.fit(X, y, time_budget=self.time_budget_sec, estimator_list=self.estimator_list)
        self._n_pos = int(np.sum(y))
        self._n_neg = len(y) - self._n_pos
        self._auroc = 1-model.best_loss
        self.meta = {
            "n_pos": self._n_pos,
            "n_neg": self._n_neg,
            "auroc": self._auroc
        }
        self.model = model.model.estimator
        self.model.fit(X, y)

    def fit_default(self, smiles, y):
        model = RandomForestClassifier()
        X = self.descriptor.fit(smiles)
        y = np.array(y)
        model.fit(X, y)
        self.model = model

    def fit(self, smiles, y):
        if self._automl:
            self.fit_automl(smiles, y)
        else:
            self.fit_default(smiles, y)

    def predict(self, smiles):
        X = self.descriptor.transform(smiles)
        return self.model.predict(X)

    def predict_proba(self, smiles):
        X = self.descriptor.transform(smiles)
        return self.model.predict_proba(X)

    def explain(self, smiles):
        X = self.descriptor.transform(smiles)
        if self.explainer is None:
            self.explainer = shap.Explainer(self.model)
        shap_values = self.explainer(X, check_additivity=False)
        return shap_values

    def save(self, path):
        joblib.dump(self, path)

    def load(self, path):
        return joblib.load(path)