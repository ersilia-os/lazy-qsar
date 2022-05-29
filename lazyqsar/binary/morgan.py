from flaml import AutoML
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import joblib

from rdkit.Chem import rdMolDescriptors as rd
from rdkit import Chem

RADIUS = 3
NBITS = 2048
DTYPE = np.uint8

def clip_sparse(vect, nbits):
    l = [0]*nbits
    for i,v in vect.GetNonzeroElements().items():
        l[i] = v if v < 255 else 255
    return l


class Descriptor(object):

    def __init__(self):
        self.nbits = NBITS
        self.radius = RADIUS

    def calc(self, mol):
        v = rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits)
        return clip_sparse(v, self.nbits)


def featurizer(smiles):
    d = Descriptor()
    X = np.zeros((len(smiles), NBITS))
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        X[i,:] = d.calc(mol)
    return X


class MorganBinaryClassifier(object):

    def __init__(self, automl=True, time_budget_sec=20, estimator_list=["rf"]):
        self.time_budget_sec=time_budget_sec
        self.estimator_list=estimator_list
        self.model = None
        self._automl = automl

    def fit_automl(self, smiles, y):
        model = AutoML(task="classification", time_budget=self.time_budget_sec)
        X = featurizer(smiles)
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
        X = featurizer(smiles)
        y = np.array(y)
        model.fit(X, y)
        self.model = model

    def fit(self, smiles, y):
        if self._automl:
            self.fit_automl(smiles, y)
        else:
            self.fit_default(smiles, y)

    def predict(self, smiles):
        X = featurizer(smiles)
        return self.model.predict(X)

    def predict_proba(self, smiles):
        X = featurizer(smiles)
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self, path)

    def load(self, path):
        return joblib.load(path)
