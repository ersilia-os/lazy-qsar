from flaml import AutoML

import numpy as np

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

    def __init__(self, time_budget_sec=20, estimator_list=["rf"]):
        self.time_budget_sec=time_budget_sec
        self.estimator_list=estimator_list
        self.model = None

    def fit(self, smiles, y):
        model = AutoML(task="classification", time_budget=self.time_budget_sec)
        X = featurizer(smiles)
        model.fit(X, y, time_budget=self.time_budget_sec, estimator_list=self.estimator_list)
        self._n_pos = int(np.sum(y))
        self._n_neg = len(y) - self._n_pos
        self._auroc = 1-self.model.best_loss
        self.meta = {
            "n_pos": self._n_pos,
            "n_neg": self._n_neg,
            "auroc": self._auroc
        }
        self.model = model.model.estimator
        self.model.fit(X)

    def predict(self, smiles):
        X = featurizer(smiles)
        return self.model.predict(X)

    def predict_proba(self, smiles):
        X = featurizer(smiles)
        return self.model.predict_proba(X)

    def save(self, path):
        pass

    def load(self, path):
        pass
