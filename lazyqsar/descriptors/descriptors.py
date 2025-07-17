import os
import joblib
import json
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from .chemeleon_descriptor import CheMeleonFingerprint


class ChemeleonDescriptor(object):
    def __init__(self):
        self.chemeleon_fingerprint = CheMeleonFingerprint()
        self.n_dim = 2048

    def fit(self, smiles):
        self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]
        print("No fitting is necessary for Chemeleon descriptor")
        return None

    def transform(self, smiles):
        if self.features is None:
            self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]
        chunk_size = 1000
        R = []
        for i in tqdm(range(0, len(smiles), chunk_size), desc="Transforming CheMeleon descriptors in chunks of 1000"):
            chunk = smiles[i:i + chunk_size]
            X_chunk = np.array(self.chemeleon_fingerprint(chunk), dtype=np.float32)
            R += [X_chunk]
        return np.concat(R, dtype=np.float32, axis=0)
    
    def save(self, dir_name: str):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        metadata = {
            "rdkit_version": Chem.rdBase.rdkitVersion,
            "features": self.features
        }
        with open(os.path.join(dir_name, "descriptor_metadata.json"), "w") as f:
            json.dump(metadata, f)
        transformer = {
            "nan_filter": self.nan_filter,
            "imputer": self.imputer,
            "variance_filter": self.variance_filter,
            "scaler": self.scaler,
        }
        joblib.dump(transformer, os.path.join(dir_name, "transformer.joblib"))

    @classmethod
    def load(cls, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} does not exist.")
        obj = cls()
        with open(os.path.join(dir_name, "descriptor_metadata.json"), "r") as f:
            metadata = json.load(f)
            rdkit_version = metadata.get("rdkit_version")
            if rdkit_version:
                print(f"Loaded RDKit version: {rdkit_version}")
            current_rdkit_version = Chem.rdBase.rdkitVersion
            if current_rdkit_version != rdkit_version:
                raise ValueError(f"RDKit version mismatch: expected {current_rdkit_version}, got {rdkit_version}")
        transformer = joblib.load(os.path.join(dir_name, "transformer.joblib"))
        obj.nan_filter = transformer["nan_filter"]
        obj.imputer = transformer["imputer"]
        obj.variance_filter = transformer["variance_filter"]
        obj.scaler = transformer["scaler"]
        obj.features = metadata.get("features", [])
        return obj