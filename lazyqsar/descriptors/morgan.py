import json
from tqdm import tqdm
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class MorganFingerprint(object):
    def __init__(self):
        """Morgan fingerprint descriptor based on RDKit's Morgan algorithm.
        Default parameters (cannot be modified):
        - n_dim: 2048
        - radius: 3

        Usage:
        >>> from lazyqsar.descriptors import MorganFingerprint
        >>> morgan = MorganFingerprint()
        >>> X = morgan.transform(smiles_list)
        """
        self.featurizer_name = "morgan"
        self.n_dim = 2048
        self.radius = 3
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_dim
        )
        self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]

    def _clip_sparse(self, vect, nbits):
        data = [0] * nbits
        for i, v in vect.GetNonzeroElements().items():
            data[i] = v if v < 255 else 255
        return data

    def _morganfp(self, smiles):
        v_ = []
        for smile in smiles:
            mol = self._mol_from_smiles(smile)
            v = self.mfpgen.GetCountFingerprint(mol)
            v = self._clip_sparse(v, self.n_dim)
            v_.append(v)
        return np.array(v_, dtype=int)

    def _mol_from_smiles(self, smiles):
        return Chem.MolFromSmiles(smiles)

    def transform(self, smiles):
        chunk_size = 100_000
        R = []
        for i in tqdm(
            range(0, len(smiles), chunk_size),
            desc="Transforming Morgan descriptors in chunks of 1000",
        ):
            chunk = smiles[i : i + chunk_size]
            X_chunk = self._morganfp(chunk)
            R += [X_chunk]
        return np.concatenate(R, axis=0)

    def save(self, dir_name: str):
        if not os.path.exists(dir_name):
            raise Exception(f"Directory {dir_name} does not exist.")
        metadata = {
            "featurizer": self.featurizer_name,
            "rdkit_version": Chem.rdBase.rdkitVersion,
        }
        with open(os.path.join(dir_name, "featurizer.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} does not exist.")
        obj = cls()
        with open(os.path.join(dir_name, "featurizer.json"), "r") as f:
            metadata = json.load(f)
            rdkit_version = metadata.get("rdkit_version")
            if rdkit_version:
                print(f"Loaded RDKit version: {rdkit_version}")
            current_rdkit_version = Chem.rdBase.rdkitVersion
            if current_rdkit_version != rdkit_version:
                raise ValueError(
                    f"RDKit version mismatch: got {current_rdkit_version}, expected {rdkit_version}"
                )
        return obj
