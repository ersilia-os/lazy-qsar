import os
import json
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from .chemeleon_descriptor import CheMeleonFingerprint

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChemeleonDescriptor(object):

    def __init__(self):
        """CheMeleon descriptor based on the CheMeleon foundational model.
        CheMeleon is based on ChemProp's MPNN model and provides a 2048-dimensional fingerprint (continuous).
        
        Usage:
        >>> from lazyqsar.descriptors import ChemeleonDescriptor
        >>> chemeleon = ChemeleonDescriptor()
        >>> X = chemeleon.transform(smiles_list)
        """

        self.chemeleon_fingerprint = CheMeleonFingerprint()
        self.n_dim = 2048
        self.features = None

    def fit(self, smiles):
        self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]
        logger.warning("No fitting is necessary for Chemeleon descriptor")
        return None

    def transform(self, smiles):
        if self.features is None:
            self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]
        chunk_size = 1000
        R = []
        for i in tqdm(
            range(0, len(smiles), chunk_size),
            desc="Transforming CheMeleon descriptors in chunks of 1000",
        ):
            chunk = smiles[i : i + chunk_size]
            X_chunk = np.array(self.chemeleon_fingerprint(chunk), dtype=np.float32)
            R += [X_chunk]
        return np.concatenate(R, dtype=np.float32, axis=0)

    def save(self, dir_name: str):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        metadata = {
            "rdkit_version": Chem.rdBase.rdkitVersion,
            "features": self.features,
        }
        with open(os.path.join(dir_name, "descriptor_metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} does not exist.")
        obj = cls()
        with open(os.path.join(dir_name, "descriptor_metadata.json"), "r") as f:
            metadata = json.load(f)
            rdkit_version = metadata.get("rdkit_version")
            if rdkit_version:
                logger.debug(f"Loaded RDKit version: {rdkit_version}")
            current_rdkit_version = Chem.rdBase.rdkitVersion
            if current_rdkit_version != rdkit_version:
                raise ValueError(
                    f"RDKit version mismatch: expected {current_rdkit_version}, got {rdkit_version}"
                )
        obj.features = metadata.get("features", [])
        return obj


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

        self.n_dim = 2048
        self.radius = 3
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_dim
        )
        self.features = None

    def _clip_sparse(self, vect, nbits):
        l = [0] * nbits
        for i, v in vect.GetNonzeroElements().items():
            l[i] = v if v < 255 else 255
        return l

    def morganfp(self, smiles):
        v_ = []
        for smile in smiles:
            mol = self._mol_from_smiles(smile)
            v = self.mfpgen.GetCountFingerprint(mol)
            v = self._clip_sparse(v, self.n_dim)
            v_.append(v)
        return np.array(v_, dtype=int)

    def fit(self, smiles):
        self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]
        logger.warning("No fitting is necessary for Morgan descriptor")
        return None

    def _mol_from_smiles(self, smiles):
        return Chem.MolFromSmiles(smiles)

    def transform(self, smiles):
        if self.features is None:
            self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]
        chunk_size = 100_000
        R = []
        for i in tqdm(
            range(0, len(smiles), chunk_size),
            desc="Transforming Morgan descriptors in chunks of 1000",
        ):
            chunk = smiles[i : i + chunk_size]
            X_chunk = self.morganfp(chunk)
            R += [X_chunk]
        return np.concatenate(R, axis=0)

    def save(self, dir_name: str):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        metadata = {
            "rdkit_version": Chem.rdBase.rdkitVersion,
            "features": self.features,
        }
        with open(os.path.join(dir_name, "descriptor_metadata.json"), "w") as f:
            json.dump(metadata, f)

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
                raise ValueError(
                    f"RDKit version mismatch: expected {current_rdkit_version}, got {rdkit_version}"
                )
        obj.features = metadata.get("features", [])
        return obj
