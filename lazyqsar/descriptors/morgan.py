import json
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from ..utils.logging import logger
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

    def _mol_from_smiles(self, smiles):
        return Chem.MolFromSmiles(smiles)

    def transform(self, smiles):
        logger.debug("Transforming Morgan fingerprints...")
        results = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            try:
                v = self.mfpgen.GetCountFingerprint(mol)
                row = [0] * self.n_dim
                for i, val in v.GetNonzeroElements().items():
                    row[i] = val if val < 255 else 255
                results.append(row)
            except Exception:
                results.append([np.nan] * self.n_dim)
        result = np.array(results, dtype=np.float32)
        nan_rows = np.where(np.isnan(result).any(axis=1))[0]
        if len(nan_rows):
            logger.warning(
                f"[morgan] {len(nan_rows)} SMILES produced NaN descriptors "
                f"and will be median-imputed (indices: {nan_rows.tolist()})"
            )
        return result

    def is_applicable(self, smiles_list: list) -> bool:
        return True

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
                logger.debug(f"Loaded RDKit version: {rdkit_version}")
            current_rdkit_version = Chem.rdBase.rdkitVersion
            if current_rdkit_version != rdkit_version:
                raise ValueError(
                    f"RDKit version mismatch: got {current_rdkit_version}, expected {rdkit_version}"
                )
        return obj
