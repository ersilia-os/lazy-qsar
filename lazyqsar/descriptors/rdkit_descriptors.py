import os
import json
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from ..utils.logging import logger

RDLogger.DisableLog("rdApp.*")


class RDKitDescriptor(object):
    def __init__(self):
        self.featurizer_name = "rdkit"
        self._descriptor_names = sorted([desc_name for desc_name, _ in Descriptors._descList])
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self._descriptor_names
        )
        self.features = [n.lower() for n in self._descriptor_names]

    def transform(self, smiles_list):
        logger.debug("Transforming RDKit descriptors...")
        results = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError("Invalid molecule")
                vals = np.array(self.calculator.CalcDescriptors(mol), dtype=np.float32)
                vals[~np.isfinite(vals)] = 0.0
            except Exception:
                vals = np.zeros(len(self._descriptor_names), dtype=np.float32)
            results.append(vals)
        return np.clip(np.array(results, dtype=np.float32), -1e5, 1e5)

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
