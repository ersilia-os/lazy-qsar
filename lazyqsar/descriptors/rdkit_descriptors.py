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
        self._descriptor_names = sorted(
            [desc_name for desc_name, _ in Descriptors._descList]
        )
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
                vals = np.array(self.calculator.CalcDescriptors(mol), dtype=np.float64)
                vals[~np.isfinite(vals)] = np.nan
                vals = np.clip(vals, -1e5, 1e5).astype(np.float32)
            except Exception:
                vals = np.full(len(self._descriptor_names), np.nan, dtype=np.float32)
            results.append(vals)
        result = np.clip(np.array(results, dtype=np.float32), -1e5, 1e5)
        nan_rows = np.where(np.isnan(result).any(axis=1))[0]
        if len(nan_rows):
            logger.warning(
                f"[rdkit] {len(nan_rows)} SMILES produced NaN descriptors "
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
