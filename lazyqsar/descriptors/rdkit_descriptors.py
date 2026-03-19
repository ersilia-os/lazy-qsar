import os
import json
import multiprocessing
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from ..utils.logging import logger

RDLogger.DisableLog("rdApp.*")

# Module-level worker state (initialised once per worker process)
_worker_calculator = None
_worker_n_desc = None


def _init_rdkit_worker(descriptor_names):
    global _worker_calculator, _worker_n_desc
    from rdkit.ML.Descriptors import MoleculeDescriptors as _MD
    _worker_calculator = _MD.MolecularDescriptorCalculator(descriptor_names)
    _worker_n_desc = len(descriptor_names)


def _compute_rdkit_worker(smiles):
    import numpy as _np
    from rdkit import Chem as _Chem
    try:
        mol = _Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid molecule")
        desc_values = _np.array(_worker_calculator.CalcDescriptors(mol), dtype=_np.float32)
        desc_values[~_np.isfinite(desc_values)] = 0.0
    except Exception:
        desc_values = _np.zeros(_worker_n_desc, dtype=_np.float32)
    return desc_values.tolist()


class RDKitDescriptor(object):
    def __init__(self):
        self.featurizer_name = "rdkit"
        self._descriptor_names = sorted([desc_name for desc_name, _ in Descriptors._descList])
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self._descriptor_names
        )
        self.features = [n.lower() for n in self._descriptor_names]

    def transform(self, smiles_list):
        n_workers = os.cpu_count() or 1
        logger.debug(
            f"Transforming RDKit descriptors using {n_workers} parallel workers..."
        )
        chunksize = max(1, len(smiles_list) // (n_workers * 4))
        with multiprocessing.Pool(
            n_workers,
            initializer=_init_rdkit_worker,
            initargs=(self._descriptor_names,),
        ) as pool:
            results = pool.map(_compute_rdkit_worker, smiles_list, chunksize=chunksize)
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
