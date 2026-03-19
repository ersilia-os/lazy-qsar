import json
import multiprocessing
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from ..utils.logging import logger

RDLogger.DisableLog("rdApp.*")

# Module-level worker state (initialised once per worker process)
_worker_mfpgen = None
_worker_fpSize = None


def _init_morgan_worker(radius, fpSize):
    global _worker_mfpgen, _worker_fpSize
    from rdkit.Chem import rdFingerprintGenerator as _rfg
    _worker_mfpgen = _rfg.GetMorganGenerator(radius=radius, fpSize=fpSize)
    _worker_fpSize = fpSize


def _compute_morgan_worker(smiles):
    from rdkit import Chem as _Chem
    mol = _Chem.MolFromSmiles(smiles)
    v = _worker_mfpgen.GetCountFingerprint(mol)
    data = [0] * _worker_fpSize
    for i, val in v.GetNonzeroElements().items():
        data[i] = val if val < 255 else 255
    return data


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
        n_workers = os.cpu_count() or 1
        logger.debug(
            f"Transforming Morgan fingerprints using {n_workers} parallel workers..."
        )
        chunksize = max(1, len(smiles) // (n_workers * 4))
        with multiprocessing.Pool(
            n_workers,
            initializer=_init_morgan_worker,
            initargs=(self.radius, self.n_dim),
        ) as pool:
            results = pool.map(_compute_morgan_worker, smiles, chunksize=chunksize)
        return np.array(results, dtype=np.uint8)

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
