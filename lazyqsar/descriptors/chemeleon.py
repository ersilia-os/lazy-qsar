import os
import json
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from ..utils.logging import logger

from pathlib import Path
from urllib.request import urlretrieve
from lazyqsar.utils._install_extras import ensure_torch_cpu, ensure_chemprop

try:
    import torch
except ImportError:
    ensure_torch_cpu()
    import torch
try:
    import chemeleon
except ImportError:
    ensure_chemprop()
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph
    from chemprop.nn import RegressionFFN
    from chemprop.models import MPNN
from rdkit.Chem import MolFromSmiles, Mol
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class _CheMeleonFingerprint:
    def __init__(self, device: str | torch.device | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        ckpt_dir = Path().home() / ".lazyqsar"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            urlretrieve(
                r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                mp_path,
            )
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = BatchMolGraph(
            [
                self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m)
                for m in molecules
            ]
        )
        bmg.to(device=self.model.device)
        return self.model.fingerprint(bmg).numpy(force=True)


class ChemeleonDescriptor(object):
    def __init__(self):
        """CheMeleon descriptor based on the CheMeleon foundational model.
        CheMeleon is based on ChemProp's MPNN model and provides a 2048-dimensional fingerprint (continuous).

        Usage:
        >>> from lazyqsar.descriptors import ChemeleonDescriptor
        >>> chemeleon = ChemeleonDescriptor()
        >>> X = chemeleon.transform(smiles_list)
        """
        self.featurizer_name = "chemeleon"
        self.chemeleon_fingerprint = _CheMeleonFingerprint()
        self.n_dim = 2048
        self.features = ["dim_{0}".format(i) for i in range(self.n_dim)]

    def transform(self, smiles):
        chunk_size = 100
        R = []
        for i in tqdm(
            range(0, len(smiles), chunk_size),
            desc="Transforming CheMeleon descriptors in chunks of 100",
        ):
            chunk = smiles[i : i + chunk_size]
            X_chunk = np.array(self.chemeleon_fingerprint(chunk), dtype=np.float32)
            R += [X_chunk]
        return np.concatenate(R, dtype=np.float32, axis=0)

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
