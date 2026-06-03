import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from ..utils.logging import logger
from ..utils.checkpoints import CHECKPOINT_DIR, CHEMELEON_MP_FILENAME, CHEMELEON_MP_URL

from urllib.request import urlretrieve

try:
    import torch
except ImportError:
    raise ImportError(
        "torch is required for ChemeleonDescriptor. "
        'Install the full descriptor extras: pip install -e ".[all]"'
    )

try:
    import chemeleon  # noqa: F401
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph
    from chemprop.nn import RegressionFFN
    from chemprop.models import MPNN
except ImportError:
    try:
        from chemprop import featurizers, nn
        from chemprop.data import BatchMolGraph
        from chemprop.nn import RegressionFFN
        from chemprop.models import MPNN
    except ImportError:
        raise ImportError(
            "chemprop is required for ChemeleonDescriptor. "
            'Install the full descriptor extras: pip install -e ".[all]"'
        )
from rdkit.Chem import MolFromSmiles, Mol
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class _CheMeleonFingerprint:
    def __init__(self, device: str | torch.device | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        ckpt_dir = CHECKPOINT_DIR
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / CHEMELEON_MP_FILENAME
        if not mp_path.exists():
            urlretrieve(CHEMELEON_MP_URL, mp_path)
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
        def _featurize(m):
            mol = MolFromSmiles(m) if isinstance(m, str) else m
            return self.featurizer(mol)

        with ThreadPoolExecutor() as ex:
            mol_graphs = list(ex.map(_featurize, molecules))
        bmg = BatchMolGraph(mol_graphs)
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
        n_total = len(smiles)
        milestones = {int(n_total * f / chunk_size) for f in (0.25, 0.5, 0.75)}
        for i in range(0, n_total, chunk_size):
            chunk = smiles[i : i + chunk_size]
            X_chunk = np.array(self.chemeleon_fingerprint(chunk), dtype=np.float32)
            R.append(X_chunk)
            done = len(R)
            if done in milestones:
                pct = int(done * chunk_size * 100 / n_total)
                logger.debug(
                    f"CheMeleon transform {pct}% ({done * chunk_size:,}/{n_total:,})"
                )
        return np.concatenate(R, dtype=np.float32, axis=0)

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
