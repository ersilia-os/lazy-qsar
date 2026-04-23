import os
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from urllib.request import urlretrieve

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings
from rdkit import RDLogger

from ..utils.logging import logger

RDLogger.DisableLog("rdApp.*")

_CLAMP_ONNX_URL = "https://ersilia-models.s3.eu-central-1.amazonaws.com/eos3l5f/model/checkpoints/clamp_clip/compound_encoder.onnx"
_FP_SIZE = 8192
_N_DIM = 768
_RADIUS = 2


def _smiles_to_fp(smi: str) -> np.ndarray:
    """Compute the 'morganc+rdkc' fingerprint used by CLAMP (8192-dim, log1p)."""
    mol = Chem.MolFromSmiles(str(smi), sanitize=False)
    if mol is None:
        return np.zeros(_FP_SIZE, dtype=np.float32)
    Chem.SanitizeMol(mol, catchErrors=True)
    FastFindRings(mol)
    mol.UpdatePropertyCache(strict=False)

    v = np.zeros(_FP_SIZE, dtype=np.float32)

    # morganc: count-based Morgan fingerprint (radius 2, with chirality/bond types/features)
    try:
        counts = AllChem.GetMorganFingerprint(
            mol,
            _RADIUS,
            useChirality=True,
            useBondTypes=True,
            useFeatures=True,
            useCounts=True,
        ).GetNonzeroElements()
        for k, c in counts.items():
            v[int(k) % _FP_SIZE] += float(c)
    except Exception:
        pass

    # rdkc: count-based RDKit path fingerprint (maxPath=6)
    try:
        counts = AllChem.UnfoldedRDKFingerprintCountBased(
            mol, maxPath=6
        ).GetNonzeroElements()
        for k, c in counts.items():
            v[int(k) % _FP_SIZE] += float(c)
    except Exception:
        pass

    return np.log1p(v)


class ClampDescriptor:
    """CLAMP 768-dimensional bioactivity embeddings.

    CLAMP (Contrastive Learning for Assay Molecules and assay Pretraining)
    encodes SMILES via a 'morganc+rdkc' fingerprint (8192-dim) fed into a
    pretrained ONNX neural network, yielding 768-dim embeddings.
    """

    def __init__(self):
        self.featurizer_name = "clamp"
        self.n_dim = _N_DIM
        self.features = [f"clamp_{i:03d}" for i in range(self.n_dim)]
        self._session = None

    def _ensure_model(self):
        if self._session is not None:
            return
        ckpt_dir = Path.home() / ".lazyqsar"
        ckpt_dir.mkdir(exist_ok=True)
        model_path = ckpt_dir / "clamp_encoder.onnx"
        if not model_path.exists():
            logger.info(
                f"Downloading CLAMP encoder model (~167 MB) to {model_path} ..."
            )
            urlretrieve(_CLAMP_ONNX_URL, model_path)
            logger.info("CLAMP model downloaded.")
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._in_name = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name

    def transform(self, smiles_list: list, chunk_size: int = 100) -> np.ndarray:
        self._ensure_model()
        n_total = len(smiles_list)
        milestones = {int(n_total * f / chunk_size) for f in (0.25, 0.5, 0.75)}
        chunks = []
        for i in range(0, n_total, chunk_size):
            chunk = smiles_list[i : i + chunk_size]
            fps = np.stack([_smiles_to_fp(s) for s in chunk], axis=0).astype(np.float32)
            emb = self._session.run([self._out_name], {self._in_name: fps})[0]
            chunks.append(emb)
            done = len(chunks)
            if done in milestones:
                pct = int(done * chunk_size * 100 / n_total)
                logger.debug(
                    f"CLAMP transform {pct}% ({done * chunk_size:,}/{n_total:,})"
                )
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def is_applicable(self, smiles_list: list) -> bool:
        return True

    def save(self, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} does not exist.")
        metadata = {"featurizer": self.featurizer_name}
        with open(os.path.join(dir_name, "featurizer.json"), "w") as f:
            json.dump(metadata, f, indent=2)
