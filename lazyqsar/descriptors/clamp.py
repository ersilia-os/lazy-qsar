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

    # rdkc: count-based RDKit path fingerprint (maxPath=6)
    counts = AllChem.UnfoldedRDKFingerprintCountBased(
        mol, maxPath=6
    ).GetNonzeroElements()
    for k, c in counts.items():
        v[int(k) % _FP_SIZE] += float(c)

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
        result = np.full((n_total, self.n_dim), np.nan, dtype=np.float32)
        chunks_done = 0
        milestones = {int(n_total * f / chunk_size) for f in (0.25, 0.5, 0.75)}
        for chunk_start in range(0, n_total, chunk_size):
            chunk = smiles_list[chunk_start : chunk_start + chunk_size]
            fps, valid_idx = [], []
            for j, s in enumerate(chunk):
                try:
                    fps.append(_smiles_to_fp(s))
                    valid_idx.append(chunk_start + j)
                except Exception:
                    pass
            if valid_idx:
                fps_arr = np.stack(fps).astype(np.float32)
                emb = self._session.run([self._out_name], {self._in_name: fps_arr})[0]
                for out_i, src_i in enumerate(valid_idx):
                    result[src_i] = emb[out_i]
            chunks_done += 1
            if chunks_done in milestones:
                pct = int(chunks_done * chunk_size * 100 / n_total)
                logger.debug(
                    f"CLAMP transform {pct}% ({chunks_done * chunk_size:,}/{n_total:,})"
                )
        nan_rows = np.where(np.isnan(result).any(axis=1))[0]
        if len(nan_rows):
            logger.warning(
                f"[clamp] {len(nan_rows)} SMILES produced NaN descriptors "
                f"and will be median-imputed (indices: {nan_rows.tolist()})"
            )
        return result

    def is_applicable(self, smiles_list: list) -> bool:
        return True

    def save(self, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} does not exist.")
        metadata = {"featurizer": self.featurizer_name}
        with open(os.path.join(dir_name, "featurizer.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} does not exist.")
        with open(os.path.join(dir_name, "featurizer.json"), "r") as f:
            metadata = json.load(f)
        if metadata.get("featurizer") != "clamp":
            raise ValueError(
                f"Expected featurizer 'clamp', got '{metadata.get('featurizer')}'"
            )
        return cls()
