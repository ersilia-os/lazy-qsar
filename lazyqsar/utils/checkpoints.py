"""Checkpoint locations for descriptor models.

Single source of truth for the download URLs and local filenames of the
descriptor model checkpoints. Both the eager ``lazyqsar setup --descriptors``
step (``lazyqsar/utils/setup.py``) and the lazy per-descriptor loaders
(``lazyqsar/descriptors/*.py``) import from here so they always agree on what
to fetch and where to store it.

This module intentionally pulls in nothing heavy (no rdkit / onnxruntime) so it
can be imported during ``setup`` before those dependencies are installed.
"""

from pathlib import Path

# Local directory where all checkpoints are stored.
CHECKPOINT_DIR = Path.home() / ".lazyqsar"

# --- Chemeleon ---
CHEMELEON_MP_FILENAME = "chemeleon_mp.pt"
CHEMELEON_MP_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

# --- CDDD ---
# The ONNX encoder plus the ChEMBL nearest-neighbour fallback database
# (fpsim index + the SMILES list it is indexed against). All three are required
# at prediction time, so all three must be downloaded by ``setup --descriptors``
# — otherwise the missing two are fetched lazily on first predict, which hangs
# on nodes with restricted internet access.
CDDD_ENCODER_FILENAME = "cddd_encoder.onnx"
CDDD_ENCODER_URL = "https://zenodo.org/records/14811055/files/encoder.onnx?download=1"

CDDD_FPSIM_FILENAME = "cddd_encoder_fpsim.h5"
CDDD_FPSIM_URL = (
    "https://ersilia-models.s3.eu-central-1.amazonaws.com/"
    "eos4rw4/model/checkpoints/fpsim2_database_chembl.h5"
)

CDDD_SMILES_FILENAME = "cddd_encoder_smiles.csv"
CDDD_SMILES_URL = (
    "https://ersilia-models.s3.eu-central-1.amazonaws.com/"
    "eos4rw4/model/checkpoints/fpsim2_database_chembl_smiles.csv"
)

# (url, filename) pairs for every file the CDDD descriptor needs at runtime.
CDDD_CHECKPOINTS = [
    (CDDD_ENCODER_URL, CDDD_ENCODER_FILENAME),
    (CDDD_FPSIM_URL, CDDD_FPSIM_FILENAME),
    (CDDD_SMILES_URL, CDDD_SMILES_FILENAME),
]
