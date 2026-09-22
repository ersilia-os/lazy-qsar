"""Checkpoint locations for the CDDD descriptor.

Single source of truth for the download URLs and local filenames of the CDDD
checkpoints. Both the eager ``lazyqsar setup --descriptors`` step
(``lazyqsar/utils/setup.py``) and the lazy loader in
``lazyqsar/descriptors/cddd.py`` import from here so they always agree on what
to fetch and where to store it — preventing the two lists from drifting apart.

This module intentionally pulls in nothing heavy (no rdkit / onnxruntime) so it
can be imported during ``setup`` before those dependencies are installed.
"""

from pathlib import Path

import os


def checkpoint_dir() -> Path:
    """Root for every cached artifact: descriptor checkpoints and the reference library.

    ``LAZYQSAR_HOME`` redirects it. That exists because ``lazyqsar setup --target-dir``
    used to move only the *download*, while the descriptors went on reading
    ``~/.lazyqsar`` -- so a redirected setup silently fetched everything twice and left the
    first copy unused. One variable now governs both, which matters more once a 267 MB
    reference bundle is also involved.
    """
    return Path(os.environ.get("LAZYQSAR_HOME", Path.home() / ".lazyqsar")).expanduser()


# Kept as a module-level alias for the one in-package importer that reads it directly.
# Prefer `checkpoint_dir()`, which honours the environment at call time rather than at
# import time.
CHECKPOINT_DIR = checkpoint_dir()

# --- CheMeleon ---
CHEMELEON_FILENAME = "chemeleon_mp.pt"
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

# --- CLAMP ---
CLAMP_FILENAME = "clamp_encoder.onnx"
CLAMP_URL = (
    "https://ersilia-models.s3.eu-central-1.amazonaws.com/"
    "eos3l5f/model/checkpoints/clamp_clip/compound_encoder.onnx"
)

# --- CDDD ---
# The ONNX encoder plus the ChEMBL nearest-neighbour fallback database
# (fpsim index + the SMILES list it is indexed against). All three are required
# at prediction time, so all three must be downloaded by
# ``lazyqsar setup --descriptors`` — otherwise the missing two are fetched
# lazily on first predict, which hangs on nodes with restricted internet access.
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


# sha256 of every checkpoint, so a truncated or substituted download is caught when it
# happens rather than surfacing later as an opaque parse error inside ONNX Runtime. These
# are also what the reference bundle's drift check reads: a changed encoder means the
# published descriptor matrices no longer describe what this install would compute.
CHECKPOINT_SHA256 = {
    "chemeleon_mp.pt": "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651",
    "clamp_encoder.onnx": "c3f30146f8691c22e520f541b854b7faef78ed1a50677872f29ae32dfa357e0e",
    "cddd_encoder.onnx": "b43c239a257a94909b20ffc77affdcfbd48301e27701b921de5961457e40a933",
    "cddd_encoder_fpsim.h5": (
        "255428e865be221385c45967b511ae01aaad58d34e1d3d25c06c7c73c85522e8"
    ),
    "cddd_encoder_smiles.csv": (
        "e92f025eab26d53aa87d199fbe68d918ec7381b40edec45b37dacf1b40d83e1d"
    ),
}
