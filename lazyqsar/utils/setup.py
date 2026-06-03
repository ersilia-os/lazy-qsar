import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

from .logging import logger
from .checkpoints import (
    CHECKPOINT_DIR,
    CHEMELEON_MP_FILENAME,
    CHEMELEON_MP_URL,
    CDDD_CHECKPOINTS,
)


def _safe_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urlretrieve(url, tmp)
    tmp.replace(dest)


def download_chemeleon():
    logger.info("Downloading Chemeleon model...")
    mp_path = CHECKPOINT_DIR / CHEMELEON_MP_FILENAME
    if not mp_path.exists():
        _safe_download(CHEMELEON_MP_URL, mp_path)


def download_cddd():
    """Download every checkpoint the CDDD descriptor needs at prediction time.

    This includes the ONNX encoder *and* the ChEMBL nearest-neighbour fallback
    database (fpsim index + SMILES list). All three must be fetched here so they
    bake into offline/air-gapped environments (e.g. Singularity SIF builds);
    otherwise the two extra files are fetched lazily on first predict, which
    hangs on nodes with restricted internet access.
    """
    logger.info("Downloading CDDD encoder and ChEMBL nearest-neighbour database...")
    for url, filename in CDDD_CHECKPOINTS:
        dest = CHECKPOINT_DIR / filename
        if not dest.exists():
            logger.info(f"Downloading {filename}...")
            _safe_download(url, dest)


def install_torch():
    logger.info("Installing PyTorch (CPU)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    )


def install_chemprop():
    logger.info("Installing chemprop...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "chemprop",
        ]
    )


def install_rdkit():
    logger.info("Installing RDKit...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "rdkit==2025.9.1",
        ]
    )


def install_fpsim2():
    logger.info("Installing FPSim2...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "FPSim2==0.7.3",
        ]
    )


def main():
    install_torch()
    install_chemprop()
    install_rdkit()
    download_chemeleon()
    download_cddd()


if __name__ == "__main__":
    main()
