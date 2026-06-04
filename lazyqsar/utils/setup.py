import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

from .logging import logger
from .checkpoints import CHECKPOINT_DIR, CDDD_CHECKPOINTS

_CLAMP_ONNX_URL = "https://ersilia-models.s3.eu-central-1.amazonaws.com/eos3l5f/model/checkpoints/clamp_clip/compound_encoder.onnx"


def _safe_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urlretrieve(url, tmp)
    tmp.replace(dest)


def download_chemeleon(target_dir: str | None = None):
    ckpt_dir = Path(target_dir) if target_dir else Path.home() / ".lazyqsar"
    mp_path = ckpt_dir / "chemeleon_mp.pt"
    if not mp_path.exists():
        logger.info("Downloading Chemeleon model...")
        _safe_download(
            "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            mp_path,
        )


def download_cddd(target_dir: str | None = None):
    """Download every checkpoint the CDDD descriptor needs at prediction time.

    This includes the ONNX encoder *and* the ChEMBL nearest-neighbour fallback
    database (fpsim index + SMILES list). All three must be fetched here so they
    bake into offline/air-gapped environments; otherwise the two extra files are
    fetched lazily on first predict, which hangs on nodes with restricted
    internet access.
    """
    ckpt_dir = Path(target_dir) if target_dir else CHECKPOINT_DIR
    logger.info("Downloading CDDD encoder and ChEMBL nearest-neighbour database...")
    for url, filename in CDDD_CHECKPOINTS:
        dest = ckpt_dir / filename
        if not dest.exists():
            logger.info(f"Downloading {filename}...")
            _safe_download(url, dest)


def download_clamp(target_dir: str | None = None):
    ckpt_dir = Path(target_dir) if target_dir else Path.home() / ".lazyqsar"
    clamp_path = ckpt_dir / "clamp_encoder.onnx"
    if not clamp_path.exists():
        logger.info("Downloading CLAMP encoder (~167 MB)...")
        _safe_download(_CLAMP_ONNX_URL, clamp_path)


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


def install_cpu_torch_force():
    logger.info("Force-reinstalling PyTorch as CPU (replacing any CUDA wheel)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "--force-reinstall",
            "torch==2.6.0",
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
