import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

from .logging import logger


def _safe_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urlretrieve(url, tmp)
    tmp.replace(dest)


def download_chemeleon():
    logger.info("Downloading Chemeleon model...")
    ckpt_dir = Path().home() / ".lazyqsar"
    mp_path = ckpt_dir / "chemeleon_mp.pt"
    if not mp_path.exists():
        _safe_download(
            "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            mp_path,
        )


def download_cddd():
    logger.info("Downloading CDDD encoder...")
    ckpt_dir = Path().home() / ".lazyqsar"
    cddd_path = ckpt_dir / "cddd_encoder.onnx"
    if not cddd_path.exists():
        _safe_download(
            "https://zenodo.org/records/14811055/files/encoder.onnx?download=1",
            cddd_path,
        )


def install_torch():
    logger.info("Installing PyTorch (CPU)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
    ])


def install_chemprop():
    logger.info("Installing chemprop...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "chemprop",
    ])


def install_rdkit():
    logger.info("Installing RDKit...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "rdkit==2025.9.1",
    ])


def install_fpsim2():
    logger.info("Installing FPSim2...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "FPSim2==0.7.3",
    ])


def main():
    install_torch()
    install_chemprop()
    install_rdkit()
    download_chemeleon()
    download_cddd()


if __name__ == "__main__":
    main()
