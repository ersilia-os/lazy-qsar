from pathlib import Path
from urllib.request import urlretrieve
from ._install_extras import ensure_torch_cpu, ensure_chemprop, ensure_rdkit
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
    ensure_torch_cpu()


def install_chemprop():
    ensure_chemprop()

def install_rdkit():
    ensure_rdkit()


def main():
    install_torch()
    install_chemprop()
    install_rdkit()
    download_chemeleon()
    download_cddd()


if __name__ == "__main__":
    main()
