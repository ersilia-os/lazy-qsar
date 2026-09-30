import subprocess
import sys
from pathlib import Path

from .logging import logger
from .checkpoints import (
    CDDD_CHECKPOINTS,
    CHECKPOINT_SHA256,
    CHEMELEON_FILENAME,
    CHEMELEON_URL,
    CLAMP_FILENAME,
    CLAMP_URL,
    checkpoint_dir,
)
from .fetch import fetch, is_cached

# Kept for callers that still import it; the canonical URL now lives in checkpoints.py
# beside its filename and checksum, so the two cannot drift apart.
_CLAMP_ONNX_URL = CLAMP_URL


def _resolve_dir(target_dir: str | None) -> Path:
    """Where checkpoints go.

    ``--target-dir`` used to move only the download while the descriptors went on reading
    ``~/.lazyqsar``, so a redirected setup fetched everything twice and used the copy it
    had not just written. It now sets ``LAZYQSAR_HOME`` for the process, which is the one
    thing both sides read.
    """
    if target_dir:
        import os

        os.environ["LAZYQSAR_HOME"] = str(Path(target_dir).expanduser())
    return checkpoint_dir()


def _safe_download(url: str, dest: Path, filename: str | None = None) -> None:
    """Deprecated shim. Use :func:`lazyqsar.utils.fetch.fetch`."""
    fetch(url, dest, sha256=CHECKPOINT_SHA256.get(filename or Path(dest).name))


def download_chemeleon(target_dir: str | None = None):
    ckpt_dir = _resolve_dir(target_dir)
    fetch(
        CHEMELEON_URL,
        ckpt_dir / CHEMELEON_FILENAME,
        sha256=CHECKPOINT_SHA256.get(CHEMELEON_FILENAME),
        description=CHEMELEON_FILENAME,
    )


def download_cddd(target_dir: str | None = None):
    """Download every checkpoint the CDDD descriptor needs at prediction time.

    This includes the ONNX encoder *and* the ChEMBL nearest-neighbour fallback
    database (fpsim index + SMILES list). All three must be fetched here so they
    bake into offline/air-gapped environments; otherwise the two extra files are
    fetched lazily on first predict, which hangs on nodes with restricted
    internet access.
    """
    ckpt_dir = _resolve_dir(target_dir)
    logger.info("Downloading CDDD encoder and ChEMBL nearest-neighbour database...")
    for url, filename in CDDD_CHECKPOINTS:
        dest = ckpt_dir / filename
        if not is_cached(dest, CHECKPOINT_SHA256.get(filename)):
            fetch(
                url,
                dest,
                sha256=CHECKPOINT_SHA256.get(filename),
                description=filename,
            )


def download_clamp(target_dir: str | None = None):
    ckpt_dir = _resolve_dir(target_dir)
    fetch(
        CLAMP_URL,
        ckpt_dir / CLAMP_FILENAME,
        sha256=CHECKPOINT_SHA256.get(CLAMP_FILENAME),
        description=CLAMP_FILENAME,
    )


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
