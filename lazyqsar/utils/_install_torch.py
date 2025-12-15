import subprocess
import sys
import logging

TORCH_VERSION = "2.8.0"
TORCH_CPU_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cpu"

logger = logging.getLogger("installer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


def run_cmd(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        logger.info(line.rstrip())

    for line in process.stderr:
        logger.error(line.rstrip())

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def ensure_torch_cpu(version=TORCH_VERSION, index_url=TORCH_CPU_EXTRA_INDEX_URL):
    try:
        import torch  # noqa: F401
        logger.info("Torch already installed; skipping installation.")
        return
    except ImportError:
        logger.warning("Torch not found, installing CPU version...")
    
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch=={version}",
        "--extra-index-url",
        index_url,
    ]

    run_cmd(cmd)
    logger.info("Torch CPU installation complete.")


def ensure_chemprop():
    try:
        import chemprop  # noqa: F401
        logger.info("Chemprop already installed; skipping installation.")
        return
    except ImportError:
        logger.warning("Chemprop not found, installing...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "chemprop",
    ]

    run_cmd(cmd)
    logger.info("Chemprop installation complete.")
