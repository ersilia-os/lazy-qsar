import subprocess
import sys

TORCH_VERSION = "2.8.0"
TORCH_CPU_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cpu"

def ensure_torch_cpu(version=TORCH_VERSION, index_url=TORCH_CPU_EXTRA_INDEX_URL, quiet=False):
    try:
        import torch
        return
    except Exception:
        pass

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch=={version}+cpu",
        "--extra-index-url",
        index_url,
    ]

    subprocess.check_call(cmd)

    import torch
