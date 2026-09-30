"""Fetch a file and prove it arrived intact.

The downloader this replaces had no integrity checking of any kind -- no checksum, no size
check -- and decided a file was already present with ``dest.exists()``. A download
interrupted halfway therefore left a truncated file that was never re-fetched, and surfaced
later as an opaque parse error inside ONNX Runtime rather than as "the download failed".
Three of these artifacts are over 100 MB, so that is not a rare case.

Everything here is standard library plus ``rich``, which is already a core dependency, so it
can run during ``lazyqsar setup`` before the descriptor stack is installed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# 256 KB: frequent enough that the progress bar moves, large enough that the syscall
# overhead disappears.
_CHUNK = 256 * 1024

# Downloads are a deliberate, slow, user-triggered thing, so their progress goes to stderr
# whatever the log level. Stdout stays clean for machine-readable CLI output.
_console = Console(stderr=True, highlight=False)


class DownloadError(RuntimeError):
    """A download failed, or arrived as something other than what was expected."""


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_cached(
    dest, sha256: str | None = None, expected_bytes: int | None = None
) -> bool:
    """Whether *dest* is present and appears to be the file that was wanted.

    Size is checked before the hash because it costs a ``stat`` rather than a pass over
    100 MB, and truncation -- the failure this exists to catch -- always changes it.
    """
    dest = Path(dest)
    if not dest.is_file():
        return False
    if expected_bytes is not None and dest.stat().st_size != expected_bytes:
        return False
    if sha256 is not None and sha256_file(dest) != sha256:
        return False
    return True


def fetch(
    url: str,
    dest,
    *,
    sha256: str | None = None,
    expected_bytes: int | None = None,
    force: bool = False,
    description: str | None = None,
) -> Path:
    """Download *url* to *dest*, atomically, verifying it if a checksum is known.

    The partial file is written beside the destination and renamed only once it has been
    verified, so an interrupted or corrupted download can never be mistaken for a complete
    one: either *dest* is the file that was wanted, or it does not exist.
    """
    dest = Path(dest)
    if not force and is_cached(dest, sha256, expected_bytes):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Beside the destination, and not via `with_suffix`: that replaces the *last* suffix, so
    # `cddd_encoder_fpsim.h5` would be staged as `cddd_encoder_fpsim.part` and a second
    # download of a differently-named file could collide with it.
    part = dest.parent / (dest.name + ".part")
    digest = hashlib.sha256()
    label = description or dest.name

    try:
        with urlopen(url) as response:  # noqa: S310 - fixed https URLs from this package
            total = response.length
            with (
                Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=_console,
                ) as progress,
                open(part, "wb") as handle,
            ):
                task = progress.add_task(label, total=total)
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
                    progress.update(task, advance=len(block))
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"Could not download {url}: {exc}") from exc

    got = digest.hexdigest()
    size = part.stat().st_size
    if sha256 is not None and got != sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{label} does not match its expected checksum.\n"
            f"  expected sha256 {sha256}\n"
            f"  got            {got}\n"
            f"  from {url}\n"
            "The file was discarded rather than cached; retry, and if it persists the "
            "published object has changed."
        )
    if expected_bytes is not None and size != expected_bytes:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{label} is {size} bytes, expected {expected_bytes}. Discarded."
        )

    os.replace(part, dest)
    return dest
