"""Every checkpoint the package will fetch has a recorded checksum.

A packaging assertion rather than a download-mechanics one: it says nothing about *how* a
file is fetched, only that everything this version is prepared to download is pinned. It
lived in ``tests/unit/test_fetch.py`` alongside the loopback-server tests, which is where
the mechanics live -- three of these artifacts are over 100 MB, so an unpinned one is a
large unverified download, and that is a property of what ships.
"""

from pathlib import Path


def test_every_shipped_checkpoint_has_a_checksum():
    """A checkpoint without one is downloaded unverified, which is the state this change
    exists to leave behind."""
    from lazyqsar.utils.checkpoints import (
        CDDD_CHECKPOINTS,
        CHECKPOINT_SHA256,
        CHEMELEON_FILENAME,
        CLAMP_FILENAME,
    )

    expected = {CHEMELEON_FILENAME, CLAMP_FILENAME} | {f for _, f in CDDD_CHECKPOINTS}
    assert expected <= set(CHECKPOINT_SHA256)
    assert all(len(v) == 64 for v in CHECKPOINT_SHA256.values())


def test_setup_fetches_every_checkpoint_from_ersilia_first(monkeypatch, tmp_path):
    """Ersilia's copy is the default and the upstream URL the fallback, for every file --
    including the CDDD and CLAMP ones that otherwise live with Model Hub models."""
    from lazyqsar.utils import setup
    from lazyqsar.utils.checkpoints import CHECKPOINT_SHA256, ERSILIA_CHECKPOINTS_URL

    sources = {}

    def record(urls, dest, **_):
        sources[Path(dest).name] = urls

    monkeypatch.setenv("LAZYQSAR_HOME", str(tmp_path))
    monkeypatch.setattr(setup, "fetch", record)
    setup.download_chemeleon()
    setup.download_cddd()
    setup.download_clamp()

    assert set(sources) == set(CHECKPOINT_SHA256)
    for name, urls in sources.items():
        assert urls[0] == ERSILIA_CHECKPOINTS_URL + name
        assert len(urls) == 2 and not urls[1].startswith(ERSILIA_CHECKPOINTS_URL)
