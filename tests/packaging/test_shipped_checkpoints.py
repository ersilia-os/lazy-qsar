"""Every checkpoint the package will fetch has a recorded checksum.

A packaging assertion rather than a download-mechanics one: it says nothing about *how* a
file is fetched, only that everything this version is prepared to download is pinned. It
lived in ``tests/unit/test_fetch.py`` alongside the loopback-server tests, which is where
the mechanics live -- three of these artifacts are over 100 MB, so an unpinned one is a
large unverified download, and that is a property of what ships.
"""


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
