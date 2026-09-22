"""Downloading a file and proving it arrived intact.

Base tier: `lazyqsar/utils/fetch.py` is standard library plus `rich`, because it runs during
`lazyqsar setup` before the descriptor stack exists.

Served from a loopback HTTP server rather than mocked. The defects this module fixes are
about what ends up *on disk* after a partial or wrong transfer, and a mock that returns
bytes cannot reproduce them.
"""

import functools
import hashlib
import http.server
import pathlib
import threading

import pytest

from lazyqsar.utils.fetch import DownloadError, fetch, is_cached, sha256_file

PAYLOAD = b"reference library bytes" * 500


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("served")
    (root / "good.bin").write_bytes(PAYLOAD)
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(root)
    )
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/good.bin"
    httpd.shutdown()


@pytest.fixture(scope="module")
def digest():
    return hashlib.sha256(PAYLOAD).hexdigest()


def test_a_verified_download_lands(server, digest, tmp_path):
    dest = tmp_path / "a.bin"
    fetch(server, dest, sha256=digest)
    assert dest.read_bytes() == PAYLOAD


def test_a_wrong_checksum_is_rejected_and_nothing_is_cached(server, tmp_path):
    """A file that is not what was asked for must not be left behind to be used later."""
    dest = tmp_path / "b.bin"
    with pytest.raises(DownloadError, match="checksum"):
        fetch(server, dest, sha256="0" * 64)
    assert not dest.exists()


def test_a_truncated_file_is_not_mistaken_for_a_complete_one(server, digest, tmp_path):
    """The defect this module exists for.

    The old downloader decided a file was present with `dest.exists()`, so a download
    interrupted halfway left a truncated file that was never re-fetched and surfaced later
    as an opaque parse error inside ONNX Runtime. Three of these artifacts are over 100 MB.
    """
    dest = tmp_path / "c.bin"
    dest.write_bytes(b"interrupted")
    assert not is_cached(dest, digest)
    fetch(server, dest, sha256=digest)
    assert dest.read_bytes() == PAYLOAD


def test_a_wrong_size_is_enough_to_invalidate_a_cache_hit(tmp_path, digest):
    """Size is checked before the hash: it costs a stat rather than a pass over 100 MB, and
    truncation always changes it."""
    dest = tmp_path / "d.bin"
    dest.write_bytes(PAYLOAD)
    assert is_cached(dest, expected_bytes=len(PAYLOAD))
    assert not is_cached(dest, expected_bytes=len(PAYLOAD) + 1)


def test_a_good_cache_hit_does_not_re_download(server, digest, tmp_path):
    dest = tmp_path / "e.bin"
    fetch(server, dest, sha256=digest)
    before = dest.stat().st_mtime_ns
    fetch(server, dest, sha256=digest)
    assert dest.stat().st_mtime_ns == before


def test_force_re_downloads_even_when_cached(server, digest, tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(PAYLOAD)
    fetch(server, dest, sha256=digest, force=True)
    assert dest.read_bytes() == PAYLOAD


def test_no_partial_file_is_left_behind(server, tmp_path):
    dest = tmp_path / "g.bin"
    with pytest.raises(DownloadError):
        fetch(server, dest, sha256="0" * 64)
    assert not list(tmp_path.glob("*.part"))


def test_a_failed_connection_raises_rather_than_writing_something(tmp_path):
    dest = tmp_path / "h.bin"
    with pytest.raises(DownloadError, match="Could not download"):
        fetch("http://127.0.0.1:1/never", dest)
    assert not dest.exists()


def test_the_staging_name_does_not_eat_a_suffix(server, digest, tmp_path):
    """`with_suffix` replaces the *last* suffix, so `cddd_encoder_fpsim.h5` would have been
    staged as `cddd_encoder_fpsim.part` and could collide with another download."""
    dest = tmp_path / "cddd_encoder_fpsim.h5"
    fetch(server, dest, sha256=digest)
    assert dest.is_file()


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "i.bin"
    path.write_bytes(PAYLOAD)
    assert sha256_file(path) == hashlib.sha256(PAYLOAD).hexdigest()


def test_lazyqsar_home_redirects_the_cache(monkeypatch, tmp_path):
    """One variable governs where checkpoints go *and* where the descriptors look.

    `--target-dir` used to move only the download, so a redirected setup fetched everything
    twice and then read the copy it had not just written.
    """
    from lazyqsar.utils.checkpoints import checkpoint_dir

    monkeypatch.setenv("LAZYQSAR_HOME", str(tmp_path / "scratch"))
    assert checkpoint_dir() == pathlib.Path(tmp_path / "scratch")
    monkeypatch.delenv("LAZYQSAR_HOME")
    assert checkpoint_dir() == pathlib.Path.home() / ".lazyqsar"


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
