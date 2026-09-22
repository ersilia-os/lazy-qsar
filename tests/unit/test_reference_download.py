"""Fetching the reference bundle, and refusing clearly when it cannot be fetched.

Nothing here reaches the network. The suite sets `LAZYQSAR_REFERENCE_OFFLINE`, and the one
test that exercises the eosvc call substitutes a fake binary on PATH -- the thing worth
pinning is the contract with eosvc (which repo, which path, anonymous), not eosvc itself.
"""

import json
import os
import pathlib
import stat

import pytest

from lazyqsar.reference import identity
from lazyqsar.reference.download import (
    EOSVC_REPO,
    EOSVC_ROOT,
    ReferenceDownloadError,
    download,
    eosvc_available,
    offline,
)


def test_offline_follows_the_environment(monkeypatch):
    monkeypatch.setenv("LAZYQSAR_REFERENCE_OFFLINE", "1")
    assert offline()
    monkeypatch.delenv("LAZYQSAR_REFERENCE_OFFLINE")
    assert not offline()


def test_offline_refuses_instead_of_hanging(monkeypatch, tmp_path):
    """An air-gapped node should get an answer, not a stalled socket."""
    monkeypatch.setenv("LAZYQSAR_REFERENCE_OFFLINE", "1")
    monkeypatch.setenv("LAZYQSAR_REFERENCE_DIR", str(tmp_path))
    with pytest.raises(ReferenceDownloadError, match="OFFLINE"):
        download(["morgan_n50000.h5"])


def test_a_cached_file_is_not_re_fetched(monkeypatch, tmp_path):
    monkeypatch.setenv("LAZYQSAR_REFERENCE_OFFLINE", "1")
    monkeypatch.setenv("LAZYQSAR_REFERENCE_DIR", str(tmp_path))
    (tmp_path / "morgan_n50000.h5").write_bytes(b"already here")
    # Offline and yet it succeeds, which is only possible if nothing was fetched.
    assert download(["morgan_n50000.h5"])[0].read_bytes() == b"already here"


def test_a_missing_eosvc_says_how_to_get_it(monkeypatch, tmp_path):
    monkeypatch.delenv("LAZYQSAR_REFERENCE_OFFLINE", raising=False)
    monkeypatch.setenv("LAZYQSAR_REFERENCE_DIR", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ReferenceDownloadError) as exc:
        download(["morgan_n50000.h5"])
    assert "lazyqsar[fit]" in str(exc.value)


def test_the_eosvc_contract(monkeypatch, tmp_path):
    """What is asked of eosvc: this repo, this path, and no credentials.

    A fake binary stands in, because the point is the arguments and the staged repo -- the
    bundle is not published yet, and a test that depended on S3 would be a flake anyway.
    """
    monkeypatch.delenv("LAZYQSAR_REFERENCE_OFFLINE", raising=False)
    cache = tmp_path / "cache"
    monkeypatch.setenv("LAZYQSAR_REFERENCE_DIR", str(cache))

    log = tmp_path / "call.json"
    fake = tmp_path / "bin" / "eosvc"
    fake.parent.mkdir()
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'cwd': os.getcwd(),\n"
        "  'repo': os.environ.get('EVC_REPO_NAME'),\n"
        "  'access': pathlib.Path('access.json').read_text()},\n"
        f"  open({str(log)!r}, 'w'))\n"
        "p = pathlib.Path(sys.argv[sys.argv.index('--path') + 1])\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_bytes(b'matrix')\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ["PATH"])

    name = identity.descriptor_filename("morgan")
    got = download([name])

    call = json.loads(log.read_text())
    assert call["repo"] == EOSVC_REPO, "the repo name is what maps to the S3 prefix"
    assert call["argv"][:2] == ["download", "--path"]
    assert call["argv"][2] == f"{EOSVC_ROOT}/{identity.REFERENCE_ID}/{name}"
    assert json.loads(call["access"]) == {"data": "public"}, (
        "the staged repo must declare data public, or eosvc resolves the wrong bucket"
    )
    assert got[0] == cache / name
    assert got[0].read_bytes() == b"matrix"


def test_the_staging_repo_does_not_outlive_the_download(monkeypatch, tmp_path):
    monkeypatch.setenv("LAZYQSAR_REFERENCE_OFFLINE", "1")
    monkeypatch.setenv("LAZYQSAR_REFERENCE_DIR", str(tmp_path))
    before = set(pathlib.Path(tempfile_dir()).glob("lazyqsar-eosvc-*"))
    with pytest.raises(ReferenceDownloadError):
        download(["morgan_n50000.h5"])
    assert set(pathlib.Path(tempfile_dir()).glob("lazyqsar-eosvc-*")) == before


def tempfile_dir():
    import tempfile

    return tempfile.gettempdir()


def test_eosvc_availability_is_reported_not_assumed():
    assert isinstance(eosvc_available(), bool)
