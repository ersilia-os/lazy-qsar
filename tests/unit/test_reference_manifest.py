"""Reading the bundle's manifest, and detecting a reference this install cannot reproduce.

A drifted reference is the quiet kind of wrong: percentiles computed against it stay
monotone, stay in [0, 1], and are simply calibrated against a population the installed
descriptor code does not produce. Nothing downstream can tell, which is why these checks
exist and why several of them are deliberately fatal.

Base tier: the manifest is JSON and the comparisons are numpy. The tests that would need
RDKit build their canary values synthetically instead.
"""

import json

import numpy as np
import pytest

from lazyqsar.reference import manifest as mf


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYQSAR_REFERENCE_DIR", str(tmp_path))
    monkeypatch.setenv("LAZYQSAR_REFERENCE_OFFLINE", "1")
    return tmp_path


def _manifest(**over):
    base = {
        "schema_version": 1,
        "reference_id": "lazyqsar_reference_v1",
        "files": {"morgan_n50000.h5": {"sha256": "a" * 64, "bytes": 10}},
        "checkpoints": {"chemeleon_mp.pt": {"sha256": "b" * 64}},
        "canary": {"file": "canary_n50000.h5", "n": 2, "smiles": ["CCO", "CCC"]},
    }
    base.update(over)
    return base


def test_a_bundle_without_a_manifest_is_not_an_error(bundle):
    """A bundle published before manifests existed still works; it just cannot be
    verified. The caller decides how much that matters."""
    assert mf.load(fetch_if_missing=False) is None


def test_a_manifest_round_trips(bundle):
    (bundle / "manifest.json").write_text(json.dumps(_manifest()))
    assert mf.load(fetch_if_missing=False)["reference_id"] == "lazyqsar_reference_v1"


def test_a_corrupt_download_is_rejected(bundle):
    path = bundle / "morgan_n50000.h5"
    path.write_bytes(b"not the published matrix")
    with pytest.raises(mf.ReferenceDriftError, match="does not match the manifest"):
        mf.verify_file(path, _manifest())


def test_a_file_the_manifest_does_not_mention_is_not_rejected(bundle):
    """Silence is not a failure -- an older manifest may simply not list everything."""
    path = bundle / "extra.h5"
    path.write_bytes(b"whatever")
    mf.verify_file(path, _manifest())


def test_verification_is_skipped_when_there_is_no_manifest(bundle):
    path = bundle / "morgan_n50000.h5"
    path.write_bytes(b"whatever")
    mf.verify_file(path, None)


# --------------------------------------------------------- the checkpoint gate


def test_a_missing_checkpoint_record_does_not_count_as_unchanged(bundle):
    """The fail-open bug this test exists for.

    The gate used to answer "unchanged" for a manifest that recorded no checkpoints at all,
    so a bundle with no provenance skipped the drift check entirely -- exactly when it was
    least deserved. Only a recorded hash matching a local file may count as proven.
    """
    proven, reason = mf._checkpoint_unchanged({"checkpoints": {}}, "chemeleon")
    assert not proven
    assert "records no checkpoint" in reason


def test_a_descriptor_without_a_checkpoint_is_always_recomputed(bundle):
    """Morgan and RDKit are a pure function of the toolkit, so nothing can vouch for them
    short of the numbers."""
    for name in ("morgan", "rdkit"):
        proven, reason = mf._checkpoint_unchanged(_manifest(), name)
        assert not proven
        assert "no checkpoint" in reason


def test_a_checkpoint_that_is_not_installed_is_not_proven(bundle, monkeypatch):
    monkeypatch.setenv("LAZYQSAR_HOME", str(bundle / "empty"))
    proven, reason = mf._checkpoint_unchanged(_manifest(), "chemeleon")
    assert not proven
    assert "not present" in reason


def test_a_matching_checkpoint_is_proven_and_skips_the_recompute(bundle, monkeypatch):
    import hashlib

    home = bundle / "home"
    home.mkdir()
    (home / "chemeleon_mp.pt").write_bytes(b"pretend checkpoint")
    digest = hashlib.sha256(b"pretend checkpoint").hexdigest()
    monkeypatch.setenv("LAZYQSAR_HOME", str(home))
    proven, _ = mf._checkpoint_unchanged(
        _manifest(checkpoints={"chemeleon_mp.pt": {"sha256": digest}}), "chemeleon"
    )
    assert proven


# --------------------------------------------------------- the value comparison


def test_a_changed_morgan_value_is_fatal():
    """Morgan is deterministic given the toolkit, so any difference is a definition change
    and the published matrices no longer describe what this install computes."""
    stored = np.zeros((2, 8))
    fresh = stored.copy()
    fresh[0, 0] = 1
    with pytest.raises(mf.ReferenceDriftError, match="deterministic"):
        mf._compare("morgan", stored, fresh)
    assert mf._compare("morgan", stored, stored.copy()) == {"status": "pass"}


def test_rdkit_is_compared_on_values_not_names():
    """RDKit's descriptor names are identical across releases that change the values, so a
    version or name check passes where the numbers do not."""
    stored = np.ones((2, 4))
    assert mf._compare("rdkit", stored, stored * (1 + 1e-9))["status"] == "pass"
    with pytest.raises(mf.ReferenceDriftError, match="beyond tolerance"):
        mf._compare("rdkit", stored, stored * 1.5)


def test_an_embedding_moved_by_a_backend_is_tolerated():
    """ONNX Runtime, BLAS and CPU architecture move embeddings at the 1e-4 level. Failing a
    fit over that would be wrong."""
    rng = np.random.default_rng(0)
    stored = rng.normal(size=(4, 32))
    assert mf._compare("chemeleon", stored, stored + 1e-7)["status"] == "pass"


def test_an_embedding_moved_by_a_changed_checkpoint_is_fatal():
    """Orders of magnitude larger than a backend difference, which is what makes cosine a
    workable separator rather than a guess."""
    rng = np.random.default_rng(0)
    stored = rng.normal(size=(4, 32))
    with pytest.raises(mf.ReferenceDriftError, match="changed checkpoint"):
        mf._compare("chemeleon", stored, -stored)


def test_a_changed_feature_count_is_fatal_for_any_descriptor():
    with pytest.raises(mf.ReferenceDriftError, match="features"):
        mf._compare("clamp", np.ones((2, 8)), np.ones((2, 9)))
