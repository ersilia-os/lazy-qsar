"""A descriptor directory the checkpoint's metadata does not list is never scored."""

from lazyqsar.ensemble import EnsembleSpec


def test_unlisted_directory_is_not_active():
    meta = {"descriptor_types": ["morgan"], "quality_aucs": {"morgan": 0.8}}
    spec, active = EnsembleSpec.from_metadata(meta, ["morgan", "rdkit"])
    assert active == ["morgan"]
    assert spec.descriptor_names == ("morgan",)


def test_regression_unlisted_directory_is_not_active():
    meta = {"task": "regression", "descriptor_types": ["rdkit"], "oof_rhos": {}}
    _, active = EnsembleSpec.from_metadata(meta, ["morgan", "rdkit"])
    assert active == ["rdkit"]


def test_checkpoints_without_the_list_keep_every_directory():
    _, active = EnsembleSpec.from_metadata({}, ["morgan", "rdkit"])
    assert active == ["morgan", "rdkit"]
