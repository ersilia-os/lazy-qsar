"""``EnsembleSpec.from_metadata`` -- how a checkpoint's metadata becomes a weighting spec.

This is the seam where the two loaders used to disagree: one read ``quality_aucs`` and the
other ``oof_aucs``, so the same checkpoint was weighted differently depending on which loader
ran. These tests pin the resolution order and the slicing of everything to the active
descriptors. Numpy only.
"""

import numpy as np

from lazyqsar.ensemble import EnsembleSpec


def test_from_metadata_prefers_quality_auc():
    """quality_aucs is the deployed convention; oof_aucs is the fallback."""
    meta = {
        "quality_aucs": {"a": 0.8, "b": 0.7},
        "oof_aucs": {"a": 0.9, "b": 0.95, "c": 0.6},
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b", "c"])
    assert active == ["a", "b", "c"]
    assert spec.oof_aucs == (0.8, 0.7, 0.6)


def test_from_metadata_honours_active_descriptors():
    meta = {
        "oof_aucs": {"a": 0.9, "b": 0.8, "c": 0.7},
        "active_descriptors": {"a": True, "b": False, "c": True},
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b", "c"])
    assert active == ["a", "c"]
    assert spec.descriptor_names == ("a", "c")
    assert spec.oof_aucs == (0.9, 0.7)


def test_from_metadata_all_inactive_falls_back_to_all():
    """A mask that excludes everything would leave nothing to predict with."""
    meta = {"active_descriptors": {"a": False, "b": False}}
    _, active = EnsembleSpec.from_metadata(meta, ["a", "b"])
    assert active == ["a", "b"]


def test_from_metadata_empty_metadata_gives_working_defaults():
    spec, active = EnsembleSpec.from_metadata({}, ["a", "b"])
    assert active == ["a", "b"]
    assert spec.oof_aucs == (1.0, 1.0)
    assert spec.proxy_aucs == (None, None)
    assert spec.rank_error_curves is None
    assert spec.ad_hard_cutoffs is None
    assert spec.population_prior == 0.5
    assert spec.decision_cutoff == 0.5
    assert spec.pooled_rank_knots is None


def test_from_metadata_reads_the_pooled_rank_reference():
    meta = {"pooled_ranker": {"knots": [0.1, 0.4, 0.9], "n_train": 3, "source": "oof"}}
    spec, _ = EnsembleSpec.from_metadata(meta, ["a"])
    assert isinstance(spec.pooled_rank_knots, np.ndarray)
    assert spec.pooled_rank_knots.tolist() == [0.1, 0.4, 0.9]


def test_from_metadata_treats_an_empty_pooled_reference_as_absent():
    """Empty knots must reach `combine` as None, not as something `prepare_knots` raises on."""
    for block in ({"knots": []}, {}, None):
        spec, _ = EnsembleSpec.from_metadata({"pooled_ranker": block}, ["a"])
        assert spec.pooled_rank_knots is None


def test_the_pooled_rank_reference_is_not_sliced_to_active_descriptors():
    """It describes the pooled probability of the active set as a whole, not one column."""
    meta = {
        "active_descriptors": {"a": True, "b": False},
        "pooled_ranker": {"knots": [0.2, 0.5, 0.8]},
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b"])
    assert active == ["a"]
    assert spec.pooled_rank_knots.tolist() == [0.2, 0.5, 0.8]


def test_from_metadata_slices_curves_and_cutoffs_to_active():
    meta = {
        "active_descriptors": {"a": True, "b": False, "c": True},
        "ad_hard_cutoffs": {"a": 0.1, "b": 0.2, "c": 0.3},
        "rank_error_curves": {
            "a": [[0.0, 1.0], [0.5, 0.1]],
            "b": [[0.0, 1.0], [0.4, 0.2]],
            "c": [[0.0, 1.0], [0.3, 0.3]],
        },
    }
    spec, active = EnsembleSpec.from_metadata(meta, ["a", "b", "c"])
    assert active == ["a", "c"]
    assert spec.ad_hard_cutoffs == (0.1, 0.3)
    assert len(spec.rank_error_curves) == 2
    assert np.array_equal(spec.rank_error_curves[1][1], np.array([0.3, 0.3]))
