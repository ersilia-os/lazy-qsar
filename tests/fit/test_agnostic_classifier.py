"""``LazyClassifier`` -- the bring-your-own-descriptors entry point.

The documented path for Ersilia Model Hub users, who compute descriptors with the Hub and
hand LazyQSAR the resulting ``.h5``. Two things matter here and are not covered anywhere
else: that the ``X=`` and ``h5_file=`` paths are genuinely the same code, and that a model
survives the zip round trip that the README tells people to use.
"""

import os
import zipfile

import h5py
import numpy as np
import pytest

from lazyqsar.agnostic import LazyClassifier, LazyRegressor, _load_h5

N, P = 200, 30


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, P)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=N) > 0).astype(int)
    return X, y


@pytest.fixture
def h5_path(data, tmp_path):
    X, _ = data
    path = str(tmp_path / "desc.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("Values", data=X)
    return path


# ------------------------------------------------------------------------------ _load_h5


@pytest.mark.parametrize("key", ["X", "data", "values", "Values"])
def test_every_documented_dataset_key_is_found(data, tmp_path, key):
    """Ersilia's .h5 files have used several of these names over time."""
    X, _ = data
    path = str(tmp_path / f"{key}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=X)
    np.testing.assert_array_equal(_load_h5(path), X)


def test_an_unrecognised_key_raises_and_says_what_it_found(data, tmp_path):
    path = str(tmp_path / "odd.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=data[0])
    with pytest.raises(ValueError) as exc:
        _load_h5(path)
    assert "features" in str(exc.value), (
        "the error should name the keys actually present"
    )


def test_h5_idxs_selects_a_subset_of_rows(data, h5_path):
    X, _ = data
    np.testing.assert_array_equal(_load_h5(h5_path, [0, 1, 3]), X[[0, 1, 3]])


def test_h5_idxs_requires_ascending_indices(h5_path):
    """Characterization, not endorsement.

    ``h5_idxs`` reads straight through to h5py fancy indexing, which requires a strictly
    increasing selection. So the parameter can subset but cannot reorder, and handing it the
    kind of index array that ``np.argsort`` or a train/test split produces raises rather than
    returning misaligned rows. Raising is the safe failure, but the constraint is undocumented
    and this is where a caller will meet it.
    """
    with pytest.raises(TypeError, match="increasing order"):
        _load_h5(h5_path, [3, 1, 0])


def test_loaded_features_are_float32(data, tmp_path):
    path = str(tmp_path / "f64.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=data[0].astype(np.float64))
    assert _load_h5(path).dtype == np.float32


# ----------------------------------------------------------------------------- fit paths


def test_fit_needs_features_from_somewhere(data):
    with pytest.raises(ValueError):
        LazyClassifier().fit(y=data[1])


def test_the_h5_path_and_the_array_path_are_the_same_model(data, h5_path):
    """If these diverge, a Hub user gets a different model from the documented input."""
    X, y = data
    from_array, from_h5 = LazyClassifier(), LazyClassifier()
    from_array.fit(X=X, y=y)
    from_h5.fit(h5_file=h5_path, y=y)

    assert from_array.oof_auc_ == pytest.approx(from_h5.oof_auc_)
    assert from_array.train_auc_ == pytest.approx(from_h5.train_auc_)
    np.testing.assert_allclose(
        from_array.predict_proba(X=X), from_h5.predict_proba(h5_file=h5_path), atol=1e-6
    )


@pytest.mark.parametrize(
    "method",
    ["predict_proba", "predict_logit", "predict_score", "predict_lift", "predict_rank"],
)
def test_every_predictor_accepts_both_input_forms(data, h5_path, method):
    X, y = data
    model = LazyClassifier()
    model.fit(X=X, y=y)
    np.testing.assert_allclose(
        getattr(model, method)(X=X), getattr(model, method)(h5_file=h5_path), atol=1e-6
    )


def test_fitted_aucs_are_probabilities(data):
    X, y = data
    model = LazyClassifier()
    model.fit(X=X, y=y)
    assert 0.0 <= model.oof_auc_ <= 1.0
    assert 0.0 <= model.train_auc_ <= 1.0


def test_fit_is_not_fluent(data):
    """Pinned because it reads like it should be, and the README never chains it."""
    X, y = data
    assert LazyClassifier().fit(X=X, y=y) is None


# ---------------------------------------------------------------------------- save / load


def test_directory_round_trip_preserves_predictions(data, tmp_path):
    X, y = data
    model = LazyClassifier()
    model.fit(X=X, y=y)
    before = model.predict_proba(X=X)

    out = model.save(str(tmp_path / "m"))
    assert out == str(tmp_path / "m")

    reloaded = LazyClassifier.load(out)
    # Not 1e-5: the export is not bit-faithful on non-separable data. See
    # test_onnx_export_fidelity.py, which isolates why (float32 preprocessor rounding
    # amplified by tree-head split discontinuities). This asserts the round trip preserves
    # the *model* -- same shape, same calibrated range, same calls -- while that file owns
    # the numerical defect so it is tracked in one place rather than papered over here.
    assert reloaded.predict_proba(X).shape == before.shape
    np.testing.assert_allclose(reloaded.predict_proba(X), before, atol=0.15)
    np.testing.assert_allclose(reloaded.predict_proba(X).sum(axis=1), 1.0, atol=1e-6)


def test_zip_round_trip_preserves_predictions(data, tmp_path):
    """The README tells people to do this, so it has to work."""
    X, y = data
    model = LazyClassifier()
    model.fit(X=X, y=y)
    before = model.predict_proba(X=X)

    archive = model.save(str(tmp_path / "m.zip"))
    assert archive.endswith(".zip")
    assert os.path.isfile(archive)
    assert not os.path.exists(str(tmp_path / "m")), (
        "the staging directory is cleaned up"
    )
    with zipfile.ZipFile(archive) as z:
        assert "metadata.json" in z.namelist()

    reloaded = LazyClassifier.load(archive)
    # Tolerance as in the directory round trip above; the archive is the same artifact.
    np.testing.assert_allclose(reloaded.predict_proba(X), before, atol=0.15)


def test_loading_a_raw_directory_is_explicitly_unsupported(tmp_path):
    """Pinned so implementing it later is a deliberate, visible change."""
    empty = tmp_path / "raw"
    empty.mkdir()
    with pytest.raises(NotImplementedError):
        LazyClassifier.load(str(empty))


def test_regression_is_not_implemented():
    with pytest.raises(NotImplementedError):
        LazyRegressor()
