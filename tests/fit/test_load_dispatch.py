"""``lazyqsar.load`` opens any checkpoint as the right inference object for its task.

One entry point for every saved model: feature-level (``LazyClassifier`` /
``LazyRegressor``) or SMILES-level, classification or regression, directory or zip.
The SMILES-level route is covered with the chem tier; here the feature-level routes are
checked to land on the artifact of their own task and to predict what the fit did.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")

import lazyqsar  # noqa: E402
from lazyqsar.agnostic import LazyClassifier, LazyRegressor  # noqa: E402
from lazyqsar.artifacts.classifier import LazyClassifierArtifact  # noqa: E402
from lazyqsar.artifacts.regressor import LazyRegressorArtifact  # noqa: E402


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _data(n=240, p=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    signal = X[:, 0] - 0.5 * X[:, 1]
    return X, signal + 0.3 * rng.normal(size=n)


@pytest.mark.parametrize("zipped", [False, True])
def test_regressor_checkpoint_loads_as_regression_artifact(tmp_path, zipped):
    X, y = _data()
    model = LazyRegressor()
    _quiet(model.fit, X=X, y=y)
    path = str(tmp_path / ("reg.zip" if zipped else "reg"))
    path = _quiet(model.save, path)
    loaded = _quiet(lazyqsar.load, path)
    assert isinstance(loaded, LazyRegressorArtifact)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X=X), atol=1e-5)


@pytest.mark.parametrize("zipped", [False, True])
def test_classifier_checkpoint_loads_as_classification_artifact(tmp_path, zipped):
    X, y = _data()
    labels = (y > np.median(y)).astype(int)
    model = LazyClassifier()
    _quiet(model.fit, X=X, y=labels)
    path = str(tmp_path / ("clf.zip" if zipped else "clf"))
    path = _quiet(model.save, path)
    loaded = _quiet(lazyqsar.load, path)
    assert isinstance(loaded, LazyClassifierArtifact)
    np.testing.assert_allclose(
        loaded.predict_proba(X)[:, 1], model.predict_proba(X=X)[:, 1], atol=1e-5
    )
