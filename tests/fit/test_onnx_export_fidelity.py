"""Does the ONNX checkpoint compute the same thing as the model it was exported from?

This is the guarantee the whole deployment story rests on: a model is validated in Python and
shipped as ONNX, and the two are supposed to be the same model.

``test_head_onnx_roundtrip.py`` checks each head against its own export, feeding both the
*same* preprocessed matrix. That is the right test for the converters, but it cannot see the
failure below, because the failure is about the heads receiving slightly *different* input.
"""

import contextlib
import io

import numpy as np
import pytest

from lazyqsar.agnostic import LazyClassifier

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")


def _noisy_dataset(n=200, p=30, seed=0):
    """Realistic data: overlapping classes, no clean separating boundary.

    Deliberately unlike the near-separable data the head round-trip tests use. When classes
    are separable every tree is confident and far from its split thresholds, which is exactly
    the regime where the discrepancy below does not appear.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    X, y = _noisy_dataset()
    model = LazyClassifier()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X=X, y=y)
    directory = str(tmp_path_factory.mktemp("export"))
    with contextlib.redirect_stdout(io.StringIO()):
        model.save(directory)
        loaded = LazyClassifier.load(directory)
    return model, loaded, X


def test_the_preprocessor_round_trips(fitted):
    """It does -- to about 4e-7, which is ordinary float32 rounding and looks harmless."""
    model, loaded, X = fitted
    mem = np.asarray(model._model.models[0].prep.transform(X))
    onnx = np.asarray(loaded._batches[0].preprocessor.run(X))
    assert np.abs(mem - onnx).max() < 1e-5


def test_labels_still_agree(fitted):
    """The practical guarantee that does hold: the shipped model calls the same compounds."""
    model, loaded, X = fitted
    assert np.array_equal(model.predict(X=X), loaded.predict(X))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect: the exported ONNX model disagrees with the model it came from by up "
        "to ~0.1 in score on non-separable data. Root cause is in "
        "test_tree_heads_amplify_preprocessor_rounding below. Remove this marker when fixed."
    ),
)
def test_exported_model_matches_the_model_it_came_from(fitted):
    model, loaded, X = fitted
    np.testing.assert_allclose(
        loaded.predict_score(X)[:, 1], model.predict_score(X=X)[:, 1], atol=1e-5
    )


def test_tree_heads_amplify_preprocessor_rounding(fitted):
    """The mechanism, pinned so the diagnosis is not lost.

    One ONNX head, two preprocessor outputs that differ only by float32 rounding (~4e-7).
    A linear or kernel head barely notices. A gradient-boosted tree is piecewise constant, so
    an input that lands a hair either side of a split threshold falls into a different leaf
    and the score jumps discontinuously -- here by around 0.4, six orders of magnitude more
    than the perturbation that caused it.

    That is why the exported model and the fitted model disagree: not because any converter
    is wrong, but because the ONNX preprocessor hands the tree heads float32 where sklearn
    handed them float64. This test documents the amplification; it does not assert that the
    amplification is acceptable.
    """
    model, loaded, X = fitted
    batch_mem, batch_onnx = model._model.models[0], loaded._batches[0]
    prep_mem = np.asarray(batch_mem.prep.transform(X))
    prep_onnx = np.asarray(batch_onnx.preprocessor.run(X))

    perturbation = np.abs(prep_mem - prep_onnx).max()
    assert perturbation < 1e-5, "the two preprocessors agree to float32 rounding"

    amplified = {}
    for name, head in zip(batch_mem.portfolio, batch_onnx.heads):
        a = head.predict_score(prep_mem)[:, 1]
        b = head.predict_score(prep_onnx)[:, 1]
        amplified[name] = float(np.abs(a - b).max())

    tree_heads = [n for n in ("xgb", "rf") if n in amplified]
    assert tree_heads, "this dataset should select at least one tree head"
    assert max(amplified[n] for n in tree_heads) > 100 * perturbation, (
        f"expected a tree head to amplify a {perturbation:.1e} input difference; "
        f"got {amplified}"
    )
