"""Does the ONNX checkpoint compute the same thing as the model it was exported from?

This is the guarantee the whole deployment story rests on: a model is validated in Python and
shipped as ONNX, and the two are supposed to be the same model.

They did not used to be. The exported preprocessor ran in float32 while scikit-learn ran it
in float64, a difference of about one float32 ULP -- and the heads downstream are piecewise
constant, so a value landing a hair either side of a learned split fell into a different
leaf and moved the score by ~0.1. Checking each head against its own export could never see
it, because that feeds both sides the *same* matrix; the failure was about the heads
receiving slightly *different* input.

The fix was to stop the two paths differing rather than to widen a tolerance: the fitted
preprocessor now runs its own exported graph (``BasePreprocessor._bind_onnx_runtime``), so
the heads are fitted on bit-identical values to the ones they are later served. Several
tests here therefore assert exact equality where they used to assert a tolerance, and one
has to synthesise the perturbation it documents, because the pipeline no longer produces it.
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
    """Bit-identically, now that the fitted preprocessor runs its own exported graph.

    Exactness is the point, not a tighter tolerance: anything above zero here is a value
    that could sit on the wrong side of a split threshold. If this starts merely *nearly*
    passing, the fit-time path has been reverted to scikit-learn's and the whole
    fit-versus-export gap is back.
    """
    model, loaded, X = fitted
    mem = np.asarray(model._model.models[0].prep.transform(X))
    onnx = np.asarray(loaded._batches[0].preprocessor.run(X))
    np.testing.assert_array_equal(mem, onnx)


def test_labels_still_agree(fitted):
    """The practical guarantee that does hold: the shipped model calls the same compounds."""
    model, loaded, X = fitted
    assert np.array_equal(model.predict(X=X), loaded.predict(X))


def test_exported_model_matches_the_model_it_came_from(fitted):
    """The headline guarantee. Was `xfail(strict=True)` until the preprocessor was bound.

    Deliberately on non-separable data: when classes separate cleanly every tree sits far
    from its thresholds and the defect this pins cannot appear.
    """
    model, loaded, X = fitted
    np.testing.assert_allclose(
        loaded.predict_score(X)[:, 1], model.predict_score(X=X)[:, 1], atol=1e-5
    )


def _pairs_that_flip(a, b):
    """Boolean matrix of pairs (i, j) that *a* and *b* strictly disagree about ordering."""
    sign_a = np.sign(np.subtract.outer(a, a))
    sign_b = np.sign(np.subtract.outer(b, b))
    return (sign_a * sign_b) < 0


def test_the_export_ranks_the_same_compounds_at_the_top(fitted):
    """The ordering counterpart to `test_labels_still_agree`, for the default output.

    `rank` used to be a mean of the batches' percentiles, which is not a monotone
    transform of the pooled probability -- so on top of inheriting the score gap above, it
    could order two compounds the other way round, and an ECDF amplified what it did
    inherit. It is now the percentile of the pooled probability, which makes it exactly as
    reproducible as `predict_proba` is and no less.
    """
    from scipy.stats import spearmanr

    model, loaded, X = fitted
    fit_rank = model.predict_rank(X=X)[:, 1]
    onnx_rank = loaded.predict_rank(X)[:, 1]
    fit_proba = model.predict_proba(X=X)[:, 1]
    onnx_proba = loaded.predict_proba(X)[:, 1]

    # Within each runtime, rank must say the same thing about order as proba does.
    for tag, proba, rank in (
        ("fit", fit_proba, fit_rank),
        ("onnx", onnx_proba, onnx_rank),
    ):
        order = np.argsort(proba, kind="stable")
        assert np.all(np.diff(rank[order]) >= -1e-12), (
            f"{tag}: rank disagrees with proba"
        )

    # And across the boundary, rank must not disagree about any pair that proba agrees
    # about. That follows from the monotonicity above and is exact: if rank strictly
    # orders i above j, so does proba, in whichever runtime. What rank may add is *ties* --
    # two molecules whose probabilities differ by less than the local knot spacing land on
    # one percentile -- so the rank correlation can sit a shade below the proba one
    # without any molecule having been reordered. Hence the pair test rather than a
    # Spearman comparison.
    new_flips = _pairs_that_flip(fit_rank, onnx_rank) & ~_pairs_that_flip(
        fit_proba, onnx_proba
    )
    s_rank = spearmanr(fit_rank, onnx_rank).statistic
    s_proba = spearmanr(fit_proba, onnx_proba).statistic
    assert not new_flips.any(), (
        f"rank reorders {new_flips.sum() // 2} pair(s) that proba agrees about "
        f"(spearman rank {s_rank:.9f}, proba {s_proba:.9f})"
    )


def test_tree_heads_amplify_a_rounding_difference(fitted):
    """Why the fix had to remove the difference rather than tolerate it.

    The pipeline no longer produces two differing preprocessor outputs, so the
    perturbation is synthesised: one ULP of float32, the scale the two paths used to
    disagree by. A linear or kernel head barely notices -- it moves by about the ULP. A
    gradient-boosted tree is piecewise constant, so an input landing a hair the other side
    of a learned split falls into a different leaf and the score jumps discontinuously,
    here by ~0.5, around a millionfold more than the nudge that caused it.

    The nudge is *downward* on purpose, and that asymmetry is the finding. XGBoost's split
    thresholds coincide with observed feature values, and its rule is ``x < threshold``, so
    a value sitting exactly on its threshold is unmoved by a nudge up and crosses on a
    nudge down. Perturbing upward flips nothing at all.

    Kept after the fix because the hazard is avoided, not removed: anything reintroducing
    a float32-scale difference upstream of a tree head reintroduces this. The test
    documents the amplification; it does not assert that it is acceptable.
    """
    model, loaded, X = fitted
    batch_mem, batch_onnx = model._model.models[0], loaded._batches[0]
    prep = np.asarray(batch_mem.prep.transform(X))

    nudged = np.nextafter(prep.astype(np.float32), np.float32(-np.inf))
    perturbation = float(
        np.abs(nudged.astype(np.float64) - prep.astype(np.float64)).max()
    )
    assert 0 < perturbation < 1e-5, (
        "one float32 ULP, the scale the two paths differed by"
    )

    amplified = {}
    for name, head in zip(batch_mem.portfolio, batch_onnx.heads):
        a = head.predict_score(prep)[:, 1]
        b = head.predict_score(nudged)[:, 1]
        amplified[name] = float(np.abs(a - b).max())

    tree_heads = [n for n in ("xgb", "rf") if n in amplified]
    assert tree_heads, "this dataset should select at least one tree head"
    assert max(amplified[n] for n in tree_heads) > 100 * perturbation, (
        f"expected a tree head to amplify a {perturbation:.1e} input difference; "
        f"got {amplified}"
    )


@pytest.mark.parametrize("raw_shape", ["two_column", "one_column", "flat"])
def test_xgboost_predict_score_normalises_shape_like_run(fitted, raw_shape):
    """`predict_score` must return (n, 2) whatever the graph emits, as `run` always did.

    It used not to. On a legacy export -- onnxmltools annotating `probabilities` with
    dim_value=2 while ORT infers {N,1} from a missing n_targets, the case
    `_build_xgb_session` exists to repair -- `run` returned (n, 2) and `predict_score`
    returned (n, 1). The caller's `[:, 1]` then raised, the scoring loop swallowed it as
    "channel unavailable", and `predict_type="score"` silently fell back to pooling the
    *calibrated* probabilities. A wrong number with no error anywhere.

    Every head shipped today emits two columns, so this is driven by substituting the
    session rather than by finding a checkpoint that still does it.
    """
    _, loaded, X = fitted
    heads = {type(h).__name__: h for b in loaded._batches for h in b.heads}
    head = next((h for n, h in heads.items() if "XGBoost" in n), None)
    if head is None:
        pytest.skip("this dataset's portfolio has no xgboost head")

    prep = loaded._batches[0].preprocessor.run(X)
    two_col = np.asarray(head.predict_score(prep), dtype=np.float64)
    assert two_col.shape == (len(X), 2), "baseline is not two columns"

    p1 = two_col[:, 1]
    substitute = {
        "two_column": two_col,
        "one_column": p1.reshape(-1, 1),
        "flat": p1,
    }[raw_shape]

    class _Stub:
        def run(self, _outputs, _feed):
            return [None, substitute]

        def get_outputs(self):
            return [
                type("M", (), {"name": "label"})(),
                type("M", (), {"name": "probabilities"})(),
            ]

    real, head._session = head._session, _Stub()
    try:
        out = head.predict_score(prep)
    finally:
        head._session = real

    assert out.shape == (len(X), 2), (
        f"a {raw_shape} graph output gave predict_score shape {out.shape}; "
        "run() normalises it to (n, 2) and predict_score must agree"
    )
    np.testing.assert_allclose(out[:, 1], p1, rtol=0, atol=0)
