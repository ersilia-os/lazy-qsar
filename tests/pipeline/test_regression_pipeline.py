"""The pipeline invariants, held for regression checkpoints too.

The classification checkpoints have long been pinned by the streaming, robustness and
runner suites. These are the same guarantees for a real ``LazyRegressorQSAR`` checkpoint
over several stubbed descriptors, with prediction intervals:

- the fitted model, the loaded wrapper and the inference runner agree on ``value``,
  ``lower`` and ``upper``;
- scoring in blocks, one molecule at a time, or a repeated molecule changes nothing;
- an empty query returns empty columns; a zip round-trips;
- a descriptor directory the metadata does not list is never scored.
"""

import contextlib
import io
import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("sklearn")

DESCRIPTORS = ("cddd", "chemeleon", "clamp", "morgan", "rdkit")  # slow mode


def _quiet(fn, *a, **kw):
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return fn(*a, **kw)


def _wrapper(cp, task):
    from lazyqsar.qsar import LazyQSAR

    return _quiet(LazyQSAR.load, os.path.join(cp["root"], task))


def test_checkpoint_uses_several_descriptors_and_carries_intervals(
    regression_checkpoint,
):
    meta = json.load(
        open(os.path.join(regression_checkpoint["root"], "alpha", "metadata.json"))
    )
    assert meta["task"] == "regression"
    assert len(meta["descriptor_types"]) >= 2
    assert meta["intervals"]["half_widths"]


def test_fitted_loaded_and_runner_agree(regression_checkpoint):
    from lazyqsar.ensemble.runner import TaskSource, predict_tasks

    cp = regression_checkpoint
    query = cp["smiles"][:40]
    for task, model in cp["models"].items():
        fitted = model.predict(query)
        f_lo, f_hi = model.predict_interval(query)
        wrapper = _wrapper(cp, task)
        loaded = wrapper.predict(query)
        l_lo, l_hi = wrapper.predict_interval(query)
        got = predict_tasks(
            [TaskSource(task, os.path.join(cp["root"], task))],
            query,
            outputs=("value", "lower", "upper"),
        )[0].values
        scale = 1e-6 * max(1.0, float(np.std(fitted)))
        np.testing.assert_allclose(loaded, fitted, rtol=0, atol=scale)
        np.testing.assert_allclose(np.c_[l_lo, l_hi], np.c_[f_lo, f_hi], atol=scale)
        np.testing.assert_array_equal(got["value"], loaded)
        np.testing.assert_array_equal(got["lower"], l_lo)
        np.testing.assert_array_equal(got["upper"], l_hi)
        assert np.all(l_lo <= loaded) and np.all(loaded <= l_hi)


@pytest.mark.parametrize("predict_type", ["value", "lower", "upper"])
def test_blocking_the_input_is_bit_identical(regression_checkpoint, predict_type):
    from lazyqsar.api.predict import predict

    from test_streaming_invariants import forced_block

    cp = regression_checkpoint
    query = (cp["smiles"] * 2)[:150]
    with forced_block(2, 200, chunk_size=50):
        whole, header = _quiet(
            predict, model_dir=cp["root"], smiles=query, predict_type=predict_type
        )
    for rows in (50, 100):
        with forced_block(2, rows, chunk_size=50):
            blocked, blocked_header = _quiet(
                predict, model_dir=cp["root"], smiles=query, predict_type=predict_type
            )
        assert blocked_header == header
        np.testing.assert_array_equal(blocked, whole)


def test_one_molecule_matches_its_batch_and_repeats_are_stable(regression_checkpoint):
    """A molecule scored alone agrees with the same molecule scored in a batch.

    To the tolerance the classification suite documents (``BATCH_SIZE_ATOL`` in
    test_checkpoint_robustness): onnxruntime's graphs see different input shapes and the
    last digits move -- tree ensembles sum in a shape-dependent order. Measured 5.8e-7 on
    x86 Linux for a value of -2.0, nothing on arm64. Bit-identity is asserted for
    blocking, which keeps the shapes, in test_blocking_the_input_is_bit_identical.
    """
    from test_checkpoint_robustness import BATCH_SIZE_ATOL

    cp = regression_checkpoint
    wrapper = _wrapper(cp, "alpha")
    query = cp["smiles"][:10]
    batch = wrapper.predict(query)
    atol = BATCH_SIZE_ATOL * max(1.0, float(np.abs(batch).max()))
    for i in (0, 4, 9):
        single = _wrapper(cp, "alpha").predict([query[i]])
        np.testing.assert_allclose(single, batch[i : i + 1], rtol=0, atol=atol)
    repeated = _wrapper(cp, "alpha").predict([query[3]] * 5)
    assert np.unique(repeated).size == 1


def test_empty_query_returns_empty(regression_checkpoint):
    wrapper = _wrapper(regression_checkpoint, "alpha")
    assert wrapper.predict([]).shape == (0,)
    lo, hi = wrapper.predict_interval([])
    assert lo.shape == hi.shape == (0,)


def test_zip_round_trip_matches_the_directory(regression_checkpoint, tmp_path):
    from lazyqsar.qsar import LazyQSAR

    cp = regression_checkpoint
    model = cp["models"]["beta"]
    path = _quiet(model.save, str(tmp_path / "beta.zip"))
    query = cp["smiles"][:20]
    np.testing.assert_array_equal(
        _quiet(LazyQSAR.load, path).predict(query), _wrapper(cp, "beta").predict(query)
    )


def test_unlisted_descriptor_directory_is_ignored(regression_checkpoint, tmp_path):
    cp = regression_checkpoint
    task_dir = str(tmp_path / "alpha")
    shutil.copytree(os.path.join(cp["root"], "alpha"), task_dir)
    meta = json.load(open(os.path.join(task_dir, "metadata.json")))
    stale = next(d for d in DESCRIPTORS if d not in meta["descriptor_types"])
    # A leftover directory from some other fit, with a copy of a real descriptor's files.
    shutil.copytree(
        os.path.join(task_dir, meta["descriptor_types"][0]),
        os.path.join(task_dir, stale),
    )
    from lazyqsar.qsar import LazyQSAR

    query = cp["smiles"][:15]
    wrapper = _quiet(LazyQSAR.load, task_dir)
    assert stale not in wrapper.spec.descriptor_names
    np.testing.assert_array_equal(
        wrapper.predict(query), _wrapper(cp, "alpha").predict(query)
    )
