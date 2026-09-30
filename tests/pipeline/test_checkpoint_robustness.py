"""What happens when a checkpoint or a query is not the happy case.

Every failure here is one a user can cause without doing anything unusual -- an interrupted
copy, a truncated write, an empty shard, a query whose rows were all filtered upstream --
and every one of them was unguarded. Two were known:

* The comment at ``LazyClassifierQSAR.load_onnx`` records that a checkpoint with no
  ``metadata.json`` once raised ``UnboundLocalError`` on ``pooled_rank_anchors``. The fix
  landed; the test did not, so nothing stops it coming back. There are *three* independent
  readers of that file (the runner's ``_read_metadata``, ``load_raw`` and ``load_onnx``),
  which is what made one of them drifting possible in the first place.
* Scoring an empty list raised ``AxisError`` out of ``np.stack`` -- a numpy internal, from
  a public entry point, for a request that has an obvious right answer.

The bar these tests set is deliberately low and deliberately explicit: *either* work, *or*
raise something that names the problem. What none of them may do is return a plausible
number, because a plausible number from a damaged checkpoint is the one outcome nobody
notices.
"""

import contextlib
import io
import json
import os
import shutil

import numpy as np
import pytest

from lazyqsar.qsar import LazyClassifierQSAR

pytest.importorskip("sklearn")

OUTPUTS = ("proba", "rank", "score", "logit", "lift")


def _task_dir(models_root, task):
    return os.path.join(models_root, task)


def _copy_task(tmp_path, models_root, task, name):
    """A writable copy, because the session checkpoints are shared and read-only."""
    dst = os.path.join(str(tmp_path), name)
    shutil.copytree(_task_dir(models_root, task), dst)
    return dst


# --------------------------------------------------------------------- damaged metadata


def test_a_checkpoint_with_no_metadata_still_loads_or_says_why(
    pooled_checkpoint, tmp_path
):
    """The regression the comment in ``load_onnx`` describes, through both loaders.

    ``metadata.json`` carries the active mask, the priors, the reference and the cutoff, so
    a checkpoint without one is impoverished but not meaningless -- the loaders are written
    to fall back. What they must not do is raise ``UnboundLocalError``, which is what
    happened when one branch read a variable the other branch assigned.
    """
    task = _copy_task(tmp_path, pooled_checkpoint["models"], "alpha", "no_meta")
    os.remove(os.path.join(task, "metadata.json"))

    loaded = []
    for loader in (LazyClassifierQSAR.load_onnx, LazyClassifierQSAR.load):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = loader(task)
        except Exception as exc:  # noqa: BLE001 - the point is *which* exception
            assert not isinstance(exc, (UnboundLocalError, AttributeError, KeyError)), (
                f"{loader.__name__} raised {type(exc).__name__} for a checkpoint with no "
                f"metadata.json, which is an internal error rather than a diagnosis: {exc}"
            )
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            proba = model.predict_proba(pooled_checkpoint["smiles"][:5])
        assert proba.shape == (5, 2)
        assert np.all(np.isfinite(proba)), (
            "a checkpoint with no metadata produced non-finite probabilities"
        )
        loaded.append(loader.__name__)

    # Without this the test would still pass if *both* loaders started raising: the
    # tolerant branch above would swallow it and nothing would have been exercised. Both
    # load today, so require both, and let a deliberate change to that be a deliberate
    # change to this line.
    assert loaded == ["load_onnx", "load"], (
        f"only {loaded} loaded a checkpoint with no metadata.json; if refusing it is now "
        "intended, say so here rather than letting the tolerant branch hide it"
    )


def test_the_runner_reads_a_checkpoint_with_no_metadata(pooled_checkpoint, tmp_path):
    """The third reader of ``metadata.json``, which the two above do not exercise."""
    from lazyqsar.ensemble.runner import _read_metadata

    task = _copy_task(tmp_path, pooled_checkpoint["models"], "alpha", "no_meta_runner")
    os.remove(os.path.join(task, "metadata.json"))
    assert _read_metadata(task) == {}, (
        "the runner's metadata reader should report an absent file as no metadata"
    )


@pytest.mark.parametrize(
    "corrupt",
    ["", "{", "not json at all", '{"mode": "fast",}'],
    ids=["empty", "truncated", "garbage", "trailing-comma"],
)
def test_a_corrupt_metadata_file_names_the_file(pooled_checkpoint, tmp_path, corrupt):
    """A half-written ``metadata.json`` must fail, and must say which file.

    An interrupted ``save``, a full disk or a partial copy all produce this. Raising is
    the right outcome -- the silent defaults are a 0.5 decision cutoff and uniform
    descriptor weights, which would score a damaged checkpoint as though it were fine.

    The message has to carry the path: a checkpoint directory holds one metadata file per
    task plus one per descriptor, and a bare ``Expecting property name ... line 1
    column 2`` leaves the user grepping for which. ``utils.archives.read_json`` adds it.
    """
    task = _copy_task(
        tmp_path, pooled_checkpoint["models"], "alpha", f"corrupt_{len(corrupt)}"
    )
    with open(os.path.join(task, "metadata.json"), "w") as fh:
        fh.write(corrupt)

    # Not `(JSONDecodeError, ValueError)`: JSONDecodeError already subclasses ValueError,
    # so the tuple only widened this to accept any unrelated ValueError.
    with pytest.raises(json.JSONDecodeError) as raised:
        with contextlib.redirect_stdout(io.StringIO()):
            LazyClassifierQSAR.load_onnx(task)
    assert "metadata.json" in str(raised.value), (
        f"the parse error does not name the file it failed on: {raised.value}"
    )


# ------------------------------------------------------------------------ degenerate query


def test_scoring_no_molecules_returns_empty_columns(pooled_checkpoint):
    """An empty query has an obvious right answer, and it used to raise ``AxisError``.

    Reached whenever an upstream filter removes every row -- a shard of a sharded library,
    a dataset whose SMILES all failed validation. Every output must come back with the
    right shape so the caller's own concatenation still works.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyClassifierQSAR.load_onnx(
            _task_dir(pooled_checkpoint["models"], "alpha")
        )
        for name in OUTPUTS:
            values = getattr(model, f"predict_{name}")([])
            assert values.shape == (0, 2), (
                f"{name} gave {values.shape} for an empty query"
            )
        assert np.asarray(model.predict([])).shape == (0,)


# The chunk-invariance tolerance the suite already uses. Not bit-identity: onnxruntime
# picks kernels by batch size, so a query of one and a query of six present different
# input shapes and the last few digits move. `_block_size` documents this, and the
# bit-identity that *is* asserted elsewhere is for blocking -- whole numbers of chunks,
# which preserves the shapes each graph actually sees.
BATCH_SIZE_ATOL = 1e-6


def test_scoring_one_molecule_agrees_with_scoring_it_in_a_batch(pooled_checkpoint):
    """A single molecule is the smallest real request, and the one most likely to hit an
    edge in code written with arrays in mind.

    Asserted to a tolerance rather than exactly, and the difference is the point: an
    earlier version of this test demanded bit-identity, passed on arm64 and failed on
    x86 at 6e-08, because it was asserting a property the code documents as false.
    """
    smiles = list(pooled_checkpoint["smiles"][:6])
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyClassifierQSAR.load_onnx(
            _task_dir(pooled_checkpoint["models"], "alpha")
        )
        alone = model.predict_proba([smiles[0]])
        batch = model.predict_proba(smiles)
    assert alone.shape == (1, 2)
    np.testing.assert_allclose(alone[0], batch[0], rtol=0, atol=BATCH_SIZE_ATOL)


def test_a_one_molecule_query_lands_on_the_same_side_of_the_cutoff(pooled_checkpoint):
    """What the tolerance above must not be allowed to hide.

    A probability that moves in its eighth decimal is harmless; a label that flips with
    the size of the batch it was scored in is not, because it makes a screening result
    depend on how the caller happened to chunk its library.
    """
    smiles = list(pooled_checkpoint["smiles"][:6])
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyClassifierQSAR.load_onnx(
            _task_dir(pooled_checkpoint["models"], "alpha")
        )
        alone = np.asarray([model.predict([s])[0] for s in smiles])
        batch = np.asarray(model.predict(smiles))
    moved = np.flatnonzero(alone != batch)
    assert moved.size == 0, (
        f"{moved.size} of {len(smiles)} molecules change label depending on whether they "
        f"were scored alone or in a batch, at positions {moved.tolist()}"
    )


def test_a_repeated_molecule_scores_the_same_every_time(pooled_checkpoint):
    """Duplicates are ordinary in a screening library and were never tested.

    Featurization is cached on a hash of the whole list, so a duplicated row is not a
    cache hit -- it is just another row. Every copy must get the identical score, and the
    molecules around them must be unaffected.
    """
    smiles = list(pooled_checkpoint["smiles"][:5])
    query = [smiles[0], smiles[1], smiles[0], smiles[2], smiles[0]]
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyClassifierQSAR.load_onnx(
            _task_dir(pooled_checkpoint["models"], "alpha")
        )
        out = model.predict_proba(query)[:, 1]
        clean = model.predict_proba(smiles[:3])[:, 1]
    # Exact within the one query: identical rows go through the same graph in the same
    # batch, so nothing may separate them.
    assert out[0] == out[2] == out[4], "copies of one molecule scored differently"
    # Across the two queries, only to tolerance -- five molecules and three are different
    # batch sizes, and onnxruntime chooses kernels by batch size. Asserting equality here
    # would be the same mistake as demanding a query of one match a query of six.
    np.testing.assert_allclose(out[[0, 1, 3]], clean, rtol=0, atol=BATCH_SIZE_ATOL)


# -------------------------------------------------------------------------- the zip route


def test_loading_a_zip_leaves_a_sibling_directory_alone(pooled_checkpoint, tmp_path):
    """``load("x.zip")`` used to ``rmtree("x")`` first, with no warning and no undo.

    ``models/tb_model/`` next to ``models/tb_model.zip`` is exactly what a user who saved
    both would have, so this deleted working checkpoints. The archive now unpacks to
    scratch; the sibling is untouched and nothing is left behind next to the archive.
    """
    task = _copy_task(tmp_path, pooled_checkpoint["models"], "alpha", "ziptask")
    archive = shutil.make_archive(os.path.join(str(tmp_path), "bundle"), "zip", task)

    sibling = os.path.join(str(tmp_path), "bundle")
    os.makedirs(sibling, exist_ok=True)
    keep = os.path.join(sibling, "precious.txt")
    with open(keep, "w") as fh:
        fh.write("a checkpoint the user is still using")

    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyClassifierQSAR.load(archive)
        proba = model.predict_proba(pooled_checkpoint["smiles"][:4])

    assert os.path.isfile(keep), "loading the archive deleted the sibling directory"
    with open(keep) as fh:
        assert fh.read() == "a checkpoint the user is still using"
    assert proba.shape == (4, 2) and np.all(np.isfinite(proba)), (
        "the model loaded from the archive does not predict, so the scratch copy was "
        "removed before its sessions were built"
    )


# ------------------------------------------------------------------- the cutoff override


def test_a_caller_supplied_threshold_is_what_binarises(pooled_checkpoint):
    """``predict(threshold=...)`` is public and was covered nowhere.

    Two thresholds either side of the scores must partition the same query differently,
    which is the only assertion that distinguishes "the argument is used" from "the
    argument is accepted and ignored".
    """
    smiles = list(pooled_checkpoint["smiles"][:20])
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyClassifierQSAR.load_onnx(
            _task_dir(pooled_checkpoint["models"], "alpha")
        )
        everything = np.asarray(model.predict(smiles, threshold=0.0))
        nothing = np.asarray(model.predict(smiles, threshold=1.0 + 1e-9))
    assert everything.sum() == len(smiles), (
        "a threshold of 0 should admit every molecule"
    )
    assert nothing.sum() == 0, "a threshold above 1 should admit none"
