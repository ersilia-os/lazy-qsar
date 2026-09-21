"""The shared inference runner: reuse, chunking and hygiene.

The runner is not wired into any entry point yet, so these tests drive it directly. They
cover the properties the CLI and Python paths will inherit from it, and the ones a
refactor could silently drop.

Channel selection and source resolution are pure functions that never open a model, so
they are tested in ``tests/unit/test_channels_selection.py`` and
``tests/unit/test_runner_sources.py`` instead -- on the base install, without paying for
the ``fit`` extra or a checkpoint build. Do not re-add them here.
"""

import json
import os

import numpy as np
import pytest
from _helpers.checkpoints import build_checkpoint
from _helpers.smiles import make_smiles

from lazyqsar.ensemble.combine import OUTPUT_NAMES
from lazyqsar.ensemble.runner import (
    TaskSource,
    predict_tasks,
    sources_from_mapping,
    sources_from_parent,
)

# ---------------------------------------------------------------------------
# Descriptor reuse
# ---------------------------------------------------------------------------


def test_featurizes_once_across_tasks(multitask_checkpoint):
    root, tasks, smiles, counter = multitask_checkpoint
    query = smiles[:40]
    counter.reset()

    results = predict_tasks(sources_from_parent(root), query, outputs=("proba",))

    assert len(results) == len(tasks)
    assert counter.calls == [len(query)], (
        f"expected one transform of {len(query)} rows, got {counter.calls}"
    )


def test_featurizes_once_for_all_requested_outputs(multitask_checkpoint):
    """Every output comes from one pass -- the property the CLI lacks today."""
    root, _, smiles, counter = multitask_checkpoint
    query = smiles[:40]
    counter.reset()

    results = predict_tasks(sources_from_parent(root), query, outputs=OUTPUT_NAMES)

    assert set(results[0].values) == set(OUTPUT_NAMES)
    assert counter.calls == [len(query)], (
        f"asking for {len(OUTPUT_NAMES)} outputs featurized {counter.calls}"
    )


def test_featurizes_once_across_separate_directories(multitask_checkpoint):
    root, tasks, smiles, counter = multitask_checkpoint
    query = smiles[:40]
    col_map = {t: os.path.join(root, t) for t in tasks}
    counter.reset()

    predict_tasks(sources_from_mapping(col_map), query, outputs=("proba",))

    assert counter.calls == [len(query)]


def test_duplicate_directory_yields_two_identical_results(multitask_checkpoint):
    root, _, smiles, counter = multitask_checkpoint
    col_map = {
        "primary": os.path.join(root, "taskA"),
        "primary_alias": os.path.join(root, "taskA"),
    }
    counter.reset()

    results = predict_tasks(
        sources_from_mapping(col_map), smiles[:20], outputs=("proba",)
    )

    assert len(results) == 2
    assert np.array_equal(results[0].values["proba"], results[1].values["proba"])
    assert counter.calls == [20], "the shared directory should still featurize once"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 10**6])
def test_chunk_size_does_not_change_results(multitask_checkpoint, chunk):
    root, _, smiles, _ = multitask_checkpoint
    query = smiles[:40]
    sources = sources_from_parent(root)

    reference = predict_tasks(sources, query, outputs=OUTPUT_NAMES, chunk_size=997)
    chunked = predict_tasks(sources, query, outputs=OUTPUT_NAMES, chunk_size=chunk)

    for name in OUTPUT_NAMES:
        a = reference[0].values[name]
        b = chunked[0].values[name]
        assert np.allclose(a, b, rtol=0, atol=1e-6), (
            f"chunk_size={chunk} moved {name} by {np.abs(a - b).max():.2e}"
        )


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def test_scratch_is_cleaned_and_never_in_the_model_dir(multitask_checkpoint):
    root, _, smiles, _ = multitask_checkpoint
    predict_tasks(sources_from_parent(root), smiles[:20], outputs=("proba",))
    leftovers = [
        os.path.join(d, f)
        for d, _, files in os.walk(root)
        for f in files
        if f.endswith(".npy")
    ]
    assert leftovers == []


def test_supplied_scratch_dir_is_left_for_the_caller(multitask_checkpoint, tmp_path):
    """A caller-owned scratch directory is not deleted, but is emptied of matrices."""
    root, _, smiles, _ = multitask_checkpoint
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    predict_tasks(
        sources_from_parent(root),
        smiles[:20],
        outputs=("proba",),
        scratch_dir=str(scratch),
    )

    assert scratch.is_dir()
    assert list(scratch.glob("*.npy")) == []


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_tasks(
            [TaskSource("x", str(tmp_path / "nope"))], ["CCO"], outputs=("proba",)
        )


def test_no_sources_raises(multitask_checkpoint):
    with pytest.raises(ValueError, match="No model sources"):
        predict_tasks([], ["CCO"], outputs=("proba",))


# ---------------------------------------------------------------------------
# Agreement with the existing prediction path
# ---------------------------------------------------------------------------


def test_runner_matches_todays_flat_average_when_weights_are_uniform(
    multitask_checkpoint,
):
    """A checkpoint with no AD gets uniform weights, where the two rules coincide.

    ``rank`` and ``score`` are weighted means of per-descriptor values, so with uniform
    weights they equal the arithmetic mean the CLI computes today. ``proba`` does not --
    the runner mixes in logit space -- which is the documented behaviour change, so it is
    deliberately not asserted here.
    """
    from lazyqsar.api.classifier_predict import predict

    root, _, smiles, _ = multitask_checkpoint
    query = smiles[:30]

    legacy, header = predict(root, smiles=query, predict_type="rank")
    new = predict_tasks(sources_from_parent(root), query, outputs=("rank",))

    for col, result in enumerate(new):
        assert np.allclose(
            legacy[:, col], result.values["rank"][:, 1], rtol=0, atol=1e-6
        ), f"rank diverged for column {header[col]}"


def test_runner_matches_artifactwrapper_on_a_single_model(multitask_checkpoint):
    """One source must reproduce what LazyClassifierQSAR.load() already returns."""
    from lazyqsar.qsar import LazyClassifierQSAR

    root, tasks, smiles, _ = multitask_checkpoint
    query = smiles[:25]
    task_dir = os.path.join(root, tasks[0])

    wrapper = LazyClassifierQSAR.load(task_dir)
    expected = {
        "proba": wrapper.predict_proba(query),
        "rank": wrapper.predict_rank(query),
        "score": wrapper.predict_score(query),
        "logit": wrapper.predict_logit(query),
        "lift": wrapper.predict_lift(query),
    }

    got = predict_tasks(
        [TaskSource(tasks[0], task_dir)], query, outputs=tuple(expected)
    )[0]

    for name, ref in expected.items():
        assert np.allclose(ref, got.values[name], rtol=0, atol=1e-6), (
            f"{name} diverged by {np.abs(ref - got.values[name]).max():.2e}"
        )


def test_inactive_descriptors_are_skipped_entirely(tmp_path, stub_descriptors):
    """A descriptor marked inactive is neither scored nor featurized.

    Today's CLI path enumerates directories and ignores the mask, so it pays to featurize
    a descriptor the model will not use -- which is both wrong and slow.
    """
    register = stub_descriptors
    counter = register("morgan", "rdkit")
    rng = np.random.default_rng(3)
    smiles = make_smiles(70)
    y = rng.integers(0, 2, len(smiles))
    y[:8] = 1
    y[-8:] = 0
    root = str(tmp_path / "m")
    task_dir = build_checkpoint(root, "task", ["morgan", "rdkit"], smiles, y)

    meta_path = os.path.join(task_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["active_descriptors"] = {"morgan": True, "rdkit": False}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    counter.reset()
    results = predict_tasks(sources_from_parent(root), smiles[:20], outputs=("proba",))

    assert len(results) == 1
    assert counter.calls == [20], (
        f"expected only the active descriptor to be featurized, got {counter.calls}"
    )
    assert results[0].weights.shape[1] == 1


def test_artifact_wrapper_never_featurizes_the_whole_list(
    multitask_checkpoint, monkeypatch
):
    """The Python API must stream too, not transform everything up front.

    ``ArtifactWrapper`` used to call ``transform(smiles_list)`` on the full input, so
    scoring a large library from Python needed the entire descriptor matrix in memory --
    roughly 8 GB for a million compounds against a 2048-dimensional descriptor. The CLI
    path had always chunked; this is the Python path catching up.
    """
    from lazyqsar.qsar import LazyClassifierQSAR

    root, tasks, smiles, counter = multitask_checkpoint
    query = smiles[:40]
    monkeypatch.setenv("LAZYQSAR_PREDICT_CHUNK", "9")

    counter.reset()
    LazyClassifierQSAR.load(os.path.join(root, tasks[0])).predict_proba(query)

    assert counter.calls, "no featurization happened at all"
    assert max(counter.calls) <= 9, (
        f"largest transform call was {max(counter.calls)} rows for chunk size 9 — "
        "the whole list is still being featurized at once"
    )
    assert sum(counter.calls) == len(query)


def test_artifact_wrapper_chunking_is_inert_at_realistic_sizes(
    multitask_checkpoint, monkeypatch
):
    """Chunk size must not change results for chunk sizes anyone would actually use.

    Not asserted for pathologically small chunks. onnxruntime selects different kernels
    and accumulation orders by batch size, moving raw head outputs by ~1e-7, and the
    applicability-domain weighting amplifies that: it feeds ranks through a 20-knot
    interpolation whose local slope can be steep, then mixes in logit space. With a chunk
    of 7 the end-to-end drift reaches ~1e-4. At the default of 1000, and at 500, measured
    output on real checkpoints is bit-identical to scoring in one pass.
    """
    from lazyqsar.qsar import LazyClassifierQSAR

    root, tasks, smiles, _ = multitask_checkpoint
    query = smiles[:60]
    task_dir = os.path.join(root, tasks[0])

    monkeypatch.setenv("LAZYQSAR_PREDICT_CHUNK", "1000000")
    reference = LazyClassifierQSAR.load(task_dir).predict_proba(query)

    for chunk in ("1000", "500", "60"):
        monkeypatch.setenv("LAZYQSAR_PREDICT_CHUNK", chunk)
        got = LazyClassifierQSAR.load(task_dir).predict_proba(query)
        assert np.array_equal(reference, got), (
            f"chunk={chunk} changed the result by {np.abs(reference - got).max():.2e}"
        )
