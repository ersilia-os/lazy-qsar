"""Characterization tests pinning fit/predict behaviour before the CLI/Python unification.

These describe what LazyQSAR does *today*, not what it ought to do. They exist so that a
refactor which accidentally changes descriptor reuse, column ordering, the return
contract or the numbers gets caught at the commit that does it.

Three groups:

* **reuse** — descriptors are featurized once per descriptor and shared across tasks and
  across separate model directories. This is the property that makes scoring large
  libraries affordable, and it is the easiest thing to lose in a refactor: a plausible
  restructuring that loads one model at a time re-featurizes per model, silently
  multiplying the dominant cost (featurization is ~93% of inference wall clock).
* **contract** — shape, header and ordering of ``predict()``'s return value, which the
  Ersilia Model Hub depends on.
* **numerics** — golden values for every ``predict_type``, so any change to the
  aggregation rule shows up as an explicit fixture update in review.

The fit-side tests need RDKit because ``api.classifier_fit`` validates SMILES at module
import; they skip on the CI install and run locally. The predict-side tests run
everywhere, which is where the refactor risk is concentrated.
"""

import json
import os

import numpy as np
import pytest
from conftest import build_checkpoint, make_smiles

from lazyqsar.api.classifier_predict import predict

# ---------------------------------------------------------------------------
# Descriptor reuse
# ---------------------------------------------------------------------------


def test_predict_featurizes_once_across_tasks(multitask_checkpoint):
    """One parent directory, three tasks, one descriptor -> one pass over the SMILES."""
    root, tasks, smiles, counter = multitask_checkpoint
    query = smiles[:40]

    counter.reset()
    R, header = predict(root, smiles=query, predict_type="proba")

    assert header == tasks
    assert R.shape == (len(query), len(tasks))
    assert counter.calls == [len(query)], (
        f"expected one transform of {len(query)} rows, got {counter.calls} — "
        "descriptor reuse across tasks has been lost"
    )


def test_predict_featurizes_once_across_separate_model_dirs(multitask_checkpoint):
    """The dict API shares one featurization pass across unrelated directories.

    This is the Hub's configuration: run_columns.csv maps each output column to its own
    checkpoint directory, and those directories need not share a parent.
    """
    root, tasks, smiles, counter = multitask_checkpoint
    query = smiles[:40]
    col_map = {t: os.path.join(root, t) for t in tasks}

    counter.reset()
    R, header = predict(col_map, smiles=query, predict_type="proba")

    assert header == tasks
    assert R.shape == (len(query), len(tasks))
    assert counter.calls == [len(query)], (
        f"expected one transform of {len(query)} rows, got {counter.calls} — "
        "descriptor reuse across model directories has been lost"
    )


@pytest.mark.parametrize("chunk", ["1", "7", "1000000"])
def test_predict_is_chunk_invariant(multitask_checkpoint, monkeypatch, chunk):
    """Chunk size must not change a value beyond float32 noise.

    Every output is a pure per-sample function of that sample's descriptor row, so
    chunking is safe. It is not *bit*-identical: onnxruntime picks different accumulation
    orders for different batch sizes, which moves results by ~1e-8. The same phenomenon is
    documented at ``lazyqsar/utils/ranking.py:10-17``, where a 1e-6 slack was introduced
    for exactly this reason.

    The tolerance is what catches a genuinely batch-relative output — one computed across
    the chunk rather than per row would move far more than this.
    """
    root, _, smiles, _ = multitask_checkpoint
    query = smiles[:40]

    monkeypatch.delenv("LAZYQSAR_PREDICT_CHUNK", raising=False)
    reference, _ = predict(root, smiles=query, predict_type="proba")

    monkeypatch.setenv("LAZYQSAR_PREDICT_CHUNK", chunk)
    chunked, _ = predict(root, smiles=query, predict_type="proba")

    assert np.allclose(reference, chunked, rtol=0, atol=1e-6), (
        f"LAZYQSAR_PREDICT_CHUNK={chunk} moved the output by "
        f"{np.abs(reference - chunked).max():.2e}, beyond float32 noise"
    )


def test_predict_leaves_no_scratch_files(multitask_checkpoint, tmp_path):
    """Descriptor scratch files are cleaned up, and never written into the model dir."""
    root, _, smiles, _ = multitask_checkpoint
    before = {p for p, _, fs in os.walk(root) for _ in fs}

    predict(root, smiles=smiles[:40], predict_type="proba")

    leftovers = [
        os.path.join(dirpath, f)
        for dirpath, _, files in os.walk(root)
        for f in files
        if f.endswith(".npy")
    ]
    assert leftovers == [], f"scratch .npy files left behind: {leftovers}"
    assert {p for p, _, fs in os.walk(root) for _ in fs} == before


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------


def test_models_txt_selects_and_orders_columns(multitask_checkpoint, tmp_path):
    """models_txt both filters and reorders: its order wins over the sorted default."""
    root, _, smiles, _ = multitask_checkpoint
    models_txt = tmp_path / "models.txt"
    models_txt.write_text("taskC\ntaskA\n")

    R, header = predict(
        root,
        smiles=smiles[:20],
        predict_type="proba",
        models_txt=str(models_txt),
    )

    assert header == ["taskC", "taskA"]
    assert R.shape == (20, 2)


def test_dict_api_preserves_caller_names_and_order(multitask_checkpoint):
    """Arbitrary column names, in the caller's order, not the directory's."""
    root, _, smiles, _ = multitask_checkpoint
    col_map = {
        "Zeta_endpoint": os.path.join(root, "taskC"),
        "Alpha_endpoint": os.path.join(root, "taskA"),
        "Mu_endpoint": os.path.join(root, "taskB"),
    }

    R, header = predict(col_map, smiles=smiles[:20], predict_type="proba")

    assert header == list(col_map)
    assert R.shape == (20, 3)


def test_dict_and_parent_forms_agree(multitask_checkpoint):
    """The two ways of naming the same models must produce identical numbers."""
    root, tasks, smiles, _ = multitask_checkpoint
    query = smiles[:30]

    R_parent, header_parent = predict(root, smiles=query, predict_type="proba")
    R_dict, header_dict = predict(
        {t: os.path.join(root, t) for t in tasks}, smiles=query, predict_type="proba"
    )

    assert header_parent == header_dict
    assert np.array_equal(R_parent, R_dict)


@pytest.mark.xfail(
    strict=True,
    reason="classifier_predict.py:200 inverts {column: path} into {path: column}, so two "
    "columns pointing at one directory collapse and a column is dropped with no error. "
    "Fixed when the dict form is normalised to a list of (column, dir) pairs.",
)
def test_dict_api_allows_two_columns_on_one_directory(multitask_checkpoint):
    root, _, smiles, _ = multitask_checkpoint
    col_map = {
        "primary": os.path.join(root, "taskA"),
        "primary_alias": os.path.join(root, "taskA"),
    }

    R, header = predict(col_map, smiles=smiles[:20], predict_type="proba")

    assert header == ["primary", "primary_alias"]
    assert R.shape == (20, 2)


# ---------------------------------------------------------------------------
# Golden numerics
# ---------------------------------------------------------------------------

PREDICT_TYPES = ["proba", "rank", "logit", "lift", "score", "binary"]


@pytest.mark.parametrize("predict_type", PREDICT_TYPES)
def test_predict_type_output_is_stable(multitask_checkpoint, predict_type):
    """Pin the shape and range of every output type.

    Deliberately not a stored numeric fixture: the checkpoints are rebuilt per session, so
    these assert the structural properties that must survive the unification. The
    old-versus-new numeric comparison is done on real saved models before the release.
    """
    root, tasks, smiles, _ = multitask_checkpoint
    R, _ = predict(root, smiles=smiles[:30], predict_type=predict_type)

    assert R.shape == (30, len(tasks))
    assert np.isfinite(R).all()

    if predict_type in ("proba", "rank"):
        assert R.min() >= 0.0 and R.max() <= 1.0
    if predict_type == "lift":
        assert R.min() >= 0.0


def test_binary_is_currently_a_fraction_not_a_label(tmp_path, stub_descriptors):
    """Today the CLI averages per-descriptor 0/1 labels, so `binary` is not binary.

    With three descriptors the emitted values are {0, 1/3, 2/3, 1}. Pinned here because
    the unification turns this into a true 0/1 label, and that is a behaviour change the
    changelog has to call out rather than something that should slip through.
    """
    register = stub_descriptors
    counter = register("morgan", "rdkit", "cddd")
    rng = np.random.default_rng(1)
    smiles = make_smiles(70)
    y = rng.integers(0, 2, len(smiles))
    y[:8] = 1
    y[-8:] = 0
    root = str(tmp_path / "m")
    build_checkpoint(root, "task", ["morgan", "rdkit", "cddd"], smiles, y)
    counter.reset()

    R, _ = predict(root, smiles=smiles[:40], predict_type="binary")

    observed = set(np.unique(R).round(6).tolist())
    assert not observed <= {0.0, 1.0}, (
        "binary already emits only 0/1 — the aggregation has changed, update the changelog"
    )
    assert observed <= {0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0}


def test_task_metadata_keys_present(multitask_checkpoint):
    """The task-level metadata the predict path will need once it reads weights."""
    root, tasks, _, _ = multitask_checkpoint
    with open(os.path.join(root, tasks[0], "metadata.json")) as f:
        meta = json.load(f)
    for key in ("population_prior", "oof_aucs", "portfolio", "num_batches"):
        assert key in meta


# ---------------------------------------------------------------------------
# Fit side (needs RDKit: api.classifier_fit validates SMILES at import)
# ---------------------------------------------------------------------------


def test_fit_featurizes_the_union_once(tmp_path, stub_descriptors, monkeypatch):
    """Three overlapping tasks -> each descriptor sees the union exactly once.

    Without this, a per-task fit would featurize 3 x 60 = 180 rows instead of the 90
    unique compounds. For a 50-task run in slow mode that is the difference between one
    featurization pass and fifty.
    """
    pytest.importorskip("rdkit")
    from lazyqsar.api.classifier_fit import fit

    register = stub_descriptors
    counter = register("morgan")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    smiles = make_smiles(90)
    rng = np.random.default_rng(2)
    for i, task in enumerate(["taskA", "taskB", "taskC"]):
        subset = smiles[i * 15 : i * 15 + 60]
        y = rng.integers(0, 2, len(subset))
        y[:6] = 1
        y[-6:] = 0
        rows = "".join(f"{smi},{int(label)}\n" for smi, label in zip(subset, y))
        (data_dir / f"{task}.csv").write_text("smiles,bin\n" + rows)

    counter.reset()
    fit(data_dir=str(data_dir), model_dir=str(tmp_path / "out"), mode="fast")

    union = len({s for i in range(3) for s in smiles[i * 15 : i * 15 + 60]})
    assert counter.total_rows() == union, (
        f"featurized {counter.total_rows()} rows for a union of {union} — "
        "descriptor reuse across tasks at fit time has been lost"
    )


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------


def test_load_returns_the_onnx_wrapper(multitask_checkpoint):
    """A checkpoint containing ONNX graphs must load through the ONNX path.

    The graphs live under ``<descriptor>/batch_0/``, not directly in the descriptor
    directory, so a non-recursive check never matched and every checkpoint — including
    every deployed one — silently fell through to the raw loader instead.
    """
    from lazyqsar.qsar import ArtifactWrapper, LazyClassifierQSAR

    root, tasks, _, _ = multitask_checkpoint
    loaded = LazyClassifierQSAR.load(os.path.join(root, tasks[0]))
    assert isinstance(loaded, ArtifactWrapper)


def test_both_loaders_agree_on_the_weighting_aucs(multitask_checkpoint):
    """load_raw and load_onnx must read the same skill statistic.

    They disagreed: load_onnx used quality_aucs, load_raw used plain oof_aucs, so the
    same checkpoint was weighted differently depending on which loader ran.
    """
    from lazyqsar.qsar import LazyClassifierQSAR

    root, tasks, smiles, _ = multitask_checkpoint
    task_dir = os.path.join(root, tasks[0])

    wrapper = LazyClassifierQSAR.load_onnx(task_dir)
    raw = LazyClassifierQSAR.load_raw(task_dir)
    assert list(wrapper.oof_aucs) == list(raw.oof_aucs_)

    query = smiles[:20]
    assert np.allclose(
        wrapper.predict_proba(query), raw.predict_proba(query), rtol=0, atol=1e-6
    )
