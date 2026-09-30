"""What may be alive at once: the memory shape of the multi-task paths.

The reuse tests next door pin that work is not *repeated*. These pin that it is not
*accumulated*, which is the other half of what lets a fifty-task run over a million
compounds finish at all:

* ``predict_tasks`` holds one ONNX checkpoint in memory at a time, and one descriptor
  matrix on disk at a time;
* ``classifier_fit.fit`` constructs one descriptor featurizer at a time — and, the
  opposite invariant for the opposite reason, keeps every staged matrix alive until the
  last task has sliced its rows out of it.

None of this shows up in the numbers. Hoisting ``_load_artifact`` out of the inner loop,
or staging all five descriptor matrices before scoring any of them, returns exactly the
same predictions and passes every other test in the suite — which is why these exist.

The instrumentation deliberately never holds a reference to an artifact, only to its
directory. A weakref-based "peak liveness" test would be a stronger statement, but it
would also fail if ``LazyClassifierArtifact`` ever grew ``__slots__``, and it would depend
on the ``gc.collect()`` calls in the runner surviving future tidying — turning a memory
regression into a failure that points somewhere else.
"""

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np
import pytest
from _helpers.smiles import make_smiles

from lazyqsar.ensemble import runner
from lazyqsar.ensemble.runner import predict_tasks, sources_from_parent


@dataclass
class _Trace:
    """What the runner did, in order, with nothing held open on its behalf."""

    events: list = field(default_factory=list)  # (kind, detail)
    disk: list = field(default_factory=list)  # (phase, descriptor, [staged matrices])
    checked_out: list = field(default_factory=list)
    peak_checked_out: int = 0

    @property
    def kinds(self):
        return [kind for kind, _ in self.events]

    def names(self, kind):
        return [detail for k, detail in self.events if k == kind]


@pytest.fixture
def trace(monkeypatch):
    """Record featurize / load / score order without keeping any artifact alive.

    Patches the three names on the runner module, not their definition sites:
    ``predict_tasks`` resolves all of them as module globals, and ``score_chunkwise`` in
    particular is from-imported, so patching ``channels`` would not be seen.
    """
    t = _Trace()
    real_persist = runner.persist_descriptors
    real_load = runner._load_artifact
    real_score = runner.score_chunkwise

    def staged(scratch):
        return sorted(
            f for f in os.listdir(scratch) if f.startswith("X_") and f.endswith(".npy")
        )

    def persist_descriptors(
        featurizer, smiles_list, out_path, chunk_size, progress=None, task_id=None
    ):
        scratch = os.path.dirname(out_path)
        name = os.path.basename(out_path)[len("X_") : -len(".npy")]
        t.disk.append(("enter", name, staged(scratch)))
        t.events.append(("featurize", name))
        real_persist(featurizer, smiles_list, out_path, chunk_size, progress, task_id)
        t.disk.append(("exit", name, staged(scratch)))

    def _load_artifact(directory):
        artifact = real_load(directory)
        t.events.append(("load", directory))
        t.checked_out.append(directory)
        t.peak_checked_out = max(t.peak_checked_out, len(t.checked_out))
        return artifact

    def score_chunkwise(artifact, ad_artifact, x_path, chunk_size, want, logger=None):
        channels = real_score(artifact, ad_artifact, x_path, chunk_size, want, logger)
        t.events.append(("score", t.checked_out.pop()))
        return channels

    monkeypatch.setattr(runner, "persist_descriptors", persist_descriptors)
    monkeypatch.setattr(runner, "_load_artifact", _load_artifact)
    monkeypatch.setattr(runner, "score_chunkwise", score_chunkwise)
    return t


# ---------------------------------------------------------------------------
# One ONNX checkpoint in memory at a time
# ---------------------------------------------------------------------------


def test_only_one_artifact_is_checked_out_at_a_time(multitask_checkpoint, trace):
    """Load, score, release — then the next one. Three tasks, one descriptor.

    Peak concurrent sessions must be a single task's worth however many task directories
    there are. Pre-loading every artifact alongside ``plans`` would be a natural-looking
    tidy-up and would change no number this suite checks.
    """
    root, tasks, smiles, _ = multitask_checkpoint

    predict_tasks(sources_from_parent(root), smiles[:30], outputs=("proba",))

    assert trace.kinds == ["featurize"] + ["load", "score"] * len(tasks), (
        "predict_tasks must release each checkpoint before loading the next; "
        f"order was {trace.kinds}"
    )
    assert trace.peak_checked_out == 1
    assert trace.checked_out == []


def test_artifacts_stay_interleaved_across_descriptors(streaming_checkpoint, trace):
    """The same, on the realistic shape: several tasks over several descriptors.

    Scoring must stay nested inside the descriptor loop. This catches a partial hoist —
    pre-loading the artifacts for one descriptor — and a hoist of the featurization
    itself, both of which the single-descriptor case above cannot see.
    """
    root = streaming_checkpoint["root"]
    tasks = streaming_checkpoint["tasks"]
    descriptors = streaming_checkpoint["descriptors"]
    smiles = streaming_checkpoint["smiles"]

    predict_tasks(sources_from_parent(root), smiles[:30], outputs=("proba",))

    expected = []
    for _ in descriptors:
        expected += ["featurize"] + ["load", "score"] * len(tasks)
    assert trace.kinds == expected, (
        f"expected each descriptor's featurization to be followed by {len(tasks)} "
        f"load/score pairs; order was {trace.kinds}"
    )
    assert trace.peak_checked_out == 1
    # `_descriptor_dirs` sorts and `EnsembleSpec.from_metadata` preserves that order.
    assert trace.names("featurize") == sorted(descriptors)
    assert len(trace.names("load")) == len(tasks) * len(descriptors), (
        "one load per (task, descriptor) pair — no caching, no double-loading"
    )


# ---------------------------------------------------------------------------
# One descriptor matrix on disk at a time
# ---------------------------------------------------------------------------


def test_only_one_descriptor_matrix_is_on_disk_at_a_time(
    streaming_checkpoint, trace, tmp_path
):
    """Descriptor k's matrix is deleted before descriptor k+1 is computed.

    ``test_scratch_is_cleaned_and_never_in_the_model_dir`` only looks at the end state,
    which a "stage all five, then delete all five" refactor would pass — while needing
    five descriptor matrices of disk instead of one. The delete is also wrapped in a bare
    ``except OSError: pass``, so a silently failing one is invisible without this.
    """
    root = streaming_checkpoint["root"]
    descriptors = streaming_checkpoint["descriptors"]
    smiles = streaming_checkpoint["smiles"]
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    predict_tasks(
        sources_from_parent(root),
        smiles[:30],
        outputs=("proba",),
        scratch_dir=str(scratch),
    )

    enters = [(name, files) for phase, name, files in trace.disk if phase == "enter"]
    exits = [(name, files) for phase, name, files in trace.disk if phase == "exit"]

    assert len(enters) == len(descriptors)
    for name, files in enters:
        assert files == [], (
            f"{name} began featurizing while {files} was still staged — the previous "
            "descriptor matrix must be removed before the next one is computed"
        )
    for name, files in exits:
        assert files == [f"X_{name}.npy"]
    assert list(scratch.glob("X_*.npy")) == []


# ---------------------------------------------------------------------------
# Fit side: one featurizer at a time, but every staged matrix kept
# ---------------------------------------------------------------------------


def test_fit_holds_one_featurizer_at_a_time(tmp_path, stub_descriptors, monkeypatch):
    """Construct, featurize, drop — then the next descriptor.

    In slow mode that is five featurizers in production, two of which (Chemeleon, CLAMP)
    carry torch models. Building them all up front reads like a tidy-up and would cost
    several GB.

    The same hook pins the opposite property at the other end: when the last descriptor
    finishes, every staged matrix must still be there, because phase 2 slices each task's
    rows out of all of them. Someone generalising the runner's delete-as-you-go rule to
    fit would break it.

    Three stubbed descriptors, not five: the assertion is that the log reads
    ``construct, persist`` repeated once per descriptor, which any count above one proves
    just as completely. The two extra fits bought no coverage.
    """
    import csv

    from lazyqsar.api import classifier_fit
    from lazyqsar.registry import DESCRIPTORS_MODE

    descriptors = ["cddd", "morgan", "rdkit"]
    monkeypatch.setitem(DESCRIPTORS_MODE, "slow", descriptors)
    stub_descriptors(*descriptors)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    smiles = make_smiles(60)
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, len(smiles))
    y[:8] = 1
    y[-8:] = 0
    with open(data_dir / "taskS.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "bin"])
        writer.writerows([[s, int(v)] for s, v in zip(smiles, y)])

    log = []
    listings = []
    real_get = classifier_fit.get_descriptor_type
    real_persist = classifier_fit.persist_descriptors

    def get_descriptor_type(name):
        base = real_get(name)

        class Observed(base):
            def __init__(self, *args, **kwargs):
                log.append(("construct", name))
                super().__init__(*args, **kwargs)

        return Observed

    def persist_descriptors(
        featurizer, smiles_list, out_path, chunk_size, progress=None, task_id=None
    ):
        log.append(("persist", os.path.basename(out_path)[: -len(".npy")]))
        real_persist(featurizer, smiles_list, out_path, chunk_size, progress, task_id)
        listings.append(sorted(os.listdir(os.path.dirname(out_path))))

    # Both names are from-imported into classifier_fit, so patching the registry or the
    # runner would leave these call sites pointing at the originals.
    monkeypatch.setattr(classifier_fit, "get_descriptor_type", get_descriptor_type)
    monkeypatch.setattr(classifier_fit, "persist_descriptors", persist_descriptors)

    classifier_fit.fit(
        data_dir=str(data_dir), model_dir=str(tmp_path / "models"), mode="slow"
    )

    assert [kind for kind, _ in log] == ["construct", "persist"] * len(descriptors), (
        f"featurizers must be built one at a time; order was {log}"
    )
    assert [name for _, name in log] == [d for d in descriptors for _ in range(2)]
    assert listings[-1] == sorted(f"{d}.npy" for d in descriptors), (
        "every staged descriptor matrix must survive phase 1 — the per-task fit in "
        "phase 2 reads its rows out of all of them"
    )


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class _FakeProgress:
    """Records what a Progress was asked to render, without rendering anything."""

    def __init__(self):
        self.added = []  # (task_id, description, total)
        self.advanced = {}  # task_id -> total advance
        self.removed = []

    def add_task(self, description, total=None):
        task_id = len(self.added)
        self.added.append((task_id, description, total))
        self.advanced[task_id] = 0
        return task_id

    def update(self, task_id, advance=0):
        self.advanced[task_id] += advance

    def remove_task(self, task_id):
        self.removed.append(task_id)

    def start(self):
        pass

    def stop(self):
        pass


def test_scoring_reports_progress(streaming_checkpoint, monkeypatch):
    """Both phases of each descriptor report progress, and finished bars are dropped.

    The pre-refactor path had a ``[<descriptor>] predicting`` bar; the runner initially
    kept only the featurization one, so a large bundle showed a completed bar and then
    nothing while it scored. Removal matters too: one Progress now serves the whole call,
    so without it the display grows two rows per descriptor.
    """
    root = streaming_checkpoint["root"]
    tasks = streaming_checkpoint["tasks"]
    descriptors = streaming_checkpoint["descriptors"]
    smiles = streaming_checkpoint["smiles"]

    fake = _FakeProgress()
    monkeypatch.setattr(runner, "new_progress", lambda: fake)

    predict_tasks(
        sources_from_parent(root), smiles[:30], outputs=("proba",), show_progress=True
    )

    predicting = [t for t in fake.added if "predicting" in t[1]]
    assert len(predicting) == len(descriptors), (
        f"expected one predicting bar per descriptor; added {fake.added}"
    )
    for task_id, _, total in predicting:
        assert total == len(tasks)
        assert fake.advanced[task_id] == len(tasks)

    assert sorted(fake.removed) == sorted(t[0] for t in fake.added), (
        "every bar must be removed when its descriptor finishes, or they pile up"
    )


# ---------------------------------------------------------------------------
# Fit-time union ordering
# ---------------------------------------------------------------------------


def test_fit_union_order_is_reproducible(tmp_path):
    """The union is first-seen order over sorted filenames, not filesystem or hash order.

    This list indexes the staged descriptor matrices, so its order is the layout every
    task then slices rows out of. It used to come from ``os.listdir`` and ``set``, which
    made the same command over the same data produce a differently-ordered — and so not
    bit-reproducible — fit from one process to the next.
    """
    from lazyqsar.api.classifier_fit import read_all_smiles

    smiles = make_smiles(6)
    # Written in an order that does not match the sorted one, so a filesystem that hands
    # back creation order would give a different answer from a sorted read.
    (tmp_path / "zebra.csv").write_text(
        "smiles,bin\n" + "".join(f"{s},1\n" for s in smiles[3:])
    )
    (tmp_path / "alpha.csv").write_text(
        "smiles,bin\n" + "".join(f"{s},0\n" for s in smiles[:4])
    )
    (tmp_path / "notes.txt").write_text("ignored")

    # alpha.csv first (sorted), then zebra.csv's compounds that alpha did not already have.
    assert read_all_smiles(str(tmp_path)) == smiles[:4] + smiles[4:]
    assert read_all_smiles(str(tmp_path)) == read_all_smiles(str(tmp_path))


# ---------------------------------------------------------------------------
# Blocking the input
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def forced_block(n_tasks, rows, chunk_size):
    """Make `predict` use blocks of `rows` molecules, by moving the budget.

    Sets the budget rather than patching `_block_size`, so the derivation under test is
    still the one that runs — a test that stubbed the function out would not notice it
    returning something that is not a whole number of chunks.
    """
    from lazyqsar.api import classifier_predict as cp

    per_row = (
        4 * cp._WIDEST_DESCRIPTOR
        + 4 * n_tasks * cp._MAX_DESCRIPTORS * cp._MAX_CHANNELS
        + n_tasks * (16 + 16 * cp._MAX_DESCRIPTORS)
    )
    saved_budget = cp._BLOCK_BUDGET_BYTES
    saved_chunk = os.environ.get("LAZYQSAR_PREDICT_CHUNK")
    cp._BLOCK_BUDGET_BYTES = per_row * rows
    # `predict` reads the chunk size from the environment, and the block is rounded down to
    # a multiple of it -- so without this the budget is quietly overridden by the 1000-row
    # default and every "blocked" run is a single block. Which is how this helper was first
    # written, and what `test_each_block_releases_its_working_set` caught.
    os.environ["LAZYQSAR_PREDICT_CHUNK"] = str(chunk_size)
    try:
        assert cp._block_size(n_tasks, chunk_size) == rows, (
            "budget did not land on `rows`"
        )
        yield
    finally:
        cp._BLOCK_BUDGET_BYTES = saved_budget
        if saved_chunk is None:
            os.environ.pop("LAZYQSAR_PREDICT_CHUNK", None)
        else:
            os.environ["LAZYQSAR_PREDICT_CHUNK"] = saved_chunk


# One per distinct channel set plus the dtype case: `proba` needs {y}, `rank` {y, r},
# `score` {y, s}, and `binary` is the one whose dtype depends on whether a row was blanked.
# `logit` and `lift` are monotone transforms of `proba` off the same channel, so they would
# re-test the same path at the cost of a third of this file's runtime.
@pytest.mark.parametrize("predict_type", ["proba", "rank", "score", "binary"])
def test_blocking_the_input_is_bit_identical(streaming_checkpoint, predict_type):
    """Scoring in blocks must return exactly the bits that scoring in one pass returns.

    Blocking is what bounds memory — without it peak RAM scales with molecules times
    endpoints and no chunk size brings it down. It is only safe because a block is a whole
    number of chunks: featurization and scoring both step in chunks, and onnxruntime picks
    kernels by batch size, so a block boundary that split a chunk would move the result by
    ~1e-7. `assert_array_equal` rather than `allclose` is the entire point.

    The dtype is part of it: `binary` is integer labels until some row carries a NaN, and a
    blocked run must agree with an unblocked one about which it is.
    """
    from lazyqsar.api.classifier_predict import predict

    root = streaming_checkpoint["root"]
    n_tasks = len(streaming_checkpoint["tasks"])
    query = (streaming_checkpoint["smiles"] * 3)[:150]

    with forced_block(n_tasks, 200, chunk_size=50):  # one block: 150 rows fit
        whole, header = predict(model_dir=root, smiles=query, predict_type=predict_type)

    for rows in (50, 100):  # 3 blocks, then 2 with a short tail
        with forced_block(n_tasks, rows, chunk_size=50):
            blocked, blocked_header = predict(
                model_dir=root, smiles=query, predict_type=predict_type
            )
        assert blocked_header == header
        assert blocked.dtype == whole.dtype, (
            f"{rows}-row blocks changed the dtype of {predict_type}: "
            f"{blocked.dtype} vs {whole.dtype}"
        )
        np.testing.assert_array_equal(
            blocked,
            whole,
            err_msg=f"{predict_type} differs when scored in {rows}-row blocks",
        )


def test_each_block_releases_its_working_set(streaming_checkpoint, trace):
    """A block's channels and combined results must not survive into the next block.

    The observable proxy: every block runs the whole featurize/load/score sequence again
    rather than one pass over the full input. If blocking ever collapsed back into a single
    `predict_tasks` call the event log would show one sequence instead of three, and peak
    memory would be back where it started.
    """
    from lazyqsar.api.classifier_predict import predict

    root = streaming_checkpoint["root"]
    tasks = streaming_checkpoint["tasks"]
    descriptors = streaming_checkpoint["descriptors"]
    query = (streaming_checkpoint["smiles"] * 3)[:150]

    with forced_block(len(tasks), 50, chunk_size=50):
        predict(model_dir=root, smiles=query, predict_type="proba")

    per_block = (["featurize"] + ["load", "score"] * len(tasks)) * len(descriptors)
    assert trace.kinds == per_block * 3, (
        "expected three blocks, each running the full per-descriptor sequence"
    )
    assert trace.peak_checked_out == 1
