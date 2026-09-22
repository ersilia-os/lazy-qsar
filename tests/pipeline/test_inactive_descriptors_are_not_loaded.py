"""A descriptor the portfolio rejected is not opened, on either Python loader.

Both ``_channels`` implementations have always honoured ``active_descriptors`` when
*scoring*. The loaders did not honour it when *loading*: every descriptor directory on disk
got its featurizer read, its ONNX sessions built and its applicability domain opened, and
the rejected ones were then never used. On a slow-mode checkpoint where the portfolio kept
two of five that is most of the load, and for chemeleon or cddd it means reading a torch
model off disk to throw it away.

The shared inference runner has resolved the active set before opening anything since the
descriptor union moved into ``ensemble.runner``; this is the Python entry point catching
up, so the two agree about what a checkpoint costs as well as what it predicts.
"""

import json
import os

import numpy as np
import pytest
from _helpers.checkpoints import build_checkpoint
from _helpers.smiles import make_smiles

from lazyqsar.qsar import LazyClassifierQSAR


@pytest.mark.parametrize("loader", ["load", "load_raw"])
def test_the_rejected_descriptor_is_never_opened(pruned_checkpoint, loader):
    """Its slot comes back empty, and its featurizer is never asked for a single row."""
    from _helpers.stubs import CountingStub

    CountingStub.reset()
    model = getattr(LazyClassifierQSAR, loader)(pruned_checkpoint["task_dir"])
    assert CountingStub.calls == [], "loading a checkpoint featurized something"

    held = model.artifacts if loader == "load" else model.models
    # Both loaders sort the descriptor directories, so positions are in sorted order --
    # ask the object rather than assuming the order the checkpoint was built in.
    position = model.descriptor_types.index(pruned_checkpoint["inactive"])
    assert held[position] is None, (
        f"{pruned_checkpoint['inactive']} was loaded despite being inactive"
    )
    assert model.descriptors[position] is None
    assert all(i == position or held[i] is not None for i in range(len(held))), (
        "an active descriptor was skipped"
    )


@pytest.mark.parametrize("loader", ["load", "load_raw"])
def test_the_lists_stay_index_aligned(pruned_checkpoint, loader):
    """The skipped slot is kept, not dropped.

    ``_spec_from_attributes`` and both ``_channels`` implementations index the
    per-descriptor lists by full position, so compacting them would silently pair a
    descriptor with another one's AUC.
    """
    model = getattr(LazyClassifierQSAR, loader)(pruned_checkpoint["task_dir"])
    n = len(pruned_checkpoint["descriptors"])
    held = model.artifacts if loader == "load" else model.models
    assert len(held) == n
    assert len(model.descriptors) == n
    assert model.descriptor_types == sorted(pruned_checkpoint["descriptors"])


def test_predictions_are_unchanged_by_not_loading_it(pruned_checkpoint):
    """Skipping the load must not move a number: it was never scored in the first place.

    Checked against the runner, which resolves the active set independently from the same
    metadata — so the two paths agreeing is evidence about the checkpoint, not just about
    one implementation repeating itself.
    """
    from lazyqsar.ensemble.runner import predict_tasks, sources_from_parent

    query = pruned_checkpoint["smiles"][:20]
    wrapper = LazyClassifierQSAR.load(pruned_checkpoint["task_dir"])
    from_wrapper = wrapper.predict_proba(query)[:, 1]

    results = predict_tasks(
        sources_from_parent(pruned_checkpoint["root"]), query, outputs=("proba",)
    )
    from_runner = results[0].values["proba"][:, 1]

    np.testing.assert_allclose(from_wrapper, from_runner, rtol=0, atol=1e-6)


def test_a_mask_that_rejects_everything_still_loads_everything(
    tmp_path, stub_descriptors
):
    """The degenerate case both ``_channels`` implementations fall back on.

    An all-false mask means the fit liked none of them; scoring then uses all of them
    rather than none, so loading none would leave the scoring loop indexing ``None``.
    """
    descriptors = ["morgan", "rdkit"]
    stub_descriptors(*descriptors)
    rng = np.random.default_rng(8)
    smiles = make_smiles(60)
    y = rng.integers(0, 2, len(smiles))
    y[:8] = 1
    y[-8:] = 0

    task_dir = build_checkpoint(str(tmp_path / "m"), "task", descriptors, smiles, y)
    meta_path = os.path.join(task_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["active_descriptors"] = {d: False for d in descriptors}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    model = LazyClassifierQSAR.load(task_dir)
    assert all(a is not None for a in model.artifacts)
    assert model.predict_proba(smiles[:10]).shape == (10, 2)


def test_a_checkpoint_with_no_mask_loads_everything(tmp_path, stub_descriptors):
    """Checkpoints predating ``active_descriptors`` must be unaffected."""
    descriptors = ["morgan", "rdkit"]
    stub_descriptors(*descriptors)
    rng = np.random.default_rng(9)
    smiles = make_smiles(60)
    y = rng.integers(0, 2, len(smiles))
    y[:8] = 1
    y[-8:] = 0

    # `build_checkpoint` writes no `active_descriptors` key at all.
    task_dir = build_checkpoint(str(tmp_path / "m"), "task", descriptors, smiles, y)
    with open(os.path.join(task_dir, "metadata.json")) as f:
        assert "active_descriptors" not in json.load(f)

    model = LazyClassifierQSAR.load(task_dir)
    assert all(a is not None for a in model.artifacts)
