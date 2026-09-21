"""The rank channel is read off the probability, not fetched from the graph again.

On a pooled-reference checkpoint a rank is the training-set ECDF evaluated at the pooled
probability — a table lookup, once that probability exists. The scoring loop computes the
probability for every chunk anyway, so asking ``predict_rank`` for the rank re-ran every
preprocessor and every head to arrive at a number already in hand.

That is not a rounding-error saving. ``rank`` is the ``predict_type`` the Ersilia template
deploys with, and a checkpoint with an applicability domain needs the rank channel even for
a plain ``proba`` request, because the per-sample reliability term is derived from it. So
the duplicate pass was most of the cost of the two most common requests.

Needs the ``fit`` extra: the checkpoint has to be fitted here, because only a checkpoint
carrying a pooled reference exercises the path at all.
"""

import json
import os

import numpy as np
import pytest

from lazyqsar.artifacts.classifier import LazyClassifierArtifact
from lazyqsar.ensemble.channels import _score_chunks
from lazyqsar.registry import get_descriptor_type


@pytest.fixture
def scored(pooled_checkpoint):
    """``(artifact, X)`` for one task of the checkpoint."""
    artifact = LazyClassifierArtifact.load(
        os.path.join(pooled_checkpoint["models"], "alpha", "morgan")
    )
    X = get_descriptor_type("morgan")().transform(pooled_checkpoint["smiles"])
    return artifact, X


@pytest.fixture
def onnx_calls(monkeypatch):
    """Counts every ``InferenceSession.run`` for the duration of a test."""
    import onnxruntime as rt

    count = {"n": 0}
    original = rt.InferenceSession.run

    def counting(self, *args, **kwargs):
        count["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rt.InferenceSession, "run", counting, raising=False)
    return count


def test_a_cli_checkpoint_carries_the_pooled_reference(pooled_checkpoint):
    """Otherwise everything below is testing the fallback, not the shortcut."""
    with open(os.path.join(pooled_checkpoint["models"], "alpha", "metadata.json")) as f:
        meta = json.load(f)
    knots = (meta.get("pooled_ranker") or {}).get("knots")
    assert knots, "no pooled reference in the checkpoint; the shortcut cannot apply"

    artifact = LazyClassifierArtifact.load(
        os.path.join(pooled_checkpoint["models"], "alpha", "morgan")
    )
    assert artifact._pooled_rank_prepared is not None


def test_the_shortcut_gives_the_graph_s_answer_exactly(scored):
    """Equal, not close. The shortcut is the same computation with the graph skipped.

    ``predict_rank`` interpolates the ECDF at ``predict_proba(X)[:, 1]``; the shortcut
    interpolates it at the probability the caller already computed from the same rows. If
    these ever diverge, the ranks LazyQSAR deploys with have changed.
    """
    artifact, X = scored
    from_proba = artifact.rank_from_proba(artifact.predict_proba(X)[:, 1])
    from_graph = artifact.predict_rank(X)[:, 1]
    np.testing.assert_array_equal(from_proba, from_graph)


def test_asking_for_the_rank_channel_costs_no_extra_onnx_runs(scored, onnx_calls):
    """The whole point: the rank channel is now free where it used to double the work."""
    artifact, X = scored

    onnx_calls["n"] = 0
    _score_chunks([X], artifact, None, {"y"})
    proba_only = onnx_calls["n"]

    onnx_calls["n"] = 0
    _score_chunks([X], artifact, None, {"y", "r"})
    with_rank = onnx_calls["n"]

    assert proba_only > 0, "nothing ran; the counter is not wired up"
    assert with_rank == proba_only, (
        f"adding the rank channel cost {with_rank - proba_only} extra ONNX runs on top of "
        f"{proba_only} — it should be read off the probability already computed"
    )


def test_a_checkpoint_without_the_reference_still_ranks(scored, monkeypatch):
    """Pre-v3.5.0 checkpoints have no ECDF, so their rank really does need the graph."""
    artifact, X = scored
    monkeypatch.setattr(artifact, "_pooled_rank_prepared", None)

    assert artifact.rank_from_proba(artifact.predict_proba(X)[:, 1]) is None
    channels = _score_chunks([X], artifact, None, {"y", "r"})
    assert channels.r is not None and channels.r.shape == (len(X),)
