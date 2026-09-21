"""The rank channel is read off the probability, not fetched from the graph again.

On a pooled-reference checkpoint a rank is the training-set ECDF evaluated at the pooled
probability — a table lookup, once that probability exists. The scoring loop computes the
probability for every chunk anyway, so asking ``_oof_percentile`` for it re-ran every
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
from _helpers.smiles import make_smiles

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
    assert artifact._oof_percentile_prepared is not None


def test_the_shortcut_gives_the_graph_s_answer_exactly(scored):
    """Equal, not close. The shortcut is the same computation with the graph skipped.

    ``_oof_percentile`` interpolates the ECDF at ``predict_proba(X)[:, 1]``; the shortcut
    interpolates it at the probability the caller already computed from the same rows. If
    these ever diverge, the ranks LazyQSAR deploys with have changed.
    """
    artifact, X = scored
    from_proba = artifact._oof_percentile_from_proba(artifact.predict_proba(X)[:, 1])
    from_graph = artifact._oof_percentile(X)[:, 1]
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
    monkeypatch.setattr(artifact, "_oof_percentile_prepared", None)

    assert artifact._oof_percentile_from_proba(artifact.predict_proba(X)[:, 1]) is None
    channels = _score_chunks([X], artifact, None, {"y", "r"})
    assert channels.r is not None and channels.r.shape == (len(X),)


# ---------------------------------------------------------------------------
# `score` costs what every other output costs
# ---------------------------------------------------------------------------


def test_score_no_longer_needs_the_raw_channel(pooled_checkpoint):
    """With a pooled score map, `score` asks for the same channels `proba` does.

    That is where the saving comes from: the raw channel was the last thing making any
    request run every preprocessor and every head twice. `required_channels` deciding this
    wrongly shows up as a slower run, not a wrong number — the fallback still produces the
    right answer — so nothing else would catch it.
    """
    from lazyqsar.ensemble.channels import required_channels

    assert required_channels(("score",), has_ad=False, has_scorer=True) == {"y"}
    assert required_channels(("score",), has_ad=True, has_scorer=True) == {
        "y",
        "r",
        "a",
    }
    # Without a map, `score` must still be pooled from the raw values.
    assert required_channels(("score",), has_ad=False, has_scorer=False) == {"y", "s"}


def test_scoring_costs_the_same_as_proba_end_to_end(pooled_checkpoint, onnx_calls):
    """The whole point, measured the way the rank shortcut is measured."""
    from lazyqsar.api.classifier_predict import predict

    models = pooled_checkpoint["models"]
    query = pooled_checkpoint["smiles"][:40]

    onnx_calls["n"] = 0
    predict(model_dir=models, smiles=query, predict_type="proba")
    proba_calls = onnx_calls["n"]

    onnx_calls["n"] = 0
    predict(model_dir=models, smiles=query, predict_type="score")
    score_calls = onnx_calls["n"]

    assert proba_calls > 0, "nothing ran; the counter is not wired up"
    assert score_calls == proba_calls, (
        f"score cost {score_calls} ONNX runs against proba's {proba_calls} — it should "
        "be read off the pooled probability, not fetched from the graphs again"
    )


def test_a_checkpoint_without_the_map_keeps_the_old_score(pooled_checkpoint, tmp_path):
    """Old checkpoints must be untouched: same pooling, same raw channel, same numbers."""
    import json
    import shutil

    from lazyqsar.api.classifier_predict import predict

    legacy = tmp_path / "legacy"
    shutil.copytree(pooled_checkpoint["models"], legacy)
    meta_path = legacy / "alpha" / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.pop("pooled_scorer", None) is not None, "fixture had no map to remove"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    query = pooled_checkpoint["smiles"][:40]
    with_map, _ = predict(
        model_dir=pooled_checkpoint["models"], smiles=query, predict_type="score"
    )
    without, _ = predict(model_dir=str(legacy), smiles=query, predict_type="score")

    assert without.shape == with_map.shape
    # The point of the change: they are different numbers, and the old one is preserved.
    assert not np.allclose(without, with_map, atol=1e-6), (
        "removing the map changed nothing; the fallback is not being taken"
    )


def test_score_agrees_with_proba_across_several_descriptors(tmp_path, stub_descriptors):
    """The multi-descriptor case, which is the only one where this can actually fail.

    With a single descriptor the weights collapse to one column and the pre-3.5.0 fallback
    degenerates to ``score == proba``, so a fast-mode fixture cannot tell a working pooled
    map from a broken one. Slow mode over five stubbed descriptors is where independent raw
    pooling genuinely reorders, and therefore where the map has to do its job.
    """
    import contextlib
    import io

    from lazyqsar.api.classifier_fit import fit
    from lazyqsar.api.classifier_predict import predict
    from lazyqsar.registry import DESCRIPTORS_MODE

    stub_descriptors(*DESCRIPTORS_MODE["slow"])
    smiles = make_smiles(80)
    rng = np.random.default_rng(17)
    y = rng.integers(0, 2, len(smiles))
    y[:12] = 1
    y[-12:] = 0
    data = tmp_path / "data"
    data.mkdir()
    (data / "alpha.csv").write_text(
        "smiles,bin\n" + "".join(f"{s},{int(v)}\n" for s, v in zip(smiles, y))
    )
    models = tmp_path / "models"
    with contextlib.redirect_stdout(io.StringIO()):
        fit(data_dir=str(data), model_dir=str(models), mode="slow")

    out = {}
    for t in ("proba", "logit", "rank", "score"):
        values, _ = predict(model_dir=str(models), smiles=smiles, predict_type=t)
        out[t] = values[:, 0]

    def flipped(a, b):
        sa = np.sign(np.subtract.outer(a, a))
        sb = np.sign(np.subtract.outer(b, b))
        return int(((sa * sb) < 0).sum() // 2)

    for other in ("logit", "rank", "score"):
        assert flipped(out["proba"], out[other]) == 0, (
            f"{other} disagrees with proba about "
            f"{flipped(out['proba'], out[other])} pairs across five descriptors"
        )
