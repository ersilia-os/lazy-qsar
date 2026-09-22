"""Does the shipped checkpoint rank the same molecules as the model it came from?

``rank`` is the default output the Ersilia template asks for, and the metric
``chembl-antimicrobial-models`` computes its headline AUROC, AUPRC and BEDROC from. Before
v3.5.0 it was also the output that survived the ONNX export least well: measured on this
data, ``proba`` agreed with its export to 9e-08 while ``rank`` moved by up to 0.09 on 500
of 665 molecules. The cause was not the export. ``rank`` was a weighted mean of
percentiles, and an ECDF is steep wherever the training scores bunch up, so it amplified a
difference ``proba`` barely registered -- and, being a different pooling from ``proba``'s,
it could order two molecules the other way round entirely.

The contract now is one sentence: ``rank`` is the training-set percentile of ``proba``.
Everything below is that sentence, checked from a different angle each time. The algebra is
asserted; the statistics are reported, because a Spearman threshold on 665 molecules is a
weaker statement than "the ordering is identical" and would only invite tuning.

Real molecules on purpose. Stub descriptors give a signal-free model whose scores bunch up
in ways real chemistry does not, which is the regime this test is least able to judge. But
*how many* descriptors survive the portfolio is data- and version-dependent -- scikit-learn
1.6 keeps rdkit on this data and 1.9 prunes it -- so the multi-descriptor pooling that
motivated the change is pinned deterministically in
``tests/unit/test_pooled_rank_reference.py`` instead. What this file is for is the
end-to-end plumbing on real chemistry: that the reference is built during a genuine fit,
survives every load route, and holds on molecules the model never trained on.
"""

import contextlib
import io
import math
import os

import numpy as np
import pytest

from _helpers.smiles import load_imbalanced_dataset, load_reference_dataset

from lazyqsar.qsar import LazyClassifierQSAR

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

CUTS = (0.01, 0.05, 0.10)

# `load_raw` and `load_onnx` are different numeric paths and differ by ~3e-08 on `proba`.
# `rank` is an ECDF of `proba`, and an ECDF's slope is the reciprocal of the local density
# of training scores, so it multiplies that gap by ~100x here. Measured max across routes
# is 4.1e-06; 1e-04 leaves a margin without being loose enough to hide a real change.
# Whether a route *kept* the reference at all is checked structurally below, not by
# tolerance -- on a single-descriptor model the two poolings coincide to ~4e-06, so no
# tolerance on this value could tell them apart.
ROUTE_RANK_ATOL = 1e-4
ROUTE_PROBA_ATOL = 1e-6

# 120 of the 233 available rows. The fixture is the only slow thing in this file and the
# assertions are about agreement between two runtimes, not about model quality, so the
# extra rows buy nothing. Matches what test_invalid_smiles_end_to_end.py trains on.
N_TRAIN = 120


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """Fit on part of the reference set, score a wider query set through model and export.

    The query set is deliberately larger than the training set, so a 1% cut is a handful
    of molecules rather than two and most of what is scored was never trained on.
    """
    train_smiles, train_y = load_reference_dataset()
    train_smiles = list(train_smiles)[:N_TRAIN]
    train_y = np.asarray(train_y[:N_TRAIN], dtype=int)
    extra_smiles, _ = load_imbalanced_dataset()
    query = list(load_reference_dataset()[0]) + list(extra_smiles)

    model = LazyClassifierQSAR(mode="fast")
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(smiles_list=train_smiles, y=train_y)
        task_dir = os.path.join(str(tmp_path_factory.mktemp("rank_parity")), "task")
        model.save(task_dir)
        onnx = LazyClassifierQSAR.load(task_dir)

        out = {
            tag: {
                channel: getattr(obj, f"predict_{channel}")(query)[:, 1]
                for channel in ("proba", "rank")
            }
            for tag, obj in (("fit", model), ("onnx", onnx))
        }
    out["task_dir"] = task_dir
    out["root"] = os.path.dirname(task_dir)
    out["query"] = query
    out["n"] = len(query)
    out["descriptors"] = list(model.descriptor_types)
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def _pairs_that_flip(a, b):
    """Boolean matrix of pairs (i, j) that *a* and *b* strictly disagree about ordering."""
    sign_a = np.sign(np.subtract.outer(a, a))
    sign_b = np.sign(np.subtract.outer(b, b))
    return (sign_a * sign_b) < 0


def _top_k(values, k):
    return set(np.argsort(-values, kind="stable")[:k].tolist())


def _churn_profile(a, b):
    """Molecules that enter or leave the top cut, per cut. Reported, never asserted on."""
    n = len(a)
    rows = []
    for cut in CUTS:
        k = max(5, math.ceil(cut * n))
        moved = len(_top_k(a, k) ^ _top_k(b, k)) // 2
        rows.append(f"top {cut:.0%} (k={k}): {moved} swapped")
    return "; ".join(rows)


@pytest.mark.parametrize("runtime", ["fit", "onnx"])
def test_rank_orders_molecules_exactly_as_proba_does(scored, runtime):
    """The contract itself, inside each runtime separately.

    This is what makes a rank-based AUROC equal to a probability-based one, and top-k by
    rank the same molecules as top-k by probability.
    """
    proba, rank = scored[runtime]["proba"], scored[runtime]["rank"]
    order = np.argsort(proba, kind="stable")
    assert np.all(np.diff(rank[order]) >= -1e-12), (
        f"{runtime}: rank disagrees with proba about the order of two molecules"
    )


def test_the_export_reorders_nothing_that_proba_agrees_about(scored):
    """The regression this file exists to prevent, stated as algebra rather than a threshold.

    Because `rank` is a monotone function of `proba` inside each runtime, a strict order
    flip in `rank` implies one in `proba`. So rank cannot disagree about a pair that proba
    agrees about -- exactly, not approximately. The one thing rank may add is a *tie*, for
    two molecules whose probabilities differ by less than the local knot spacing, which is
    why this is a pair test and not a Spearman comparison.
    """
    fit_rank, onnx_rank = scored["fit"]["rank"], scored["onnx"]["rank"]
    fit_proba, onnx_proba = scored["fit"]["proba"], scored["onnx"]["proba"]
    new_flips = _pairs_that_flip(fit_rank, onnx_rank) & ~_pairs_that_flip(
        fit_proba, onnx_proba
    )
    assert not new_flips.any(), (
        f"rank reorders {new_flips.sum() // 2} pair(s) proba agrees about "
        f"(spearman rank {_spearman(fit_rank, onnx_rank):.9f}, "
        f"proba {_spearman(fit_proba, onnx_proba):.9f}); "
        f"{_churn_profile(fit_rank, onnx_rank)}"
    )


def test_the_top_of_the_screen_is_the_same_set_of_molecules(scored):
    """What a screening campaign actually acts on."""
    fit_rank, onnx_rank = scored["fit"]["rank"], scored["onnx"]["rank"]
    k = max(5, math.ceil(0.01 * scored["n"]))
    assert _top_k(fit_rank, k) == _top_k(onnx_rank, k), (
        f"top 1% differs between the model and its export; "
        f"{_churn_profile(fit_rank, onnx_rank)}"
    )


def test_the_pooled_reference_is_uniform_on_the_training_set(scored):
    """The one check that can catch a reference calibrated against the wrong distribution.

    A reference built with the wrong weights, or with the prior correction applied in the
    wrong place, still yields a rank that is monotone in proba and still lies in [0, 1] --
    every other test here passes. What it does not do is spread the training molecules
    evenly, because the knots would no longer be that distribution's own quantiles.
    """
    import json

    with open(os.path.join(scored["task_dir"], "metadata.json")) as f:
        knots = np.asarray(json.load(f)["pooled_ranker"]["knots"], dtype=float)

    from lazyqsar.utils.ranking import rank_from_knots

    self_ranks = rank_from_knots(knots, knots)
    assert abs(self_ranks.mean() - 0.5) < 0.05, f"mean {self_ranks.mean():.3f}"
    counts = np.histogram(self_ranks, bins=10, range=(0.0, 1.0))[0]
    expected = len(knots) / 10
    assert counts.min() > 0.4 * expected, f"deciles are lumpy: {counts.tolist()}"


def test_the_checkpoint_carries_the_reference_and_a_matching_cutoff(scored):
    """Without the stored reference a loaded model silently reverts to the old rank."""
    import json

    from lazyqsar.utils.ranking import prepare_knots, rank_from_reference

    with open(os.path.join(scored["task_dir"], "metadata.json")) as f:
        meta = json.load(f)

    knots = meta["pooled_ranker"]["knots"]
    assert len(knots) > 0
    assert knots == sorted(knots), "knots must be stored ascending"
    # decision_cutoff_rank is reported, never thresholded on, but it should still be the
    # learned probability cutoff expressed in the units `rank` now uses -- which means
    # through the tail anchors as well, not just the reference knots. Recomputing it
    # without them is how a checkpoint ends up disagreeing with its own scale.
    block = meta["pooled_ranker"]
    anchors = (
        block.get("anchor_low") if block.get("anchor_low_used") else None,
        block.get("anchor_high") if block.get("anchor_high_used") else None,
    )
    expected = float(
        rank_from_reference(
            meta["decision_cutoff_proba"],
            prepared=prepare_knots(np.asarray(knots)),
            anchors=anchors if any(a is not None for a in anchors) else None,
        )
    )
    assert meta["decision_cutoff_rank"] == pytest.approx(expected)
    assert meta["pooled_ranker"]["source"] == "reference_library"
    assert meta["pooled_ranker"]["descriptors"]


def test_every_load_route_carries_the_reference(scored):
    """Structural, because no tolerance on the values could tell you this.

    Four places parse the task metadata -- `EnsembleSpec.from_metadata`, `load_raw`,
    `load_onnx`, and the runner through the first of them -- so a route can silently keep
    the pre-v3.5.0 pooling by simply not reading the new key. On a single-descriptor model
    the two poolings agree to ~4e-06, far inside any sane tolerance, so this has to be
    checked by looking for the reference rather than by comparing numbers.
    """
    probe = scored["query"][:5]
    with contextlib.redirect_stdout(io.StringIO()):
        specs = {
            name: getattr(LazyClassifierQSAR, name)(scored["task_dir"])._channels(
                probe
            )[-1]
            for name in ("load", "load_onnx", "load_raw")
        }
    for name, spec in specs.items():
        assert spec.pooled_rank_knots is not None, (
            f"{name} did not read the pooled reference, so it is still using the "
            "pre-v3.5.0 rank"
        )


def test_every_load_route_agrees_on_rank(scored):
    """And the values agree, to a tolerance the ECDF's slope explains.

    `load_raw` is a different numeric path from `load_onnx` and differs from it by ~3e-08
    on `proba`; ranking amplifies that by the reciprocal of the local training-score
    density. The tolerance is sized for that, not tuned to pass.
    """
    from lazyqsar.api.classifier_predict import predict

    expected = scored["onnx"]["rank"]
    with contextlib.redirect_stdout(io.StringIO()):
        for name in ("load_onnx", "load_raw"):
            obj = getattr(LazyClassifierQSAR, name)(scored["task_dir"])
            np.testing.assert_allclose(
                obj.predict_proba(scored["query"])[:, 1],
                scored["onnx"]["proba"],
                atol=ROUTE_PROBA_ATOL,
                err_msg=f"{name} disagrees about proba",
            )
            np.testing.assert_allclose(
                obj.predict_rank(scored["query"])[:, 1],
                expected,
                atol=ROUTE_RANK_ATOL,
                err_msg=f"{name} disagrees about rank",
            )
        # The runner -- the path the Ersilia template actually takes.
        runner_rank, _ = predict(
            model_dir=scored["root"], smiles=scored["query"], predict_type="rank"
        )
    np.testing.assert_allclose(
        np.asarray(runner_rank).ravel(),
        expected,
        atol=ROUTE_RANK_ATOL,
        err_msg="the runner disagrees about rank",
    )


def test_a_checkpoint_without_the_reference_refuses_to_rank(scored, tmp_path):
    """Strip the reference and `rank` must refuse, while everything else keeps working.

    Every checkpoint published before v3.6 is in this state. Its `pooled_ranker` holds
    out-of-fold knots, which are indistinguishable from library knots once read, so
    reporting them would answer "beats 99% of drug-like space" with a training-set
    percentile. The other five outputs did not change meaning and are not withheld.
    """
    import json
    import shutil

    legacy_dir = str(tmp_path / "legacy")
    shutil.copytree(scored["task_dir"], legacy_dir)
    meta_path = os.path.join(legacy_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.pop("pooled_ranker", None) is not None, "fixture should have had one"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with contextlib.redirect_stdout(io.StringIO()):
        legacy = LazyClassifierQSAR.load(legacy_dir)
        proba = legacy.predict_proba(scored["query"])[:, 1]
        legacy.predict_logit(scored["query"])
        legacy.predict_lift(scored["query"])
        legacy.predict(scored["query"])
        with pytest.raises(ValueError, match="no reference-library rank"):
            legacy.predict_rank(scored["query"])

    assert np.all(np.isfinite(proba)), "proba must survive a missing reference"
