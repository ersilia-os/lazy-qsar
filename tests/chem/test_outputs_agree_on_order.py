"""Every ``predict_type`` must order molecules the same way.

A screening pipeline that ranks by one output and a report that ranks by another have to
agree about which compounds are at the top. Until v3.5.0 two of them did not: ``rank`` was a
weighted mean of per-descriptor percentiles, and ``score`` a weighted mean of per-descriptor
raw values, neither of which is a monotone transform of the pooled probability. Measured on
these fixtures, ``score`` disagreed with ``proba`` about 581 of 79,800 pairs.

The cause is subtle enough to be worth writing down, because it is not the obvious one.
Calibration does not reorder anything: every head's calibrator is monotone and never moves
that head's own molecules. It is that the heads get *different* curves, so calibration
changes how far apart each head's opinions sit -- how loudly it votes -- and a weighted
average of differently-stretched monotone curves is not a monotone function of the weighted
average of the originals. No way of pooling raw values fixes that; the fix is to derive
``score`` from the pooled probability through a stored monotone map, as ``rank`` is.

Strict inversions, not Spearman: the map's running-max step creates ties, which drag a rank
correlation below 1.0 while leaving every pair correctly ordered. A tie is not a
disagreement.

Needs RDKit and the ``fit`` extra: only a fitted checkpoint carries the references.

One prevalence regime, not two. This used to fit a balanced checkpoint as well, at ~8 s,
and both fixtures use ``mode="fast"`` -- a single descriptor, where the weights collapse to
one column and the pooling defect above *cannot* recur. The regression is pinned properly,
and for free, by ``tests/unit/test_combine_score_ordering.py`` and
``tests/unit/test_pooled_rank_reference.py``, which build the disagreeing columns by hand.
What is left here is the end-to-end check that a real fitted checkpoint writes a usable
map -- worth one fixture, and the low-prevalence one is the regime Ersilia deploys in.
"""

import contextlib
import io
import itertools

import numpy as np
import pytest
from _helpers.smiles import load_imbalanced_dataset

ORDERED = ("proba", "logit", "lift", "rank", "score")


def _flipped_pairs(a, b):
    """Pairs that *a* and *b* strictly disagree about ordering. Ties are not disagreement."""
    sign_a = np.sign(np.subtract.outer(a, a))
    sign_b = np.sign(np.subtract.outer(b, b))
    return int(((sign_a * sign_b) < 0).sum() // 2)


def _fit_and_score(tmp_path, smiles, y, query):
    from lazyqsar.api.classifier_fit import fit
    from lazyqsar.api.classifier_predict import predict

    data = tmp_path / "data"
    data.mkdir()
    (data / "alpha.csv").write_text(
        "smiles,bin\n" + "".join(f"{s},{int(v)}\n" for s, v in zip(smiles, y))
    )
    models = tmp_path / "models"
    with contextlib.redirect_stdout(io.StringIO()):
        fit(data_dir=str(data), model_dir=str(models), mode="fast")
    out = {}
    for t in (*ORDERED, "binary"):
        values, _ = predict(model_dir=str(models), smiles=query, predict_type=t)
        out[t] = values[:, 0]
    return str(models), out


@pytest.fixture(scope="module")
def imbalanced(tmp_path_factory):
    smiles, y = load_imbalanced_dataset()
    return _fit_and_score(tmp_path_factory.mktemp("imb"), smiles, y, smiles[:250])


@pytest.mark.parametrize("a,b", list(itertools.combinations(ORDERED, 2)))
def test_no_two_outputs_disagree_about_order(imbalanced, a, b):
    """Not one inverted pair."""
    _, out = imbalanced
    flipped = _flipped_pairs(out[a], out[b])
    assert flipped == 0, (
        f"{a} and {b} disagree about the order of {flipped} pairs; every predict_type "
        f"must rank molecules identically"
    )


@pytest.mark.parametrize("other", ORDERED)
def test_binary_never_contradicts_a_continuous_output(imbalanced, other):
    """``binary`` is a threshold on the same quantity, so it may tie but never invert."""
    _, out = imbalanced
    assert _flipped_pairs(out["binary"], out[other]) == 0


def test_the_checkpoint_carries_the_score_reference(imbalanced):
    """Otherwise the tests above are passing on the fallback path, not the fix."""
    import json
    import os

    models, _ = imbalanced
    with open(os.path.join(models, "alpha", "metadata.json")) as f:
        meta = json.load(f)
    block = meta.get("pooled_scorer") or {}
    assert block.get("knots_x"), "no pooled score map written at fit time"
    assert len(block["knots_x"]) == len(block["knots_y"])
    assert np.all(np.diff(block["knots_x"]) > 0), "probability knots must be ascending"
    assert np.all(np.diff(block["knots_y"]) >= 0), "score knots must be non-decreasing"


def test_score_still_lands_near_the_raw_scale(imbalanced):
    """The map reproduces the values it replaced; it is not a rescale of proba.

    Derived from the probability, but fitted to the raw scores, so ``score`` keeps meaning
    "roughly what the heads said before calibration" rather than becoming ``proba`` under
    another name.
    """
    _, out = imbalanced
    assert _flipped_pairs(out["score"], out["proba"]) == 0
    assert not np.allclose(out["score"], out["proba"], atol=1e-3), (
        "score has collapsed onto proba; the map is no longer carrying the raw scale"
    )
