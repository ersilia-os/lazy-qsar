"""Fitting severely imbalanced data end to end, which nothing used to do.

Above 100 negatives per positive the trainer stops fitting one model and fits one per
negative slice, each seeing all the positives. Every model is then calibrated to a base rate
its batch invented, so ``predict_proba`` corrects each one back to the population prior
before averaging. None of that ran in the suite: the committed fixtures are 3.3:1 and 6.7:1,
so they take the single-batch path, ``batch_priors`` equals ``[population_prior]``, and the
correction returns its input untouched.

This is the one test here that has to fit a real model, about 18 seconds, so it is a single
session-scoped fixture that several assertions share. The cheap parts of the same machinery
are covered without fitting in ``tests/fit/test_batch_planning.py`` and
``tests/unit/test_prior_correction.py``.

Needs RDKit and the ``fit`` extra.
"""

import contextlib
import io
import json
import os

import numpy as np
import pytest
from _helpers.smiles import load_severely_imbalanced_dataset

ORDERED = ("proba", "logit", "lift", "rank", "score")


def _flipped_pairs(a, b):
    sign_a = np.sign(np.subtract.outer(a, a))
    sign_b = np.sign(np.subtract.outer(b, b))
    return int(((sign_a * sign_b) < 0).sum() // 2)


@pytest.fixture(scope="module")
def two_batch(tmp_path_factory):
    """A checkpoint fitted on >100:1 data, so it really does carry several batches."""
    from lazyqsar.api.classifier_fit import fit

    root = tmp_path_factory.mktemp("imbalanced")
    smiles, y = load_severely_imbalanced_dataset()
    data = root / "data"
    data.mkdir()
    (data / "alpha.csv").write_text(
        "smiles,bin\n" + "".join(f"{s},{v}\n" for s, v in zip(smiles, y))
    )
    models = root / "models"
    with contextlib.redirect_stdout(io.StringIO()):
        fit(data_dir=str(data), model_dir=str(models), mode="fast")
    return str(models), smiles


@pytest.fixture(scope="module")
def descriptor_meta(two_batch):
    models, _ = two_batch
    with open(os.path.join(models, "alpha", "morgan", "metadata.json")) as f:
        return json.load(f)


def test_the_fit_really_produced_several_batches(descriptor_meta):
    """Everything below is vacuous on a single-batch checkpoint, so check first."""
    assert descriptor_meta["num_batches"] > 1, (
        "the fixture no longer trips the imbalance branch; this file is testing the "
        "ordinary single-batch path and proving nothing"
    )
    assert len(descriptor_meta["batch_priors"]) == descriptor_meta["num_batches"]


def test_the_batches_are_enriched_relative_to_the_library(descriptor_meta):
    """Which is why the correction exists: each batch is a denser slice than the whole."""
    population = descriptor_meta["population_prior"]
    assert all(p > population for p in descriptor_meta["batch_priors"]), (
        f"batch priors {descriptor_meta['batch_priors']} should all exceed the population "
        f"prior {population}"
    )
    assert population < 0.02, "the fixture is no longer severely imbalanced"


def test_the_prior_correction_actually_moves_the_numbers(two_batch, descriptor_meta):
    """The arithmetic that was dead code whenever a checkpoint had one batch."""
    from lazyqsar.artifacts.classifier import LazyClassifierArtifact
    from lazyqsar.registry import get_descriptor_type

    models, smiles = two_batch
    artifact = LazyClassifierArtifact.load(os.path.join(models, "alpha", "morgan"))
    X = get_descriptor_type("morgan")().transform(smiles[:60])

    corrected = artifact.predict_proba(X)[:, 1]
    uncorrected = np.array([b.predict_proba(X)[:, 1] for b in artifact._batches]).mean(
        axis=0
    )
    assert not np.allclose(corrected, uncorrected, atol=1e-6), (
        "the prior correction is a no-op on a checkpoint that should need it"
    )
    assert np.all(corrected < uncorrected), (
        "correcting an enriched batch toward a rarer population must lower probabilities"
    )
    assert np.all(np.isfinite(corrected))
    assert np.all((corrected >= 0) & (corrected <= 1))


@pytest.mark.parametrize("predict_type", [*ORDERED, "binary"])
def test_every_output_is_well_formed_across_batches(two_batch, predict_type):
    from lazyqsar.api.classifier_predict import predict

    models, smiles = two_batch
    values, header = predict(
        model_dir=models, smiles=smiles[:60], predict_type=predict_type
    )
    assert values.shape == (60, 1)
    assert header == ["alpha"]
    assert np.all(np.isfinite(values))


def test_the_outputs_still_agree_on_order_across_batches(two_batch):
    """The regime the ordering fix could not previously be checked in.

    ``predict_proba`` applies a *per-batch* correction and then averages, so with several
    batches the averaged result is not obviously a monotone function of the uncorrected
    one. If the corrections ever reordered molecules, ``score`` -- which is read off the
    pooled probability through a stored map -- would inherit that and the outputs would
    part company again.
    """
    from lazyqsar.api.classifier_predict import predict

    models, smiles = two_batch
    out = {}
    for t in ORDERED:
        values, _ = predict(model_dir=models, smiles=smiles[:120], predict_type=t)
        out[t] = values[:, 0]

    for other in ORDERED[1:]:
        flipped = _flipped_pairs(out["proba"], out[other])
        assert flipped == 0, (
            f"{other} disagrees with proba about {flipped} pairs on a multi-batch "
            f"checkpoint"
        )
