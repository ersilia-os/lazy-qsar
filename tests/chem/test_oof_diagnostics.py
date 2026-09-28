"""Advisory numbers describing how a fitted model treats molecules with known labels.

None of this enters the rank scale, and that is the whole design. Anchoring the scale on the
out-of-fold actives -- so "0.95 means looks like a known active" -- was considered and
rejected: it would put *every* model's median active at 0.95 by construction, and a model
with AUC 0.95 would read the same as one with AUC 0.55. Reporting the numbers instead keeps
the signal, which is exactly what `test_a_worthless_model_says_so` below pins. That test
could not exist under the anchoring design.
"""

import json
import os

import numpy as np
import pytest

from _helpers.smiles import load_reference_dataset
from lazyqsar.qsar import LazyClassifierQSAR
from lazyqsar.utils.ranking import DECISION_RANK


def _fit(tmp_path, y, name="m"):
    smiles, _ = load_reference_dataset()
    model = LazyClassifierQSAR(mode="fast")
    model.fit(smiles_list=smiles, y=np.asarray(y))
    directory = str(tmp_path / name)
    model.save_raw(directory)
    with open(os.path.join(directory, "metadata.json")) as handle:
        return model, smiles, json.load(handle)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    _, y = load_reference_dataset()
    return _fit(tmp_path_factory.mktemp("real"), y)


def test_the_checkpoint_carries_the_diagnostics(real):
    _, _, meta = real
    diag = meta["oof_diagnostics"]
    assert set(diag) == {
        "actives",
        "inactives",
        "screening_auc",
        "generic_hit_rate",
        "decision_cutoff_rank",
        "sensitivity_at_cutoff",
    }
    assert diag["actives"]["n"] + diag["inactives"]["n"] == 233


def test_the_diagnostics_are_not_part_of_the_ranker(real):
    """They describe the model, not the scale. Nothing may read them as a reference."""
    _, _, meta = real
    assert "oof_diagnostics" not in meta["pooled_ranker"]
    assert meta["pooled_ranker"]["source"] == "reference_library"


def test_the_checkpoint_names_the_bundle_it_was_ranked_against(real):
    """The id is only a promise; the manifest hash is what can be checked."""
    _, _, meta = real
    assert "manifest_sha256" in meta["pooled_ranker"]["library"]


def test_known_actives_sit_above_known_inactives(real):
    _, _, meta = real
    diag = meta["oof_diagnostics"]
    assert diag["actives"]["rank_p50"] > diag["inactives"]["rank_p75"]


def test_the_rates_are_proportions(real):
    _, _, meta = real
    diag = meta["oof_diagnostics"]
    assert 0.0 <= diag["screening_auc"] <= 1.0
    assert 0.0 <= diag["generic_hit_rate"] <= 1.0


def test_the_hit_rate_is_fixed_by_the_cutoff(real):
    """`generic_hit_rate` is now 1% by construction, and kept for exactly that reason.

    The cutoff is `DECISION_RANK` inverted against the reference, so the share of drug-like
    chemical space called active is pinned. This assertion is therefore a check that the
    inverse landed where it claims, not a statement about the model -- what discriminates
    between models is `sensitivity_at_cutoff`.

    The tolerance is the knot resolution: the reference is subsampled, so the percentile can
    only be hit to within one knot spacing.
    """
    diag = real[2]["oof_diagnostics"]
    assert diag["decision_cutoff_rank"] == DECISION_RANK
    assert diag["generic_hit_rate"] == pytest.approx(0.01, abs=2e-3)


def test_a_working_model_keeps_most_of_its_actives_at_that_hit_rate(real):
    """The number that makes a model's selectivity legible before anyone screens with it.

    Sensitivity at a fixed generic hit rate: what share of its own known actives the model
    still catches while calling only 1% of drug-like space active.
    """
    assert real[2]["oof_diagnostics"]["sensitivity_at_cutoff"] > 0.5


def test_the_band_is_on_the_scale_the_user_sees(real):
    """The band must be in the units `predict_rank` returns, or it cannot be compared
    against a query's rank -- which is the entire point of reporting it."""
    model, smiles, meta = real
    _, y = load_reference_dataset()
    ranks = model.predict_rank(smiles_list=smiles)[:, 1]
    expected = float(np.percentile(ranks[np.asarray(y) == 1], 50))
    assert meta["oof_diagnostics"]["actives"]["rank_p50"] == pytest.approx(
        expected, abs=0.05
    )


def test_a_worthless_model_says_so(tmp_path):
    """The reason the scale is not anchored on the out-of-fold actives.

    Labels are shuffled, so the model has learned nothing. Its actives must land low and
    overlap its inactives, and its screening AUC must be near chance. Anchoring would have
    pinned the actives at 0.95 and made this indistinguishable from a model that works.
    """
    _, y = load_reference_dataset()
    shuffled = np.random.default_rng(0).permutation(np.asarray(y))
    _, _, meta = _fit(tmp_path, shuffled, name="shuffled")
    diag = meta["oof_diagnostics"]

    assert diag["screening_auc"] < 0.8, (
        "a shuffled model must not separate from generic"
    )
    assert diag["actives"]["rank_p50"] < 0.85, "its actives must not look like actives"
    # The two classes must be indistinguishable, which is what "learned nothing" means.
    assert diag["actives"]["rank_p25"] < diag["inactives"]["rank_p75"]
    # And the assertion `generic_hit_rate` used to carry, now that it is fixed by the
    # cutoff: at a 1% hit rate a model that learned nothing catches almost none of its
    # actives, where the real model above keeps more than half.
    assert diag["sensitivity_at_cutoff"] < 0.25, (
        "a shuffled model must not catch its actives at a 1% generic hit rate"
    )


def test_the_checkpoint_carries_the_rank_anchors(real):
    """The tails are pinned on out-of-fold molecules, which do not travel in a checkpoint,
    so the anchors themselves must."""
    _, _, meta = real
    block = meta["pooled_ranker"]
    assert block["anchor_high"] is not None and block["anchor_low"] is not None
    # Only the high anchor shapes the scale: the tail table has no low anchor, so
    # `anchor_low` is recorded as a diagnostic and `anchor_low_used` is always False.
    assert block["anchor_high_used"] and not block["anchor_low_used"]
    assert block["n_actives"] == 54 and block["n_inactives"] == 179


def test_the_anchors_put_the_actives_p95_at_the_top_of_the_scale(real):
    model, smiles, _ = real
    _, y = load_reference_dataset()
    ranks = model.predict_rank(smiles_list=smiles)[:, 1]
    actives = ranks[np.asarray(y) == 1]
    assert float(np.percentile(actives, 95)) == pytest.approx(0.95, abs=0.01)


def test_a_reloaded_checkpoint_ranks_like_the_model_it_came_from(real, tmp_path):
    """The regression this exists for.

    `ArtifactWrapper` -- the ONNX path, which is what the Ersilia Model Hub deploys --
    built its spec through a different call site than the fitted estimator, and silently
    dropped the anchors. Probabilities still agreed to 9e-09 while ranks moved by 0.096,
    because the unanchored fallback is a different function, not a rounding difference.
    Nothing in the output would have shown it.
    """
    model, smiles, _ = real
    directory = str(tmp_path / "roundtrip")
    model.save_raw(directory)
    loaded = LazyClassifierQSAR.load(directory)

    query = smiles[:60]
    fitted_rank = model.predict_rank(smiles_list=query)[:, 1]
    loaded_rank = loaded.predict_rank(smiles_list=query)[:, 1]
    assert np.allclose(fitted_rank, loaded_rank, atol=1e-4)
    assert loaded.pooled_rank_anchors is not None, (
        "the ONNX path must carry the anchors"
    )
