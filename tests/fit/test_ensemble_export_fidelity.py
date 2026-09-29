"""Does the shipped checkpoint return what the fitted ensemble returned?

``tests/fit/test_onnx_export_fidelity.py`` asks this of one :class:`LazyClassifier` --
preprocessor, heads, calibration. This file asks it of :class:`LazyClassifierQSAR`, the
object the runner and the Ersilia template actually call, across all six outputs it
publishes. Nothing covered that before: ``tests/chem/test_rank_parity_fit_vs_onnx.py``
compares the fitted model to its export only on *ordering* and on the top-1% set, and its
other comparisons are between ``load_raw`` and ``load_onnx`` -- which are both ONNX, since
``agnostic.LazyClassifier.load`` raises ``NotImplementedError`` for a genuinely raw load.
So the numeric agreement between scikit-learn and onnxruntime, at the level a caller sees,
was measured nowhere.

It is not exact, and the tolerances below say by how much. The preprocessor *is* exact --
``BasePreprocessor._bind_onnx_runtime`` makes the fitted pipeline run its own exported
graph -- but the heads are not bound, so they compute float32 in ONNX against float64 in
scikit-learn. Measured on three antimicrobial datasets in fast mode, over 4,000 DrugBank
molecules: every molecule's ``proba`` differs, by at most 1.8e-07, and no label moves.

Raising the export to double precision would not fix it: ``svc`` accepts
``DoubleTensorType`` and silently stays float32, ``xgb`` has no double converter, and
``lr`` in double still lands at 2.2e-16 rather than 0 because the op order differs. Only
binding the heads the way the preprocessor is bound would make the two paths identical,
and that changes every fitted number. Until that is done the guarantee is bounded rather
than exact, and these tolerances are what makes it enforceable: they sit about an order of
magnitude above the measurement, close enough that a real regression fails rather than
being absorbed.
"""

import contextlib
import io
import os

import numpy as np
import pytest

from _helpers.smiles import load_imbalanced_dataset, load_reference_dataset

from lazyqsar.qsar import LazyClassifierQSAR

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")

# Measured maxima over three datasets x 4,000 molecules, fast mode, all four head types:
# proba 1.8e-07, rank 2.9e-07, logit 1.6e-06, lift 1.7e-06, score 8.0e-06. Each tolerance
# is the next round number about 10x above its measurement. They are deliberately not
# tight to the measurement: a tolerance that only just passes turns every reordering of a
# floating-point sum into a failing test.
CHANNEL_ATOL = {
    "proba": 1e-06,
    "rank": 1e-06,
    "logit": 1e-05,
    "lift": 1e-05,
    "score": 1e-04,
}

N_TRAIN = 120


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """Fit, save, load the export, and score one query set through both.

    The query set is wider than the training set and mostly unseen, because agreement on
    the training molecules is the easy case: those sit where the heads are most confident
    and furthest from the split thresholds that make the two runtimes diverge.
    """
    train_smiles, train_y = load_reference_dataset()
    train_smiles = list(train_smiles)[:N_TRAIN]
    train_y = np.asarray(train_y[:N_TRAIN], dtype=int)
    extra_smiles, _ = load_imbalanced_dataset()
    query = list(load_reference_dataset()[0]) + list(extra_smiles)

    model = LazyClassifierQSAR(mode="fast")
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(smiles_list=train_smiles, y=train_y)
        task_dir = os.path.join(str(tmp_path_factory.mktemp("export_fidelity")), "task")
        model.save(task_dir)
        exported = LazyClassifierQSAR.load_onnx(task_dir)

        values = {
            tag: {
                channel: getattr(obj, f"predict_{channel}")(query)[:, 1]
                for channel in CHANNEL_ATOL
            }
            for tag, obj in (("fit", model), ("onnx", exported))
        }
        labels = {tag: np.asarray(obj.predict(query))
                  for tag, obj in (("fit", model), ("onnx", exported))}

    return {"values": values, "labels": labels, "query": query, "n": len(query),
            "descriptors": list(model.descriptor_types)}


@pytest.mark.parametrize("channel", sorted(CHANNEL_ATOL))
def test_the_export_agrees_with_the_model_it_came_from(scored, channel):
    """The headline guarantee, per output, at the level a caller sees."""
    a = scored["values"]["fit"][channel]
    b = scored["values"]["onnx"][channel]
    diff = np.abs(a - b)
    worst = int(np.argmax(diff))
    np.testing.assert_allclose(
        b, a, rtol=0, atol=CHANNEL_ATOL[channel],
        err_msg=(
            f"{channel}: export differs from the fitted model by "
            f"{diff.max():.3e} (tolerance {CHANNEL_ATOL[channel]:.0e}) on "
            f"{int((diff > CHANNEL_ATOL[channel]).sum())} of {len(diff)} molecules; "
            f"worst at index {worst}: fit={a[worst]!r} onnx={b[worst]!r}"
        ),
    )


def test_the_export_calls_exactly_the_same_compounds_active(scored):
    """The guarantee that is exact, and the one a screening decision rests on.

    A tolerance on ``proba`` permits a label to move when a molecule sits within that
    tolerance of the cutoff. Nothing here forces that not to happen, so it is asserted
    rather than assumed -- if it ever fails, the binary output has become runtime-
    dependent and no tolerance above makes that acceptable.
    """
    fit, onnx = scored["labels"]["fit"], scored["labels"]["onnx"]
    moved = np.flatnonzero(fit != onnx)
    assert moved.size == 0, (
        f"{moved.size} of {scored['n']} molecules change label between the fitted "
        f"model and its export, at indices {moved[:10].tolist()}"
    )


def test_rank_is_not_more_export_sensitive_than_proba_in_absolute_terms(scored):
    """Rank amplifies whatever gap proba has, so pin the amplification, not just the gap.

    Rank is proba read through the reference library, and that transfer is steep wherever
    the reference bunches up -- historically ~100x, which is how a 1e-08 proba difference
    became a visible rank difference before v3.5. The tail-anchored table flattened the
    steepest part, and this fails if some future change restores it.
    """
    values = scored["values"]
    proba_gap = float(np.abs(values["fit"]["proba"] - values["onnx"]["proba"]).max())
    rank_gap = float(np.abs(values["fit"]["rank"] - values["onnx"]["rank"]).max())
    if proba_gap == 0.0:
        pytest.skip("proba is bit-identical here; there is no gap to amplify")
    assert rank_gap / proba_gap < 50.0, (
        f"the reference transfer amplifies the export gap {rank_gap / proba_gap:.1f}x "
        f"(proba {proba_gap:.3e} -> rank {rank_gap:.3e}); it was ~100x before the "
        "scale was anchored on the reference's tail"
    )
