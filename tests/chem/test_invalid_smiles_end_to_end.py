"""An unparseable molecule is blanked, not dropped and not imputed.

Its descriptor row comes back all-NaN, and the imputer inside the exported preprocessor would
otherwise fill it with the training median -- turning a string that is not a molecule into an
ordinary-looking score. The predict path writes NaN there instead.

Three things have to hold, and only the first is obvious:

1. the row is NaN;
2. the output still has one row per input, in the original order, so a caller can line the
   results back up against their CSV;
3. the other rows are *unchanged* by the bad row's presence -- which is what distinguishes
   blanking from dropping, and proves nothing leaked through the imputer into its neighbours.

Uses real RDKit and a real Morgan checkpoint, because the whole point is what happens when
featurization genuinely fails.
"""

import os

import numpy as np
import pytest

from _helpers.smiles import load_invalid_smiles, load_reference_dataset

from lazyqsar.api.classifier_predict import predict
from lazyqsar.ensemble.combine import OUTPUT_NAMES
from lazyqsar.qsar import LazyClassifierQSAR

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def morgan_checkpoint(tmp_path_factory):
    """A real, small Morgan model. Built once -- it is the expensive part of this file."""
    smiles, y = load_reference_dataset()
    smiles, y = smiles[:120], np.asarray(y[:120], dtype=int)
    model = LazyClassifierQSAR(mode="fast")
    model.fit(smiles_list=smiles, y=y)
    root = tmp_path_factory.mktemp("chem_ckpt")
    task_dir = os.path.join(str(root), "task")
    model.save(task_dir)
    return {"root": str(root), "task_dir": task_dir, "smiles": smiles}


@pytest.fixture(scope="module")
def probe(morgan_checkpoint):
    """Two good molecules with one unparseable string wedged between them."""
    good = morgan_checkpoint["smiles"][:2]
    bad = load_invalid_smiles()[0]
    return {"good": list(good), "mixed": [good[0], bad, good[1]]}


@pytest.mark.parametrize("predict_type", OUTPUT_NAMES)
def test_bad_row_is_blanked_without_disturbing_its_neighbours(
    morgan_checkpoint, probe, predict_type
):
    root = morgan_checkpoint["root"]
    mixed, _ = predict(model_dir=root, smiles=probe["mixed"], predict_type=predict_type)
    clean, _ = predict(model_dir=root, smiles=probe["good"], predict_type=predict_type)

    assert mixed.shape[0] == 3, (
        "rows are blanked, never dropped -- the caller aligns by index"
    )
    assert np.isnan(mixed[1]).all(), f"{predict_type}: the unparseable row is not NaN"
    assert not np.isnan(mixed[[0, 2]]).any(), (
        f"{predict_type}: a good row was blanked too"
    )
    np.testing.assert_allclose(
        mixed[[0, 2]],
        clean,
        atol=1e-6,
        err_msg=(
            f"{predict_type}: the good rows moved when a bad one was present. They must not "
            "-- that would mean the bad row leaked through the imputer or the batching."
        ),
    )


def test_predict_returns_nan_rather_than_a_confident_negative(morgan_checkpoint, probe):
    """The one outcome this path must never produce.

    ``NaN >= threshold`` is False, so a naive implementation would label an unparseable
    molecule 0 -- an inactive call on a string that is not a molecule.
    """
    model = LazyClassifierQSAR.load(morgan_checkpoint["task_dir"])
    labels = model.predict(probe["mixed"])

    assert labels.dtype.kind == "f", "the array is promoted so it can carry NaN"
    assert np.isnan(labels[1])
    assert labels[1] != 0, (
        "an unparseable molecule must not come back as a negative call"
    )
    assert set(labels[[0, 2]]) <= {0.0, 1.0}


def test_all_good_input_still_returns_integer_labels(morgan_checkpoint, probe):
    """The promotion happens only when there is a NaN to carry."""
    model = LazyClassifierQSAR.load(morgan_checkpoint["task_dir"])
    labels = model.predict(probe["good"])
    assert labels.dtype.kind == "i"
    assert set(labels) <= {0, 1}


def test_a_warning_names_how_many_rows_were_blanked(morgan_checkpoint, probe):
    """Silence would be the worst outcome: a screen with holes and nothing saying so.

    The package logs through loguru, which pytest's ``caplog`` does not see, so the sink is
    attached directly rather than trusting a fixture that would capture nothing and make this
    assertion vacuous.
    """
    from loguru import logger as loguru_logger

    records = []
    sink_id = loguru_logger.add(records.append, level="WARNING")
    try:
        predict(model_dir=morgan_checkpoint["root"], smiles=probe["mixed"])
    finally:
        loguru_logger.remove(sink_id)

    warnings = [str(r) for r in records]
    assert any("could not be parsed" in w for w in warnings), (
        f"no warning was emitted for the unparseable row; got {warnings}"
    )
    assert any("1 SMILES" in w for w in warnings), "the warning should say how many"
