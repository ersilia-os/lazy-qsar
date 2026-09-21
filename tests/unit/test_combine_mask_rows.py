"""Blanking the rows of molecules that could not be parsed.

An unparseable SMILES produces an all-NaN descriptor row, and the imputer inside the exported
preprocessor would otherwise fill it with the training median -- turning a string that is not
a molecule into an ordinary-looking score. ``mask_rows`` writes NaN there instead, keeping the
row aligned with the input while making the gap unmissable.

Two things make this delicate, and both are pinned below: the mask must not disturb its
neighbours, and ``binary`` is integer-valued so it has to be promoted rather than silently
truncating NaN to zero. Numpy only.
"""

import numpy as np
import pytest

from lazyqsar.ensemble import mask_rows


def _values(n=6):
    rng = np.random.default_rng(0)
    return {
        "proba": rng.uniform(0.01, 0.99, size=(n, 2)),
        "logit": rng.normal(size=(n, 2)),
        "rank": rng.uniform(0.0, 1.0, size=(n, 2)),
        "binary": np.ones(n, dtype=np.int64),
    }


def test_masked_rows_are_nan_in_every_output():
    values = _values()
    mask_rows(values, [1, 4])
    for name, arr in values.items():
        assert np.isnan(np.asarray(arr)[1]).all(), f"{name} row 1 not blanked"
        assert np.isnan(np.asarray(arr)[4]).all(), f"{name} row 4 not blanked"


def test_other_rows_are_untouched():
    """A masked row must not perturb its neighbours -- they stay bit-identical."""
    values = _values()
    before = {k: np.array(v, copy=True) for k, v in values.items()}
    mask_rows(values, [1, 4])
    keep = [0, 2, 3, 5]
    for name, arr in values.items():
        assert np.array_equal(
            np.asarray(arr)[keep], before[name][keep].astype(float)
        ), f"{name} changed outside the masked rows"


def test_binary_is_promoted_to_float():
    """Integer arrays cannot hold NaN, so the documented trade is a dtype promotion.

    A caller checking ``== 1`` still behaves correctly, and a NaN is visible where a 0 would
    have been silently wrong.
    """
    values = _values()
    assert values["binary"].dtype.kind == "i"
    mask_rows(values, [2])
    assert values["binary"].dtype.kind == "f"
    assert np.isnan(values["binary"][2])


def test_empty_rows_leaves_dtypes_alone():
    """With nothing to carry, the usual integer return must survive."""
    values = _values()
    out = mask_rows(values, [])
    assert out is values
    assert values["binary"].dtype.kind == "i"
    assert not np.isnan(values["proba"]).any()


def test_mutates_in_place_and_returns_the_same_dict():
    """Callers rely on this: api.classifier_predict masks results it already holds."""
    values = _values()
    out = mask_rows(values, [0])
    assert out is values
    assert np.isnan(values["proba"][0]).all()


@pytest.mark.parametrize(
    "rows", [[1, 3], np.array([1, 3]), range(1, 4, 2), (1, 3)], ids=type
)
def test_accepts_any_sequence_of_row_indices(rows):
    values = _values()
    mask_rows(values, rows)
    assert np.isnan(values["proba"][1]).all()
    assert np.isnan(values["proba"][3]).all()
    assert not np.isnan(values["proba"][0]).any()
