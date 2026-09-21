"""Two ways the predict path avoids re-parsing molecules, and the property each rests on.

Predict used to parse the whole library with RDKit *after* every descriptor had already
parsed every molecule, purely to find the rows to blank. It now re-checks only the rows a
descriptor already returned as all-NaN. That is exact rather than approximate, but only
because of a property of the descriptors themselves, which is what the first test here
pins: a descriptor in ``_NAN_FAITHFUL`` returns an all-NaN row for every molecule RDKit
cannot parse. If that ever stops holding, unparseable molecules would start coming back
with an ordinary-looking score.

Also covers Morgan's preallocated construction, which replaced a list of lists and has to
produce the same array to the bit.

Needs RDKit.
"""

import numpy as np
import pytest
from _helpers.smiles import load_invalid_smiles, load_reference_dataset

from lazyqsar.api.classifier_predict import _NAN_FAITHFUL, _unparseable
from lazyqsar.descriptors._validate import invalid_smiles_indices
from lazyqsar.registry import get_descriptor_type


@pytest.fixture(scope="module")
def mixed():
    """Valid ChEMBL molecules with the committed bad strings interleaved."""
    good, _ = load_reference_dataset()
    bad = load_invalid_smiles()
    out = []
    for i, smi in enumerate(good[: len(bad) * 3]):
        out.append(smi)
        if i % 3 == 2 and bad:
            out.append(bad.pop())
    return out


# ---------------------------------------------------------------------------
# The property the shortcut rests on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["morgan", "rdkit"])
def test_nan_faithful_descriptors_flag_every_unparseable_molecule(name, mixed):
    """All-NaN rows must be a superset of the rows RDKit rejects.

    ``_unparseable`` re-checks only the all-NaN rows, so anything RDKit rejects has to turn
    up among them. A superset is fine — the re-check discards the extras — but a subset
    would silently let an unparseable string through with a median-imputed score.

    ``cddd`` is deliberately not in :data:`_NAN_FAITHFUL` and not tested here: it repairs
    NaN rows from a ChEMBL nearest neighbour, so a clean row from cddd is not evidence that
    the molecule parsed.
    """
    assert name in _NAN_FAITHFUL
    X = get_descriptor_type(name)().transform(mixed)
    all_nan = set(np.flatnonzero(np.isnan(X).all(axis=1)).tolist())
    unparseable = set(invalid_smiles_indices(mixed))

    assert unparseable, (
        "fixture produced no invalid molecules; the test would be vacuous"
    )
    assert unparseable <= all_nan, (
        f"{name} returned a non-NaN row for molecules RDKit rejects: "
        f"{sorted(unparseable - all_nan)}"
    )


# ---------------------------------------------------------------------------
# The shortcut itself
# ---------------------------------------------------------------------------


def test_unparseable_agrees_with_a_full_scan(mixed):
    """Same answer as parsing every molecule, which is what it replaced."""
    scan = {
        "descriptors": ["morgan"],
        "nan_rows": set(invalid_smiles_indices(mixed)) | {0, 1},  # plus harmless extras
    }
    assert _unparseable(mixed, scan) == invalid_smiles_indices(mixed)


def test_only_the_candidates_are_parsed(mixed, monkeypatch):
    """The whole point: a clean library costs zero parses, not one per molecule."""
    from lazyqsar.api import classifier_predict as cp

    seen = []

    def counting(smiles_list):
        seen.append(len(smiles_list))
        return invalid_smiles_indices(smiles_list)

    monkeypatch.setattr(cp, "invalid_smiles_indices", counting)

    candidates = set(invalid_smiles_indices(mixed))
    cp._unparseable(mixed, {"descriptors": ["morgan"], "nan_rows": candidates})
    assert seen == [len(candidates)], (
        f"parsed {seen} molecules for {len(candidates)} candidates out of {len(mixed)}"
    )

    seen.clear()
    cp._unparseable(mixed, {"descriptors": ["morgan"], "nan_rows": set()})
    assert seen == [], "a library with no NaN rows must cost no parses at all"


def test_it_falls_back_to_a_full_scan_without_a_faithful_descriptor(mixed, monkeypatch):
    """A cddd-only run gets the old behaviour rather than a wrong answer."""
    from lazyqsar.api import classifier_predict as cp

    seen = []

    def counting(smiles_list):
        seen.append(len(smiles_list))
        return invalid_smiles_indices(smiles_list)

    monkeypatch.setattr(cp, "invalid_smiles_indices", counting)

    # cddd reports no NaN rows because it repaired them; the answer must still be right.
    bad = cp._unparseable(mixed, {"descriptors": ["cddd"], "nan_rows": set()})
    assert seen == [len(mixed)], (
        "expected the full scan when no faithful descriptor ran"
    )
    assert bad == invalid_smiles_indices(mixed)


# ---------------------------------------------------------------------------
# Morgan's preallocated construction
# ---------------------------------------------------------------------------


def test_morgan_matches_the_list_of_lists_it_replaced(mixed):
    """Preallocating changed the speed, not the array.

    The reference below is the previous implementation verbatim. Counts are clamped to 255
    and small integers are exact in float32, so "close enough" is not the standard here —
    the arrays have to be equal, NaN placement included.
    """
    from rdkit import Chem

    from lazyqsar.descriptors.morgan import MorganFingerprint

    fp = MorganFingerprint()
    new = fp.transform(mixed)

    reference = []
    for smi in mixed:
        mol = Chem.MolFromSmiles(smi)
        try:
            v = fp.mfpgen.GetCountFingerprint(mol)
            row = [0] * fp.n_dim
            for i, val in v.GetNonzeroElements().items():
                row[i] = val if val < 255 else 255
            reference.append(row)
        except Exception:
            reference.append([np.nan] * fp.n_dim)
    old = np.array(reference, dtype=np.float32)

    assert new.dtype == old.dtype
    assert new.shape == old.shape
    np.testing.assert_array_equal(new, old)
