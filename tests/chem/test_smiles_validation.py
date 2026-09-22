"""Parsing SMILES: what fit refuses, and what predict merely reports.

Fit and predict want different things from the same check. Training on a molecule that cannot
be featurized is meaningless, so ``validate_smiles`` raises. Screening a library should not
abort on one malformed row, but it must not invent a number for it either -- so
``invalid_smiles_indices`` returns the positions and predict writes NaN there.

Needs RDKit.
"""

import pytest

from _helpers.smiles import load_invalid_smiles, load_reference_dataset

from lazyqsar.qsar import invalid_smiles_indices, validate_smiles


@pytest.fixture(scope="module")
def good():
    smiles, _ = load_reference_dataset()
    return smiles[:5]


@pytest.fixture(scope="module")
def bad():
    return load_invalid_smiles()


def test_every_committed_bad_string_is_rejected(bad):
    """The corpus is the specification; an RDKit bump that accepts one should fail here."""
    assert invalid_smiles_indices(bad) == list(range(len(bad))), (
        "RDKit now parses something tests/data/invalid_smiles.txt says it must not"
    )


def test_valid_smiles_report_no_bad_positions(good):
    assert invalid_smiles_indices(good) == []


def test_indices_are_positions_in_the_input(good, bad):
    mixed = [good[0], bad[0], good[1], bad[1], good[2]]
    assert invalid_smiles_indices(mixed) == [1, 3]


@pytest.mark.parametrize(
    "hostile", [[None], [42], [3.5], [None, 42]], ids=["none", "int", "float", "mixed"]
)
def test_non_string_input_is_reported_not_raised(hostile):
    """Predict calls this on every batch, so it must survive anything a CSV can hold.

    RDKit raises ``TypeError`` on a non-string; the check swallows that and reports the
    position, because an exception here would abort a whole library screen over one
    malformed cell.
    """
    assert invalid_smiles_indices(hostile) == list(range(len(hostile)))


def test_the_empty_string_is_accepted_as_a_zero_atom_molecule():
    """Characterization, not endorsement.

    ``Chem.MolFromSmiles("")`` returns a real Mol with no atoms, so an empty cell in an input
    CSV is *not* flagged and does not get the NaN treatment that an unparseable string gets.
    It is featurized as an empty molecule and scored like any other row.

    That is arguably the same failure the NaN masking exists to prevent -- something that is
    not a molecule receiving an ordinary-looking score -- but it is current behaviour, and
    changing it is a decision about the API rather than something a test should assume. This
    pins it so the decision is at least visible.
    """
    assert invalid_smiles_indices([""]) == []
    assert validate_smiles([""]) is None


def test_empty_input_is_not_an_error():
    assert invalid_smiles_indices([]) == []
    assert validate_smiles([]) is None


def test_validate_smiles_accepts_valid_input(good):
    assert validate_smiles(good) is None


def test_validate_smiles_raises_naming_every_offender(good, bad):
    mixed = [good[0], bad[0], good[1], bad[1]]
    with pytest.raises(ValueError) as exc:
        validate_smiles(mixed)
    message = str(exc.value)
    assert "1" in message and "3" in message, "the message must name every position"
    assert bad[0] in message and bad[1] in message, "and show the offending string"


def test_the_two_checks_agree(good, bad):
    """They must stay derived from one another, not drift into two notions of 'invalid'."""
    mixed = [good[0], bad[0], good[1], bad[2], bad[3]]
    positions = invalid_smiles_indices(mixed)
    with pytest.raises(ValueError) as exc:
        validate_smiles(mixed)
    for i in positions:
        assert f"[{i}]" in str(exc.value)


def test_every_descriptor_reports_its_width_the_same_way():
    """`n_dim` is what callers read, and RDKit's was missing.

    `LazyClassifierQSAR` asks the reference library for a matrix of the right width via
    `descriptors[i].n_dim`, so a slow-mode fit with RDKit active raised AttributeError the
    moment it reached the reference. Nothing caught it because the fitting tests run in fast
    mode; the bundle verification gate did, by comparing the published dimension against the
    installed descriptor's.

    RDKit's count is not a constant -- it is however many descriptors the installed version
    defines -- so this asserts the attribute agrees with `features`, not a number.
    """
    from lazyqsar.registry import DESCRIPTOR_TYPES, get_descriptor_type

    for name in sorted(DESCRIPTOR_TYPES):
        descriptor = get_descriptor_type(name)()
        n_dim = getattr(descriptor, "n_dim", None)
        assert isinstance(n_dim, int) and n_dim > 0, f"{name} has no usable n_dim"
        # `features` is not universal -- CDDD has none -- so it is only cross-checked
        # where it exists, which is where the two could disagree.
        features = getattr(descriptor, "features", None)
        if features is not None:
            assert n_dim == len(features), (
                f"{name}: n_dim {n_dim} disagrees with {len(features)} features"
            )
