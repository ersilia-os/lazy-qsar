from rdkit import Chem


def parse_molecule(smiles):
    """Parse one SMILES, returning ``None`` for anything that is not a molecule.

    ``Chem.MolFromSmiles`` rejects most junk by returning ``None``, but it accepts the
    empty string and returns a valid **zero-atom** molecule. That is not a molecule for
    any purpose here: Morgan hands back an all-zero fingerprint for it, which is a
    perfectly plausible-looking feature vector, so a blank cell -- a trailing comma, a
    shifted column, a NaN written as "" -- used to be scored like a real compound instead
    of being NaN'd. Whitespace, ``"."`` and ``"()"`` already parse to ``None``.

    Non-string input raises inside RDKit rather than returning ``None``, so that is caught
    here too and reported the same way.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def invalid_smiles_indices(smiles_list):
    """Return the positions of SMILES RDKit cannot parse, without raising.

    Separated from :func:`validate_smiles` because fit and predict want different things
    from the same check. Training on a molecule that cannot be featurized is meaningless,
    so fit raises. Prediction over a large library should not abort on one malformed row,
    but it must not invent a number for it either -- so predict asks for the positions and
    writes NaN there.
    """
    return [i for i, smi in enumerate(smiles_list) if parse_molecule(smi) is None]


def validate_smiles(smiles_list):
    """Raise if any SMILES cannot be parsed, naming every offending position."""
    invalid = invalid_smiles_indices(smiles_list)
    if invalid:
        details = ", ".join(f"[{i}] {repr(smiles_list[i])}" for i in invalid)
        raise ValueError(f"Invalid SMILES at position(s): {details}")
