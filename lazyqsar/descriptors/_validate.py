from rdkit import Chem


def invalid_smiles_indices(smiles_list):
    """Return the positions of SMILES RDKit cannot parse, without raising.

    Separated from :func:`validate_smiles` because fit and predict want different things
    from the same check. Training on a molecule that cannot be featurized is meaningless,
    so fit raises. Prediction over a large library should not abort on one malformed row,
    but it must not invent a number for it either -- so predict asks for the positions and
    writes NaN there.
    """
    invalid = []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            invalid.append(i)
    return invalid


def validate_smiles(smiles_list):
    """Raise if any SMILES cannot be parsed, naming every offending position."""
    invalid = invalid_smiles_indices(smiles_list)
    if invalid:
        details = ", ".join(f"[{i}] {repr(smiles_list[i])}" for i in invalid)
        raise ValueError(f"Invalid SMILES at position(s): {details}")
