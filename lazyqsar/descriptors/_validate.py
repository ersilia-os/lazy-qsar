from rdkit import Chem


def validate_smiles(smiles_list):
    invalid = []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            invalid.append((i, smi))
    if invalid:
        details = ", ".join(f"[{i}] {repr(s)}" for i, s in invalid)
        raise ValueError(f"Invalid SMILES at position(s): {details}")
