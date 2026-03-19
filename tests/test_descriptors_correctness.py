"""Verify parallel output matches the old sequential implementation."""
import numpy as np


SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC(=O)NC1=CC=C(C=C1)O", "CCCCCCCCCCCCCCCC(=O)O",
    "c1ccc2c(c1)ccc3cccc4cccc2c34",
]


def _morgan_sequential(smiles_list, radius=3, n_dim=2048):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_dim)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        v = mfpgen.GetCountFingerprint(mol)
        data = [0] * n_dim
        for i, val in v.GetNonzeroElements().items():
            data[i] = val if val < 255 else 255
        results.append(data)
    return np.array(results, dtype=int)


def _rdkit_sequential(smiles_list):
    from rdkit import Chem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem import Descriptors
    descriptor_names = sorted([n for n, _ in Descriptors._descList])
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError
            vals = np.array(calc.CalcDescriptors(mol), dtype=float)
            vals[~np.isfinite(vals)] = 0.0
        except Exception:
            vals = np.zeros(len(descriptor_names), dtype=float)
        results.append(vals)
    return np.clip(np.vstack(results), -1e5, 1e5)


def test_morgan_correctness():
    from lazyqsar.descriptors.morgan import MorganFingerprint
    X_parallel = MorganFingerprint().transform(SMILES)
    X_seq = _morgan_sequential(SMILES)
    assert np.array_equal(X_parallel, X_seq), "Morgan parallel != sequential"
    print("Morgan correctness OK")


def test_rdkit_correctness():
    from lazyqsar.descriptors.rdkit_descriptors import RDKitDescriptor
    X_parallel = RDKitDescriptor().transform(SMILES)
    X_seq = _rdkit_sequential(SMILES)
    assert np.allclose(X_parallel, X_seq), "RDKit parallel != sequential"
    print("RDKit correctness OK")


if __name__ == "__main__":
    test_morgan_correctness()
    test_rdkit_correctness()
    print("All correctness checks passed.")
