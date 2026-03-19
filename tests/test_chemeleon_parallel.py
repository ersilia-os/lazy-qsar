"""Verify Chemeleon parallel featurization matches sequential reference."""
import numpy as np


SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC(=O)NC1=CC=C(C=C1)O", "CCCCCCCCCCCCCCCC(=O)O",
    "c1ccc2c(c1)ccc3cccc4cccc2c34",
]


def _chemeleon_sequential(fp, molecules):
    """Old sequential featurization path."""
    from chemprop.data import BatchMolGraph
    from rdkit.Chem import MolFromSmiles
    bmg = BatchMolGraph([
        fp.featurizer(MolFromSmiles(m) if isinstance(m, str) else m)
        for m in molecules
    ])
    bmg.to(device=fp.model.device)
    return fp.model.fingerprint(bmg).numpy(force=True)


def test_chemeleon():
    from lazyqsar.descriptors.chemeleon import ChemeleonDescriptor
    desc = ChemeleonDescriptor()

    X_parallel = desc.transform(SMILES)
    assert X_parallel.shape == (10, 2048), f"Unexpected shape: {X_parallel.shape}"
    assert np.issubdtype(X_parallel.dtype, np.floating)
    print(f"Chemeleon shape OK: {X_parallel.shape}")

    X_seq = _chemeleon_sequential(desc.chemeleon_fingerprint, SMILES)
    assert np.allclose(X_parallel, X_seq, atol=1e-5), "Parallel output differs from sequential"
    print("Chemeleon correctness OK")


if __name__ == "__main__":
    test_chemeleon()
    print("All Chemeleon tests passed.")
