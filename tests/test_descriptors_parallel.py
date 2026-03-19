"""Quick smoke-test for the parallelised descriptor transforms."""
import numpy as np


SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC(=O)NC1=CC=C(C=C1)O",
    "CCCCCCCCCCCCCCCC(=O)O",
    "c1ccc2c(c1)ccc3cccc4cccc2c34",
]


def test_morgan():
    from lazyqsar.descriptors.morgan import MorganFingerprint
    m = MorganFingerprint()
    X = m.transform(SMILES)
    assert X.shape == (10, 2048), f"Unexpected shape: {X.shape}"
    assert np.issubdtype(X.dtype, np.integer)
    assert (X >= 0).all()
    print(f"Morgan OK  shape={X.shape}  min={X.min()}  max={X.max()}")


def test_rdkit():
    from lazyqsar.descriptors.rdkit_descriptors import RDKitDescriptor
    r = RDKitDescriptor()
    X = r.transform(SMILES)
    assert X.shape[0] == 10, f"Unexpected row count: {X.shape[0]}"
    assert np.issubdtype(X.dtype, np.floating)
    assert np.isfinite(X).all(), "Non-finite values in RDKit output"
    print(f"RDKit OK   shape={X.shape}  min={X.min():.2f}  max={X.max():.2f}")


if __name__ == "__main__":
    test_morgan()
    test_rdkit()
    print("All tests passed.")
