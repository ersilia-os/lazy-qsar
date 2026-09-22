"""A reference library for the suite, built from committed ChEMBL fixtures.

Fitting requires a reference library, so every test that fits a model needs one. The suite
cannot use the published bundle: it is 267 MB, it would have to be downloaded, and a test
that silently reaches the network is a flake waiting to happen.

Instead it builds a small one from the molecules already committed under ``tests/data``,
using whichever descriptor classes are registered at the time -- the real ones in the
``chem`` tier, the stubs elsewhere -- so the matrices always have the dimensions the model
being fitted expects. ``LAZYQSAR_REFERENCE_N`` selects the small tier by name.

The molecules come from ``reference_imbalanced.csv`` while models are fitted on
``reference_binary.csv``, so the reference population is not the training set. That is the
distinction the whole feature rests on, and a fixture that blurred it would let a
training-relative rank pass every test here.
"""

from __future__ import annotations

import pathlib

import numpy as np

from _helpers.smiles import load_imbalanced_dataset


def build_reference(directory, descriptor_names, n=None, smiles=None):
    """Write a reference bundle for *descriptor_names* into *directory*.

    Returns the tier size, which the caller puts in ``LAZYQSAR_REFERENCE_N``.
    """
    import h5py

    from lazyqsar.registry import get_descriptor_type

    if smiles is None:
        smiles, _ = load_imbalanced_dataset()
    smiles = list(smiles)[: n or len(smiles)]
    n = len(smiles)

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"reference_smiles_n{n}.csv").write_text(
        "smiles\n" + "\n".join(smiles) + "\n"
    )

    for name in descriptor_names:
        X = np.asarray(get_descriptor_type(name)().transform(smiles), dtype=np.float32)
        # A NaN row would be imputed at fit time and smear the ECDF with a fabricated
        # point, so drop it here rather than let it into the reference.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        with h5py.File(directory / f"{name}_n{n}.h5", "w") as f:
            dset = f.create_dataset("X", data=X)
            dset.attrs["descriptor"] = name
            dset.attrs["n_rows"] = X.shape[0]
    return n
