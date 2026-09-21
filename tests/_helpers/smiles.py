"""SMILES sources: synthetic strings for the stubbed tiers, real ChEMBL data for the chem tier.

Two different jobs, deliberately kept apart. ``make_smiles`` is for tests that run with stub
descriptors, where chemistry buys nothing and all that matters is validity and distinctness.
The committed ChEMBL datasets are for the ``chem`` tier, where the thing under test *is* the
chemistry -- ``make_smiles`` emits a family of alkyl ethers whose fingerprints are nearly
constant, so testing a descriptor against it measures the generator, not the code.
"""

import csv
import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

REFERENCE_BINARY = os.path.join(DATA_DIR, "reference_binary.csv")
REFERENCE_IMBALANCED = os.path.join(DATA_DIR, "reference_imbalanced.csv")
INVALID_SMILES_FILE = os.path.join(DATA_DIR, "invalid_smiles.txt")


def make_smiles(n):
    """n distinct, genuinely valid SMILES (simple ethers and alcohols).

    They have to parse: ``api.classifier_fit`` runs ``validate_smiles`` over the union before
    featurizing, so placeholder strings would be rejected before the stub is ever reached.
    Chemistry is irrelevant here -- only validity and distinctness.
    """
    if n > 90:
        raise ValueError("make_smiles supports up to 90 distinct strings")
    return [("C" * (i // 10 + 1)) + "O" + ("C" * (i % 10)) for i in range(n)]


def load_dataset(path):
    """Read a committed ``smiles,bin`` CSV into ``(smiles_list, y)``."""
    smiles, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            smiles.append(row["smiles"])
            y.append(int(row["bin"]))
    return smiles, y


def load_reference_dataset():
    """The balanced ChEMBL reference set (see ``tests/data/README.md``)."""
    return load_dataset(REFERENCE_BINARY)


def load_imbalanced_dataset():
    """The low-prevalence ChEMBL set, which reaches the imbalance-batching branch."""
    return load_dataset(REFERENCE_IMBALANCED)


def load_invalid_smiles():
    """Strings RDKit must reject.

    Committed rather than generated on purpose: which strings RDKit rejects is RDKit's
    business and shifts between versions, so this file is the specification. An RDKit bump
    that starts accepting one of these should fail a test, not silently weaken validation.
    """
    with open(INVALID_SMILES_FILE) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
