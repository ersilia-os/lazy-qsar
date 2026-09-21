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


def load_severely_imbalanced_dataset(n_positives=2, n_negatives=201):
    """A >100:1 set, built by downsampling the committed ChEMBL files. No new data.

    ``reference_imbalanced.csv`` is low-prevalence at 13% but only 6.7:1, so it takes the
    single-batch branch exactly as the balanced set does. ``_plan_batches`` splits into
    several batches only above 100:1, and nothing in the suite reached that — which left
    the imbalance-batching path, and the per-batch prior correction downstream of it,
    with no coverage at all.

    Rather than commit a third file, this pools the negatives of both committed sets
    (deduplicated; the two overlap by 7 molecules) and keeps a handful of positives. The
    defaults are the smallest set that still trips the branch: 2 positives against 201
    negatives is 100.5:1, which plans two batches. Real molecules throughout, and no
    bioactivity source other than ChEMBL.

    The labels are of course no longer the measured ones -- discarding positives makes
    this a structural fixture for the batching machinery, not a dataset anything should
    be evaluated on.

    Parameters
    ----------
    n_positives, n_negatives : int
        Kept from the pooled sets. Their ratio must exceed 100 to trip the branch.

    Returns
    -------
    (list of str, list of int)
    """
    pos_a, y_a = load_reference_dataset()
    pos_b, y_b = load_imbalanced_dataset()
    positives = list(
        dict.fromkeys(
            [s for s, v in zip(pos_a, y_a) if v == 1]
            + [s for s, v in zip(pos_b, y_b) if v == 1]
        )
    )
    negatives = list(
        dict.fromkeys(
            [s for s, v in zip(pos_a, y_a) if v == 0]
            + [s for s, v in zip(pos_b, y_b) if v == 0]
        )
    )
    if n_positives > len(positives) or n_negatives > len(negatives):
        raise ValueError(
            f"only {len(positives)} positives and {len(negatives)} negatives available"
        )
    smiles = positives[:n_positives] + negatives[:n_negatives]
    y = [1] * n_positives + [0] * n_negatives
    return smiles, y
