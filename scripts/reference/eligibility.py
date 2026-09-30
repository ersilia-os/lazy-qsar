"""Which molecules may enter the reference library, and when each test is paid.

Two tiers of test, deliberately separated by cost:

**Whole-library (stage A)** -- parse and deduplicate only, 0.086 ms/mol. Over 1,355,109
rows that is about two minutes.

**Pick-time (stage D)** -- the CDDD gate, 0.342 ms/mol. Run only on the ~51,000 molecules
actually chosen as cluster representatives, so it costs ~20 seconds instead of the 7.7
minutes a whole-library pass would. A representative that fails is replaced by another
member of *its own cluster*, so a chemotype is never dropped merely because its most
central molecule happens to be one CDDD cannot encode.

Note what is *not* here: the CDDD encoder itself, at 31.4 ms/mol, is never run over the
library. That would be 11.8 hours. It runs once, over the chosen 50,000, in the descriptor
stage.

The CDDD gate matters more than it looks. ``ContinuousDataDrivenDescriptor.is_applicable``
(``lazyqsar/descriptors/cddd.py:463-470``) refuses a dataset when more than 0.1% of it
fails these predicates, and the library's own failure rate is 1.52% -- so a *random* 50k
slice would make CDDD inapplicable to the reference set. Selecting with replacement drives
the final set to 0% by construction.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Iterable, Iterator

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Restated from the local in `ContinuousDataDrivenDescriptor.is_applicable`
# (lazyqsar/descriptors/cddd.py:464). Kept in sync by
# tests/... -- see scripts/README.md; there is no importable constant to read.
MAX_SMILES_LEN = 150


def canonicalize(smiles: str) -> str | None:
    """Canonical SMILES under the *lazyqsar* RDKit pin, or ``None`` if unparseable.

    Deliberately not eosquality's RDKit: the descriptors that will be computed from these
    strings run under this package's pin, and a canonical form is only meaningful relative
    to the toolkit that produced it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def parse_and_dedup(
    smiles: Iterable[str], processes: int | None = None, chunksize: int = 2048
) -> tuple[list[str], list[int], dict[str, int]]:
    """Canonicalise and deduplicate, preserving input order.

    ``imap`` rather than ``imap_unordered`` because the first occurrence of a duplicate is
    the one kept, which is only deterministic if order is preserved.

    Callers must guard their entry point with ``if __name__ == "__main__":``. macOS and
    Windows start workers with *spawn*, which re-imports the parent module in every child;
    without the guard each child re-enters this function and forks again, and the run dies
    in a fork bomb rather than in anything that names the cause.

    Returns
    -------
    canonical : list of str
        Canonical SMILES, deduplicated, in first-occurrence order.
    source_index : list of int
        Row of the source CSV each survivor came from. These index
        ``physchem_scaled.npy`` and the kNN graph too, which are aligned to the same file.
    counts : dict
        ``{"input", "unparseable", "duplicate", "kept"}``.
    """
    seen: set[str] = set()
    canonical: list[str] = []
    source_index: list[int] = []
    counts = {"input": 0, "unparseable": 0, "duplicate": 0, "kept": 0}

    with mp.Pool(processes) as pool:
        for row, smi in enumerate(pool.imap(canonicalize, smiles, chunksize=chunksize)):
            counts["input"] += 1
            if smi is None:
                counts["unparseable"] += 1
                continue
            if smi in seen:
                counts["duplicate"] += 1
                continue
            seen.add(smi)
            canonical.append(smi)
            source_index.append(row)
    counts["kept"] = len(canonical)
    return canonical, source_index, counts


def is_cddd_calculable(smiles: str) -> bool:
    """Whether CDDD encodes this molecule natively, rather than substituting a neighbour.

    ``ContinuousDataDrivenDescriptor.transform`` (``cddd.py:436-450``) replaces a molecule
    it cannot preprocess with the embedding of *a different molecule* found by FPSim2
    nearest-neighbour search. A reference row carrying another molecule's embedding is a
    silently wrong ECDF point, so those rows are excluded rather than substituted.

    The predicate is pure RDKit -- it never touches the 182 MB FPSim2 database -- because
    ``preprocess_smiles`` returning a non-string is exactly the condition that triggers the
    fallback upstream.
    """
    from lazyqsar.descriptors.cddd import preprocess_smiles

    if len(smiles) > MAX_SMILES_LEN:
        return False
    try:
        return isinstance(preprocess_smiles(smiles), str)
    except Exception:
        return False


def iter_cddd_calculable(smiles: Iterable[str]) -> Iterator[bool]:
    """``is_cddd_calculable`` over an iterable, importing the predicate once."""
    from lazyqsar.descriptors.cddd import preprocess_smiles

    for smi in smiles:
        if len(smi) > MAX_SMILES_LEN:
            yield False
            continue
        try:
            yield isinstance(preprocess_smiles(smi), str)
        except Exception:
            yield False
