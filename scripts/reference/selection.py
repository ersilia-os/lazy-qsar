"""Choose exactly N reference molecules: cluster, apportion, then pick.

The reference set is the ECDF every ``predict_rank`` percentile is read against, so how
it is chosen *is* the calibration. Three steps:

1. **Cluster** with BitBIRCH over binary Morgan fingerprints. O(N), iSIM/Tanimoto, built
   for million-scale libraries.
2. **Apportion** exactly N slots across the clusters, proportional to ``size ** ALPHA``,
   using largest-remainder so the total is exact.
3. **Pick** within each cluster, medoid first and then MaxMin, skipping molecules the
   CDDD gate rejects and taking the next candidate from the same cluster instead.

Why an exponent rather than one-representative-per-cluster: a rank is a percentile, so
flattening cluster density flattens the distribution the percentile is quoted against.
``ALPHA = 0.5`` compresses giant combinatorial series without erasing the fact that
crowded drug-like regions are where real screening compounds actually live.

Determinism comes from fixing the input order (BitBIRCH is a BIRCH variant, and every
BIRCH is insertion-order dependent) and from using no RNG anywhere in the pick.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from . import config
from .picking import allocate, order_cluster, tanimoto_to

RDLogger.DisableLog("rdApp.*")


# --- fingerprints ------------------------------------------------------------------


def morgan_bits(smiles: list[str], packed: bool = True) -> np.ndarray:
    """Binary Morgan fingerprints, matching ``lazyqsar/descriptors/morgan.py``.

    Radius 3 and 2048 bits are taken from :mod:`config` rather than bblean's ``ecfp4``
    preset, which is radius 2 -- the clusters must live in the same geometry the fitted
    descriptor sees.

    Binary rather than counts because BitBIRCH's iSIM formalism is defined over bits.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_N_BITS
    )
    out = np.zeros((len(smiles), config.MORGAN_N_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:  # pragma: no cover - callers pass canonicalised input
            raise ValueError(f"row {i}: unparseable SMILES {smi!r}")
        out[i] = gen.GetFingerprintAsNumPy(mol)
    return np.packbits(out, axis=1) if packed else out


# --- clustering ---------------------------------------------------------------------


def cluster(
    packed_fps: np.ndarray, threshold: float, branching_factor: int | None = None
):
    """Fit BitBIRCH and return it. Input order is the caller's responsibility."""
    from bblean.bitbirch import BitBirch

    tree = BitBirch(
        threshold=threshold,
        branching_factor=branching_factor or config.BITBIRCH_BRANCHING_FACTOR,
    )
    tree.fit(packed_fps, input_is_packed=True, n_features=config.MORGAN_N_BITS)
    return tree


def cluster_medoids(packed_fps: np.ndarray, clusters: list[list[int]]) -> np.ndarray:
    """The most central member of each cluster.

    Computed directly rather than through ``BitBirch.get_medoids_mol_ids`` so the tree does
    not have to be kept alive or refitted. Cheap because the clusters are tiny: 90.5% are
    singletons and the largest holds 29 molecules, so the quadratic inner loop never sees a
    meaningful cluster.
    """
    out = np.empty(len(clusters), dtype=np.int64)
    for i, members in enumerate(clusters):
        if len(members) == 1:
            out[i] = members[0]
            continue
        ids = np.asarray(members, dtype=np.int64)
        fps = packed_fps[ids]
        # Mean similarity to the rest; the medoid maximises it.
        total = np.zeros(len(ids), dtype=np.float64)
        for j in range(len(ids)):
            total += tanimoto_to(fps, fps[j])
        out[i] = ids[int(np.argmax(total))]
    return out


# --- stratification -------------------------------------------------------------------


def stratify(
    X: np.ndarray,
    n_strata: int,
    seed: int = config.SEED,
    pca_components: int | None = None,
    log=print,
):
    """Partition rows of *X* into *n_strata* density-reflecting strata.

    BitBIRCH cannot supply these. Measured over the whole library, it returns 1,193,292
    clusters from 1,355,109 molecules at threshold 0.65 -- 90.5% singletons, largest 29 --
    because this library was already diversity-curated upstream. Cluster sizes that flat
    carry no density signal, so ``size ** ALPHA`` would be a no-op.

    k-means in physchem space does carry it: Voronoi cells are roughly equal in *volume*,
    so their populations vary with how crowded that region of chemical space is. That is
    exactly the quantity ``ALPHA`` tempers. Equal-frequency binning would not work -- it
    produces equal-sized strata by construction, which is the same no-op.

    PCA first, because k-means on 217 correlated, heavy-tailed descriptors spends its
    variance on a few dominant axes.

    Returns
    -------
    labels : ndarray of int
        Stratum index per row.
    seeds : ndarray of int
        Row index of the member nearest each stratum's centroid, or -1 if the stratum is
        empty.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    k = pca_components or config.PCA_COMPONENTS
    t = _now()
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    Z = pca.fit_transform(X).astype(np.float32)
    log(
        f"  PCA -> {Z.shape}, {pca.explained_variance_ratio_.sum():.1%} variance, {_now() - t:.0f}s"
    )

    t = _now()
    km = MiniBatchKMeans(
        n_clusters=n_strata,
        random_state=seed,
        batch_size=config.KMEANS_BATCH_SIZE,
        n_init=3,
        max_iter=100,
    )
    labels = km.fit_predict(Z)
    log(f"  MiniBatchKMeans k={n_strata:,} in {_now() - t:.0f}s")

    # The row of each stratum nearest its own centroid. Used to seed MaxMin, so the first
    # molecule taken from a stratum is a typical member of it rather than whichever row
    # happened to come first in the input order.
    seeds = np.full(n_strata, -1, dtype=np.int64)
    d2 = ((Z - km.cluster_centers_[labels]) ** 2).sum(axis=1)
    order = np.lexsort((d2, labels))
    first = np.searchsorted(labels[order], np.arange(n_strata), side="left")
    present = np.bincount(labels, minlength=n_strata) > 0
    seeds[present] = order[first[present]]
    return labels, seeds


def _now() -> float:
    import time

    return time.time()


# --- apportionment -------------------------------------------------------------------


__all__ = [
    "allocate",
    "cluster",
    "cluster_medoids",
    "morgan_bits",
    "order_cluster",
    "stratify",
    "tanimoto_to",
]
