"""Selection primitives that need only numpy.

Split out of :mod:`selection` so they can be imported -- and regression-tested -- without
RDKit or bblean. That matters more than tidiness: these three functions decide which
molecules become the reference library, and a fault in them produces a set that looks
entirely healthy (right row count, right file size, plausible report) while being wrong.
Both bugs found during the first build were here.
"""

from __future__ import annotations

import numpy as np

from . import config


def tanimoto_to(packed: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Tanimoto of every row of *packed* against one *query* row, on packed bits.

    ``np.bitwise_count`` popcounts in C, so this stays on the 256-byte packed form rather
    than unpacking to 2048 bytes per molecule.
    """
    inter = np.bitwise_count(packed & query).sum(axis=1, dtype=np.int32)
    union = np.bitwise_count(packed | query).sum(axis=1, dtype=np.int32)
    return np.divide(
        inter, union, out=np.zeros(len(packed), dtype=np.float64), where=union > 0
    )


def allocate(
    sizes: np.ndarray, n_total: int, alpha: float = config.ALPHA
) -> np.ndarray:
    """Split *n_total* slots across clusters, proportional to ``size ** alpha``.

    Largest-remainder apportionment, so the slots sum to *n_total* exactly rather than to
    whatever rounding produces. Ties break on cluster index, so the result is a pure
    function of the inputs.

    No cluster is allocated more slots than it has members; surplus from a cluster that is
    too small is redistributed over the rest in the same pass.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    if n_total > sizes.sum():
        raise ValueError(
            f"cannot allocate {n_total:,} slots across clusters holding "
            f"{sizes.sum():,} molecules"
        )

    alloc = np.zeros(len(sizes), dtype=np.int64)
    remaining = n_total
    active = np.ones(len(sizes), dtype=bool)

    # Redistribution can itself overflow another cluster, so iterate to a fixed point.
    while remaining > 0 and active.any():
        weights = np.where(active, sizes.astype(np.float64) ** alpha, 0.0)
        total = weights.sum()
        if total <= 0:
            break
        quota = remaining * weights / total
        floor = np.floor(quota).astype(np.int64)
        short = remaining - floor.sum()
        if short > 0:
            # Largest remainder; index order breaks ties deterministically.
            remainder = quota - floor
            order = np.lexsort((np.arange(len(sizes)), -remainder))
            floor[order[:short]] += 1

        proposed = alloc + floor
        capped = np.minimum(proposed, sizes)
        if (capped == alloc).all():
            break  # no progress possible
        remaining -= int((capped - alloc).sum())
        alloc = capped
        active = alloc < sizes

    return alloc


def order_cluster(
    packed_fps: np.ndarray, member_ids: np.ndarray, medoid_id: int, limit: int
) -> np.ndarray:
    """Order a cluster's members medoid-first, then by MaxMin, up to *limit*.

    The medoid leads so the first representative of a chemotype is a typical member rather
    than an outlier; MaxMin then spreads the remaining picks across the cluster instead of
    clustering them around its centre.

    Large clusters are truncated to a deterministic candidate pool -- the members nearest
    the medoid -- before MaxMin, because MaxMin is O(pool x picks) and a few clusters hold
    six figures of molecules. Truncation only removes candidates that would have been
    picked last, if at all.
    """
    if limit >= len(member_ids):
        limit = len(member_ids)
    if limit <= 0:
        return np.empty(0, dtype=np.int64)

    member_ids = np.asarray(member_ids, dtype=np.int64)
    pool_cap = max(4 * limit, 512)
    if len(member_ids) > pool_cap:
        sim_to_medoid = tanimoto_to(packed_fps[member_ids], packed_fps[medoid_id])
        keep = np.argsort(-sim_to_medoid, kind="stable")[:pool_cap]
        member_ids = member_ids[np.sort(keep)]

    fps = packed_fps[member_ids]
    start = (
        int(np.flatnonzero(member_ids == medoid_id)[0])
        if medoid_id in member_ids
        else 0
    )

    picked = [start]
    # `chosen` is kept separate from `min_sim` on purpose. Masking a pick by writing inf
    # into `min_sim` does not survive the next update: `np.minimum(min_sim, new)` restores
    # it to a finite value, so already-picked positions become selectable again and the
    # same molecule is returned several times.
    chosen = np.zeros(len(fps), dtype=bool)
    chosen[start] = True
    min_sim = tanimoto_to(fps, fps[start])
    while len(picked) < limit:
        nxt = int(np.argmin(np.where(chosen, np.inf, min_sim)))
        if chosen[nxt]:
            break  # every candidate taken
        chosen[nxt] = True
        picked.append(nxt)
        np.minimum(min_sim, tanimoto_to(fps, fps[nxt]), out=min_sim)
    return member_ids[np.asarray(picked, dtype=np.int64)]
