"""The numpy-only half of the reference-library selection.

These functions are maintainer tooling, not shipped code, but they decide which molecules
become the reference library that every ``predict_rank`` percentile is quoted against. A
fault here produces a set that looks completely healthy -- right row count, right file size,
plausible report -- and is wrong, which is the one failure mode nothing downstream can
detect. Two such faults were found during the first build; both are pinned below.

Base tier on purpose: ``scripts/reference/picking.py`` imports nothing but numpy, so this
runs on an install with neither RDKit nor bblean.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from reference.picking import allocate, order_cluster, tanimoto_to  # noqa: E402


def _random_fps(n, seed=0, n_bits=2048):
    rng = np.random.default_rng(seed)
    return np.packbits(rng.integers(0, 2, (n, n_bits), dtype=np.uint8), axis=1)


# --------------------------------------------------------------------- tanimoto


def test_tanimoto_of_a_fingerprint_with_itself_is_one():
    fps = _random_fps(8)
    assert tanimoto_to(fps, fps[3])[3] == pytest.approx(1.0)


def test_tanimoto_matches_the_definition():
    a = np.packbits(np.array([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8), axis=1)
    b = np.packbits(np.array([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=np.uint8), axis=1)
    # |A and B| = 1, |A or B| = 3
    assert tanimoto_to(a, b[0])[0] == pytest.approx(1 / 3)


def test_two_empty_fingerprints_do_not_divide_by_zero():
    z = np.zeros((2, 4), dtype=np.uint8)
    assert list(tanimoto_to(z, z[0])) == [0.0, 0.0]


# --------------------------------------------------------------------- order_cluster


def test_maxmin_never_returns_the_same_molecule_twice():
    """The first build shipped 50,000 rows holding 33,797 distinct molecules.

    Masking a pick by writing ``inf`` into the running minimum does not survive the next
    ``np.minimum(min_sim, new, out=min_sim)``, which restores it to a finite value and makes
    an already-picked position selectable again. The median within-set nearest-neighbour
    Tanimoto was 1.000 -- every molecule had an identical twin -- and nothing about the
    output's shape gave it away.
    """
    fps = _random_fps(300)
    ids = np.arange(300)
    out = order_cluster(fps, ids, medoid_id=7, limit=120)
    assert len(out) == 120
    assert len(np.unique(out)) == 120


def test_maxmin_starts_from_the_medoid():
    fps = _random_fps(50)
    ids = np.arange(50)
    assert order_cluster(fps, ids, medoid_id=11, limit=5)[0] == 11


def test_a_limit_beyond_the_cluster_returns_every_member_once():
    fps = _random_fps(40)
    ids = np.arange(40)
    out = order_cluster(fps, ids, medoid_id=0, limit=500)
    assert sorted(out) == list(range(40))


def test_member_ids_are_returned_not_positions():
    """`order_cluster` indexes into the full fingerprint array, so it must map back."""
    fps = _random_fps(100)
    ids = np.array([90, 91, 92, 93, 94], dtype=np.int64)
    out = order_cluster(fps, ids, medoid_id=92, limit=3)
    assert set(out) <= set(ids.tolist())


def test_an_empty_limit_returns_nothing():
    fps = _random_fps(10)
    assert len(order_cluster(fps, np.arange(10), medoid_id=0, limit=0)) == 0


def test_maxmin_is_deterministic():
    fps = _random_fps(200)
    ids = np.arange(200)
    a = order_cluster(fps, ids, medoid_id=5, limit=60)
    b = order_cluster(fps, ids, medoid_id=5, limit=60)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------- allocate


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_allocation_totals_exactly_the_requested_count(alpha):
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 500, 1000)
    assert allocate(sizes, 50_000, alpha).sum() == 50_000


def test_no_cluster_is_allocated_more_than_it_holds():
    sizes = np.array([3, 3, 100_000])
    alloc = allocate(sizes, 50_000, 0.5)
    assert (alloc <= sizes).all()
    assert alloc.sum() == 50_000


def test_proportional_allocation_starves_small_clusters_and_sqrt_does_not():
    """Why ALPHA is 0.5 rather than 1.0.

    Measured on the real strata: at alpha=1 fifty-two of ten thousand strata received no
    representative at all, which is chemistry the reference library would simply not
    describe. At alpha=0.5 none did.
    """
    rng = np.random.default_rng(0)
    sizes = np.maximum(1, (rng.pareto(1.2, 5_000) * 40).astype(int))
    starved_proportional = int((allocate(sizes, 50_000, 1.0) == 0).sum())
    starved_sqrt = int((allocate(sizes, 50_000, 0.5) == 0).sum())
    assert starved_proportional > starved_sqrt
    assert starved_sqrt == 0


def test_a_larger_cluster_never_gets_fewer_slots():
    sizes = np.array([10, 100, 1000, 10_000])
    alloc = allocate(sizes, 5_000, 0.5)
    assert list(alloc) == sorted(alloc)


def test_allocation_is_deterministic():
    rng = np.random.default_rng(1)
    sizes = rng.integers(1, 200, 500)
    assert np.array_equal(allocate(sizes, 10_000, 0.5), allocate(sizes, 10_000, 0.5))


def test_asking_for_more_than_exists_is_an_error():
    with pytest.raises(ValueError, match="cannot allocate"):
        allocate(np.array([5, 5]), 50)
