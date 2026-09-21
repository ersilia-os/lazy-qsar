"""Identity and paths for the LazyQSAR reference library build.

One source of truth for what is being built, where the inputs live, and where the
staged outputs go. Imported by every script under ``scripts/``; imported by nothing
inside ``lazyqsar/``.

The published bundle is immutable: any change to the molecule set or to any descriptor
matrix means a new ``REFERENCE_ID``, never an edit in place. Clients cache by filename
and have no way to notice an object that changed underneath them.
"""

from __future__ import annotations

import os
import pathlib

# --- identity -------------------------------------------------------------------

REFERENCE_ID = "lazyqsar_reference_v1"
DEFAULT_N = 50_000

# Headroom over DEFAULT_N. Stage D emits this many picks so the descriptor stage can
# drop a straggler without re-running selection. 2% of 50k is 1000 spare rows against a
# measured post-gate failure rate of ~0.
HEADROOM = 1.02

# The source library, from the eosquality project.
SOURCE_LIBRARY_ID = "ersilia_reference_library_v0"
SOURCE_LIBRARY_URL = (
    "https://eosvc-public.s3.amazonaws.com/eosquality/libraries/"
    "ersilia_reference_library_v0.csv"
)

# Where the published bundle will live. eosvc maps the repo name to the S3 prefix, so
# `eosvc upload --path data/reference/<REFERENCE_ID>` lands here.
PUBLIC_BASE_URL = "https://eosvc-public.s3.amazonaws.com/lazy-qsar/reference/"

# --- clustering and selection defaults -------------------------------------------

# Slots per cluster are proportional to size ** ALPHA.
#   1.0 -> proportional: a faithful subsample, density-preserving.
#   0.5 -> sqrt: the standard compromise. A cluster 10,000x larger contributes ~100x
#          more representatives, so combinatorial series are compressed but the crowded
#          regions real screening compounds occupy stay denser than rare chemotypes.
#   0.0 -> equal: one representative per cluster, maximum diversity, flattest density.
# The reference set *is* the ECDF a rank is read against, so this exponent is the single
# knob trading "beats 99% of drug-like compounds" against "beats 99% of chemotypes".
ALPHA = 0.5

# BitBIRCH threshold, fixed rather than searched. Measured over the whole library:
#
#   threshold   clusters      % of n   mean   max   singletons
#   0.65        1,193,292     88.1%    1.14    29       90.5%
#   0.55        1,078,515     79.6%    1.26    64       84.9%
#   0.45          951,089     70.2%    1.42   160       79.4%
#
# There is no threshold at which this library forms large clusters -- it was already
# diversity-curated upstream. So BitBIRCH's job here is *not* stratification, it is
# collapsing the ~10% of molecules that have a near-duplicate, which is what stops an
# analogue pair from occupying two reference slots. 0.65 is the tightest of the three and
# the most conservative about calling two molecules the same chemotype.
BITBIRCH_THRESHOLD = 0.65
BITBIRCH_BRANCHING_FACTOR = 50

# Second level: density strata in physchem space, over the BitBIRCH medoids. This is what
# ALPHA acts on -- k-means Voronoi cells are roughly equal in volume, so their populations
# track how crowded that region of chemical space is. ~119 medoids per stratum at 1.19M.
N_STRATA = 10_000
PCA_COMPONENTS = 50
KMEANS_BATCH_SIZE = 8192

# The physchem matrix has tails out to ~49 sigma; clip before PCA so a handful of outliers
# do not each capture their own stratum.
CLIP_SIGMA = 10.0

# Morgan parameters for the clustering space. These MUST match
# lazyqsar/descriptors/morgan.py (radius 3, 2048 bits) so the geometry the clusters are
# built in is the geometry a fitted model sees. Binary, not counts: BitBIRCH's iSIM
# formalism is defined over bits.
MORGAN_RADIUS = 3
MORGAN_N_BITS = 2048

SEED = 42

# --- paths ------------------------------------------------------------------------


def eosquality_home() -> pathlib.Path:
    """Local eosquality cache, populated by ``eosquality download``."""
    return pathlib.Path(
        os.environ.get("EOSQUALITY_HOME", pathlib.Path.home() / ".eosquality")
    )


def source_csv() -> pathlib.Path:
    return eosquality_home() / "libraries" / f"{SOURCE_LIBRARY_ID}.csv"


def source_index_dir() -> pathlib.Path:
    return eosquality_home() / "indices" / SOURCE_LIBRARY_ID


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def staging_dir() -> pathlib.Path:
    """Gitignored staging area that ``eosvc upload`` publishes from."""
    return repo_root() / "data" / "reference" / REFERENCE_ID


def work_dir() -> pathlib.Path:
    """Intermediates that are never published."""
    return staging_dir() / "_work"


def descriptor_filename(descriptor: str, n: int = DEFAULT_N) -> str:
    """Published name for one descriptor matrix.

    The tier is in the filename rather than a subfolder because tiers nest: a later
    ``*_n100000.h5`` sits in the same prefix, and a cached file is never ambiguous about
    how many rows it holds.
    """
    return f"{descriptor}_n{n}.h5"


def smiles_filename(n: int = DEFAULT_N) -> str:
    return f"reference_smiles_n{n}.csv"
