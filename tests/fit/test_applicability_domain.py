"""The applicability domain: reproducible to fit, and identical through ONNX.

Two properties, neither previously covered.

**Reproducibility.** Two fits of the same data must give the same scores. The ensemble's
veto compares an AD score against a fixed cutoff, so a model that scored a screening library
differently on each retrain would silently promote a different set of compounds.

A note on what that test does and does not prove. ``ApplicabilityDomain`` passes
``random_state`` to PCA, and the comment at that line says the seed is load-bearing because
``svd_solver="auto"`` resolves to ``"randomized"`` at these widths. The first half is true --
the solver *is* randomized here, which ``test_the_solver_here_really_is_randomized`` shows.
The second half does not follow: the score comes out of a Mahalanobis distance taken with the
precision matrix of the *projected* data, and that quantity is invariant to any orthogonal
change of basis within the retained subspace. Rotate the components and the distance is
unchanged. Measured across Morgan-like sparse binary, rank-deficient and tiny-n-wide-p
inputs, removing the seed changes nothing at all.

So the seed is insurance rather than a fix: it makes reproducibility a guarantee of the code
instead of a consequence of an algebraic accident that a future change to the distance metric
could remove. The test below therefore asserts the property, not the mechanism -- it would
still be the right test if the seed were removed, and it would catch a change to the metric
that made seeding matter.

**Round trip.** Fitting happens in sklearn and scoring happens in ONNX, and the fit-time
cutoff is derived from the sklearn side. If the two disagree, the veto fires on the wrong
rows.
"""

import json
import os

import numpy as np
import pytest

from lazyqsar.applicability import ApplicabilityDomain, ApplicabilityDomainArtifact

# Wide enough that PCA picks the randomized solver -- the whole point of the seeding test.
N_TRAIN, N_FEATURES = 300, 1000


@pytest.fixture(scope="module")
def train_X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float32)


@pytest.fixture(scope="module")
def query_X(train_X):
    rng = np.random.default_rng(1)
    return rng.normal(size=(25, N_FEATURES)).astype(np.float32)


def test_two_fits_of_the_same_data_score_identically(train_X, query_X):
    """The ambient seed is varied on purpose -- a fixed one would prove nothing."""
    np.random.seed(11)
    first = ApplicabilityDomain().fit(train_X).score(query_X)
    np.random.seed(97)
    second = ApplicabilityDomain().fit(train_X).score(query_X)

    assert np.array_equal(first, second), (
        "two fits of identical data gave different AD scores. The ensemble would veto a "
        "different set of descriptors on every retrain, which is invisible on held-out "
        "folds and shows up as a screening library that reorders itself between runs."
    )


def test_the_solver_here_really_is_randomized(train_X):
    """Pin the premise the seeding comment rests on.

    ``svd_solver="auto"`` resolving to the randomized solver is what makes seeding worth
    doing at all. Shown through the public API rather than a private attribute: an unseeded
    PCA at this width must give different components for different ambient state. (That the
    *AD score* is nonetheless unchanged is the point made in the module docstring.)
    """
    from sklearn.decomposition import PCA

    k = min(max(1, int(N_FEATURES**0.5)), 100, N_FEATURES, N_TRAIN - 1)
    X = (train_X - train_X.mean(0)) / (train_X.std(0) + 1e-12)

    np.random.seed(11)
    a = PCA(n_components=k, random_state=None).fit(X).components_
    np.random.seed(97)
    b = PCA(n_components=k, random_state=None).fit(X).components_

    assert not np.array_equal(a, b), (
        f"PCA is already deterministic at n={N_TRAIN}, p={N_FEATURES}, so the seeding test "
        "above no longer exercises the randomized solver. Raise N_FEATURES."
    )


def test_scores_are_probabilities(train_X, query_X):
    scores = ApplicabilityDomain().fit(train_X).score(query_X)
    assert scores.shape == (len(query_X),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_out_of_domain_scores_lower_than_in_domain(train_X):
    """The whole purpose: chemistry unlike the training set must score lower."""
    ad = ApplicabilityDomain().fit(train_X)
    in_domain = ad.score(train_X)
    out_of_domain = ad.score(train_X + 50.0)
    assert out_of_domain.mean() < in_domain.mean()


def test_an_all_nan_query_row_is_imputed_not_propagated(train_X, query_X):
    """Descriptors do return NaN; the AD must still produce a score for that row."""
    probe = query_X.copy()
    probe[0] = np.nan
    scores = ApplicabilityDomain().fit(train_X).score(probe)
    assert np.isfinite(scores).all()


def test_fitting_needs_at_least_two_samples(train_X):
    with pytest.raises(ValueError):
        ApplicabilityDomain().fit(train_X[:1])


# --------------------------------------------------------------------------- round trip


def test_onnx_artifact_agrees_with_the_sklearn_estimator(train_X, query_X, tmp_path):
    ad = ApplicabilityDomain().fit(train_X)
    directory = str(tmp_path / "ad")
    ad.save(directory)

    artifact = ApplicabilityDomainArtifact.load(directory)
    np.testing.assert_allclose(
        artifact.score(query_X),
        ad.score(query_X),
        atol=1e-5,
        err_msg=(
            "the exported graph disagrees with the estimator it came from, so the ensemble "
            "would veto on different rows than the fit-time cutoff was derived for"
        ),
    )


def test_saved_metadata_describes_the_fitted_model(train_X, tmp_path):
    ad = ApplicabilityDomain().fit(train_X)
    directory = str(tmp_path / "ad")
    ad.save(directory)

    with open(os.path.join(directory, "applicability_domain.json")) as f:
        meta = json.load(f)

    assert meta["n_components"] == ad.pca_.n_components_
    assert meta["n_features_in"] == N_FEATURES
    assert meta["n_cal_knots"] == len(ad.cal_knots_)
    assert meta["cal_min"] <= meta["cal_max"]
    assert meta["scaler"] == "standard"
    assert os.path.isfile(os.path.join(directory, "applicability_domain.onnx"))


def test_loading_without_metadata_is_an_error(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ApplicabilityDomainArtifact.load(str(empty))
