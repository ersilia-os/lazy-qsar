import tempfile

import numpy as np

from lazyqsar.base.randomforest import BaseRFArtifact, BaseRFClassifier


def make_clf_data(n=400, p=20, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.4 > 0).astype(int)
    return X, y


def make_imbalanced_clf_data(n=1000, p=20, pos_frac=0.1, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3
    threshold = np.quantile(signal, 1.0 - pos_frac)
    y = (signal >= threshold).astype(int)
    return X, y


def test_rf_classifier_predict_proba_shape():
    X, y = make_clf_data()
    clf = BaseRFClassifier()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_rf_classifier_calibrated_learns_cutoff():
    X, y = make_clf_data()
    clf = BaseRFClassifier(calibrated=True)
    clf.fit(X, y)
    assert hasattr(clf, "decision_cutoff_")
    assert clf.decision_cutoff_source_ in {"oof_balanced_accuracy", "default_0.5"}


def test_rf_classifier_save_load_artifact_roundtrip():
    X, y = make_clf_data()
    clf = BaseRFClassifier()
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        clf.save(d)
        artifact = BaseRFArtifact.load(d)
        proba = artifact.run(X)
        preds = artifact.predict(X)
    assert proba.shape == (len(X), 2)
    assert preds.shape == (len(X),)


def test_rf_classifier_switches_to_balanced_subsample_on_imbalance():
    X, y = make_imbalanced_clf_data(n=1200, pos_frac=0.05)
    clf = BaseRFClassifier(calibrated=False)
    clf.fit(X, y)
    assert clf.imbalance_ratio_ > 3.0
    assert clf.class_weight_ == "balanced_subsample"


def test_rf_classifier_imbalanced_predict_proba_is_finite():
    X, y = make_imbalanced_clf_data(n=1200, pos_frac=0.05)
    clf = BaseRFClassifier(calibrated=True)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
