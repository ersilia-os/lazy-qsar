"""CheMeleon's forward pass must not record an autograd graph.

``_CheMeleonFingerprint.__init__`` calls ``model.eval()``, which switches layers to inference
behaviour but does nothing about autograd. Without an explicit ``torch.no_grad()`` every
batch builds a graph that is discarded one line later, for a model that is never trained
here — wasted memory and time on the most expensive descriptor in the package, and invisible
in the output, since forward values are identical either way. That is exactly the kind of
change nothing else would catch.

Skipped rather than tiered. The ``deep`` tier was removed in 2940126, so the whole suite is
now collected with no ``-m`` filter and a module-scope ``import torch`` would fail collection
on the CI runner, which installs only the ``fit`` extra and RDKit. ``importorskip`` keeps this
file collectable everywhere and running wherever torch and chemprop are present.

The fingerprint object is built through ``__new__`` with a stub model, so this needs neither
the 100 MB CheMeleon checkpoint nor a network round trip.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="CheMeleon needs torch")
pytest.importorskip("chemprop", reason="CheMeleon needs chemprop")


@pytest.fixture
def recording_fingerprint():
    """A real ``_CheMeleonFingerprint`` whose model records the autograd state it sees."""
    from chemprop import featurizers

    from lazyqsar.descriptors.chemeleon import _CheMeleonFingerprint

    n_molecules, n_dim = 3, 4

    class _RecordingModel:
        device = torch.device("cpu")

        def __init__(self):
            self.grad_enabled_during_forward = None
            self.calls = 0

        def fingerprint(self, _bmg):
            self.grad_enabled_during_forward = torch.is_grad_enabled()
            self.calls += 1
            return torch.zeros((n_molecules, n_dim))

    obj = _CheMeleonFingerprint.__new__(_CheMeleonFingerprint)
    obj.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    obj.model = _RecordingModel()
    return obj, n_molecules, n_dim


def test_the_forward_pass_runs_with_autograd_off(recording_fingerprint):
    """The assertion. `eval()` alone does not do this."""
    obj, n_molecules, _ = recording_fingerprint
    assert torch.is_grad_enabled(), (
        "autograd is already off in this process, so the test cannot tell whether the "
        "code under test turned it off"
    )

    obj(["CCO", "c1ccccc1", "CC(=O)O"])

    assert obj.model.calls == 1
    assert obj.model.grad_enabled_during_forward is False, (
        "the CheMeleon forward pass recorded an autograd graph it immediately discards"
    )


def test_autograd_is_left_as_it_was_found(recording_fingerprint):
    """A context manager, not a global switch: the caller's state must survive."""
    obj, _, _ = recording_fingerprint
    obj(["CCO", "c1ccccc1", "CC(=O)O"])
    assert torch.is_grad_enabled(), "grad was disabled for the rest of the process"


def test_the_output_is_still_the_right_shape(recording_fingerprint):
    """Disabling autograd must not change what comes back."""
    obj, n_molecules, n_dim = recording_fingerprint
    out = obj(["CCO", "c1ccccc1", "CC(=O)O"])
    assert out.shape == (n_molecules, n_dim)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_unparseable_molecules_still_come_back_as_nan_rows(recording_fingerprint):
    """The NaN contract the predict path's candidate-confirm shortcut depends on.

    ``api.classifier_predict`` treats CheMeleon as NaN-faithful: a molecule RDKit cannot
    parse must produce an all-NaN row, so that re-checking only the NaN rows finds every
    unparseable one.
    """
    obj, _, n_dim = recording_fingerprint
    out = obj(["CCO", "not_a_molecule", "c1ccccc1", "((((", "CC(=O)O"])

    assert out.shape == (5, n_dim)
    assert np.isnan(out[1]).all() and np.isnan(out[3]).all()
    assert not np.isnan(out[[0, 2, 4]]).any()
