"""CDDD inference without the checkpoints: what each batch does and what the fallback accepts.

The encoder session and the ChEMBL index are stand-ins, so nothing here downloads the
431 MB of CDDD files or needs FPSim2. What is tested is the code around them, which is
where the bugs were.

Needs RDKit (for preprocessing).
"""

import numpy as np
import pytest

from lazyqsar.descriptors import cddd

# Valid and in CDDD's domain, so preprocessing keeps them.
IN_DOMAIN = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "c1ccc2[nH]ccc2c1"]


class _Session:
    """Returns each molecule's first token id as its embedding; can fail on one batch."""

    def __init__(self, fail_on_batch=None):
        self.calls = 0
        self.fail_on_batch = fail_on_batch

    def run(self, _, feeds):
        batch = self.calls
        self.calls += 1
        if batch == self.fail_on_batch:
            raise RuntimeError("simulated ONNX failure")
        seq = feeds["Input/Placeholder:0"]
        return [np.repeat(seq[:, 1:2].astype(np.float32), 512, axis=1)]


def _model(session, batch_size=128):
    model = cddd.InferenceModel.__new__(cddd.InferenceModel)
    model.hparams = cddd.HParams(batch_size=batch_size)
    model.encoder_session = session
    return model


def test_the_pipeline_does_not_preprocess_a_second_time(monkeypatch):
    """`seq_to_emb` preprocesses once; the pipeline takes its input as given.

    cddd-onnx preprocessed again inside the pipeline. A molecule dropped by that second
    pass misaligned every row of the call.
    """
    calls = []
    real = cddd.preprocess_smiles
    monkeypatch.setattr(cddd, "preprocess_smiles", lambda s: calls.append(s) or real(s))
    _model(_Session()).seq_to_emb(IN_DOMAIN)
    assert calls == IN_DOMAIN


def test_a_failing_batch_leaves_only_its_own_rows_nan():
    """One error used to make the whole call NaN, and the ChEMBL fallback then replaced
    every molecule with a neighbour's embedding."""
    smiles = IN_DOMAIN * 2  # six molecules, batches of two
    X = _model(_Session(fail_on_batch=1), batch_size=2).seq_to_emb(smiles)
    assert np.isnan(X).all(axis=1).tolist() == [False, False, True, True, False, False]


def test_out_of_domain_rows_stay_nan_and_aligned():
    smiles = [IN_DOMAIN[0], "C", IN_DOMAIN[1]]  # "C" fails CDDD's heavy-atom filter
    X = _model(_Session()).seq_to_emb(smiles)
    assert np.isnan(X).all(axis=1).tolist() == [False, True, False]


class _Engine:
    def __init__(self, hits):
        self.hits = hits
        self.thresholds = []

    def top_k(self, smiles, k, threshold, metric, n_workers):
        self.thresholds.append(threshold)
        return [h for h in self.hits if h[1] >= threshold]


@pytest.mark.parametrize("similarity, expected", [(0.69, None), (0.7, 5), (0.95, 5)])
def test_the_fallback_ignores_neighbours_below_the_threshold(similarity, expected):
    """`similarity_threshold=0.7` existed but was never used: a neighbour at Tanimoto 0.4
    still donated its embedding."""
    nn = cddd.ChemblNearestNeighbour.__new__(cddd.ChemblNearestNeighbour)
    nn.similarity_threshold = 0.7
    nn.fpe = _Engine([(5, similarity)])
    assert nn.highest_similarity("CCO") == expected
    assert nn.fpe.thresholds == [0.7]


def test_the_chembl_index_is_loaded_once(monkeypatch):
    """It was rebuilt (~0.5 s) for every predict chunk with an out-of-domain molecule."""
    built = []

    class _Neighbours:
        def __init__(self):
            built.append(1)

        def highest_similarity(self, smiles):
            return None

    monkeypatch.setattr(cddd, "ChemblNearestNeighbour", _Neighbours)
    descriptor = cddd.ContinuousDataDrivenDescriptor.__new__(
        cddd.ContinuousDataDrivenDescriptor
    )
    descriptor.model = _model(_Session())
    descriptor.smiles_indexed = []
    descriptor._chembl = None
    for _ in range(3):
        descriptor.transform([IN_DOMAIN[0], "C"])
    assert len(built) == 1
