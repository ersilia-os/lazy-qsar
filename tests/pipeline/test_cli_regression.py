"""``lazyqsar fit --task regression`` then ``lazyqsar predict``, actually run.

The regression counterpart of ``test_cli_end_to_end``: stubbed descriptors, small fit,
assertions on the files. The target is a function of the stub's features, so the
predictions can be checked for signal as well as shape. Also pinned:

- ``predict`` reads the task from the checkpoint and defaults to ``value``,
- a classification-only ``--predict_type`` on a regression model is a clean exit 1,
- a model directory mixing tasks is refused rather than half-predicted,
- the CLI output agrees with ``LazyQSAR.load(...).predict``.
"""

import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd
import pytest

from _helpers.smiles import make_smiles

from lazyqsar.cli.main import main as cli_main

pytest.importorskip("sklearn")

N_TRAIN = 90


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["lazyqsar", *argv])
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main()


@pytest.fixture
def fitted(tmp_path, monkeypatch, stub_descriptors):
    stub_descriptors("morgan")
    from lazyqsar.registry import get_descriptor_type

    smiles = make_smiles(N_TRAIN)
    F = get_descriptor_type("morgan")().transform(smiles)
    y = 3.0 + 2.0 * F[:, 0] - F[:, 1]
    data = tmp_path / "data"
    data.mkdir()
    (data / "solubility.csv").write_text(
        "smiles,value\n" + "".join(f"{s},{v:.6f}\n" for s, v in zip(smiles, y))
    )
    models = tmp_path / "models"
    _run(
        monkeypatch,
        "fit",
        "--task",
        "regression",
        "--input",
        str(data),
        "--output",
        str(models),
        "--mode",
        "fast",
    )
    query = tmp_path / "query.csv"
    query.write_text("smiles\n" + "".join(f"{s}\n" for s in smiles))
    return {"models": models, "smiles": smiles, "y": y, "query": query, "tmp": tmp_path}


def test_fit_writes_a_regression_checkpoint(fitted):
    meta = json.loads((fitted["models"] / "solubility" / "metadata.json").read_text())
    assert meta["task"] == "regression"
    assert (fitted["models"] / "solubility" / "morgan").is_dir()


def test_predict_defaults_to_value(fitted, monkeypatch):
    out = fitted["tmp"] / "out.csv"
    _run(
        monkeypatch,
        "predict",
        "--input",
        str(fitted["query"]),
        "--model",
        str(fitted["models"]),
        "--output",
        str(out),
    )
    df = pd.read_csv(out)
    assert list(df.columns) == ["solubility"]
    assert len(df) == N_TRAIN
    pred = df["solubility"].to_numpy()
    assert np.isfinite(pred).all()
    assert np.corrcoef(pred, fitted["y"])[0, 1] > 0.8


def test_cli_agrees_with_the_python_api(fitted, monkeypatch):
    from lazyqsar.api.predict import predict
    from lazyqsar.qsar import LazyQSAR

    R, header = predict(model_dir=str(fitted["models"]), smiles=fitted["smiles"])
    assert header == ["solubility"]
    with contextlib.redirect_stdout(io.StringIO()):
        direct = LazyQSAR.load(str(fitted["models"] / "solubility")).predict(
            fitted["smiles"]
        )
    np.testing.assert_allclose(R[:, 0], direct, rtol=0, atol=1e-6 * fitted["y"].std())


def test_classification_output_on_a_regression_model_exits_one(
    fitted, monkeypatch, capsys
):
    with pytest.raises(SystemExit) as exc:
        _run(
            monkeypatch,
            "predict",
            "--input",
            str(fitted["query"]),
            "--model",
            str(fitted["models"]),
            "--output",
            str(fitted["tmp"] / "x.csv"),
            "--predict_type",
            "proba",
        )
    assert exc.value.code == 1
    assert "not available for regression" in capsys.readouterr().err


def test_mixed_task_directory_is_refused(fitted, monkeypatch, capsys):
    # A sibling checkpoint with no `task` key reads as classification.
    legacy = fitted["models"] / "legacy_classifier"
    legacy.mkdir()
    (legacy / "metadata.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        _run(
            monkeypatch,
            "predict",
            "--input",
            str(fitted["query"]),
            "--model",
            str(fitted["models"]),
            "--output",
            str(fitted["tmp"] / "x.csv"),
        )
    assert exc.value.code == 1
    assert "mix classification and regression" in capsys.readouterr().err


def test_non_numeric_target_is_reported_with_its_line(tmp_path):
    from lazyqsar.api.classifier_fit import get_task_data

    (tmp_path / "t.csv").write_text("smiles,value\nCCO,1.5\nCCC,high\n")
    with pytest.raises(ValueError, match="line 3: expected a numeric target"):
        get_task_data(str(tmp_path), "t", task="regression")
