"""``lazyqsar fit`` then ``lazyqsar predict``, actually run.

``tests/unit/test_cli_argparse.py`` covers the argument surface and deliberately patches
every command body, so until now nothing executed a CLI command at all. The CLI's whole
job is file I/O -- read a directory of CSVs, write a CSV of predictions -- and that was
the part with no test: reading the header, picking the SMILES column, naming the output
columns, and the exit code.

Stubbed descriptors, so this is about the plumbing rather than chemistry; ``tests/chem``
covers the real featurizers. The fit is small on purpose -- the assertions are about the
shape and content of two files, not about model quality.
"""

import contextlib
import io
import sys

import numpy as np
import pytest

from _helpers.smiles import make_smiles

from lazyqsar.cli.main import main as cli_main

pytest.importorskip("sklearn")

N_TRAIN = 80
N_QUERY = 12


def _write_task(data_dir, name, smiles, seed):
    """One task CSV in the layout `fit` documents: SMILES first, label second."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, len(smiles))
    y[:10] = 1
    y[-10:] = 0
    (data_dir / f"{name}.csv").write_text(
        "smiles,bin\n" + "".join(f"{s},{int(v)}\n" for s, v in zip(smiles, y))
    )


def _predict_parser():
    """The live ``predict`` subparser, captured from ``main`` rather than rebuilt."""
    import argparse

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        for action in self._actions:
            if getattr(action, "choices", None) and hasattr(action.choices, "keys"):
                captured["sub"] = action.choices
        return real_parse(self, *a, **k)

    import sys as _sys

    argv, _sys.argv = _sys.argv, ["lazyqsar", "predict", "--help"]
    argparse.ArgumentParser.parse_args = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(SystemExit):
            cli_main()
    finally:
        argparse.ArgumentParser.parse_args = real_parse
        _sys.argv = argv
    return captured["sub"]["predict"]


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["lazyqsar", *argv])
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main()


@pytest.fixture
def fitted_via_cli(tmp_path, monkeypatch, stub_descriptors):
    """Fit two tasks through ``lazyqsar fit`` and return the paths it wrote.

    ``stub_descriptors`` is function-scoped, so the stub stays registered for the whole
    test that uses this fixture -- which is what lets `predict` read back a checkpoint the
    stub wrote.
    """
    stub_descriptors("morgan")
    data = tmp_path / "data"
    data.mkdir()
    smiles = make_smiles(N_TRAIN)
    _write_task(data, "alpha", smiles, seed=1)
    _write_task(data, "beta", smiles, seed=2)
    models = tmp_path / "models"

    _run(
        monkeypatch,
        "fit",
        "--task",
        "classification",
        "--input",
        str(data),
        "--output",
        str(models),
        "--mode",
        "fast",
    )
    return {"models": models, "smiles": smiles, "tmp": tmp_path}


def test_fit_writes_one_checkpoint_per_task(fitted_via_cli):
    """The contract `--input` documents: one CSV in, one task directory out."""
    models = fitted_via_cli["models"]
    assert models.is_dir()
    assert sorted(p.name for p in models.iterdir() if p.is_dir()) == ["alpha", "beta"]
    for task in ("alpha", "beta"):
        assert (models / task / "metadata.json").is_file(), (
            f"{task} has no metadata.json, so nothing downstream can read its cutoff"
        )


@pytest.mark.parametrize("predict_type", ["proba", "rank", "binary"])
def test_predict_writes_a_row_per_input_molecule(
    fitted_via_cli, tmp_path, monkeypatch, predict_type
):
    """The output CSV is the deliverable, so assert on the file, not on a return value."""
    import pandas as pd

    query = list(fitted_via_cli["smiles"][:N_QUERY])
    in_csv = tmp_path / "query.csv"
    out_csv = tmp_path / f"out_{predict_type}.csv"
    in_csv.write_text("smiles\n" + "".join(f"{s}\n" for s in query))

    _run(
        monkeypatch,
        "predict",
        "--input",
        str(in_csv),
        "--model",
        str(fitted_via_cli["models"]),
        "--output",
        str(out_csv),
        "--predict_type",
        predict_type,
    )

    assert out_csv.is_file(), "predict exited without writing its --output"
    frame = pd.read_csv(out_csv)
    assert len(frame) == N_QUERY, (
        f"{len(frame)} rows written for {N_QUERY} molecules; the CSV must line up with "
        "its input row for row"
    )
    # One column per fitted task, named after it, plus whatever identifier columns the
    # writer prepends. Both tasks must appear: a silently single-task output is the
    # failure mode that matters here.
    for task in ("alpha", "beta"):
        assert any(task in c for c in frame.columns), (
            f"no column for task {task!r} in {list(frame.columns)}"
        )
    values = frame.select_dtypes("number").to_numpy()
    assert np.isfinite(values).all(), "the CLI wrote non-finite predictions"
    if predict_type == "binary":
        assert set(np.unique(values)) <= {0, 1}, "binary output is not 0/1"
    else:
        assert (values >= 0).all() and (values <= 1).all(), (
            f"{predict_type} is out of [0, 1]"
        )


def test_predict_refuses_an_unknown_output_type(fitted_via_cli, tmp_path, monkeypatch):
    """Argparse rejects it before any model is loaded, and names the valid choices.

    ``--predict_type``'s metavar is ``TYPE``, so ``--help`` does not show the choice list
    and the option looks unvalidated. It is not: the parser carries ``choices``, so a typo
    costs nothing and the error tells the user what to type instead. Pinned because the
    cheap way to add a new output is to widen the API and forget the parser, which would
    turn this clean exit into a ValueError from inside the ensemble.
    """
    query = list(fitted_via_cli["smiles"][:4])
    in_csv = tmp_path / "q2.csv"
    in_csv.write_text("smiles\n" + "".join(f"{s}\n" for s in query))
    out_csv = tmp_path / "unused.csv"

    with pytest.raises(SystemExit) as exit_info:
        _run(
            monkeypatch,
            "predict",
            "--input",
            str(in_csv),
            "--model",
            str(fitted_via_cli["models"]),
            "--output",
            str(out_csv),
            "--predict_type",
            "not_an_output",
        )
    assert exit_info.value.code == 2, (
        "an unusable argument should exit 2, as argparse does"
    )
    assert not out_csv.exists(), "predict wrote an output file for an invalid request"


def test_every_output_the_api_accepts_is_offered_by_the_cli():
    """The parser's choices and the API's accepted outputs must not drift apart.

    Two lists of the same thing in two files is how ``--predict_type rank`` would come to
    be rejected by the CLI months after the API grew it.
    """
    from lazyqsar.ensemble.combine import OUTPUT_NAMES

    parser = _predict_parser()
    action = next(a for a in parser._actions if "--predict_type" in a.option_strings)
    assert set(action.choices) == set(OUTPUT_NAMES), (
        f"CLI offers {sorted(action.choices)} but the ensemble produces "
        f"{sorted(OUTPUT_NAMES)}"
    )


def test_fit_refuses_to_overwrite_an_existing_output_directory(
    tmp_path, monkeypatch, stub_descriptors
):
    """`fit` raises rather than merging into a directory that already holds a model.

    A merge would leave a checkpoint whose task directories came from different runs, and
    whose metadata therefore describes neither.
    """
    data = tmp_path / "data2"
    data.mkdir()
    _write_task(data, "alpha", make_smiles(40), seed=3)
    existing = tmp_path / "already_there"
    existing.mkdir()

    stub_descriptors("morgan")
    with pytest.raises(FileExistsError):
        _run(
            monkeypatch,
            "fit",
            "--task",
            "classification",
            "--input",
            str(data),
            "--output",
            str(existing),
            "--mode",
            "fast",
        )
