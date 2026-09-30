"""The CLI's argument surface: defaults, exit codes, and what it forwards.

``lazyqsar fit`` and ``lazyqsar predict`` are what the Ersilia pipelines shell out to, so
their defaults *are* deployed behaviour -- dropping ``--mode`` has to keep meaning ``slow``,
and dropping ``--predict_type`` has to keep meaning ``proba``.

Everything here patches the function the command dispatches to, so nothing is fitted and
nothing is installed. ``subprocess.check_call`` is patched throughout and asserted never to
run: a test that pip-installs into the developer's environment is not a test.
"""

import subprocess
import sys

import pytest

from lazyqsar.cli import main as cli
from lazyqsar.ensemble import OUTPUT_NAMES
from lazyqsar.registry import DESCRIPTORS_MODE


@pytest.fixture(autouse=True)
def no_pip(monkeypatch):
    """No test in this module may install anything."""

    def boom(*a, **k):  # pragma: no cover - only runs if a test regresses
        raise AssertionError(f"the CLI tried to run a subprocess: {a!r}")

    monkeypatch.setattr(subprocess, "check_call", boom)
    monkeypatch.setattr(subprocess, "run", boom)


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["lazyqsar", *argv])
    cli.main()


@pytest.fixture
def recorder():
    calls = []

    def record(**kwargs):
        calls.append(kwargs)

    record.calls = calls
    return record


# ------------------------------------------------------------------ dispatch and defaults


def test_fit_forwards_its_arguments_with_slow_as_the_default(monkeypatch, recorder):
    import lazyqsar.api.classifier_fit as fit_api

    monkeypatch.setattr(fit_api, "fit", recorder)
    run(monkeypatch, "fit", "--task", "classification", "--input", "d", "--output", "m")

    assert recorder.calls == [
        {"data_dir": "d", "model_dir": "m", "models_txt": None, "mode": "slow"}
    ]


def test_predict_forwards_its_arguments_with_proba_as_the_default(
    monkeypatch, recorder
):
    import lazyqsar.api.classifier_predict as predict_api

    monkeypatch.setattr(predict_api, "predict", recorder)
    run(monkeypatch, "predict", "--input", "i.csv", "--model", "m", "--output", "o.csv")

    assert recorder.calls == [
        {
            "model_dir": "m",
            "input_csv": "i.csv",
            "output_csv": "o.csv",
            "models_txt": None,
            "predict_type": "proba",
        }
    ]


# ---------------------------------------------------------------------------- exit codes


def test_no_subcommand_is_a_usage_error(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch)
    assert exc.value.code == 2


def test_regression_is_rejected_with_a_message(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "fit", "--task", "regression", "--input", "d", "--output", "m")
    assert exc.value.code == 1
    assert "not yet implemented" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ("fit", "--task", "classification", "--input", "d"),  # missing --output
        ("fit", "--task", "bogus", "--input", "d", "--output", "m"),
        ("predict", "--input", "i", "--model", "m"),  # missing --output
        (
            "predict",
            "--input",
            "i",
            "--model",
            "m",
            "--output",
            "o",
            "--predict_type",
            "bogus",
        ),
    ],
    ids=[
        "fit_missing_output",
        "fit_bad_task",
        "predict_missing_output",
        "predict_bad_type",
    ],
)
def test_malformed_invocations_are_usage_errors(monkeypatch, argv):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, *argv)
    assert exc.value.code == 2


def test_setup_with_nothing_to_do_exits_one(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "setup")
    assert exc.value.code == 1
    assert "Nothing to do" in capsys.readouterr().err


def test_setup_rejects_an_unknown_descriptor(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "setup", "--descriptors", "--only", "bogus")
    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert "bogus" in err
    for valid in ("chemeleon", "cddd", "clamp"):
        assert valid in err, "the error should name the valid options"


def test_setup_warns_when_descriptor_flags_have_no_effect(monkeypatch, capsys):
    """`--only` without `--descriptors` is inert; say so rather than ignoring it."""
    monkeypatch.setattr(cli, "_setup_fit", lambda: None)
    called = []
    monkeypatch.setattr(cli, "_setup_descriptors", lambda args: called.append(args))

    run(monkeypatch, "setup", "--fit", "--only", "chemeleon")

    assert "--only" in capsys.readouterr().err
    assert called == [], "descriptor setup must not run without --descriptors"


# ------------------------------------------------------------------------- drift guards
#
# The CLI restates two lists the library already owns. These fail when they drift, which is
# the class of bug 3.5.0 was about: an entry point quietly offering something different from
# what the library supports.


def option_choices(subcommand, option):
    """The `choices` argparse actually enforces, read off the live parser.

    Shares ``_build_parser`` below rather than repeating the capture: the two were
    near-identical twenty-line copies of the same trick, which is one place too many for
    something whose whole purpose is to avoid working against a drifting copy.
    """
    import argparse

    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    action = next(
        a for a in sub.choices[subcommand]._actions if option in a.option_strings
    )
    return set(action.choices)


def test_predict_type_choices_match_the_library(monkeypatch, recorder):
    """Every documented output name must be accepted, and no others."""
    import lazyqsar.api.classifier_predict as predict_api

    monkeypatch.setattr(predict_api, "predict", recorder)
    for name in OUTPUT_NAMES:
        run(
            monkeypatch,
            "predict",
            "--input",
            "i",
            "--model",
            "m",
            "--output",
            "o",
            "--predict_type",
            name,
        )
    assert [c["predict_type"] for c in recorder.calls] == list(OUTPUT_NAMES)


def test_mode_choices_match_the_descriptor_registry(monkeypatch, recorder):
    import lazyqsar.api.classifier_fit as fit_api

    monkeypatch.setattr(fit_api, "fit", recorder)
    for mode in sorted(DESCRIPTORS_MODE):
        run(
            monkeypatch,
            "fit",
            "--task",
            "classification",
            "--input",
            "d",
            "--output",
            "m",
            "--mode",
            mode,
        )
    assert [c["mode"] for c in recorder.calls] == sorted(DESCRIPTORS_MODE)


def test_predict_type_offers_exactly_the_library_outputs():
    """Not just "the valid ones work" -- also that no stale extra is still offered."""
    assert option_choices("predict", "--predict_type") == set(OUTPUT_NAMES)


def test_mode_offers_exactly_the_registered_modes():
    assert option_choices("fit", "--mode") == set(DESCRIPTORS_MODE)


# --------------------------------------------------------------- reference command


def _build_parser():
    """The live parser, captured from ``main()`` rather than rebuilt.

    Same reason as ``_choices_for`` above: a copy could drift from what ships.
    """
    import argparse

    captured = {}
    real = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        captured.setdefault("parser", self)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        try:
            cli.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real
    return captured["parser"]


def test_reference_subcommand_exists_with_its_four_actions():
    """The errors raised elsewhere name these commands, so they have to exist.

    `store.descriptor_path` tells a user to run `lazyqsar setup --reference`, and the
    agnostic entry point tells them to run `lazyqsar reference smiles`. Both were written
    against the intended design before it existed.
    """
    parser = _build_parser()
    for action in ("status", "fetch", "verify", "smiles"):
        args = parser.parse_args(["reference", action])
        assert args.command == "reference"
        assert args.action == action


def test_reference_rejects_an_unknown_action():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["reference", "publish"])


def test_setup_accepts_reference_and_it_is_not_implied_by_descriptors():
    """A fast-mode model needs one 6.5 MB matrix; the bundle is 267 MB. Opting in is the
    point."""
    parser = _build_parser()
    assert parser.parse_args(["setup", "--reference"]).reference is True
    assert parser.parse_args(["setup", "--descriptors"]).reference is False


def test_reference_fetch_takes_only_and_force():
    parser = _build_parser()
    args = parser.parse_args(["reference", "fetch", "--only", "morgan", "--force"])
    assert args.only == "morgan" and args.force is True
