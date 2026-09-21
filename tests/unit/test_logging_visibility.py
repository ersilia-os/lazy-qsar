"""Warnings must reach stderr without the caller opting in.

This exists because they did not. ``utils.logging`` drops loguru's default sink at import
and used to add one only under ``set_verbosity(True)``, so every ``logger.warning`` in the
package was a no-op in the default configuration -- including the message that explains why
a prediction came back blank. The CLI has no verbosity flag, so on the deployed path that
warning could not be seen at all. A unit test is the only thing that keeps it visible,
because nothing else fails when a warning silently stops being printed.
"""

import pytest

from lazyqsar.utils.logging import logger as _singleton


@pytest.fixture
def log(capsys):
    """The package's own logger, left exactly as it was found.

    Deliberately not a fresh ``Logger()``. ``loguru.logger`` is process-wide, so what is
    under test is the sink configuration of the real singleton -- the thing a user
    actually gets. Verbosity is restored afterwards because it is global state.
    """
    was_verbose = _singleton.verbose
    _singleton.set_verbosity(False)
    capsys.readouterr()
    yield _singleton
    _singleton.set_verbosity(was_verbose)


def _stderr(capsys):
    return capsys.readouterr().err


def test_a_warning_is_printed_without_enabling_verbosity(log, capsys):
    log.warning("the-warning")
    assert "the-warning" in _stderr(capsys)


@pytest.mark.parametrize("level", ["error", "critical"])
def test_levels_above_warning_are_printed_too(log, capsys, level):
    getattr(log, level)("loud-message")
    assert "loud-message" in _stderr(capsys)


@pytest.mark.parametrize("level", ["debug", "info"])
def test_levels_below_warning_stay_quiet_by_default(log, capsys, level):
    getattr(log, level)("chatter")
    assert "chatter" not in _stderr(capsys)


def test_verbose_lowers_the_floor_rather_than_switching_output_on(log, capsys):
    log.set_verbosity(True)
    log.debug("now-visible")
    assert "now-visible" in _stderr(capsys)


def test_a_warning_is_not_printed_twice_while_verbose(log, capsys):
    """The two sinks must cover disjoint level ranges."""
    log.set_verbosity(True)
    log.warning("exactly-once")
    assert _stderr(capsys).count("exactly-once") == 1


def test_turning_verbosity_off_leaves_warnings_on(log, capsys):
    log.set_verbosity(True)
    log.set_verbosity(False)
    log.debug("hidden-again")
    log.warning("still-shown")
    err = _stderr(capsys)
    assert "hidden-again" not in err
    assert "still-shown" in err


def test_nan_descriptor_rows_truncates_a_long_index_list(log, capsys):
    """A 300k-row screen must not put five figures of indices on one line."""
    log.nan_descriptor_rows("morgan", range(500), 300_000)
    err = _stderr(capsys)
    assert "500/300000" in err
    assert "490 more" in err
    assert "499" not in err


def test_nan_descriptor_rows_says_nothing_when_there_is_nothing_to_say(log, capsys):
    log.nan_descriptor_rows("morgan", [], 100)
    assert _stderr(capsys) == ""
