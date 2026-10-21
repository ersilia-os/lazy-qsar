"""Warnings must reach stderr without the caller opting in.

This exists because they did not. ``utils.logging`` used to emit only under
``set_verbosity(True)``, so every ``logger.warning`` in the package was a no-op in the
default configuration -- including the message that explains why a prediction came back
blank. The CLI has no verbosity flag, so on the deployed path that
warning could not be seen at all. A unit test is the only thing that keeps it visible,
because nothing else fails when a warning silently stops being printed.
"""

import pytest

from lazyqsar.utils.logging import logger as _singleton


@pytest.fixture
def log(capsys):
    """The package's own logger, left exactly as it was found.

    Deliberately not a fresh ``Logger()``: what is under test is the handler configuration
    of the real singleton -- the thing a user actually gets. Verbosity is restored
    afterwards because it is process-wide state.
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


# ------------------------------------------------------------------ host isolation


_HOST_SCRIPT = r"""
import logging, sys
from io import StringIO

# A host application configures logging before importing lazyqsar.
root_stream = StringIO()
logging.basicConfig(stream=root_stream, level=logging.DEBUG, force=True)
try:
    from loguru import logger as host_loguru
    host_sink = []
    host_loguru.add(host_sink.append, level="DEBUG")
except ImportError:
    host_loguru = None

import lazyqsar  # noqa: F401
from lazyqsar.utils.logging import logger

logger.set_verbosity(True)
logger.warning("from-lazyqsar")
logging.getLogger("host").info("from-host")
if host_loguru is not None:
    host_loguru.info("from-host-loguru")

out = root_stream.getvalue()
assert "from-host" in out, "the host's root logging handler was disturbed"
assert "from-lazyqsar" not in out, "lazyqsar records leaked into the host's root logger"
if host_loguru is not None:
    got = [str(m) for m in host_sink]
    assert any("from-host-loguru" in m for m in got), "importing lazyqsar removed loguru handlers"
    assert not any("from-lazyqsar" in m for m in got), "lazyqsar wrote through loguru"
print("ok")
"""


def test_importing_lazyqsar_leaves_host_logging_alone():
    """Importing the package must not remove, reconfigure or write into a host's logging.

    It used to call ``loguru.logger.remove()`` at import, deleting every handler the host
    (the Ersilia CLI, a notebook, a service) had installed on loguru's shared logger. Run
    in a fresh interpreter because the import has to be the first one in the process.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _HOST_SCRIPT], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
