"""``lazyqsar setup --descriptors`` installs what the ``[descriptors]`` extra pins.

The setup command runs its own ``pip install`` calls rather than installing the extra, so
the two can drift apart. They did in #52: an unpinned ``chemprop`` began resolving to 2.3,
which pulls in cuik-molmaker and rdkit 2026.03, and the very next step forced rdkit back
to 2025.9.1 underneath it.
"""

import tomllib
from pathlib import Path

import pytest

from lazyqsar.utils import setup

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _descriptor_pins():
    with open(_PYPROJECT, "rb") as f:
        extra = tomllib.load(f)["project"]["optional-dependencies"]["descriptors"]
    return {spec.split("==")[0].lower(): spec for spec in extra if "==" in spec}


@pytest.mark.parametrize(
    "installer, package",
    [
        ("install_chemprop", "chemprop"),
        ("install_rdkit", "rdkit"),
        ("install_fpsim2", "fpsim2"),
    ],
)
def test_setup_installs_the_pinned_version(monkeypatch, installer, package):
    calls = []
    monkeypatch.setattr(setup.subprocess, "check_call", calls.append)
    getattr(setup, installer)()
    (argv,) = calls
    assert _descriptor_pins()[package] in argv
