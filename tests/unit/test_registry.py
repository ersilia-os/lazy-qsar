"""The descriptor registry: names, modes, and lookup without importing anything heavy.

This module exists so the inference stack can learn *which* descriptors a checkpoint uses
without importing any of them -- the descriptor modules pull in RDKit and torch, which no
Ersilia Model Hub deployment has. Everything here therefore has to hold on a base install,
and the resolution tests check module paths with ``find_spec`` rather than importing them.
"""

import importlib.util

import pytest

from lazyqsar.registry import DESCRIPTOR_TYPES, DESCRIPTORS_MODE, get_descriptor_type


def test_every_mode_names_only_registered_descriptors():
    for mode, names in DESCRIPTORS_MODE.items():
        unknown = set(names) - set(DESCRIPTOR_TYPES)
        assert not unknown, f"mode {mode!r} names unregistered descriptors: {unknown}"


def test_slow_mode_covers_every_registered_descriptor():
    """`slow` is the "everything" mode; a new descriptor that skips it is unreachable."""
    assert set(DESCRIPTORS_MODE["slow"]) == set(DESCRIPTOR_TYPES)


def test_mode_lists_are_sorted():
    """The module re-sorts at import, and checkpoint layouts depend on that order."""
    for mode, names in DESCRIPTORS_MODE.items():
        assert list(names) == sorted(names), f"mode {mode!r} is not sorted"


@pytest.mark.parametrize("name", sorted(DESCRIPTOR_TYPES))
def test_registered_module_paths_resolve(name, shipped_descriptor_types):
    """Catch a renamed or moved descriptor module without importing it.

    Importing would need RDKit or torch; ``find_spec`` only needs the file to exist. Reads
    the shipped snapshot rather than the live dict, which the stub fixtures patch in place.
    """
    module_path, class_name = shipped_descriptor_types[name]
    assert importlib.util.find_spec(module_path) is not None, (
        f"{name!r} points at {module_path!r}, which does not exist"
    )
    assert class_name


def test_unknown_descriptor_raises_key_error():
    with pytest.raises(KeyError):
        get_descriptor_type("no_such_descriptor")


def test_get_descriptor_type_returns_the_registered_class(stub_descriptors):
    """Resolution is checked through a stub, so this holds without RDKit installed."""
    stub_descriptors("morgan")
    cls = get_descriptor_type("morgan")
    assert cls.__name__ == DESCRIPTOR_TYPES["morgan"][1], (
        "resolves through the live registry"
    )
    assert isinstance(cls, type), "the class itself is returned, not an instance"
