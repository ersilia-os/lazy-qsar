"""The RDKit-backed SMILES checks must be reachable through a known, finite set of names.

Only the ``chem`` tier installs RDKit. Every other tier neutralises the SMILES checks through
``_helpers.stubs.STUBBED_BINDINGS`` -- but a from-import copies the function object at import
time, so patching the definition site does not rebind a module that imported the name.

That is not hypothetical. ``api/classifier_fit.py`` and ``api/classifier_predict.py`` both
hold their own bindings, and the suite this replaced patched only ``lazyqsar.qsar``, so those
two modules called straight into RDKit on an install without it.

The scan is static. Importing every module to look for bindings would need torch and chemprop
to be installed, would take tens of seconds, and would silently skip any module that failed to
import -- which is exactly where an unnoticed binding is most likely to hide.
"""

import ast
import importlib
import os

import pytest

from _helpers.stubs import CHEM_FUNCTION_NAMES, STUBBED_BINDINGS

import lazyqsar

PACKAGE_ROOT = os.path.dirname(os.path.abspath(lazyqsar.__file__))

# Where the functions are defined, and the wrappers that deliberately re-export them. These
# are the bindings the stubs already cover or that must not be patched.
DEFINITION_SITE = "lazyqsar.descriptors._validate"


def _module_name(path):
    rel = os.path.relpath(path, os.path.dirname(PACKAGE_ROOT))
    return rel[: -len(".py")].replace(os.sep, ".").removesuffix(".__init__")


def _package_files():
    for dirpath, _, filenames in os.walk(PACKAGE_ROOT):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _module_level_chem_bindings(path):
    """Names bound at module level that refer to a chem check.

    Covers both ``from X import validate_smiles`` and ``def validate_smiles(...)``, which are
    the two ways a module ends up with an attribute the stubs would have to patch. Imports
    inside a function body are not bindings -- they are the lazy-import pattern this package
    uses deliberately, and they resolve through the patched module at call time.
    """
    found = set()
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in CHEM_FUNCTION_NAMES:
                    found.add(bound)
        elif isinstance(node, ast.FunctionDef) and node.name in CHEM_FUNCTION_NAMES:
            found.add(node.name)
    return found


def test_every_chem_binding_is_stubbed():
    declared = set(STUBBED_BINDINGS)
    found = set()
    for path in _package_files():
        module = _module_name(path)
        if module == DEFINITION_SITE:
            continue
        for name in _module_level_chem_bindings(path):
            found.add((module, name))

    missing = found - declared
    assert not missing, (
        "these modules bind an RDKit-backed SMILES check that the test stubs do not patch: "
        f"{sorted(missing)}. Add them to _helpers.stubs.STUBBED_BINDINGS, or every test "
        "outside the chem tier will call straight into RDKit."
    )


def test_the_scan_finds_the_known_bindings():
    """Guard the guard: a broken walk would make the assertion above vacuous."""
    found = {
        (_module_name(p), n)
        for p in _package_files()
        for n in _module_level_chem_bindings(p)
    }
    assert ("lazyqsar.qsar", "validate_smiles") in found
    assert ("lazyqsar.api.classifier_fit", "validate_smiles") in found


def test_stub_list_has_no_stale_entries():
    """A binding that was removed should not linger in the list, pretending to be covered."""
    for module_name, attr in STUBBED_BINDINGS:
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr, None)), (
            f"{module_name}.{attr} is in STUBBED_BINDINGS but no longer exists"
        )


@pytest.mark.parametrize("module_name,attr", STUBBED_BINDINGS)
def test_stubs_actually_take_effect(module_name, attr, stub_descriptors):
    """The fixture must rebind every declared name, not just the definition site."""
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if attr == "validate_smiles":
        assert fn(["not a molecule at all ((("]) is None
    else:
        assert fn(["not a molecule at all ((("]) == []
