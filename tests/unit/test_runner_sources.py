"""Resolving task directories into ordered, named prediction sources.

These functions decide what the output CSV's columns are and what order they come in, which
is the contract every Ersilia Model Hub template depends on. They inspect directory *shape*
and never open a model, so they are tested against empty directories -- that keeps them on
the base tier and saves a model fit per task.
"""

import os

import pytest

from _helpers.trees import fake_task_tree, write_models_txt

from lazyqsar.ensemble.runner import sources_from_mapping, sources_from_parent

TASKS = ["taskA", "taskB", "taskC"]


@pytest.fixture
def tree(tmp_path):
    return fake_task_tree(str(tmp_path / "models"), TASKS)


def test_sources_from_parent_sorted(tree):
    """Without a selection file, columns are alphabetical -- not filesystem order."""
    assert [s.column_name for s in sources_from_parent(tree)] == sorted(TASKS)


def test_sources_from_parent_models_txt_filters_and_reorders(tree, tmp_path):
    models_txt = write_models_txt(tmp_path / "m.txt", ["taskC", "taskA"])
    sources = sources_from_parent(tree, models_txt)
    assert [s.column_name for s in sources] == ["taskC", "taskA"]


def test_sources_from_mapping_keeps_caller_order(tree):
    """A dict maps column names to directories, and the caller's order is the output order."""
    col_map = {
        "Zeta": os.path.join(tree, "taskC"),
        "Alpha": os.path.join(tree, "taskA"),
    }
    assert [s.column_name for s in sources_from_mapping(col_map)] == ["Zeta", "Alpha"]


def test_sources_from_mapping_keeps_duplicate_directories(tree):
    """Two columns may name one directory; a dict keyed by path would lose one."""
    col_map = {
        "primary": os.path.join(tree, "taskA"),
        "primary_alias": os.path.join(tree, "taskA"),
    }
    sources = sources_from_mapping(col_map)
    assert [s.column_name for s in sources] == ["primary", "primary_alias"]
    assert sources[0].task_dir == sources[1].task_dir


def test_sources_from_mapping_filters_but_does_not_reorder(tree, tmp_path):
    """models_txt selects for a mapping; it does not reorder it.

    This is the one place the two resolvers deliberately differ. For a parent directory the
    file is the only thing that can express an order, so it sets one. For a mapping the
    caller has already expressed an order by constructing the dict, and honouring the file's
    order instead would silently permute their columns.
    """
    col_map = {
        "Zeta": os.path.join(tree, "taskC"),
        "Alpha": os.path.join(tree, "taskA"),
        "Mid": os.path.join(tree, "taskB"),
    }
    models_txt = write_models_txt(tmp_path / "m.txt", ["Alpha", "Zeta"])
    sources = sources_from_mapping(col_map, models_txt)
    assert [s.column_name for s in sources] == ["Zeta", "Alpha"]
