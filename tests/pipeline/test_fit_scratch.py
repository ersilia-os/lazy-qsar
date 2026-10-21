"""The fit's scratch space is its own: cleanup can only remove what the fit created.

``LAZYQSAR_FIT_SCRATCH`` used to be used as the scratch directory itself and removed with
``rmtree`` when the fit finished -- so pointing it at a shared scratch disk, as the README
suggests for large fits, deleted everything in it. It now names a parent directory, inside
which each fit makes (and removes) its own ``lazyqsar-fit-*`` subdirectory.
"""

import contextlib
import io

import numpy as np

from _helpers.smiles import make_smiles


def _write_task(data_dir, smiles):
    y = np.zeros(len(smiles), dtype=int)
    y[: len(smiles) // 2] = 1
    (data_dir / "task.csv").write_text(
        "smiles,bin\n" + "".join(f"{s},{v}\n" for s, v in zip(smiles, y))
    )


def test_fit_leaves_existing_scratch_contents_alone(
    tmp_path, monkeypatch, stub_descriptors
):
    from lazyqsar.api.classifier_fit import fit

    stub_descriptors("morgan")
    scratch = tmp_path / "shared-scratch"
    scratch.mkdir()
    keep = scratch / "someone-elses-results.txt"
    keep.write_text("precious")
    data = tmp_path / "data"
    data.mkdir()
    _write_task(data, make_smiles(60))
    monkeypatch.setenv("LAZYQSAR_FIT_SCRATCH", str(scratch))

    with contextlib.redirect_stdout(io.StringIO()):
        fit(data_dir=str(data), model_dir=str(tmp_path / "models"), mode="fast")

    assert keep.read_text() == "precious"
    assert sorted(p.name for p in scratch.iterdir()) == [keep.name]


def test_fit_creates_a_missing_scratch_parent(tmp_path, monkeypatch, stub_descriptors):
    from lazyqsar.api.classifier_fit import fit

    stub_descriptors("morgan")
    scratch = tmp_path / "does" / "not" / "exist"
    data = tmp_path / "data"
    data.mkdir()
    _write_task(data, make_smiles(60))
    monkeypatch.setenv("LAZYQSAR_FIT_SCRATCH", str(scratch))

    with contextlib.redirect_stdout(io.StringIO()):
        fit(data_dir=str(data), model_dir=str(tmp_path / "models"), mode="fast")

    assert scratch.is_dir() and list(scratch.iterdir()) == []
