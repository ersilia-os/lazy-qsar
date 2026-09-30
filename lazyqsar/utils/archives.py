"""Reading a checkpoint off disk: unpacking an archive, and parsing its metadata.

Both jobs are about not losing information the caller needs -- one about not destroying
their files, the other about not throwing away which file failed.

Both loaders accept ``model.zip``. Both used to unpack it to the sibling path -- so
``load("models/tb.zip")`` ran ``shutil.rmtree("models/tb")`` first, deleting whatever was
there. That is very plausibly the caller's working checkpoint, and nothing asked.

Nothing heavy is imported here so this stays usable from the base tier.
"""

import json
import os
import shutil
import tempfile


def unpack_to_scratch(archive_path: str) -> tuple[str, str]:
    """Unpack *archive_path* into a fresh scratch directory.

    Returns ``(scratch_root, checkpoint_dir)``. The caller removes ``scratch_root`` once
    loading is done -- which is safe, because onnxruntime reads a graph into memory when
    the session is constructed, so the loaded sessions outlive the files they came from.

    ``save`` writes the archive with ``shutil.make_archive(..., root_dir=model_dir)``, so
    the checkpoint's contents sit at the archive root. An archive packed elsewhere may
    instead carry a single top-level folder; both layouts are accepted.
    """
    scratch = tempfile.mkdtemp(prefix="lazyqsar-load-")
    try:
        shutil.unpack_archive(archive_path, scratch)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    entries = os.listdir(scratch)
    if len(entries) == 1 and "metadata.json" not in entries:
        nested = os.path.join(scratch, entries[0])
        if os.path.isdir(nested):
            return scratch, nested
    return scratch, scratch


def read_json(path: str):
    """``json.load`` that names the file it failed on.

    A truncated ``metadata.json`` -- an interrupted save, a full disk, a partial copy --
    raises ``JSONDecodeError: Expecting property name ... line 1 column 2``, which says
    nothing about *which* file. A checkpoint directory holds one per task plus one per
    descriptor, so the bare message leaves the user grepping.
    """
    with open(path) as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"{exc.msg} (in {path})", exc.doc, exc.pos
            ) from None
