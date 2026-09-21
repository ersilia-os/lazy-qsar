"""Directory trees shaped like a checkpoint, with no models in them.

``sources_from_parent`` and ``sources_from_mapping`` resolve task directories by *shape*:
they list directories, apply ``models_txt`` filtering and ordering, and never open a model.
Testing them against a real checkpoint costs a model fit per task for no added coverage, so
they get empty directories instead -- which also moves those tests down to the base tier.
"""

import json
import os


def fake_task_tree(root, tasks, descriptors=("morgan",)):
    """Create ``root/<task>/<descriptor>/`` with a plausible ``metadata.json`` per task.

    Nothing here is loadable. Use it only for code that inspects layout.
    """
    os.makedirs(root, exist_ok=True)
    for task in tasks:
        task_dir = os.path.join(root, task)
        for d in descriptors:
            os.makedirs(os.path.join(task_dir, d), exist_ok=True)
        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump({"descriptor_types": list(descriptors)}, f)
    return root


def write_models_txt(path, names):
    """Write a ``models_txt`` selection file, one task name per line."""
    with open(path, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    return str(path)
