"""Task-neutral name for the CLI fit entry point (classification and regression).

``lazyqsar.api.classifier_fit`` keeps working; it predates regression and is where the
implementation lives.
"""

from .classifier_fit import fit, get_task_data, get_task_names  # noqa: F401

__all__ = ["fit", "get_task_data", "get_task_names"]
