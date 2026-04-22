"""Re-export XGB inspector — SVC reuses the same dataset profiling."""

from lazyqsar.base.xgboost.inspector import inspect, DatasetProfile  # noqa: F401

__all__ = ["inspect", "DatasetProfile"]
