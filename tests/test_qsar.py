"""
Tests for lazyqsar.qsar.

Covers:
  - LazyQSAR dispatcher: returns correct concrete class
  - LazyRegressorQSAR stub raises NotImplementedError
  - LazyClassifierQSAR instantiation and basic attribute checks
  - LazyQSAR with unknown task raises ValueError

Note: descriptor instances are populated during fit(), not __init__(), so
      model.descriptors is empty until fit() is called. Use mode="fast"
      (rdkit + morgan only) to stay lightweight in unit tests.
"""

import pytest

from lazyqsar.qsar import LazyClassifierQSAR, LazyRegressorQSAR, LazyQSAR


# ---------------------------------------------------------------------------
# LazyQSAR dispatcher
# ---------------------------------------------------------------------------

class TestLazyQSARDispatcher:

    def test_classification_type(self):
        obj = LazyQSAR(task="classification", mode="fast")
        assert isinstance(obj, LazyClassifierQSAR)

    def test_regression_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            LazyQSAR(task="regression")

    def test_unknown_task_raises_value_error(self):
        with pytest.raises(ValueError):
            LazyQSAR(task="multiclass")

    def test_default_task_is_classification(self):
        # Default task should be classification; use fast mode to avoid heavy imports
        obj = LazyQSAR(mode="fast")
        assert isinstance(obj, LazyClassifierQSAR)


# ---------------------------------------------------------------------------
# LazyRegressorQSAR stub
# ---------------------------------------------------------------------------

class TestLazyRegressorQSAR:

    def test_raises_on_instantiation(self):
        with pytest.raises(NotImplementedError):
            LazyRegressorQSAR()

    def test_raises_with_mode_arg(self):
        with pytest.raises(NotImplementedError):
            LazyRegressorQSAR(mode="slow")


# ---------------------------------------------------------------------------
# LazyClassifierQSAR instantiation
# ---------------------------------------------------------------------------

class TestLazyClassifierQSAR:

    def test_instantiate_fast_mode(self):
        model = LazyClassifierQSAR(mode="fast")
        assert model.mode == "fast"

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError):
            LazyClassifierQSAR(mode="turbo")

    def test_descriptor_types_set_for_fast(self):
        model = LazyClassifierQSAR(mode="fast")
        assert len(model.descriptor_types) > 0
        assert all(isinstance(d, str) for d in model.descriptor_types)

    def test_fast_mode_uses_morgan_only(self):
        model = LazyClassifierQSAR(mode="fast")
        assert "morgan" in model.descriptor_types
        assert "rdkit" not in model.descriptor_types

    def test_descriptors_empty_before_fit(self):
        # Descriptor instances are populated during fit(), not __init__().
        model = LazyClassifierQSAR(mode="fast")
        assert model.descriptors == []

    def test_descriptor_types_set_before_fit(self):
        model = LazyClassifierQSAR(mode="fast")
        assert len(model.descriptor_types) == len(["morgan"])
