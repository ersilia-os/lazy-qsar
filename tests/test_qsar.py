"""
Tests for lazyqsar.qsar.

Covers:
  - LazyQSAR dispatcher: returns correct concrete class
  - LazyRegressorQSAR stub raises NotImplementedError
  - LazyClassifierQSAR instantiation and basic attribute checks
  - LazyQSAR with unknown task raises ValueError

Note: LazyClassifierQSAR.__init__ immediately imports descriptor modules.
      Tests that instantiate with mode="default" or mode="slow" require
      chemeleon/cddd; use mode="fast" (rdkit + morgan only) to stay lightweight.
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
            LazyRegressorQSAR(mode="default")


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

    def test_fast_mode_uses_rdkit_and_morgan(self):
        model = LazyClassifierQSAR(mode="fast")
        assert "rdkit" in model.descriptor_types
        assert "morgan" in model.descriptor_types

    def test_descriptors_list_length_matches_types(self):
        model = LazyClassifierQSAR(mode="fast")
        assert len(model.descriptors) == len(model.descriptor_types)
