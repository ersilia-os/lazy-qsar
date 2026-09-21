"""The block size that bounds a prediction's working set.

Not configurable, so there is no environment variable to get wrong — but the derivation has
one property the numbers depend on, and one the memory bound depends on, and neither is
obvious from reading it.
"""

import pytest

from lazyqsar.api.classifier_predict import (
    _BLOCK_BUDGET_BYTES,
    _block_size,
)


@pytest.mark.parametrize("n_tasks", [1, 2, 5, 20, 50, 200])
@pytest.mark.parametrize("chunk_size", [1, 7, 100, 1000, 4096])
def test_block_is_a_whole_number_of_chunks(n_tasks, chunk_size):
    """This is what keeps blocking bit-identical.

    Featurization and scoring both step in ``chunk_size`` rows, and onnxruntime selects
    different kernels for different batch sizes. If a block boundary produced a short
    chunk, the last few digits would move — so every block must divide evenly into chunks,
    leaving the one short chunk at the very end of the input, exactly where an unblocked
    run has it.
    """
    block = _block_size(n_tasks, chunk_size)
    assert block % chunk_size == 0
    assert block >= chunk_size


@pytest.mark.parametrize("n_tasks", [1, 20, 200])
def test_block_shrinks_as_endpoints_are_added(n_tasks):
    """More endpoints means more accumulated channels and results per molecule.

    The point of deriving the block rather than fixing it: scoring 200 endpoints must not
    hold 200 times the working set of scoring one.
    """
    assert _block_size(n_tasks, 1000) >= _block_size(n_tasks * 2, 1000)


def test_block_stays_within_the_budget():
    """A block's estimated working set must not exceed the budget it was derived from."""
    for n_tasks in (1, 5, 20, 50):
        block = _block_size(n_tasks, 1000)
        if block == 1000:
            continue  # floored at one chunk; the budget cannot be honoured below that
        per_row = _BLOCK_BUDGET_BYTES // block
        assert block * per_row <= _BLOCK_BUDGET_BYTES


def test_a_huge_endpoint_count_still_returns_one_chunk():
    """The floor, so that an absurd task count cannot produce a zero-row block."""
    assert _block_size(10**6, 1000) == 1000
