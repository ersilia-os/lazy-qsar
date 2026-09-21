"""``LAZYQSAR_PREDICT_CHUNK`` -- the documented knob for inference batch size.

Scoring a million compounds against a 2048-dimensional descriptor is only tractable because
inference streams in chunks, and this variable is how a deployment tunes that. A malformed
value must fall back to the default rather than crash a library screen or, worse, silently
set the batch size to zero.
"""

import pytest

from lazyqsar.ensemble.runner import get_chunk_size

DEFAULT = 1000


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv("LAZYQSAR_PREDICT_CHUNK", raising=False)
    assert get_chunk_size() == DEFAULT


def test_honours_a_valid_value(monkeypatch):
    monkeypatch.setenv("LAZYQSAR_PREDICT_CHUNK", "13")
    assert get_chunk_size() == 13


@pytest.mark.parametrize("value", ["0", "-5", "abc", "", "1.5"])
def test_falls_back_on_a_useless_value(monkeypatch, value):
    monkeypatch.setenv("LAZYQSAR_PREDICT_CHUNK", value)
    assert get_chunk_size() == DEFAULT, f"{value!r} should fall back to the default"
