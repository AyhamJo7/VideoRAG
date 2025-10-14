"""Tests for video chunking module."""

import pytest

from videorag.io.chunking import compute_chunk_times


def test_compute_chunk_times_basic():
    """Test basic chunking without overlap."""
    chunks = compute_chunk_times(video_duration=100.0, chunk_length=30.0, overlap=0.0)
    assert len(chunks) == 4
    assert chunks[0] == (0.0, 30.0)
    assert chunks[1] == (30.0, 60.0)
    assert chunks[2] == (60.0, 90.0)
    assert chunks[3] == (90.0, 100.0)


def test_compute_chunk_times_with_overlap():
    """Test chunking with overlap."""
    chunks = compute_chunk_times(video_duration=100.0, chunk_length=30.0, overlap=5.0)
    assert len(chunks) == 5
    assert chunks[0] == (0.0, 30.0)
    assert chunks[1] == (25.0, 55.0)
    assert chunks[2] == (50.0, 80.0)


def test_compute_chunk_times_short_video():
    """Test chunking with video shorter than chunk length."""
    chunks = compute_chunk_times(video_duration=20.0, chunk_length=30.0, overlap=0.0)
    assert len(chunks) == 1
    assert chunks[0] == (0.0, 20.0)


def test_compute_chunk_times_invalid_params():
    """Test error handling for invalid parameters."""
    with pytest.raises(ValueError):
        compute_chunk_times(100.0, chunk_length=-10.0, overlap=0.0)

    with pytest.raises(ValueError):
        compute_chunk_times(100.0, chunk_length=30.0, overlap=30.0)
