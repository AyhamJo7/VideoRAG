"""Video chunking utilities for splitting videos into temporal segments."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import ffmpeg
from loguru import logger

from videorag.config.settings import settings
from videorag.io.video import VideoInfo, get_video_info
from videorag.utils.paths import get_chunk_path


@dataclass
class VideoChunk:
    """Metadata for a video chunk."""

    video_id: str
    chunk_idx: int
    start_time: float
    end_time: float
    duration: float
    chunk_path: Path
    original_video_path: Path

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "chunk_idx": self.chunk_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "chunk_path": str(self.chunk_path),
            "original_video_path": str(self.original_video_path),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoChunk":
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            chunk_idx=data["chunk_idx"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            duration=data["duration"],
            chunk_path=Path(data["chunk_path"]),
            original_video_path=Path(data["original_video_path"]),
        )


def compute_chunk_times(
    video_duration: float,
    chunk_length: float,
    overlap: float,
) -> List[tuple[float, float]]:
    """
    Compute start and end times for overlapping chunks.

    Args:
        video_duration: Total video duration in seconds
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between consecutive chunks in seconds

    Returns:
        List of (start_time, end_time) tuples
    """
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if overlap < 0 or overlap >= chunk_length:
        raise ValueError("overlap must be in range [0, chunk_length)")

    chunks = []
    stride = chunk_length - overlap
    current_start = 0.0

    while current_start < video_duration:
        current_end = min(current_start + chunk_length, video_duration)
        chunks.append((current_start, current_end))

        # Break if we've reached the end
        if current_end >= video_duration:
            break

        current_start += stride

    return chunks


def chunk_video(
    video_path: Path,
    output_dir: Path,
    chunk_length: Optional[float] = None,
    overlap: Optional[float] = None,
) -> List[VideoChunk]:
    """
    Split a video into temporal chunks using ffmpeg.

    Args:
        video_path: Path to input video
        output_dir: Directory to save chunks
        chunk_length: Length of each chunk (uses settings default if None)
        overlap: Overlap between chunks (uses settings default if None)

    Returns:
        List of VideoChunk objects

    Raises:
        ValueError: If video cannot be processed
    """
    chunk_length = chunk_length or settings.chunk_length_sec
    overlap = overlap or settings.chunk_overlap_sec

    video_info = get_video_info(video_path)
    chunk_times = compute_chunk_times(video_info.duration, chunk_length, overlap)

    logger.info(
        f"Chunking {video_path.name}: {len(chunk_times)} chunks "
        f"(length={chunk_length}s, overlap={overlap}s)"
    )

    chunks = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (start_time, end_time) in enumerate(chunk_times):
        chunk_path = get_chunk_path(video_info.video_id, idx, output_dir)
        duration = end_time - start_time

        try:
            # Use ffmpeg to extract chunk
            # -ss: start time, -t: duration, -c copy: copy codec (fast)
            # For precise cutting, use -c:v libx264 -c:a aac instead
            stream = ffmpeg.input(str(video_path), ss=start_time, t=duration)
            stream = ffmpeg.output(
                stream,
                str(chunk_path),
                c="copy",  # Fast copy, may not be frame-accurate
                avoid_negative_ts="make_zero",
            )
            ffmpeg.run(stream, quiet=True, overwrite_output=True)

            chunk = VideoChunk(
                video_id=video_info.video_id,
                chunk_idx=idx,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                chunk_path=chunk_path,
                original_video_path=video_path,
            )
            chunks.append(chunk)

            logger.debug(f"Created chunk {idx}: {start_time:.1f}s - {end_time:.1f}s")

        except ffmpeg.Error as e:
            logger.error(f"Failed to create chunk {idx}: {e.stderr.decode()}")
            continue

    logger.info(f"Successfully created {len(chunks)} chunks for {video_path.name}")
    return chunks


def save_chunk_metadata(chunks: List[VideoChunk], output_path: Path) -> None:
    """
    Save chunk metadata to JSON file.

    Args:
        chunks: List of VideoChunk objects
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [chunk.to_dict() for chunk in chunks]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved chunk metadata to {output_path}")


def load_chunk_metadata(metadata_path: Path) -> List[VideoChunk]:
    """
    Load chunk metadata from JSON file.

    Args:
        metadata_path: Path to JSON file

    Returns:
        List of VideoChunk objects
    """
    with open(metadata_path) as f:
        data = json.load(f)
    return [VideoChunk.from_dict(item) for item in data]
