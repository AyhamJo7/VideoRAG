"""Keyframe extraction from video chunks."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from videorag.config.settings import settings
from videorag.io.video import extract_frame_at_time, get_video_info, save_frame
from videorag.utils.paths import get_keyframe_path


@dataclass
class Keyframe:
    """Metadata for a single keyframe."""

    video_id: str
    chunk_idx: int
    frame_idx: int
    timestamp: float
    keyframe_path: Path

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "chunk_idx": self.chunk_idx,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "keyframe_path": str(self.keyframe_path),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Keyframe":
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            chunk_idx=data["chunk_idx"],
            frame_idx=data["frame_idx"],
            timestamp=data["timestamp"],
            keyframe_path=Path(data["keyframe_path"]),
        )


def extract_keyframes_uniform(
    video_path: Path,
    video_id: str,
    chunk_idx: int,
    chunk_start_time: float,
    chunk_end_time: float,
    output_dir: Path,
    sample_rate: Optional[float] = None,
) -> List[Keyframe]:
    """
    Extract keyframes at uniform intervals from a video chunk.

    Args:
        video_path: Path to video file (can be chunk or original)
        video_id: Unique video identifier
        chunk_idx: Chunk index
        chunk_start_time: Start time of chunk in original video
        chunk_end_time: End time of chunk in original video
        output_dir: Directory to save keyframes
        sample_rate: Frames per second to sample (uses settings default if None)

    Returns:
        List of Keyframe objects
    """
    sample_rate = sample_rate or settings.keyframe_sample_rate
    chunk_duration = chunk_end_time - chunk_start_time

    # Compute timestamps to sample
    num_frames = max(1, int(chunk_duration * sample_rate))
    timestamps = np.linspace(0, chunk_duration, num_frames, endpoint=False)

    logger.info(
        f"Extracting {num_frames} keyframes from chunk {chunk_idx} "
        f"at {sample_rate} fps"
    )

    keyframes = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, relative_timestamp in enumerate(timestamps):
        # Absolute timestamp in original video
        absolute_timestamp = chunk_start_time + relative_timestamp

        # Extract frame from chunk video (timestamp is relative to chunk)
        frame = extract_frame_at_time(video_path, relative_timestamp)
        if frame is None:
            logger.warning(f"Failed to extract frame {frame_idx} at {relative_timestamp}s")
            continue

        # Save keyframe
        keyframe_path = get_keyframe_path(video_id, chunk_idx, frame_idx, output_dir)
        save_frame(frame, keyframe_path, quality=85)

        keyframe = Keyframe(
            video_id=video_id,
            chunk_idx=chunk_idx,
            frame_idx=frame_idx,
            timestamp=absolute_timestamp,
            keyframe_path=keyframe_path,
        )
        keyframes.append(keyframe)

    logger.info(f"Extracted {len(keyframes)} keyframes for chunk {chunk_idx}")
    return keyframes


def save_keyframe_metadata(keyframes: List[Keyframe], output_path: Path) -> None:
    """
    Save keyframe metadata to JSON file.

    Args:
        keyframes: List of Keyframe objects
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [kf.to_dict() for kf in keyframes]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved keyframe metadata to {output_path}")


def load_keyframe_metadata(metadata_path: Path) -> List[Keyframe]:
    """
    Load keyframe metadata from JSON file.

    Args:
        metadata_path: Path to JSON file

    Returns:
        List of Keyframe objects
    """
    with open(metadata_path) as f:
        data = json.load(f)
    return [Keyframe.from_dict(item) for item in data]
