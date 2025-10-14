"""Video I/O utilities for loading and processing video files."""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from videorag.utils.paths import get_video_id


class VideoInfo:
    """Container for video metadata."""

    def __init__(
        self,
        path: Path,
        fps: float,
        duration: float,
        width: int,
        height: int,
        frame_count: int,
    ):
        self.path = path
        self.video_id = get_video_id(path)
        self.fps = fps
        self.duration = duration
        self.width = width
        self.height = height
        self.frame_count = frame_count

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "video_id": self.video_id,
            "fps": self.fps,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoInfo":
        """Create from dictionary."""
        return cls(
            path=Path(data["path"]),
            fps=data["fps"],
            duration=data["duration"],
            width=data["width"],
            height=data["height"],
            frame_count=data["frame_count"],
        )


def get_video_info(video_path: Path) -> VideoInfo:
    """
    Extract metadata from a video file.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo object with metadata

    Raises:
        ValueError: If video cannot be opened
    """
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    logger.info(
        f"Video info: {video_path.name} - {duration:.1f}s, {fps:.1f} fps, "
        f"{width}x{height}, {frame_count} frames"
    )

    return VideoInfo(
        path=video_path,
        fps=fps,
        duration=duration,
        width=width,
        height=height,
        frame_count=frame_count,
    )


def extract_frame_at_time(video_path: Path, timestamp: float) -> np.ndarray | None:
    """
    Extract a single frame at a specific timestamp.

    Args:
        video_path: Path to video file
        timestamp: Time in seconds

    Returns:
        Frame as numpy array (BGR format) or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    # Set position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.warning(f"Failed to extract frame at {timestamp}s from {video_path.name}")
        return None

    return frame


def save_frame(frame: np.ndarray, output_path: Path, quality: int = 90) -> None:
    """
    Save a frame as JPEG image.

    Args:
        frame: Frame as numpy array (BGR format)
        output_path: Path to save image
        quality: JPEG quality (0-100)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
