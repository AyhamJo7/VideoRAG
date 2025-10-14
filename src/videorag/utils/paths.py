"""Path utilities for VideoRAG."""

import hashlib
from pathlib import Path


def get_video_id(video_path: Path) -> str:
    """
    Generate a unique ID for a video based on its path.

    Args:
        video_path: Path to video file

    Returns:
        Hexadecimal string ID
    """
    return hashlib.md5(str(video_path).encode()).hexdigest()[:12]


def get_chunk_path(video_id: str, chunk_idx: int, output_dir: Path) -> Path:
    """
    Get path for a video chunk.

    Args:
        video_id: Unique video identifier
        chunk_idx: Chunk index
        output_dir: Directory to store chunks

    Returns:
        Path to chunk file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_id}_chunk_{chunk_idx:04d}.mp4"


def get_keyframe_path(video_id: str, chunk_idx: int, frame_idx: int, output_dir: Path) -> Path:
    """
    Get path for a keyframe image.

    Args:
        video_id: Unique video identifier
        chunk_idx: Chunk index
        frame_idx: Frame index within chunk
        output_dir: Directory to store keyframes

    Returns:
        Path to keyframe file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_id}_c{chunk_idx:04d}_f{frame_idx:04d}.jpg"


def get_transcript_path(video_id: str, chunk_idx: int, output_dir: Path) -> Path:
    """
    Get path for a transcript JSON file.

    Args:
        video_id: Unique video identifier
        chunk_idx: Chunk index
        output_dir: Directory to store transcripts

    Returns:
        Path to transcript file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_id}_chunk_{chunk_idx:04d}.json"


def get_embedding_path(video_id: str, modality: str, output_dir: Path) -> Path:
    """
    Get path for embeddings file.

    Args:
        video_id: Unique video identifier
        modality: 'clip' or 'text'
        output_dir: Directory to store embeddings

    Returns:
        Path to embeddings file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_id}_{modality}_embeddings.npy"
