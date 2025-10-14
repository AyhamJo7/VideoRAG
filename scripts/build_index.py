#!/usr/bin/env python3
"""Build Milvus index from processed video embeddings and metadata."""
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from videorag.config.settings import settings
from videorag.index.milvus_client import MilvusClient
from videorag.index.schema import ChunkDocument
from videorag.utils.logging import setup_logging
from videorag.utils.paths import get_embedding_path, get_transcript_path, get_video_id
from loguru import logger


def collect_documents_for_video(video_path: Path) -> list[ChunkDocument]:
    """Collect all chunk documents for a single video."""
    video_id = get_video_id(video_path)

    # Load chunk metadata
    chunk_metadata_path = settings.chunk_dir / f"{video_id}_chunks.json"
    if not chunk_metadata_path.exists():
        logger.warning(f"No chunk metadata found for {video_id}")
        return []

    with open(chunk_metadata_path) as f:
        chunks_data = json.load(f)

    # Load keyframe metadata
    keyframe_metadata_path = settings.keyframe_dir / f"{video_id}_keyframes.json"
    keyframes_by_chunk = {}
    if keyframe_metadata_path.exists():
        with open(keyframe_metadata_path) as f:
            keyframes_data = json.load(f)
            for kf in keyframes_data:
                chunk_idx = kf["chunk_idx"]
                if chunk_idx not in keyframes_by_chunk:
                    keyframes_by_chunk[chunk_idx] = []
                keyframes_by_chunk[chunk_idx].append(kf["keyframe_path"])

    # Load embeddings
    clip_emb_path = get_embedding_path(video_id, "clip", settings.embedding_dir)
    text_emb_path = get_embedding_path(video_id, "text", settings.embedding_dir)

    if not clip_emb_path.exists() or not text_emb_path.exists():
        logger.warning(f"Missing embeddings for {video_id}")
        return []

    clip_embeddings = np.load(clip_emb_path)
    text_embeddings = np.load(text_emb_path)

    documents = []
    for idx, chunk_data in enumerate(chunks_data):
        chunk_idx = chunk_data["chunk_idx"]

        # Load transcript
        transcript_path = get_transcript_path(video_id, chunk_idx, settings.transcript_dir)
        if not transcript_path.exists():
            logger.warning(f"Missing transcript for {video_id} chunk {chunk_idx}")
            continue

        with open(transcript_path) as f:
            transcript_data = json.load(f)

        # Create document
        doc = ChunkDocument(
            id=f"{video_id}_chunk_{chunk_idx:04d}",
            video_id=video_id,
            chunk_idx=chunk_idx,
            start_time=chunk_data["start_time"],
            end_time=chunk_data["end_time"],
            video_path=chunk_data["original_video_path"],
            chunk_path=chunk_data["chunk_path"],
            transcript=transcript_data["full_text"],
            language=transcript_data.get("language", "unknown"),
            clip_embedding=clip_embeddings[idx].tolist(),
            text_embedding=text_embeddings[idx].tolist(),
            keyframe_paths=keyframes_by_chunk.get(chunk_idx, []),
            num_keyframes=len(keyframes_by_chunk.get(chunk_idx, [])),
        )
        documents.append(doc)

    logger.info(f"Collected {len(documents)} documents for {video_id}")
    return documents


def main():
    """Build Milvus index from all processed videos."""
    setup_logging()
    settings.ensure_dirs()

    # Connect to Milvus
    logger.info("Connecting to Milvus...")
    client = MilvusClient()

    # Create collection
    logger.info("Creating collection...")
    client.create_collection(drop_existing=True)

    # Collect documents from all videos
    video_files = list(settings.video_dir.glob("*.*"))
    video_files = [
        f for f in video_files
        if f.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov", ".webm"]
    ]

    all_documents = []
    for video_path in tqdm(video_files, desc="Collecting documents"):
        try:
            documents = collect_documents_for_video(video_path)
            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Failed to collect documents for {video_path.name}: {e}")
            continue

    if not all_documents:
        logger.error("No documents to index!")
        return

    logger.info(f"Total documents to index: {len(all_documents)}")

    # Insert documents
    logger.info("Inserting documents into Milvus...")
    client.insert(all_documents)

    # Create indexes
    logger.info("Creating vector indexes...")
    client.create_indexes()

    # Load collection
    logger.info("Loading collection...")
    client.load_collection()

    logger.info(f"Index built successfully! Total entities: {client.count()}")
    client.close()


if __name__ == "__main__":
    main()
