#!/usr/bin/env python3
"""
Complete video processing pipeline: chunking → keyframes → transcription → embeddings.
Run this script after placing videos in data/videos/.
"""
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from videorag.asr.whisper_transcriber import WhisperTranscriber, save_transcript
from videorag.config.settings import settings
from videorag.io.chunking import chunk_video, save_chunk_metadata
from videorag.text.embedder import TextEmbedder
from videorag.utils.logging import setup_logging
from videorag.utils.paths import get_embedding_path, get_transcript_path, get_video_id
from videorag.vision.clip_embedder import CLIPEmbedder
from videorag.vision.keyframes import extract_keyframes_uniform, save_keyframe_metadata
from loguru import logger


def process_single_video(video_path: Path, skip_existing: bool = True):
    """Process a single video through the complete pipeline."""
    logger.info(f"Processing video: {video_path.name}")
    video_id = get_video_id(video_path)

    # Step 1: Chunk video
    chunks = chunk_video(video_path, settings.chunk_dir)
    chunk_metadata_path = settings.chunk_dir / f"{video_id}_chunks.json"
    save_chunk_metadata(chunks, chunk_metadata_path)

    # Step 2: Extract keyframes from each chunk
    all_keyframes = []
    for chunk in tqdm(chunks, desc="Extracting keyframes"):
        keyframes = extract_keyframes_uniform(
            video_path=chunk.chunk_path,
            video_id=chunk.video_id,
            chunk_idx=chunk.chunk_idx,
            chunk_start_time=chunk.start_time,
            chunk_end_time=chunk.end_time,
            output_dir=settings.keyframe_dir,
        )
        all_keyframes.extend(keyframes)

    keyframe_metadata_path = settings.keyframe_dir / f"{video_id}_keyframes.json"
    save_keyframe_metadata(all_keyframes, keyframe_metadata_path)

    # Step 3: Transcribe chunks
    transcriber = WhisperTranscriber()
    for chunk in tqdm(chunks, desc="Transcribing"):
        transcript_path = get_transcript_path(chunk.video_id, chunk.chunk_idx, settings.transcript_dir)
        if skip_existing and transcript_path.exists():
            logger.debug(f"Skipping existing transcript: {transcript_path.name}")
            continue

        transcript = transcriber.transcribe_to_transcript(
            audio_path=chunk.chunk_path,
            video_id=chunk.video_id,
            chunk_idx=chunk.chunk_idx,
        )
        save_transcript(transcript, transcript_path)

    # Step 4: Compute CLIP embeddings for keyframes
    clip_embedder = CLIPEmbedder()
    # Group keyframes by chunk
    keyframes_by_chunk = {}
    for kf in all_keyframes:
        key = (kf.video_id, kf.chunk_idx)
        if key not in keyframes_by_chunk:
            keyframes_by_chunk[key] = []
        keyframes_by_chunk[key].append(kf)

    clip_embeddings = {}
    for (vid_id, chunk_idx), kf_list in tqdm(keyframes_by_chunk.items(), desc="CLIP embeddings"):
        kf_paths = [kf.keyframe_path for kf in kf_list]
        embeddings = clip_embedder.embed_images_batch(kf_paths, batch_size=16)
        # Average embeddings for the chunk
        avg_embedding = embeddings.mean(axis=0)
        clip_embeddings[(vid_id, chunk_idx)] = avg_embedding

    clip_emb_path = get_embedding_path(video_id, "clip", settings.embedding_dir)
    np.save(clip_emb_path, np.array(list(clip_embeddings.values())))
    logger.info(f"Saved CLIP embeddings: {clip_emb_path}")

    # Step 5: Compute text embeddings for transcripts
    text_embedder = TextEmbedder()
    transcripts = []
    for chunk in chunks:
        transcript_path = get_transcript_path(chunk.video_id, chunk.chunk_idx, settings.transcript_dir)
        with open(transcript_path) as f:
            data = json.load(f)
            transcripts.append(data["full_text"])

    text_embeddings = text_embedder.embed_batch(transcripts, batch_size=32)
    text_emb_path = get_embedding_path(video_id, "text", settings.embedding_dir)
    np.save(text_emb_path, text_embeddings)
    logger.info(f"Saved text embeddings: {text_emb_path}")

    logger.info(f"Completed processing: {video_path.name}")


def main():
    """Process all videos in the video directory."""
    setup_logging()
    settings.ensure_dirs()

    video_files = list(settings.video_dir.glob("*.*"))
    video_files = [
        f for f in video_files
        if f.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov", ".webm"]
    ]

    if not video_files:
        logger.warning(f"No video files found in {settings.video_dir}")
        logger.info("Please add videos to data/videos/ directory")
        return

    logger.info(f"Found {len(video_files)} videos to process")

    for video_path in video_files:
        try:
            process_single_video(video_path)
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            continue

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
