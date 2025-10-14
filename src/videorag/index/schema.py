"""Milvus collection schema definitions for VideoRAG."""
from dataclasses import dataclass
from typing import List

from pymilvus import CollectionSchema, DataType, FieldSchema

from videorag.config.settings import settings


@dataclass
class ChunkDocument:
    """Document representing a video chunk with multimodal embeddings."""

    id: str  # Unique ID: {video_id}_chunk_{chunk_idx}
    video_id: str
    chunk_idx: int
    start_time: float
    end_time: float
    video_path: str
    chunk_path: str

    # Text data
    transcript: str
    language: str

    # Embeddings
    clip_embedding: List[float]  # Visual embedding (averaged keyframes)
    text_embedding: List[float]  # Transcript embedding

    # Optional metadata
    keyframe_paths: List[str]  # JSON-encoded list of keyframe paths
    num_keyframes: int


def create_collection_schema() -> CollectionSchema:
    """
    Create Milvus collection schema for VideoRAG.

    Returns:
        CollectionSchema with fields for multimodal video chunks
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_idx", dtype=DataType.INT32),
        FieldSchema(name="start_time", dtype=DataType.FLOAT),
        FieldSchema(name="end_time", dtype=DataType.FLOAT),
        FieldSchema(name="video_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_path", dtype=DataType.VARCHAR, max_length=512),
        # Text fields
        FieldSchema(name="transcript", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
        # Vector fields
        FieldSchema(
            name="clip_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.clip_dim,
        ),
        FieldSchema(
            name="text_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.text_dim,
        ),
        # Metadata
        FieldSchema(name="keyframe_paths", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="num_keyframes", dtype=DataType.INT32),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="VideoRAG multimodal video chunk collection",
        enable_dynamic_field=False,
    )

    return schema


def get_index_params() -> dict:
    """
    Get index parameters for vector fields.

    Uses IVF_FLAT for simplicity and good recall.
    For production, consider HNSW for better performance.

    Returns:
        Dictionary of index parameters
    """
    return {
        "metric_type": "COSINE",  # Cosine similarity (assumes normalized vectors)
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},  # Number of clusters
    }


def get_search_params() -> dict:
    """
    Get search parameters for ANN queries.

    Returns:
        Dictionary of search parameters
    """
    return {
        "metric_type": "COSINE",
        "params": {"nprobe": 16},  # Number of clusters to search
    }
