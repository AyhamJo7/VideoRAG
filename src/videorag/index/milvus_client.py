"""Milvus client for VideoRAG index operations."""

import json
from typing import Dict, List, Optional

from loguru import logger
from pymilvus import Collection, connections, utility

from videorag.config.settings import settings
from videorag.index.schema import (
    ChunkDocument,
    create_collection_schema,
    get_index_params,
    get_search_params,
)


class MilvusClient:
    """Client for managing Milvus collections and vector operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize Milvus client.

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of collection to use
        """
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port
        self.collection_name = collection_name or settings.collection_name
        self.collection: Optional[Collection] = None

        self._connect()

    def _connect(self) -> None:
        """Establish connection to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=settings.milvus_user or "",
                password=settings.milvus_password or "",
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self, drop_existing: bool = False) -> Collection:
        """
        Create collection with VideoRAG schema.

        Args:
            drop_existing: Whether to drop existing collection

        Returns:
            Created collection
        """
        if drop_existing and utility.has_collection(self.collection_name):
            logger.warning(f"Dropping existing collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        if utility.has_collection(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists")
            self.collection = Collection(self.collection_name)
        else:
            schema = create_collection_schema()
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using="default",
            )
            logger.info(f"Created collection: {self.collection_name}")

        return self.collection

    def create_indexes(self) -> None:
        """Create indexes on vector fields for efficient search."""
        if self.collection is None:
            raise ValueError("Collection not initialized")

        index_params = get_index_params()

        # Create index for CLIP embeddings
        if not self.collection.has_index(index_name="clip_index"):
            logger.info("Creating index for clip_embedding...")
            self.collection.create_index(
                field_name="clip_embedding",
                index_params=index_params,
                index_name="clip_index",
            )

        # Create index for text embeddings
        if not self.collection.has_index(index_name="text_index"):
            logger.info("Creating index for text_embedding...")
            self.collection.create_index(
                field_name="text_embedding",
                index_params=index_params,
                index_name="text_index",
            )

        logger.info("Indexes created successfully")

    def load_collection(self) -> None:
        """Load collection into memory for searching."""
        if self.collection is None:
            raise ValueError("Collection not initialized")

        self.collection.load()
        logger.info(f"Collection {self.collection_name} loaded into memory")

    def insert(self, documents: List[ChunkDocument]) -> List[str]:
        """
        Insert documents into collection.

        Args:
            documents: List of ChunkDocument objects

        Returns:
            List of inserted IDs
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")

        if not documents:
            logger.warning("No documents to insert")
            return []

        # Prepare data
        data = [
            {
                "id": doc.id,
                "video_id": doc.video_id,
                "chunk_idx": doc.chunk_idx,
                "start_time": doc.start_time,
                "end_time": doc.end_time,
                "video_path": doc.video_path,
                "chunk_path": doc.chunk_path,
                "transcript": doc.transcript[:8191],  # Truncate if needed
                "language": doc.language[:15],
                "clip_embedding": doc.clip_embedding,
                "text_embedding": doc.text_embedding,
                "keyframe_paths": json.dumps(doc.keyframe_paths)[:2047],
                "num_keyframes": doc.num_keyframes,
            }
            for doc in documents
        ]

        result = self.collection.insert(data)
        logger.info(f"Inserted {len(documents)} documents")

        return result.primary_keys

    def search_hybrid(
        self,
        clip_query: Optional[List[float]] = None,
        text_query: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        clip_weight: Optional[float] = None,
        text_weight: Optional[float] = None,
    ) -> List[Dict]:
        """
        Perform hybrid search across CLIP and text embeddings.

        Args:
            clip_query: CLIP query embedding (optional)
            text_query: Text query embedding (optional)
            top_k: Number of results to return
            clip_weight: Weight for CLIP scores
            text_weight: Weight for text scores

        Returns:
            List of result dictionaries with scores and metadata
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")

        if clip_query is None and text_query is None:
            raise ValueError("At least one of clip_query or text_query must be provided")

        top_k = top_k or settings.top_k
        clip_weight, text_weight = settings.normalized_weights

        search_params = get_search_params()
        output_fields = [
            "video_id",
            "chunk_idx",
            "start_time",
            "end_time",
            "video_path",
            "chunk_path",
            "transcript",
            "language",
            "keyframe_paths",
            "num_keyframes",
        ]

        results_map: Dict[str, Dict] = {}

        # Search CLIP embeddings
        if clip_query is not None:
            clip_results = self.collection.search(
                data=[clip_query],
                anns_field="clip_embedding",
                param=search_params,
                limit=top_k * 2,  # Fetch more for fusion
                output_fields=output_fields,
            )[0]

            for hit in clip_results:
                doc_id = hit.id
                if doc_id not in results_map:
                    results_map[doc_id] = {
                        "id": doc_id,
                        "entity": hit.entity,
                        "clip_score": hit.score * clip_weight,
                        "text_score": 0.0,
                    }
                else:
                    results_map[doc_id]["clip_score"] = hit.score * clip_weight

        # Search text embeddings
        if text_query is not None:
            text_results = self.collection.search(
                data=[text_query],
                anns_field="text_embedding",
                param=search_params,
                limit=top_k * 2,
                output_fields=output_fields,
            )[0]

            for hit in text_results:
                doc_id = hit.id
                if doc_id not in results_map:
                    results_map[doc_id] = {
                        "id": doc_id,
                        "entity": hit.entity,
                        "clip_score": 0.0,
                        "text_score": hit.score * text_weight,
                    }
                else:
                    results_map[doc_id]["text_score"] = hit.score * text_weight

        # Compute combined scores and sort
        results = []
        for doc_id, data in results_map.items():
            combined_score = data["clip_score"] + data["text_score"]
            entity = data["entity"]

            results.append(
                {
                    "id": doc_id,
                    "score": combined_score,
                    "clip_score": data["clip_score"],
                    "text_score": data["text_score"],
                    "video_id": entity.get("video_id"),
                    "chunk_idx": entity.get("chunk_idx"),
                    "start_time": entity.get("start_time"),
                    "end_time": entity.get("end_time"),
                    "video_path": entity.get("video_path"),
                    "chunk_path": entity.get("chunk_path"),
                    "transcript": entity.get("transcript"),
                    "language": entity.get("language"),
                    "keyframe_paths": json.loads(entity.get("keyframe_paths", "[]")),
                    "num_keyframes": entity.get("num_keyframes"),
                }
            )

        # Sort by combined score
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        logger.info(f"Hybrid search returned {len(results)} results")
        return results

    def count(self) -> int:
        """Get number of entities in collection."""
        if self.collection is None:
            raise ValueError("Collection not initialized")

        return self.collection.num_entities

    def close(self) -> None:
        """Close connection to Milvus."""
        if self.collection is not None:
            self.collection.release()
        connections.disconnect(alias="default")
        logger.info("Disconnected from Milvus")
