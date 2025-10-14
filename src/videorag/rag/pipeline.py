"""RAG pipeline for multimodal video retrieval and grounded answer generation."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from videorag.config.settings import settings
from videorag.index.milvus_client import MilvusClient
from videorag.text.embedder import TextEmbedder
from videorag.vision.clip_embedder import CLIPEmbedder


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""

    chunks: List[Dict]
    query: str
    clip_query_used: bool
    text_query_used: bool


class VideoRAGPipeline:
    """End-to-end RAG pipeline for video Q&A."""

    def __init__(
        self,
        milvus_client: MilvusClient,
        text_embedder: TextEmbedder,
        clip_embedder: Optional[CLIPEmbedder] = None,
    ):
        """
        Initialize VideoRAG pipeline.

        Args:
            milvus_client: Milvus client for retrieval
            text_embedder: Text embedder for query encoding
            clip_embedder: Optional CLIP embedder for text-to-image queries
        """
        self.milvus_client = milvus_client
        self.text_embedder = text_embedder
        self.clip_embedder = clip_embedder

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_clip: bool = True,
        use_text: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant video chunks for a query.

        Args:
            query: User query text
            top_k: Number of results to return
            use_clip: Whether to use CLIP text embeddings for visual search
            use_text: Whether to use text embeddings for transcript search

        Returns:
            RetrievalResult with retrieved chunks
        """
        top_k = top_k or settings.top_k

        # Encode query
        clip_query = None
        text_query = None

        if use_clip and self.clip_embedder is not None:
            clip_query = self.clip_embedder.embed_text(query).tolist()
            logger.info("Encoded query with CLIP")

        if use_text:
            text_query = self.text_embedder.embed(query).tolist()
            logger.info("Encoded query with text embedder")

        # Hybrid search
        chunks = self.milvus_client.search_hybrid(
            clip_query=clip_query,
            text_query=text_query,
            top_k=top_k,
        )

        return RetrievalResult(
            chunks=chunks,
            query=query,
            clip_query_used=clip_query is not None,
            text_query_used=text_query is not None,
        )

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_clip: bool = True,
        use_text: bool = True,
        generate_answer: bool = True,
    ) -> Dict:
        """
        End-to-end query: retrieve chunks and optionally generate answer.

        Args:
            query: User query text
            top_k: Number of chunks to retrieve
            use_clip: Whether to use CLIP for visual search
            use_text: Whether to use text for transcript search
            generate_answer: Whether to generate LLM answer

        Returns:
            Dictionary with 'chunks', 'answer' (if generated), and metadata
        """
        # Retrieve
        result = self.retrieve(query=query, top_k=top_k, use_clip=use_clip, use_text=use_text)

        response = {
            "query": query,
            "chunks": result.chunks,
            "num_results": len(result.chunks),
        }

        # Generate answer if requested
        if generate_answer and result.chunks:
            from videorag.rag.generator import generate_grounded_answer

            answer = generate_grounded_answer(query=query, chunks=result.chunks)
            response["answer"] = answer
        else:
            response["answer"] = None

        return response
