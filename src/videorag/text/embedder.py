"""Text embeddings using sentence transformers for transcript retrieval."""

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from videorag.config.settings import settings


class TextEmbedder:
    """Wrapper for sentence transformer model to generate text embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        """
        Initialize text embedder.

        Args:
            model_name: Sentence transformer model name
            device: Device to run on (cpu, cuda, mps)
        """
        self.model_name = model_name or settings.text_embedding_model
        self.device = device or settings.embedding_device

        logger.info(f"Loading text embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Text embedding model loaded successfully")

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            normalize: Whether to L2 normalize the embedding

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            normalize: Whether to L2 normalize embeddings

        Returns:
            Array of embeddings with shape (num_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )

        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
