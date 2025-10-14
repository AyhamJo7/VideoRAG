"""CLIP-based visual embeddings for keyframes."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from videorag.config.settings import settings


class CLIPEmbedder:
    """Wrapper for CLIP model to generate image and text embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        """
        Initialize CLIP embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cpu, cuda, mps)
        """
        self.model_name = model_name or settings.clip_model
        self.device = device or settings.embedding_device

        logger.info(f"Loading CLIP model: {self.model_name} on {self.device}")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully")

    @torch.no_grad()
    def embed_image(self, image_path: Path) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image_path: Path to image file

        Returns:
            Normalized embedding vector

        Raises:
            ValueError: If image cannot be loaded
        """
        if not image_path.exists():
            raise ValueError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().flatten()
        except Exception as e:
            raise ValueError(f"Failed to embed image {image_path}: {e}")

    @torch.no_grad()
    def embed_images_batch(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple images in batches.

        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once

        Returns:
            Array of embeddings with shape (num_images, embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch_paths)} images")

            # Load batch of images
            images = []
            valid_indices = []
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_indices.append(i + idx)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    continue

            if not images:
                continue

            # Process batch
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
                self.device
            )
            embeddings = self.model.get_image_features(**inputs)
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())

        if not all_embeddings:
            raise ValueError("No valid images could be processed")

        embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings for {len(embeddings)} images")
        return embeddings

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text query (for CLIP-text retrieval).

        Args:
            text: Text query

        Returns:
            Normalized embedding vector
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        embeddings = self.model.get_text_features(**inputs)
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_texts_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple text strings in batches.

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once

        Returns:
            Array of embeddings with shape (num_texts, embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(
                self.device
            )
            embeddings = self.model.get_text_features(**inputs)
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated CLIP text embeddings for {len(embeddings)} texts")
        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get dimension of embeddings."""
        return self.model.config.projection_dim
