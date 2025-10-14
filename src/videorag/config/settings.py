"""
Configuration management for VideoRAG using Pydantic settings.
Loads from environment variables and .env files.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main configuration class for VideoRAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    video_dir: Path = Field(default=Path("./data/videos"))
    chunk_dir: Path = Field(default=Path("./data/chunks"))
    keyframe_dir: Path = Field(default=Path("./data/keyframes"))
    transcript_dir: Path = Field(default=Path("./data/transcripts"))
    embedding_dir: Path = Field(default=Path("./data/embeddings"))
    clip_output_dir: Path = Field(default=Path("./data/clips"))

    # Video Processing
    chunk_length_sec: int = Field(default=30, ge=5, le=300)
    chunk_overlap_sec: int = Field(default=5, ge=0, le=60)
    keyframe_sample_rate: float = Field(default=2.0, ge=0.1, le=30.0)
    video_quality: Literal["low", "medium", "high"] = Field(default="medium")

    # Models
    clip_model: str = Field(default="openai/clip-vit-base-patch32")
    text_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    whisper_model: str = Field(default="base")
    whisper_device: str = Field(default="cpu")
    embedding_device: str = Field(default="cpu")

    # Vector Database
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_user: str = Field(default="")
    milvus_password: str = Field(default="")
    collection_name: str = Field(default="videorag_collection")
    clip_dim: int = Field(default=512)
    text_dim: int = Field(default=384)

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=100)
    clip_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_rerank: bool = Field(default=False)

    # LLM
    llm_provider: Literal["openai", "anthropic"] = Field(default="openai")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4-turbo-preview")
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, ge=100, le=8000)

    # UI
    ui_port: int = Field(default=8501)
    ui_theme: Literal["light", "dark"] = Field(default="light")
    show_debug_info: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Path = Field(default=Path("./logs/videorag.log"))

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.video_dir,
            self.chunk_dir,
            self.keyframe_dir,
            self.transcript_dir,
            self.embedding_dir,
            self.clip_output_dir,
            self.log_file.parent,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def milvus_uri(self) -> str:
        """Get Milvus connection URI."""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def normalized_weights(self) -> tuple[float, float]:
        """Get normalized CLIP and text weights that sum to 1.0."""
        total = self.clip_weight + self.text_weight
        if total == 0:
            return 0.5, 0.5
        return self.clip_weight / total, self.text_weight / total


# Global settings instance
settings = Settings()
