"""Logging utilities for VideoRAG."""
import sys
from pathlib import Path

from loguru import logger

from videorag.config.settings import settings


def setup_logging() -> None:
    """Configure loguru logger with file and console outputs."""
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # File handler
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )

    logger.info(f"Logging initialized: level={settings.log_level}, file={settings.log_file}")
