#!/usr/bin/env python3
"""Download demo videos for VideoRAG testing."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from videorag.config.settings import settings
from videorag.utils.logging import setup_logging


def create_demo_placeholder():
    """
    Create placeholder for demo videos.

    For production use, download from licensed sources or use your own content.
    This script creates a README explaining where to add demo videos.
    """
    settings.ensure_dirs()

    readme_path = settings.video_dir / "README.md"
    readme_content = """# Demo Videos

Please add your own video files to this directory for testing VideoRAG.

## Supported Formats
- MP4, AVI, MKV, MOV, WEBM

## Recommended Test Videos
1. Educational lectures (e.g., from your own content or CC-licensed sources)
2. Conference talks
3. Tutorial videos
4. Documentary clips

## Example: Public Domain Sources
- Archive.org (archive.org)
- Wikimedia Commons (commons.wikimedia.org)

## Usage
1. Place video files in this directory
2. Run: `make ingest` to process them
3. Run: `make embed` to generate embeddings
4. Run: `make index` to build the search index

IMPORTANT: Only use videos you have rights to use!
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)

    logger.info(f"Created demo data README at {readme_path}")
    logger.info("Please add your own video files to data/videos/")
    logger.info("You can then run: make ingest embed index")


if __name__ == "__main__":
    setup_logging()
    create_demo_placeholder()
