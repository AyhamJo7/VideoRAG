.PHONY: help setup install install-dev clean test lint format ingest embed index ui up down logs

help:
	@echo "VideoRAG Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Create virtual environment and install dependencies"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with dev dependencies"
	@echo "  clean        - Remove build artifacts and caches"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run linters (ruff, mypy)"
	@echo "  format       - Format code (black, isort)"
	@echo "  ingest       - Download and chunk demo videos"
	@echo "  embed        - Extract keyframes and compute embeddings"
	@echo "  index        - Build vector index"
	@echo "  ui           - Launch Streamlit UI"
	@echo "  up           - Start Docker services"
	@echo "  down         - Stop Docker services"
	@echo "  logs         - Show Docker logs"

setup:
	python3.11 -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -e ".[dev]"
	. venv/bin/activate && pre-commit install

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov

test:
	pytest

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

ingest:
	python scripts/download_demo_data.py
	python scripts/chunk_videos.py

embed:
	python scripts/extract_keyframes.py
	python scripts/embed_clip.py
	python scripts/transcribe_whisper.py
	python scripts/embed_text.py

index:
	python scripts/build_index.py

ui:
	streamlit run src/videorag/ui/app.py

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f
