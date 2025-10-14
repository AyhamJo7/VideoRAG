# VideoRAG: Multimodal Retrieval-Augmented Generation over Video Corpora

[![CI](https://github.com/AyhamJo7/VideoRAG/actions/workflows/ci.yaml/badge.svg)](https://github.com/AyhamJo7/VideoRAG/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VideoRAG** is a production-grade system for multimodal retrieval-augmented generation over video collections. Ask natural language questions about your videos and get timestamped, grounded answers with source clips.

## Features

- **Multimodal Retrieval**: Hybrid search across visual (CLIP) and textual (transcript) modalities
- **Automatic Transcription**: Whisper-based ASR with speaker diarization support
- **Temporal Chunking**: Intelligent video segmentation with configurable overlap
- **Grounded Generation**: LLM answers with source citations and exact timestamps
- **Interactive UI**: Streamlit interface with video playback and keyframe previews
- **Vector Search**: Milvus-powered efficient similarity search at scale
- **Docker Ready**: Complete stack deployment with compose
- **Evaluation Tools**: Built-in metrics (Hit@K, Precision@K, MRR)

## Architecture

```
┌─────────────┐
│ Video Files │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Ingestion Pipeline                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Chunking │→ │ Keyframes│→ │ Transcription│  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Embedding & Indexing                           │
│  ┌───────────┐     ┌──────────────┐            │
│  │CLIP (img) │     │Text Embedder │            │
│  └─────┬─────┘     └───────┬──────┘            │
│        │                   │                    │
│        └────────┬──────────┘                    │
│                 ▼                                │
│         ┌──────────────┐                        │
│         │Milvus Index  │                        │
│         └──────────────┘                        │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Retrieval & Generation                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Query    │→ │ Hybrid   │→ │ Grounded LLM │  │
│  │ Encoding │  │ Search   │  │ Answer       │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
                     │
                     ▼
               ┌──────────┐
               │Streamlit │
               │    UI    │
               └──────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg
- Docker & Docker Compose (for Milvus)
- API key for OpenAI or Anthropic (for answer generation)

### Installation

```bash
# Clone repository
git clone https://github.com/AyhamJo7/VideoRAG.git
cd VideoRAG

# Create virtual environment and install
make setup

# Or manually:
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings:
# - Add OpenAI or Anthropic API key
# - Adjust chunk length, embedding models, etc.
```

### Usage

#### 1. Start Milvus

```bash
make up  # Starts Milvus with docker-compose
```

#### 2. Add Videos

Place your video files in `data/videos/`. Supported formats: MP4, AVI, MKV, MOV, WEBM.

**IMPORTANT**: Only use videos you have rights to! See `data/videos/README.md` for recommendations.

#### 3. Process Videos

```bash
# Process all videos: chunk → extract keyframes → transcribe → embed
python scripts/process_videos.py
```

This pipeline:
- Chunks videos into 30s segments (configurable)
- Extracts keyframes at 2 FPS (configurable)
- Transcribes audio with Whisper
- Computes CLIP and text embeddings

#### 4. Build Index

```bash
# Insert all processed data into Milvus
python scripts/build_index.py
```

#### 5. Launch UI

```bash
make ui  # Or: streamlit run src/videorag/ui/app.py
```

Open http://localhost:8501 and start querying!

### Example Queries

- "Find where the lecture explains backpropagation"
- "Show clips about data preprocessing"
- "When does the speaker discuss ethical AI?"

## Development

### Run Tests

```bash
make test
```

### Linting & Formatting

```bash
make format  # Run black + isort
make lint    # Run ruff + mypy
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_LENGTH_SEC` | Video chunk duration | 30 |
| `CHUNK_OVERLAP_SEC` | Chunk overlap | 5 |
| `KEYFRAME_SAMPLE_RATE` | Keyframes per second | 2.0 |
| `CLIP_MODEL` | HuggingFace CLIP model | `openai/clip-vit-base-patch32` |
| `TEXT_EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `WHISPER_MODEL` | Whisper size | `base` |
| `TOP_K` | Number of retrieval results | 5 |
| `CLIP_WEIGHT` | Visual similarity weight | 0.5 |
| `TEXT_WEIGHT` | Text similarity weight | 0.5 |
| `LLM_PROVIDER` | `openai` or `anthropic` | `openai` |

## API Usage

```python
from videorag.index.milvus_client import MilvusClient
from videorag.rag.pipeline import VideoRAGPipeline
from videorag.text.embedder import TextEmbedder
from videorag.vision.clip_embedder import CLIPEmbedder

# Initialize components
milvus_client = MilvusClient()
milvus_client.collection = milvus_client.create_collection()
milvus_client.load_collection()

text_embedder = TextEmbedder()
clip_embedder = CLIPEmbedder()

# Create pipeline
pipeline = VideoRAGPipeline(
    milvus_client=milvus_client,
    text_embedder=text_embedder,
    clip_embedder=clip_embedder,
)

# Query
result = pipeline.query(
    query="Find lectures about neural networks",
    top_k=5,
    generate_answer=True,
)

print(result["answer"])
for chunk in result["chunks"]:
    print(f"{chunk['video_id']} @ {chunk['start_time']:.1f}s: {chunk['transcript'][:100]}")
```

## Evaluation

The system includes retrieval metrics:

- **Hit@K**: Recall of any relevant item in top-k
- **Precision@K**: Fraction of top-k that are relevant
- **Recall@K**: Fraction of all relevant items in top-k
- **MRR**: Mean Reciprocal Rank

See `src/videorag/eval/metrics.py` for implementations.

## Docker Deployment

```bash
# Start full stack
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

## Project Structure

```
VideoRAG/
├── src/videorag/          # Main package
│   ├── config/            # Settings & configuration
│   ├── io/                # Video I/O and chunking
│   ├── asr/               # Whisper transcription
│   ├── vision/            # CLIP embeddings & keyframes
│   ├── text/              # Text embeddings
│   ├── index/             # Milvus client & schema
│   ├── rag/               # Retrieval & generation pipeline
│   ├── ui/                # Streamlit app
│   ├── eval/              # Evaluation metrics
│   └── utils/             # Logging, paths, etc.
├── scripts/               # CLI processing scripts
├── tests/                 # Pytest unit tests
├── docker/                # Dockerfile
├── compose.yaml           # Docker Compose for Milvus
└── data/                  # Data directories (gitignored)
```

## Troubleshooting

**Milvus connection failed**
- Ensure Milvus is running: `docker compose ps`
- Check logs: `docker compose logs milvus`

**Out of memory during embedding**
- Reduce batch sizes in scripts
- Use smaller models (e.g., `WHISPER_MODEL=tiny`)
- Set `EMBEDDING_DEVICE=cpu` if GPU OOM

**Poor retrieval quality**
- Increase `TOP_K` for more candidates
- Adjust `CLIP_WEIGHT` / `TEXT_WEIGHT` balance
- Use longer context with larger `CHUNK_LENGTH_SEC`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use VideoRAG in research, please cite:

```bibtex
@software{videorag2025,
  author = {Ayham Jo},
  title = {VideoRAG: Multimodal Retrieval-Augmented Generation over Video Corpora},
  year = {2025},
  url = {https://github.com/AyhamJo7/VideoRAG}
}
```

## Acknowledgments

Built with:
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Milvus](https://milvus.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)

---

For questions or issues, please open a [GitHub issue](https://github.com/AyhamJo7/VideoRAG/issues).
