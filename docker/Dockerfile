# Dockerfile for VideoRAG application
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p data/videos data/chunks data/keyframes data/transcripts data/embeddings data/clips logs

# Expose Streamlit port
EXPOSE 8501

# Set environment
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["streamlit", "run", "src/videorag/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
