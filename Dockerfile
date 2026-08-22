# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — HH Goa 2026 Voice RAG
# Builds a single container: FastAPI backend + HTML/JS frontend on port 8000
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps: ffmpeg for audio decoding, libgomp for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed for chunking
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY frontend/ ./frontend/

# Copy config files
COPY pyproject.toml .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
