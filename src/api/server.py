"""
FastAPI server wrapping the RAG pipeline.

Endpoints:
    POST /query/text       — text query input
    POST /query/voice      — voice (audio file upload) input
    GET  /health           — health check
    GET  /stats            — pipeline statistics
    GET  /                 — serves the frontend HTML UI

Run with: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.logger import get_logger
from src.models import AudioInput, PipelineResponse, TextQuery

log = get_logger("api.server")

# ── Frontend static files ─────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the pipeline in background on startup to reduce first-request latency."""
    import asyncio
    import threading

    def _warm():
        try:
            get_pipeline()
        except Exception as exc:
            log.warning("pipeline_warmup_failed", error=str(exc))

    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    title="Voice-Enabled RAG API",
    description="HH Goa 2026 — Voice-Enabled RAG pipeline with hybrid retrieval and guardrails",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy pipeline initialization ─────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.harness.pipeline import PipelineHarness
        from src.retrieval.retriever import HybridRetriever
        from src.stt.elevenlabs import get_stt_tool

        log.info("initializing_pipeline")
        retriever = HybridRetriever()
        stt = get_stt_tool()
        _pipeline = PipelineHarness(retriever=retriever, stt_tool=stt)
        log.info("pipeline_ready")
    return _pipeline


# ── Request/Response models ───────────────────────────────────────────────────


class TextQueryRequest(BaseModel):
    query: str
    language_filter: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "voice-rag-pipeline"}


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend index.html at the root URL."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"message": "Frontend not found. Run with frontend/ directory present."}


@app.post("/query/text", response_model=PipelineResponse)
async def query_text(request: TextQueryRequest):
    """
    Process a text query through the full RAG pipeline.

    Returns a structured PipelineResponse with answer, sources, and latency breakdown.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    pipeline = get_pipeline()
    response = pipeline.run(
        TextQuery(text=request.query),
        language_filter=request.language_filter,
    )
    return response


@app.post("/query/voice", response_model=PipelineResponse)
async def query_voice(
    audio: UploadFile = File(...),
    language_filter: Optional[str] = Form(None),
):
    """
    Process a voice query (audio file) through the full RAG pipeline.

    Accepts WAV or MP3 audio files.
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    audio_bytes = await audio.read()
    fmt = audio.filename.rsplit(".", 1)[-1].lower() if "." in audio.filename else "wav"

    audio_input = AudioInput(audio_bytes=audio_bytes, format=fmt)
    pipeline = get_pipeline()
    response = pipeline.run(audio_input, language_filter=language_filter)
    return response


@app.get("/stats")
async def get_stats():
    """Return pipeline and index statistics."""
    try:
        pipeline = get_pipeline()
        collection_count = 0
        if pipeline._retriever:
            collection_count = pipeline._retriever._collection.count()
        return {
            "status": "ok",
            "index_size": collection_count,
            "bm25_docs": len(pipeline._retriever._bm25_docs) if pipeline._retriever else 0,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Serve frontend static assets (JS, SVG, etc.) ─────────────────────────────
# Mounted last so API routes take priority over file paths.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
