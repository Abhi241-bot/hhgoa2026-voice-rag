"""
Shared Pydantic data models for structured I/O across all pipeline stages.
Every stage accepts and returns typed models — no raw dicts passed between components.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import time


# ── Enums ────────────────────────────────────────────────────────────────────


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    METADATA_AWARE = "metadata_aware"


class GuardrailStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"


# ── Stage Models ─────────────────────────────────────────────────────────────


class AudioInput(BaseModel):
    """Raw audio input — either a file path or bytes."""

    audio_bytes: Optional[bytes] = Field(default=None, description="Raw audio bytes")
    file_path: Optional[str] = Field(default=None, description="Path to audio file")
    format: str = Field(default="wav", description="Audio format: wav, mp3")

    model_config = {"arbitrary_types_allowed": True}


class TranscriptionResult(BaseModel):
    """Output of the STT stage."""

    transcript: str = Field(description="Transcribed text")
    confidence: float = Field(default=1.0, description="Confidence score 0-1")
    language: Optional[str] = Field(default=None, description="Detected language code")
    latency_ms: float = Field(description="STT latency in milliseconds")


class TextQuery(BaseModel):
    """User query (text), either from STT or direct input."""

    text: str = Field(description="The query text")
    source: str = Field(default="text", description="Source: 'stt' or 'text'")


class Chunk(BaseModel):
    """A single text chunk from the dataset."""

    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    source_passage_id: Optional[str] = Field(default=None, description="Original passage ID")
    language: Optional[str] = Field(default=None, description="Language code")
    strategy: ChunkingStrategy = Field(description="Chunking strategy used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")
    char_count: int = Field(default=0, description="Character count")

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.text)


class RetrievedContext(BaseModel):
    """A retrieved chunk with relevance scores."""

    chunk: Chunk
    dense_score: float = Field(default=0.0, description="Cosine similarity score")
    sparse_score: float = Field(default=0.0, description="BM25 score")
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion score")
    rank: int = Field(description="Final rank after fusion")


class RetrievalResult(BaseModel):
    """Output of the retrieval stage."""

    query: str = Field(description="The query string")
    contexts: list[RetrievedContext] = Field(description="Retrieved and ranked contexts")
    latency_ms: float = Field(description="Retrieval latency in milliseconds")
    num_results: int = Field(default=0)

    def model_post_init(self, __context: Any) -> None:
        if not self.num_results:
            self.num_results = len(self.contexts)


class GuardrailResult(BaseModel):
    """Result of a single guardrail check."""

    name: str = Field(description="Guardrail name")
    status: GuardrailStatus = Field(description="passed or blocked")
    reason: Optional[str] = Field(default=None, description="Reason if blocked")
    score: Optional[float] = Field(default=None, description="Numeric score if applicable")
    latency_ms: float = Field(default=0.0)


class LLMResponse(BaseModel):
    """Raw LLM generation output."""

    answer: str = Field(description="Generated answer text")
    model: str = Field(description="Model name used")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    latency_ms: float = Field(description="Generation latency in milliseconds")


class PipelineResponse(BaseModel):
    """Final structured response from the full pipeline."""

    status: PipelineStatus = Field(description="Overall pipeline status")
    answer: Optional[str] = Field(default=None, description="Final answer text")
    sources: list[str] = Field(default_factory=list, description="Source passage IDs used")
    confidence: float = Field(default=0.0, description="Answer confidence 0-1")
    guardrail_results: list[GuardrailResult] = Field(default_factory=list)
    latency_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Latency per stage in ms"
    )
    total_latency_ms: float = Field(default=0.0, description="Total pipeline latency in ms")
    error: Optional[str] = Field(default=None, description="Error message if status=error")
    query: Optional[str] = Field(default=None, description="The original query")

    def compute_total_latency(self) -> None:
        """Sum all stage latencies into total_latency_ms."""
        self.total_latency_ms = sum(self.latency_breakdown.values())


# ── Latency Record ────────────────────────────────────────────────────────────


class LatencyRecord(BaseModel):
    """Single latency measurement for analytics."""

    query_id: int
    query: str
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    guardrails_ms: float = 0.0
    total_ms: float = 0.0
    status: PipelineStatus = PipelineStatus.SUCCESS
    timestamp: float = Field(default_factory=time.time)
