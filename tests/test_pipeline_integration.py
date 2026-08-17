"""
Integration tests for the full RAG pipeline using mocked external services.

Tests the complete flow from text input → retrieval → generation → response,
without requiring real API keys (Groq, ElevenLabs are mocked).

Run with: pytest tests/test_pipeline_integration.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models import (
    Chunk,
    ChunkingStrategy,
    GuardrailStatus,
    LLMResponse,
    PipelineResponse,
    PipelineStatus,
    RetrievedContext,
    RetrievalResult,
    TextQuery,
    AudioInput,
    TranscriptionResult,
)
from src.harness.pipeline import PipelineHarness
from src.guardrails.manager import GuardrailManager


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_context(text: str, passage_id: str = "p1", rank: int = 1) -> RetrievedContext:
    chunk = Chunk(
        chunk_id=f"chunk_{passage_id}",
        text=text,
        source_passage_id=passage_id,
        strategy=ChunkingStrategy.METADATA_AWARE,
    )
    return RetrievedContext(chunk=chunk, rank=rank, rrf_score=0.1)


def make_retrieval_result(query: str, texts: list[str]) -> RetrievalResult:
    contexts = [make_context(t, f"p{i+1}", i+1) for i, t in enumerate(texts)]
    return RetrievalResult(query=query, contexts=contexts, latency_ms=15.0)


def make_llm_response(answer: str) -> LLMResponse:
    return LLMResponse(
        answer=answer,
        model="llama-3.1-8b-instant",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=80.0,
    )


# ── Full pipeline integration tests ──────────────────────────────────────────


class TestPipelineIntegration:
    """End-to-end pipeline tests using mocked retriever and generator."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock HybridRetriever that returns controlled results."""
        retriever = MagicMock()
        retriever.retrieve.return_value = make_retrieval_result(
            query="What is the capital of France?",
            texts=["Paris is the capital of France, located in Western Europe."],
        )
        return retriever

    @pytest.fixture
    def mock_generator(self):
        """Mock GeneratorTool that returns a controlled answer."""
        generator = MagicMock()
        generator.generate.return_value = make_llm_response(
            "Paris is the capital of France."
        )
        return generator

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_generator):
        """Full pipeline with mocked retriever and generator, real guardrails."""
        guardrails = GuardrailManager(
            use_openai_moderation=False,
            use_faithfulness_judge=False,
        )
        return PipelineHarness(
            retriever=mock_retriever,
            generator=mock_generator,
            guardrail_manager=guardrails,
        )

    # ── Success path ──────────────────────────────────────────────────────

    def test_text_query_success(self, pipeline):
        """Full pipeline succeeds for a valid text query."""
        response = pipeline.run(TextQuery(text="What is the capital of France?"))
        assert response.status == PipelineStatus.SUCCESS
        assert response.answer is not None
        assert len(response.answer) > 0

    def test_string_input_works(self, pipeline):
        """Pipeline accepts raw strings, not just TextQuery objects."""
        response = pipeline.run("What is the capital of France?")
        assert response.status == PipelineStatus.SUCCESS

    def test_response_has_sources(self, pipeline):
        """Response includes source passage IDs."""
        response = pipeline.run("What is the capital of France?")
        assert response.status == PipelineStatus.SUCCESS
        assert len(response.sources) > 0

    def test_response_has_latency_breakdown(self, pipeline):
        """Response includes per-stage latency breakdown."""
        response = pipeline.run("What is the capital of France?")
        assert "retrieval_ms" in response.latency_breakdown
        assert "generation_ms" in response.latency_breakdown
        assert response.total_latency_ms > 0

    def test_confidence_is_in_range(self, pipeline):
        """Confidence score is between 0 and 1."""
        response = pipeline.run("What is the capital of France?")
        assert 0.0 <= response.confidence <= 1.0

    def test_guardrail_results_attached(self, pipeline):
        """Response includes guardrail results."""
        response = pipeline.run("What is the capital of France?")
        assert len(response.guardrail_results) > 0

    # ── Guardrail blocking tests ──────────────────────────────────────────

    def test_unsafe_query_blocked(self, pipeline):
        """Unsafe queries are blocked by the safety guardrail."""
        response = pipeline.run("How to make a bomb at home?")
        assert response.status == PipelineStatus.BLOCKED
        assert response.error is not None
        # Retriever should NOT have been called
        pipeline._retriever.retrieve.assert_not_called()

    def test_empty_query_returns_error(self, pipeline):
        """Empty queries return an error, not a crash."""
        response = pipeline.run("   ")
        assert response.status == PipelineStatus.ERROR

    # ── No context path ───────────────────────────────────────────────────

    def test_no_context_returns_blocked(self, mock_generator):
        """Pipeline blocks gracefully when no context is retrieved."""
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            query="obscure query", contexts=[], latency_ms=5.0
        )
        guardrails = GuardrailManager(use_openai_moderation=False, use_faithfulness_judge=False)
        pipeline = PipelineHarness(
            retriever=retriever,
            generator=mock_generator,
            guardrail_manager=guardrails,
        )
        response = pipeline.run("obscure query with no results")
        assert response.status == PipelineStatus.BLOCKED
        # Generator should NOT have been called — no context = no generation
        mock_generator.generate.assert_not_called()

    # ── Error recovery tests ──────────────────────────────────────────────

    def test_retriever_failure_returns_error(self, mock_generator):
        """Retriever API failure returns error status, not exception."""
        retriever = MagicMock()
        retriever.retrieve.side_effect = Exception("Connection timeout")
        guardrails = GuardrailManager(use_openai_moderation=False, use_faithfulness_judge=False)
        pipeline = PipelineHarness(
            retriever=retriever,
            generator=mock_generator,
            guardrail_manager=guardrails,
        )
        response = pipeline.run("What is AI?")
        assert response.status == PipelineStatus.ERROR
        assert "Retrieval failed" in response.error

    def test_generator_failure_returns_error(self, mock_retriever):
        """LLM failure returns error status, not exception."""
        generator = MagicMock()
        generator.generate.side_effect = Exception("Groq API error")
        guardrails = GuardrailManager(use_openai_moderation=False, use_faithfulness_judge=False)
        pipeline = PipelineHarness(
            retriever=mock_retriever,
            generator=generator,
            guardrail_manager=guardrails,
        )
        response = pipeline.run("What is AI?")
        assert response.status == PipelineStatus.ERROR
        assert "LLM generation failed" in response.error

    # ── STT integration tests (with mock STT) ─────────────────────────────

    def test_audio_input_with_mock_stt(self, mock_retriever, mock_generator):
        """Audio input works end-to-end with MockSTT."""
        from src.stt.elevenlabs import MockSTT

        mock_stt = MockSTT(mock_transcript="What is the capital of France?")
        guardrails = GuardrailManager(use_openai_moderation=False, use_faithfulness_judge=False)
        pipeline = PipelineHarness(
            retriever=mock_retriever,
            generator=mock_generator,
            guardrail_manager=guardrails,
            stt_tool=mock_stt,
        )
        audio = AudioInput(audio_bytes=b"fake_audio_bytes", format="wav")
        response = pipeline.run(audio)
        assert response.status == PipelineStatus.SUCCESS
        assert "stt_ms" in response.latency_breakdown

    def test_no_stt_configured_returns_error(self, mock_retriever, mock_generator):
        """Audio input without STT tool returns error gracefully."""
        guardrails = GuardrailManager(use_openai_moderation=False, use_faithfulness_judge=False)
        pipeline = PipelineHarness(
            retriever=mock_retriever,
            generator=mock_generator,
            guardrail_manager=guardrails,
            stt_tool=None,  # No STT
        )
        audio = AudioInput(audio_bytes=b"fake_audio_bytes", format="wav")
        response = pipeline.run(audio)
        assert response.status == PipelineStatus.ERROR
        assert "STT failed" in response.error


# ── Pydantic model tests ──────────────────────────────────────────────────────


class TestPipelineModels:
    """Verify Pydantic model correctness."""

    def test_chunk_auto_computes_char_count(self):
        chunk = Chunk(
            chunk_id="abc",
            text="Hello world",
            strategy=ChunkingStrategy.FIXED_SIZE,
        )
        assert chunk.char_count == 11

    def test_retrieval_result_auto_computes_num_results(self):
        ctx = make_context("text")
        result = RetrievalResult(query="q", contexts=[ctx, ctx], latency_ms=10.0)
        assert result.num_results == 2

    def test_pipeline_response_compute_total(self):
        resp = PipelineResponse(
            status=PipelineStatus.SUCCESS,
            answer="Yes",
            latency_breakdown={"retrieval_ms": 50.0, "generation_ms": 80.0},
        )
        resp.compute_total_latency()
        assert resp.total_latency_ms == 130.0
