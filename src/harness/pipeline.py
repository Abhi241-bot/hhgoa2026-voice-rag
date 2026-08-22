"""
Pipeline Orchestration Harness.

The PipelineHarness is the central orchestrator that:
    1. Accepts voice (AudioInput) or text (TextQuery) input.
    2. Runs STT if audio input is provided.
    3. Applies input guardrails (safety + off-topic check).
    4. Executes hybrid retrieval.
    5. Generates an answer via LLM.
    6. Applies output guardrails (faithfulness + grounding check).
    7. Returns a fully structured PipelineResponse.

Features:
    - Retry with exponential backoff on each tool call (via tenacity).
    - Structured JSON logging at each stage.
    - Graceful error recovery — partial failures return status=error, not crash.
    - Full latency breakdown per stage.
"""

from __future__ import annotations

import time
from typing import Optional, Union

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.guardrails.manager import GuardrailManager
from src.harness.generator import GeneratorTool
from src.logger import get_logger
from src.models import (
    AudioInput,
    GuardrailStatus,
    PipelineResponse,
    PipelineStatus,
    RetrievalResult,
    TextQuery,
)
from src.retrieval.retriever import HybridRetriever

log = get_logger("harness.pipeline")


class PipelineHarness:
    """
    End-to-end RAG pipeline orchestrator.

    Args:
        retriever: HybridRetriever instance (injected for testability).
        generator: GeneratorTool instance (injected for testability).
        guardrail_manager: GuardrailManager instance.
        stt_tool: STT tool instance (optional; text-only if None).
        top_k: Override retrieval top-k.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[GeneratorTool] = None,
        guardrail_manager: Optional["GuardrailManager"] = None,
        stt_tool=None,
        top_k: Optional[int] = None,
    ) -> None:
        self.top_k = top_k or settings.top_k_retrieval
        self._retriever = retriever
        # Delay creating GeneratorTool until generation stage to avoid
        # allocating external clients at pipeline construction time.
        self._generator = generator
        self._guardrails = guardrail_manager or GuardrailManager()
        self._stt = stt_tool
        log.info("harness_init", top_k=self.top_k)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        input: Union[TextQuery, AudioInput, str],
        language_filter: Optional[str] = None,
    ) -> PipelineResponse:
        """
        Execute the full RAG pipeline.

        Args:
            input: TextQuery, AudioInput, or raw query string.
            language_filter: Optional language filter for retrieval.

        Returns:
            PipelineResponse with answer, sources, latency breakdown.
        """
        pipeline_start = time.perf_counter()
        latency_breakdown: dict[str, float] = {}
        query_text: Optional[str] = None

        # ── Normalise input ────────────────────────────────────────────
        if isinstance(input, str):
            input = TextQuery(text=input)

        # ── Stage 1: STT (optional) ────────────────────────────────────
        if isinstance(input, AudioInput):
            try:
                stt_start = time.perf_counter()
                transcript = self._run_stt(input)
                latency_breakdown["stt_ms"] = round((time.perf_counter() - stt_start) * 1000, 2)
                query_text = transcript.transcript
                log.info("stt_complete", transcript=query_text[:100], latency_ms=latency_breakdown["stt_ms"])
            except Exception as e:
                log.error("stt_failed", error=str(e))
                return PipelineResponse(
                    status=PipelineStatus.ERROR,
                    error=f"STT failed: {e}",
                    latency_breakdown=latency_breakdown,
                )
        else:
            query_text = input.text

        if not query_text or not query_text.strip():
            return PipelineResponse(
                status=PipelineStatus.ERROR,
                error="Empty query after STT",
                query=query_text,
                latency_breakdown=latency_breakdown,
            )

        # ── Stage 2: Input guardrails ──────────────────────────────────
        guard_start = time.perf_counter()
        input_guard_results = self._guardrails.check_input(query_text)
        latency_breakdown["input_guardrails_ms"] = round(
            (time.perf_counter() - guard_start) * 1000, 2
        )

        blocked_input = [g for g in input_guard_results if g.status == GuardrailStatus.BLOCKED]
        if blocked_input:
            block_reason = blocked_input[0].reason or "Guardrail blocked"
            log.warning("input_blocked", reason=block_reason, query=query_text[:80])
            return PipelineResponse(
                status=PipelineStatus.BLOCKED,
                error=block_reason,
                query=query_text,
                guardrail_results=input_guard_results,
                latency_breakdown=latency_breakdown,
            )

        # ── Stage 3: Retrieval ─────────────────────────────────────────
        try:
            retrieval_start = time.perf_counter()
            retrieval_result = self._run_retrieval(query_text, language_filter)
            latency_breakdown["retrieval_ms"] = round(
                (time.perf_counter() - retrieval_start) * 1000, 2
            )
        except Exception as e:
            log.error("retrieval_failed", error=str(e))
            return PipelineResponse(
                status=PipelineStatus.ERROR,
                error=f"Retrieval failed: {e}",
                query=query_text,
                latency_breakdown=latency_breakdown,
            )

        # ── Stage 4: Grounding check (pre-generation) ──────────────────
        if not retrieval_result.contexts:
            log.warning("no_context_found", query=query_text[:80])
            return PipelineResponse(
                status=PipelineStatus.BLOCKED,
                error="No relevant context found in the knowledge base.",
                query=query_text,
                latency_breakdown=latency_breakdown,
            )

        # ── Stage 5: LLM generation ────────────────────────────────────
        try:
            gen_start = time.perf_counter()
            llm_response = self._run_generation(query_text, retrieval_result)
            latency_breakdown["generation_ms"] = round(
                (time.perf_counter() - gen_start) * 1000, 2
            )
        except Exception as e:
            log.error("generation_failed", error=str(e))
            return PipelineResponse(
                status=PipelineStatus.ERROR,
                error=f"LLM generation failed: {e}",
                query=query_text,
                latency_breakdown=latency_breakdown,
            )

        # ── Stage 6: Output guardrails ─────────────────────────────────
        out_guard_start = time.perf_counter()
        output_guard_results = self._guardrails.check_output(
            query=query_text,
            answer=llm_response.answer,
            contexts=retrieval_result.contexts,
        )
        latency_breakdown["output_guardrails_ms"] = round(
            (time.perf_counter() - out_guard_start) * 1000, 2
        )

        blocked_output = [g for g in output_guard_results if g.status == GuardrailStatus.BLOCKED]
        if blocked_output:
            block_reason = blocked_output[0].reason or "Output guardrail blocked"
            log.warning("output_blocked", reason=block_reason)
            return PipelineResponse(
                status=PipelineStatus.BLOCKED,
                error=block_reason,
                query=query_text,
                guardrail_results=input_guard_results + output_guard_results,
                latency_breakdown=latency_breakdown,
            )

        # ── Assemble final response ────────────────────────────────────
        all_guardrails = input_guard_results + output_guard_results
        sources = list({
            ctx.chunk.source_passage_id
            for ctx in retrieval_result.contexts
            if ctx.chunk.source_passage_id
        })

        # Confidence: average of top RRF scores (normalised)
        top_rrf = [ctx.rrf_score for ctx in retrieval_result.contexts[:3]]
        confidence = round(min(1.0, sum(top_rrf) / max(1, len(top_rrf)) * 10), 3)

        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        latency_breakdown["total_ms"] = total_ms

        log.info(
            "pipeline_success",
            query=query_text[:80],
            answer_length=len(llm_response.answer),
            confidence=confidence,
            total_ms=total_ms,
        )

        return PipelineResponse(
            status=PipelineStatus.SUCCESS,
            answer=llm_response.answer,
            sources=sources,
            confidence=confidence,
            guardrail_results=all_guardrails,
            latency_breakdown=latency_breakdown,
            total_latency_ms=total_ms,
            query=query_text,
        )

    # ── Internal tool methods (with retry) ────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def _run_stt(self, audio: AudioInput):
        """STT with exponential backoff retry."""
        if self._stt is None:
            raise RuntimeError("No STT tool configured")
        return self._stt.transcribe(audio)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def _run_retrieval(self, query: str, language_filter: Optional[str]) -> RetrievalResult:
        """Retrieval with exponential backoff retry."""
        if self._retriever is None:
            raise RuntimeError("No retriever configured. Call build_index first.")
        return self._retriever.retrieve(query, language_filter=language_filter)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def _run_generation(self, query: str, retrieval_result: RetrievalResult):
        """LLM generation with exponential backoff retry."""
        if self._generator is None:
            self._generator = GeneratorTool()
        return self._generator.generate(
            query=query,
            contexts=retrieval_result.contexts,
        )

    def __repr__(self) -> str:
        return f"PipelineHarness(top_k={self.top_k})"
