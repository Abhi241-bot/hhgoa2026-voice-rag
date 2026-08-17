"""
LLM generation module using Groq's Llama-3.1-8b-instant.

Provides the GeneratorTool that takes a query + retrieved context and produces
a grounded, structured answer via a carefully crafted prompt template.

Design decisions:
    - Uses Groq for ultra-low latency (<100ms typical generation time).
    - Temperature=0.1 for factual, deterministic outputs.
    - System prompt enforces grounding — model must cite retrieved passages.
    - Returns structured LLMResponse with token counts for monitoring.
"""

from __future__ import annotations

import time
from typing import Optional

from groq import Groq

from src.config import settings
from src.logger import get_logger
from src.models import LLMResponse, RetrievedContext

log = get_logger("harness.generator")

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, factual question-answering assistant.

RULES:
1. Answer ONLY based on the provided context passages.
2. If the context does not contain enough information, say: "I cannot answer this based on the provided context."
3. Keep answers concise and direct (2-4 sentences max).
4. Do not hallucinate facts not present in the context.
5. Cite which passage supports your answer when possible.
"""

CONTEXT_TEMPLATE = """CONTEXT PASSAGES:
{context_block}

QUESTION: {query}

ANSWER:"""


def _format_context(contexts: list[RetrievedContext]) -> str:
    """Format retrieved contexts into a numbered block for the prompt."""
    if not contexts:
        return "No context available."
    parts = []
    for i, ctx in enumerate(contexts, start=1):
        passage_id = ctx.chunk.source_passage_id or f"chunk_{i}"
        parts.append(f"[{i}] (source: {passage_id})\n{ctx.chunk.text}")
    return "\n\n".join(parts)


class GeneratorTool:
    """
    LLM generation tool using Groq.

    Args:
        model: Groq model name (default: llama-3.1-8b-instant).
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature (low = more deterministic).
        api_key: Groq API key (defaults to settings).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self._client = Groq(api_key=api_key or settings.groq_api_key)
        log.info("generator_init", model=self.model)

    def generate(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> LLMResponse:
        """
        Generate a grounded answer from query + retrieved contexts.

        Args:
            query: The user's question.
            contexts: Retrieved context chunks.

        Returns:
            LLMResponse with answer text and metadata.
        """
        context_block = _format_context(contexts)
        user_message = CONTEXT_TEMPLATE.format(
            context_block=context_block,
            query=query,
        )

        start = time.perf_counter()

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        answer = completion.choices[0].message.content or ""

        log.info(
            "generation_complete",
            model=self.model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            latency_ms=round(latency_ms, 2),
        )

        return LLMResponse(
            answer=answer.strip(),
            model=self.model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            latency_ms=round(latency_ms, 2),
        )

    def __repr__(self) -> str:
        return f"GeneratorTool(model='{self.model}', max_tokens={self.max_tokens})"
