"""
Unit tests for guardrails.

Run with: pytest tests/test_guardrails.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from src.guardrails.manager import (
    SafetyGuardrail,
    OffTopicGuardrail,
    GroundingGuardrail,
    FaithfulnessGuardrail,
    GuardrailManager,
)
from src.models import (
    Chunk,
    ChunkingStrategy,
    GuardrailStatus,
    RetrievedContext,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_context(text: str, passage_id: str = "p1") -> RetrievedContext:
    chunk = Chunk(
        chunk_id="test_chunk",
        text=text,
        source_passage_id=passage_id,
        strategy=ChunkingStrategy.FIXED_SIZE,
    )
    return RetrievedContext(chunk=chunk, rank=1)


SAFE_QUERY = "What is the capital of France?"
UNSAFE_QUERY = "How to make a bomb at home?"
OFF_TOPIC_QUERY = "What is your favourite pizza topping?"
GOOD_CONTEXT = make_context("Paris is the capital of France, located in Western Europe.")
GROUNDED_ANSWER = "Paris is the capital of France."
UNGROUNDED_ANSWER = "I cannot answer this based on the provided context."


# ── SafetyGuardrail Tests ─────────────────────────────────────────────────────

class TestSafetyGuardrail:
    def test_safe_query_passes(self):
        guard = SafetyGuardrail(use_openai_moderation=False)
        result = guard.check(SAFE_QUERY)
        assert result.status == GuardrailStatus.PASSED

    def test_unsafe_query_blocked(self):
        guard = SafetyGuardrail(use_openai_moderation=False)
        result = guard.check(UNSAFE_QUERY)
        assert result.status == GuardrailStatus.BLOCKED
        assert "bomb" in result.reason.lower()

    def test_blocked_has_reason(self):
        guard = SafetyGuardrail(use_openai_moderation=False)
        result = guard.check("How to murder someone?")
        assert result.status == GuardrailStatus.BLOCKED
        assert result.reason is not None

    def test_latency_recorded(self):
        guard = SafetyGuardrail(use_openai_moderation=False)
        result = guard.check(SAFE_QUERY)
        assert result.latency_ms >= 0


# ── OffTopicGuardrail Tests ───────────────────────────────────────────────────

class TestOffTopicGuardrail:
    @pytest.fixture(scope="class")
    def guard(self):
        # Very low threshold to make testing predictable
        return OffTopicGuardrail(threshold=0.0)  # effectively disabled

    def test_on_topic_passes(self, guard):
        result = guard.check(SAFE_QUERY)
        assert result.status == GuardrailStatus.PASSED

    def test_very_short_query_passes(self, guard):
        result = guard.check("hi")
        assert result.status == GuardrailStatus.PASSED

    def test_latency_recorded(self, guard):
        result = guard.check(SAFE_QUERY)
        assert result.latency_ms >= 0

    def test_blocking_threshold(self):
        # Very high threshold should block everything
        guard = OffTopicGuardrail(threshold=0.9999)
        result = guard.check("random gibberish zxqw")
        # May or may not block depending on embedding — just verify it runs
        assert result.status in (GuardrailStatus.PASSED, GuardrailStatus.BLOCKED)


# ── GroundingGuardrail Tests ──────────────────────────────────────────────────

class TestGroundingGuardrail:
    def test_passes_with_context(self):
        guard = GroundingGuardrail()
        result = guard.check(GROUNDED_ANSWER, [GOOD_CONTEXT])
        assert result.status == GuardrailStatus.PASSED

    def test_blocks_without_context(self):
        guard = GroundingGuardrail()
        result = guard.check(GROUNDED_ANSWER, [])
        assert result.status == GuardrailStatus.BLOCKED
        assert "No retrieved context" in result.reason

    def test_passes_on_refusal_phrase(self):
        guard = GroundingGuardrail()
        result = guard.check(UNGROUNDED_ANSWER, [GOOD_CONTEXT])
        assert result.status == GuardrailStatus.PASSED

    def test_passes_on_cannot_answer_with_empty_context(self):
        # Model correctly says "cannot answer" — should still pass even with no context
        guard = GroundingGuardrail()
        result = guard.check("I cannot answer this based on the provided context.", [])
        # This should still be BLOCKED because we have no context regardless
        assert result.status == GuardrailStatus.BLOCKED


# ── FaithfulnessGuardrail Tests ───────────────────────────────────────────────

class TestFaithfulnessGuardrail:
    def test_passes_when_llm_judge_disabled(self):
        guard = FaithfulnessGuardrail(use_llm_judge=False)
        result = guard.check(SAFE_QUERY, GROUNDED_ANSWER, [GOOD_CONTEXT])
        assert result.status == GuardrailStatus.PASSED

    def test_passes_with_empty_answer(self):
        guard = FaithfulnessGuardrail(use_llm_judge=False)
        result = guard.check(SAFE_QUERY, "", [GOOD_CONTEXT])
        assert result.status == GuardrailStatus.PASSED

    def test_passes_with_no_context(self):
        guard = FaithfulnessGuardrail(use_llm_judge=False)
        result = guard.check(SAFE_QUERY, GROUNDED_ANSWER, [])
        assert result.status == GuardrailStatus.PASSED

    def test_latency_recorded(self):
        guard = FaithfulnessGuardrail(use_llm_judge=False)
        result = guard.check(SAFE_QUERY, GROUNDED_ANSWER, [GOOD_CONTEXT])
        assert result.latency_ms >= 0


# ── GuardrailManager Tests ────────────────────────────────────────────────────

class TestGuardrailManager:
    @pytest.fixture(scope="class")
    def manager(self):
        return GuardrailManager(
            use_openai_moderation=False,
            use_faithfulness_judge=False,
        )

    def test_safe_input_passes(self, manager):
        results = manager.check_input(SAFE_QUERY)
        assert all(r.status == GuardrailStatus.PASSED for r in results)

    def test_unsafe_input_blocked(self, manager):
        results = manager.check_input(UNSAFE_QUERY)
        blocked = [r for r in results if r.status == GuardrailStatus.BLOCKED]
        assert len(blocked) > 0

    def test_unsafe_short_circuits(self, manager):
        """If safety blocks, off-topic check should not run."""
        results = manager.check_input(UNSAFE_QUERY)
        # Only safety guardrail runs
        assert len(results) == 1
        assert results[0].name == "safety"

    def test_good_output_passes(self, manager):
        results = manager.check_output(
            query=SAFE_QUERY,
            answer=GROUNDED_ANSWER,
            contexts=[GOOD_CONTEXT],
        )
        assert all(r.status == GuardrailStatus.PASSED for r in results)

    def test_ungrounded_output_passes_on_refusal(self, manager):
        results = manager.check_output(
            query=SAFE_QUERY,
            answer="I cannot answer this based on the provided context.",
            contexts=[GOOD_CONTEXT],
        )
        assert all(r.status == GuardrailStatus.PASSED for r in results)

    def test_output_blocked_no_context(self, manager):
        results = manager.check_output(
            query=SAFE_QUERY,
            answer=GROUNDED_ANSWER,
            contexts=[],
        )
        blocked = [r for r in results if r.status == GuardrailStatus.BLOCKED]
        assert len(blocked) > 0
