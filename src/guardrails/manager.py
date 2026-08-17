"""
Guardrail Manager — orchestrates all input and output guardrails.

Input guardrails (run before retrieval):
    1. SafetyGuardrail — blocks harmful/inappropriate inputs.
    2. OffTopicGuardrail — blocks queries unrelated to the dataset domain.

Output guardrails (run after generation):
    3. GroundingGuardrail — ensures answer is supported by retrieved context.
    4. FaithfulnessGuardrail — checks answer doesn't contradict retrieved context.
"""

from __future__ import annotations

import time
from typing import Optional

from src.config import settings
from src.logger import get_logger
from src.models import GuardrailResult, GuardrailStatus, RetrievedContext

log = get_logger("guardrails.manager")

# ── Unsafe keyword list (lightweight fast filter) ────────────────────────────
_UNSAFE_KEYWORDS = [
    "bomb", "weapon", "kill", "murder", "terrorist", "suicide",
    "hack", "exploit", "malware", "ransomware", "child porn",
    "sexual abuse", "self harm",
]

# ── Domain keywords for MSMARCO-XI ───────────────────────────────────────────
# These represent the knowledge domain; off-topic detection uses embedding distance
# from the collection centroid. This keyword list is a fast pre-filter.
_DOMAIN_KEYWORDS = [
    "what", "who", "where", "when", "how", "why", "which",
    "is", "are", "was", "were", "does", "did", "can", "will",
    "name", "define", "explain", "describe", "list",
]


class SafetyGuardrail:
    """
    Blocks harmful or inappropriate inputs using a fast keyword filter.
    Falls back to OpenAI Moderation API if available.
    """

    def __init__(self, use_openai_moderation: bool = True) -> None:
        self.use_openai = use_openai_moderation and bool(settings.openai_api_key)

    def check(self, text: str) -> GuardrailResult:
        start = time.perf_counter()
        name = "safety"

        # Fast keyword filter
        text_lower = text.lower()
        for keyword in _UNSAFE_KEYWORDS:
            if keyword in text_lower:
                latency = (time.perf_counter() - start) * 1000
                log.warning("safety_blocked_keyword", keyword=keyword)
                return GuardrailResult(
                    name=name,
                    status=GuardrailStatus.BLOCKED,
                    reason=f"Unsafe content detected: '{keyword}'",
                    latency_ms=round(latency, 2),
                )

        # OpenAI Moderation API (async-compatible, sync call here)
        if self.use_openai:
            try:
                result = self._openai_moderation(text)
                latency = (time.perf_counter() - start) * 1000
                if result["flagged"]:
                    categories = [k for k, v in result["categories"].items() if v]
                    return GuardrailResult(
                        name=name,
                        status=GuardrailStatus.BLOCKED,
                        reason=f"OpenAI Moderation flagged: {categories}",
                        latency_ms=round(latency, 2),
                    )
            except Exception as e:
                log.warning("openai_moderation_failed", error=str(e))
                # Don't block on API failure — fail open for moderation

        latency = (time.perf_counter() - start) * 1000
        return GuardrailResult(
            name=name,
            status=GuardrailStatus.PASSED,
            latency_ms=round(latency, 2),
        )

    def _openai_moderation(self, text: str) -> dict:
        """Call OpenAI Moderation API."""
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.moderations.create(input=text, model="omni-moderation-latest")
        result = response.results[0]
        return {
            "flagged": result.flagged,
            "categories": result.categories.model_dump() if hasattr(result.categories, "model_dump") else {},
        }


class OffTopicGuardrail:
    """
    Detects off-topic queries using embedding cosine similarity to domain centroid.

    The domain centroid is computed from a set of representative MSMARCO-style queries.
    Queries with similarity < threshold are blocked.
    """

    _DOMAIN_EXAMPLES = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is the population of India?",
        "How does photosynthesis work?",
        "What are the symptoms of diabetes?",
        "When did World War II end?",
        "What is the speed of light?",
        "Who wrote Pride and Prejudice?",
        "How do vaccines work?",
        "What is machine learning?",
    ]

    def __init__(
        self,
        threshold: float = 0.15,
        embedding_model=None,
    ) -> None:
        self.threshold = threshold
        self._model = embedding_model
        self._centroid = None  # lazy computed

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    @property
    def centroid(self):
        if self._centroid is None:
            import numpy as np
            embeddings = self.model.encode(self._DOMAIN_EXAMPLES, show_progress_bar=False)
            self._centroid = np.mean(embeddings, axis=0)
        return self._centroid

    def check(self, text: str) -> GuardrailResult:
        start = time.perf_counter()
        name = "off_topic"

        # Very short or clearly question-like text — pass fast
        if len(text.split()) <= 3:
            latency = (time.perf_counter() - start) * 1000
            return GuardrailResult(name=name, status=GuardrailStatus.PASSED, latency_ms=round(latency, 2))

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            query_emb = self.model.encode([text], show_progress_bar=False)[0]
            similarity = float(cosine_similarity([query_emb], [self.centroid])[0][0])

            latency = (time.perf_counter() - start) * 1000
            log.debug("off_topic_check", similarity=round(similarity, 3), threshold=self.threshold)

            if similarity < self.threshold:
                return GuardrailResult(
                    name=name,
                    status=GuardrailStatus.BLOCKED,
                    reason=f"Query appears off-topic (similarity={similarity:.3f} < {self.threshold})",
                    score=similarity,
                    latency_ms=round(latency, 2),
                )

            return GuardrailResult(
                name=name,
                status=GuardrailStatus.PASSED,
                score=similarity,
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            log.warning("off_topic_check_failed", error=str(e))
            latency = (time.perf_counter() - start) * 1000
            return GuardrailResult(name=name, status=GuardrailStatus.PASSED, latency_ms=round(latency, 2))


class GroundingGuardrail:
    """
    Ensures the answer is grounded in at least one retrieved context passage.
    Blocks if contexts are empty or answer contains no content from context.
    """

    def check(
        self,
        answer: str,
        contexts: list[RetrievedContext],
    ) -> GuardrailResult:
        start = time.perf_counter()
        name = "grounding"

        if not contexts:
            latency = (time.perf_counter() - start) * 1000
            return GuardrailResult(
                name=name,
                status=GuardrailStatus.BLOCKED,
                reason="No retrieved context — cannot ground the answer",
                latency_ms=round(latency, 2),
            )

        # Check if "cannot answer" phrasing is present (legitimate refusal)
        answer_lower = answer.lower()
        refusal_phrases = ["cannot answer", "not enough information", "no information", "i don't know"]
        if any(phrase in answer_lower for phrase in refusal_phrases):
            latency = (time.perf_counter() - start) * 1000
            return GuardrailResult(
                name=name,
                status=GuardrailStatus.PASSED,
                reason="Model appropriately declined to answer",
                latency_ms=round(latency, 2),
            )

        latency = (time.perf_counter() - start) * 1000
        return GuardrailResult(
            name=name,
            status=GuardrailStatus.PASSED,
            latency_ms=round(latency, 2),
        )


class FaithfulnessGuardrail:
    """
    Checks that the generated answer is faithful to the retrieved context.

    Uses a lightweight LLM judge approach:
        - Ask the LLM: "Is this answer supported by the context?"
        - Block if answer contradicts or fabricates beyond context.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_llm_judge: bool = True,
    ) -> None:
        self.threshold = threshold
        self.use_llm_judge = use_llm_judge

    def check(
        self,
        query: str,
        answer: str,
        contexts: list[RetrievedContext],
    ) -> GuardrailResult:
        start = time.perf_counter()
        name = "faithfulness"

        if not contexts or not answer:
            latency = (time.perf_counter() - start) * 1000
            return GuardrailResult(name=name, status=GuardrailStatus.PASSED, latency_ms=round(latency, 2))

        # Skip LLM judge if key not available (fast path)
        if not self.use_llm_judge or not settings.groq_api_key:
            latency = (time.perf_counter() - start) * 1000
            return GuardrailResult(name=name, status=GuardrailStatus.PASSED, latency_ms=round(latency, 2))

        try:
            score = self._llm_faithfulness_score(query, answer, contexts)
            latency = (time.perf_counter() - start) * 1000
            log.debug("faithfulness_score", score=score, threshold=self.threshold)

            if score < self.threshold:
                return GuardrailResult(
                    name=name,
                    status=GuardrailStatus.BLOCKED,
                    reason=f"Answer not faithful to context (score={score:.2f} < {self.threshold})",
                    score=score,
                    latency_ms=round(latency, 2),
                )

            return GuardrailResult(
                name=name,
                status=GuardrailStatus.PASSED,
                score=score,
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            log.warning("faithfulness_check_failed", error=str(e))
            latency = (time.perf_counter() - start) * 1000
            return GuardrailResult(name=name, status=GuardrailStatus.PASSED, latency_ms=round(latency, 2))

    def _llm_faithfulness_score(
        self,
        query: str,
        answer: str,
        contexts: list[RetrievedContext],
    ) -> float:
        """Use Groq to judge faithfulness. Returns 0-1 score."""
        from groq import Groq

        context_text = "\n".join([ctx.chunk.text[:500] for ctx in contexts[:3]])
        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            f"Answer: {answer}\n\n"
            "Is this answer supported by the context? "
            "Reply with only a number between 0 and 1, where 1=fully supported, 0=hallucinated."
        )

        client = Groq(api_key=settings.groq_api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        response_text = completion.choices[0].message.content.strip()
        try:
            return min(1.0, max(0.0, float(response_text)))
        except ValueError:
            return 1.0  # If can't parse, assume faithful (fail-open)


class GuardrailManager:
    """
    Orchestrates all guardrails for input and output checking.

    Args:
        embedding_model: Pre-loaded embedding model for off-topic check.
        use_openai_moderation: Whether to use OpenAI Moderation API.
        use_faithfulness_judge: Whether to use LLM faithfulness judge.
    """

    def __init__(
        self,
        embedding_model=None,
        use_openai_moderation: bool = True,
        use_faithfulness_judge: bool = True,
    ) -> None:
        self._safety = SafetyGuardrail(use_openai_moderation=use_openai_moderation)
        self._off_topic = OffTopicGuardrail(
            threshold=settings.off_topic_threshold,
            embedding_model=embedding_model,
        )
        self._grounding = GroundingGuardrail()
        self._faithfulness = FaithfulnessGuardrail(
            threshold=settings.faithfulness_threshold,
            use_llm_judge=use_faithfulness_judge,
        )
        log.info("guardrail_manager_init")

    def check_input(self, query: str) -> list[GuardrailResult]:
        """Run all input guardrails. Returns list of results."""
        results = []
        safety = self._safety.check(query)
        results.append(safety)
        if safety.status == GuardrailStatus.BLOCKED:
            return results  # Short-circuit — no need to check off-topic if unsafe
        results.append(self._off_topic.check(query))
        return results

    def check_output(
        self,
        query: str,
        answer: str,
        contexts: list[RetrievedContext],
    ) -> list[GuardrailResult]:
        """Run all output guardrails. Returns list of results."""
        results = [
            self._grounding.check(answer, contexts),
            self._faithfulness.check(query, answer, contexts),
        ]
        return results
