"""
Semantic chunking strategy.

Splits text at semantic boundaries detected by a drop in cosine similarity
between consecutive sentence embeddings. This produces semantically coherent
chunks rather than arbitrarily cut pieces.

Algorithm:
    1. Tokenize text into sentences using NLTK.
    2. Embed each sentence with a lightweight model.
    3. Compute cosine similarity between consecutive sentence embeddings.
    4. Find "breakpoints" where similarity drops below a threshold.
    5. Group sentences between breakpoints into chunks.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.models import Chunk, ChunkingStrategy

# Download sentence tokenizer data once
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class SemanticChunker:
    """
    Splits text at semantic boundaries using sentence embedding similarity.

    Args:
        similarity_threshold: Similarity drop threshold to trigger a split (0-1).
            Lower = more chunks (finer granularity).
            Higher = fewer chunks (coarser granularity).
        max_chunk_size: Hard cap on chunk character length (safety net).
        min_sentences: Minimum sentences per chunk.
        embedding_model: Pre-loaded sentence transformer model. If None, a
            lightweight default model is loaded lazily.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_chunk_size: int = 1024,
        min_sentences: int = 2,
        embedding_model=None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_sentences = min_sentences
        self._model = embedding_model  # injected to avoid re-loading

    @property
    def model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def chunk(
        self,
        text: str,
        passage_id: Optional[str] = None,
        language: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Semantically split text and return Chunk objects."""
        if not text or not text.strip():
            return []

        sentences = self._tokenize_sentences(text.strip())
        if len(sentences) <= self.min_sentences:
            # Too short to semantically split — return as single chunk
            return self._make_chunks(
                [text.strip()], passage_id, language, extra_metadata
            )

        # Embed all sentences in one batch (fast)
        embeddings = self.model.encode(sentences, show_progress_bar=False, batch_size=64)

        # Compute pairwise similarity between consecutive sentences
        similarities = [
            float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
            for i in range(len(embeddings) - 1)
        ]

        # Find breakpoints where similarity drops below threshold
        breakpoints: set[int] = set()
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.add(i + 1)  # split AFTER sentence i

        # Group sentences into chunks by breakpoints
        groups: list[list[str]] = []
        current: list[str] = []
        for i, sent in enumerate(sentences):
            if i in breakpoints and len(current) >= self.min_sentences:
                groups.append(current)
                current = []
            current.append(sent)
        if current:
            groups.append(current)

        # Join each group and enforce max_chunk_size
        raw_chunks: list[str] = []
        for group in groups:
            combined = " ".join(group)
            if len(combined) <= self.max_chunk_size:
                raw_chunks.append(combined)
            else:
                # Hard-split oversized chunks
                for i in range(0, len(combined), self.max_chunk_size):
                    raw_chunks.append(combined[i : i + self.max_chunk_size])

        return self._make_chunks(raw_chunks, passage_id, language, extra_metadata)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _tokenize_sentences(self, text: str) -> list[str]:
        """Tokenize text into sentences using NLTK."""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def _make_chunks(
        self,
        texts: list[str],
        passage_id: Optional[str],
        language: Optional[str],
        extra_metadata: Optional[dict],
    ) -> list[Chunk]:
        """Convert raw text pieces into Chunk objects."""
        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(texts):
            if not chunk_text.strip():
                continue
            chunk_id = hashlib.sha256(
                f"{passage_id or ''}:semantic:{idx}:{chunk_text[:50]}".encode()
            ).hexdigest()[:16]
            metadata = extra_metadata.copy() if extra_metadata else {}
            metadata["chunk_index"] = idx
            metadata["similarity_threshold"] = self.similarity_threshold
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_passage_id=passage_id,
                    language=language,
                    strategy=ChunkingStrategy.SEMANTIC,
                    metadata=metadata,
                )
            )
        return chunks

    def __repr__(self) -> str:
        return (
            f"SemanticChunker(threshold={self.similarity_threshold}, "
            f"max_chunk_size={self.max_chunk_size})"
        )
