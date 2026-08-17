"""
Metadata-aware chunking strategy.

Designed specifically for the MSMARCO-XI dataset structure.
Each passage is treated as an atomic unit; chunking preserves and injects
rich metadata: passage_id, query_id, language, source URL, query context.

When passages are long enough to need splitting, recursive character splitting
is applied, but ALL metadata is propagated to every child chunk.

This strategy enables downstream metadata filtering during retrieval.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from src.models import Chunk, ChunkingStrategy
from src.chunking.recursive import RecursiveChunker


class MetadataAwareChunker:
    """
    Chunks documents while preserving and enriching structured metadata.

    Specifically tuned for MSMARCO-XI records:
        {
            "id": "...",
            "passage": "...",
            "query": "...",
            "language": "en",
            ...
        }

    Args:
        max_chunk_size: Maximum chunk character size before sub-splitting.
        overlap: Overlap for sub-splitting.
    """

    def __init__(self, max_chunk_size: int = 800, overlap: int = 80) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self._recursive = RecursiveChunker(chunk_size=max_chunk_size, overlap=overlap)

    def chunk_record(self, record: dict[str, Any]) -> list[Chunk]:
        """
        Chunk a single MSMARCO-XI dataset record.

        Expected record keys (all optional except 'passage'):
            - passage / text / passage_text: The passage text.
            - id / passage_id: Passage identifier.
            - query / question: Associated query string.
            - language / lang: Language code (e.g., 'en', 'hi').
            - url: Source URL.
        """
        # ── Extract text ──────────────────────────────────────────────────
        text = (
            record.get("passage")
            or record.get("text")
            or record.get("passage_text")
            or ""
        )
        if not text or not text.strip():
            return []

        text = text.strip()

        # ── Extract metadata ──────────────────────────────────────────────
        passage_id = str(record.get("id") or record.get("passage_id") or "")
        language = record.get("language") or record.get("lang") or "en"
        query_context = record.get("query") or record.get("question") or ""
        url = record.get("url") or ""

        base_metadata: dict[str, Any] = {
            "passage_id": passage_id,
            "language": language,
            "query_context": query_context[:200],  # truncate long queries
            "url": url,
            "original_length": len(text),
        }

        # ── Single chunk if short enough ───────────────────────────────────
        if len(text) <= self.max_chunk_size:
            chunk_id = self._make_id(passage_id, 0, text)
            metadata = {**base_metadata, "chunk_index": 0, "is_single_chunk": True}
            return [
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_passage_id=passage_id,
                    language=language,
                    strategy=ChunkingStrategy.METADATA_AWARE,
                    metadata=metadata,
                )
            ]

        # ── Sub-split long passages ────────────────────────────────────────
        sub_chunks_raw = self._recursive.chunk(
            text=text,
            passage_id=passage_id,
            language=language,
        )

        result: list[Chunk] = []
        for sub_chunk in sub_chunks_raw:
            # Merge base metadata with sub-chunk metadata
            enriched_metadata = {**base_metadata, **sub_chunk.metadata, "is_single_chunk": False}
            chunk_id = self._make_id(passage_id, sub_chunk.metadata.get("chunk_index", 0), sub_chunk.text)
            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=sub_chunk.text,
                    source_passage_id=passage_id,
                    language=language,
                    strategy=ChunkingStrategy.METADATA_AWARE,
                    metadata=enriched_metadata,
                )
            )

        return result

    def chunk_batch(self, records: list[dict[str, Any]]) -> list[Chunk]:
        """Chunk a list of records and return all chunks flat."""
        all_chunks: list[Chunk] = []
        for record in records:
            all_chunks.extend(self.chunk_record(record))
        return all_chunks

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(passage_id: str, idx: int, text: str) -> str:
        return hashlib.sha256(
            f"{passage_id}:meta:{idx}:{text[:50]}".encode()
        ).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"MetadataAwareChunker(max_chunk_size={self.max_chunk_size})"
