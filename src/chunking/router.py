"""
Chunking Router.

Selects the best chunking strategy for each record based on heuristics:
    - Very short texts (< 200 chars)     → MetadataAware (atomic, no splitting)
    - Structured/long passages (≥ 200)   → MetadataAware with recursive sub-split
    - Query-answering passages (MSMARCO) → MetadataAware (default for this dataset)
    - Rich multi-paragraph text          → Recursive
    - Technical / dense text             → Semantic
    - Uniform paragraphs / raw text      → FixedSize

For the MSMARCO-XI dataset, MetadataAware is almost always the right choice
since each record carries rich metadata (passage_id, query, language).
The router supports explicit strategy override for benchmarking purposes.
"""

from __future__ import annotations

from typing import Any, Optional

from src.models import Chunk, ChunkingStrategy
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.logger import get_logger

log = get_logger("chunking.router")


class ChunkingRouter:
    """
    Routes records to the appropriate chunking strategy.

    Args:
        chunk_size: Base chunk size for fixed/recursive strategies.
        overlap: Overlap for fixed/recursive strategies.
        semantic_threshold: Similarity threshold for semantic chunker.
        embedding_model: Pre-loaded model for semantic chunker (optional).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        semantic_threshold: float = 0.75,
        embedding_model=None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Instantiate all strategies once
        self._fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        self._recursive = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
        self._semantic = SemanticChunker(
            similarity_threshold=semantic_threshold,
            max_chunk_size=chunk_size * 2,
            embedding_model=embedding_model,
        )
        self._metadata_aware = MetadataAwareChunker(
            max_chunk_size=chunk_size, overlap=overlap
        )

    def route_record(
        self,
        record: dict[str, Any],
        strategy_override: Optional[ChunkingStrategy] = None,
    ) -> list[Chunk]:
        """
        Chunk a single dataset record using the best-fit strategy.

        Args:
            record: Dataset record dict.
            strategy_override: Force a specific strategy (for benchmarking).

        Returns:
            List of Chunk objects.
        """
        if strategy_override:
            return self._apply_strategy(strategy_override, record)

        # ── Heuristic routing ─────────────────────────────────────────────
        text = (
            record.get("passage")
            or record.get("text")
            or record.get("passage_text")
            or ""
        )
        text_len = len(text.strip())
        has_query = bool(record.get("query") or record.get("question"))
        has_structured_id = bool(record.get("id") or record.get("passage_id"))

        # MSMARCO-XI records: always use MetadataAware
        if has_query or has_structured_id:
            strategy = ChunkingStrategy.METADATA_AWARE
        elif text_len > 2000:
            # Long unstructured text → semantic chunking
            strategy = ChunkingStrategy.SEMANTIC
        elif text_len > 500:
            # Medium text with paragraph structure → recursive
            strategy = ChunkingStrategy.RECURSIVE
        else:
            # Short, uniform → fixed size
            strategy = ChunkingStrategy.FIXED_SIZE

        log.debug("routing_decision", text_len=text_len, strategy=strategy.value)
        return self._apply_strategy(strategy, record)

    def route_batch(
        self,
        records: list[dict[str, Any]],
        strategy_override: Optional[ChunkingStrategy] = None,
    ) -> list[Chunk]:
        """Chunk a batch of records and return all chunks flat."""
        all_chunks: list[Chunk] = []
        for record in records:
            all_chunks.extend(self.route_record(record, strategy_override))
        return all_chunks

    # ── Internal ──────────────────────────────────────────────────────────

    def _apply_strategy(
        self, strategy: ChunkingStrategy, record: dict[str, Any]
    ) -> list[Chunk]:
        text = (
            record.get("passage")
            or record.get("text")
            or record.get("passage_text")
            or ""
        )
        passage_id = str(record.get("id") or record.get("passage_id") or "")
        language = record.get("language") or record.get("lang") or "en"

        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed.chunk(text, passage_id=passage_id, language=language)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive.chunk(text, passage_id=passage_id, language=language)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic.chunk(text, passage_id=passage_id, language=language)
        elif strategy == ChunkingStrategy.METADATA_AWARE:
            return self._metadata_aware.chunk_record(record)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def __repr__(self) -> str:
        return (
            f"ChunkingRouter(chunk_size={self.chunk_size}, overlap={self.overlap})"
        )
