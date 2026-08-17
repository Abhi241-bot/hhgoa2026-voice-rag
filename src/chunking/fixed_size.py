"""
Fixed-size chunking strategy.

Splits text into chunks of a fixed character size with configurable overlap.
Simple, predictable, and fast — good baseline and suitable for uniform passages.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from src.models import Chunk, ChunkingStrategy


class FixedSizeChunker:
    """
    Splits text into fixed-size character chunks with overlap.

    Args:
        chunk_size: Maximum characters per chunk (default 512).
        overlap: Number of characters to overlap between adjacent chunks (default 64).
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        passage_id: Optional[str] = None,
        language: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Split text into fixed-size chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Generate a stable chunk_id from content
            chunk_id = hashlib.sha256(
                f"{passage_id or ''}:{idx}:{chunk_text[:50]}".encode()
            ).hexdigest()[:16]

            metadata = extra_metadata.copy() if extra_metadata else {}
            metadata["chunk_index"] = idx
            metadata["start_char"] = start
            metadata["end_char"] = end

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_passage_id=passage_id,
                    language=language,
                    strategy=ChunkingStrategy.FIXED_SIZE,
                    metadata=metadata,
                )
            )

            if end == len(text):
                break

            start += step
            idx += 1

        return chunks

    def __repr__(self) -> str:
        return f"FixedSizeChunker(chunk_size={self.chunk_size}, overlap={self.overlap})"
