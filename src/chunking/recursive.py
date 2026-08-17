"""
Recursive character text splitter.

Splits text using a hierarchy of separators: paragraph → sentence → word.
This preserves logical text structure better than naive fixed-size splitting.
Inspired by LangChain's RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from src.models import Chunk, ChunkingStrategy


# Default separator hierarchy: paragraphs → sentences → clauses → words → chars
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


class RecursiveChunker:
    """
    Recursively splits text using a hierarchy of separators.

    Tries each separator in order; if any resulting piece exceeds chunk_size,
    that piece is recursively split using the next separator in the hierarchy.

    Args:
        chunk_size: Target maximum characters per chunk.
        overlap: Characters of overlap between adjacent final chunks.
        separators: Ordered list of separators to try (most to least semantic).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: Optional[list[str]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else _DEFAULT_SEPARATORS

    # ── Public API ────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        passage_id: Optional[str] = None,
        language: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split text recursively and return a list of Chunks."""
        if not text or not text.strip():
            return []

        raw_chunks = self._split_text(text.strip(), self.separators)
        merged = self._merge_chunks(raw_chunks)

        result: list[Chunk] = []
        for idx, chunk_text in enumerate(merged):
            chunk_id = hashlib.sha256(
                f"{passage_id or ''}:recursive:{idx}:{chunk_text[:50]}".encode()
            ).hexdigest()[:16]

            metadata = extra_metadata.copy() if extra_metadata else {}
            metadata["chunk_index"] = idx
            metadata["separator_strategy"] = "recursive"

            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_passage_id=passage_id,
                    language=language,
                    strategy=ChunkingStrategy.RECURSIVE,
                    metadata=metadata,
                )
            )

        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split using separator hierarchy."""
        final_chunks: list[str] = []

        # Find the first separator that actually splits this text
        separator = separators[-1]  # fallback: char-by-char
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split on the chosen separator
        if separator:
            splits = re.split(re.escape(separator), text)
        else:
            # No separator works → split by chunk_size directly
            splits = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # Classify each split: good (fits) vs. too large (needs recursion)
        good_splits: list[str] = []
        next_separators = separators[separators.index(separator) + 1 :] if separator in separators and separator != separators[-1] else []

        for split in splits:
            if not split.strip():
                continue
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # Recurse with remaining separators
                if next_separators:
                    final_chunks.extend(self._split_text(split, next_separators))
                else:
                    # Hard split as last resort
                    for i in range(0, len(split), self.chunk_size):
                        final_chunks.append(split[i : i + self.chunk_size])
                # Flush accumulated good_splits first
                final_chunks.extend(good_splits)
                good_splits = []

        final_chunks.extend(good_splits)
        return [c for c in final_chunks if c.strip()]

    def _merge_chunks(self, splits: list[str]) -> list[str]:
        """
        Merge small splits into chunks that stay within chunk_size,
        adding overlap between consecutive chunks.
        """
        merged: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len > self.chunk_size and current_parts:
                # Emit current chunk
                merged.append(" ".join(current_parts).strip())
                # Keep overlap: drop parts from front until we're within overlap budget
                while current_parts and current_len > self.overlap:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + 1

            current_parts.append(split)
            current_len += split_len + 1  # +1 for the space separator

        if current_parts:
            merged.append(" ".join(current_parts).strip())

        return [c for c in merged if c.strip()]

    def __repr__(self) -> str:
        return f"RecursiveChunker(chunk_size={self.chunk_size}, overlap={self.overlap})"
