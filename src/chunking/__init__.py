"""
src/chunking/__init__.py

Public API for the chunking module.
"""

from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.router import ChunkingRouter

__all__ = [
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "MetadataAwareChunker",
    "ChunkingRouter",
]
