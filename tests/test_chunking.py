"""
Unit tests for all chunking strategies and the router.

Run with: pytest tests/test_chunking.py -v
"""

import pytest
from src.models import ChunkingStrategy
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.router import ChunkingRouter


# ── Fixtures ──────────────────────────────────────────────────────────────────

SHORT_TEXT = "The quick brown fox jumps over the lazy dog."
MEDIUM_TEXT = (
    "Artificial intelligence is transforming the world. "
    "Machine learning models can now process natural language with remarkable accuracy. "
    "These advances have enabled applications in healthcare, finance, and education. "
    "Researchers continue to push the boundaries of what is possible. "
    "Deep learning, in particular, has achieved superhuman performance on many tasks."
)
LONG_TEXT = MEDIUM_TEXT * 10  # ~2000 chars

MSMARCO_RECORD = {
    "id": "passage_001",
    "passage": MEDIUM_TEXT,
    "query": "What is artificial intelligence?",
    "language": "en",
    "url": "https://example.com",
}


# ── FixedSizeChunker Tests ────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_basic_split(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(SHORT_TEXT)
        assert len(chunks) == 1
        assert chunks[0].text == SHORT_TEXT.strip()

    def test_long_text_splits_correctly(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(LONG_TEXT)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 100

    def test_chunk_strategy_label(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(SHORT_TEXT)
        assert all(c.strategy == ChunkingStrategy.FIXED_SIZE for c in chunks)

    def test_empty_text_returns_empty(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunk_ids_are_unique(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=5)
        chunks = chunker.chunk(LONG_TEXT, passage_id="test_passage")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)  # overlap >= chunk_size

    def test_metadata_propagated(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(SHORT_TEXT, passage_id="p1", language="en")
        assert chunks[0].source_passage_id == "p1"
        assert chunks[0].language == "en"


# ── RecursiveChunker Tests ────────────────────────────────────────────────────

class TestRecursiveChunker:
    def test_basic_chunk(self):
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(MEDIUM_TEXT)
        assert len(chunks) >= 1
        assert all(c.strategy == ChunkingStrategy.RECURSIVE for c in chunks)

    def test_respects_paragraphs(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunker = RecursiveChunker(chunk_size=40, overlap=5)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []

    def test_chunk_size_respected(self):
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            # Allow slight overage for word-boundary preservation
            assert len(chunk.text) <= 200

    def test_metadata_propagated(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk(MEDIUM_TEXT, passage_id="r1", language="hi")
        assert all(c.source_passage_id == "r1" for c in chunks)
        assert all(c.language == "hi" for c in chunks)


# ── SemanticChunker Tests ─────────────────────────────────────────────────────

class TestSemanticChunker:
    @pytest.fixture(scope="class")
    def chunker(self):
        """Create chunker once per class (model loading is expensive)."""
        return SemanticChunker(similarity_threshold=0.7, max_chunk_size=1024)

    def test_basic_chunk(self, chunker):
        chunks = chunker.chunk(MEDIUM_TEXT)
        assert len(chunks) >= 1

    def test_strategy_label(self, chunker):
        chunks = chunker.chunk(MEDIUM_TEXT)
        assert all(c.strategy == ChunkingStrategy.SEMANTIC for c in chunks)

    def test_empty_text(self, chunker):
        assert chunker.chunk("") == []

    def test_short_text_single_chunk(self, chunker):
        chunks = chunker.chunk(SHORT_TEXT)
        assert len(chunks) == 1

    def test_max_chunk_size_respected(self, chunker):
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert len(chunk.text) <= chunker.max_chunk_size * 1.1  # small tolerance


# ── MetadataAwareChunker Tests ────────────────────────────────────────────────

class TestMetadataAwareChunker:
    def test_chunk_record(self):
        chunker = MetadataAwareChunker(max_chunk_size=600)
        chunks = chunker.chunk_record(MSMARCO_RECORD)
        assert len(chunks) >= 1
        assert all(c.strategy == ChunkingStrategy.METADATA_AWARE for c in chunks)

    def test_metadata_preserved(self):
        chunker = MetadataAwareChunker()
        chunks = chunker.chunk_record(MSMARCO_RECORD)
        for chunk in chunks:
            assert chunk.source_passage_id == "passage_001"
            assert chunk.language == "en"
            assert chunk.metadata.get("query_context") is not None
            assert chunk.metadata.get("url") is not None

    def test_short_passage_single_chunk(self):
        record = {**MSMARCO_RECORD, "passage": SHORT_TEXT}
        chunker = MetadataAwareChunker(max_chunk_size=600)
        chunks = chunker.chunk_record(record)
        assert len(chunks) == 1
        assert chunks[0].metadata.get("is_single_chunk") is True

    def test_empty_passage_returns_empty(self):
        chunker = MetadataAwareChunker()
        assert chunker.chunk_record({"id": "x", "passage": ""}) == []

    def test_batch_chunking(self):
        chunker = MetadataAwareChunker()
        records = [MSMARCO_RECORD, {**MSMARCO_RECORD, "id": "passage_002"}]
        chunks = chunker.chunk_batch(records)
        assert len(chunks) >= 2

    def test_missing_optional_fields(self):
        """Must not crash on records with only passage text."""
        chunker = MetadataAwareChunker()
        record = {"passage": MEDIUM_TEXT}
        chunks = chunker.chunk_record(record)
        assert len(chunks) >= 1


# ── ChunkingRouter Tests ──────────────────────────────────────────────────────

class TestChunkingRouter:
    @pytest.fixture(scope="class")
    def router(self):
        return ChunkingRouter(chunk_size=256, overlap=32)

    def test_msmarco_record_routes_to_metadata_aware(self, router):
        chunks = router.route_record(MSMARCO_RECORD)
        assert all(c.strategy == ChunkingStrategy.METADATA_AWARE for c in chunks)

    def test_strategy_override_fixed(self, router):
        chunks = router.route_record(
            MSMARCO_RECORD,
            strategy_override=ChunkingStrategy.FIXED_SIZE,
        )
        assert all(c.strategy == ChunkingStrategy.FIXED_SIZE for c in chunks)

    def test_strategy_override_recursive(self, router):
        chunks = router.route_record(
            MSMARCO_RECORD,
            strategy_override=ChunkingStrategy.RECURSIVE,
        )
        assert all(c.strategy == ChunkingStrategy.RECURSIVE for c in chunks)

    def test_batch_routing(self, router):
        records = [MSMARCO_RECORD, {**MSMARCO_RECORD, "id": "p2"}]
        chunks = router.route_batch(records)
        assert len(chunks) >= 2

    def test_short_text_without_query(self, router):
        record = {"passage": SHORT_TEXT}  # no query → routing falls to fixed-size
        chunks = router.route_record(record)
        assert len(chunks) >= 1
