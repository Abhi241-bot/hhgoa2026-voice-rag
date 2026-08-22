"""
Hybrid retriever combining dense ANN (ChromaDB) + BM25 sparse search.

Retrieval strategy:
    1. Dense retrieval: query → embedding → cosine ANN search in ChromaDB.
    2. Sparse retrieval: query → BM25 term-frequency scoring over indexed texts.
    3. Reciprocal Rank Fusion (RRF): merge ranked lists from both sources.
    4. Return top-k fused results as RetrievedContext objects.

The retriever is initialized from a persisted ChromaDB collection and
a pre-built BM25 corpus. Both can be loaded from disk (no re-indexing needed).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger
from src.models import Chunk, ChunkingStrategy, RetrievedContext, RetrievalResult
from src.retrieval.indexer import get_chroma_client, get_or_create_collection, COLLECTION_NAME

log = get_logger("retrieval.retriever")

# RRF constant (k=60 is standard)
RRF_K = 60


class HybridRetriever:
    """
    Hybrid dense + sparse retriever with Reciprocal Rank Fusion.

    Args:
        collection_name: ChromaDB collection name.
        persist_path: ChromaDB persistence directory.
        embedding_model_name: Sentence transformer model name.
        top_k: Number of results to return after fusion.
        dense_candidates: Number of dense results to fetch before fusion.
        sparse_candidates: Number of BM25 results to fetch before fusion.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_path: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        top_k: int = 5,
        dense_candidates: int = 20,
        sparse_candidates: int = 20,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates

        self._model_name = embedding_model_name or settings.embedding_model
        log.info("embedding_model_configured", model=self._model_name)
        # Delay heavy SentenceTransformer instantiation until actually needed
        self._model: Optional[SentenceTransformer] = None

        log.info("connecting_to_chroma", collection=collection_name)
        client = get_chroma_client(persist_path)
        self._collection = get_or_create_collection(client, collection_name)

        # Build BM25 index from ChromaDB documents
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metadatas: list[dict] = []
        # _load_bm25_index() is intentionally not called here to avoid
        # loading the entire corpus into memory during startup. The
        # index will be built lazily on first retrieval request.

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        language_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Execute hybrid retrieval for a query.

        Args:
            query: The search query string.
            top_k: Override number of results.
            language_filter: If set, only return chunks of this language.

        Returns:
            RetrievalResult with ranked RetrievedContext objects.
        """
        k = top_k or self.top_k
        start = time.perf_counter()

        # Ensure BM25 index is built lazily to avoid loading all documents
        # into memory at pipeline startup on memory-constrained hosts.
        if settings.use_bm25 and self._bm25 is None:
            self._load_bm25_index()

        # ── 1. Dense retrieval ─────────────────────────────────────────
        dense_results = self._dense_retrieve(query, self.dense_candidates, language_filter)

        # ── 2. Sparse retrieval ────────────────────────────────────────
        sparse_results = self._sparse_retrieve(query, self.sparse_candidates)

        # ── 3. Reciprocal Rank Fusion ──────────────────────────────────
        fused = self._rrf_merge(dense_results, sparse_results, top_k=k)

        latency_ms = (time.perf_counter() - start) * 1000
        log.info(
            "retrieval_complete",
            query=query[:80],
            num_results=len(fused),
            latency_ms=round(latency_ms, 2),
        )

        return RetrievalResult(
            query=query,
            contexts=fused,
            latency_ms=round(latency_ms, 2),
        )

    # ── Dense retrieval ───────────────────────────────────────────────────

    def _dense_retrieve(
        self,
        query: str,
        n_results: int,
        language_filter: Optional[str],
    ) -> list[tuple[str, float, str, dict]]:
        """
        Query ChromaDB for dense ANN results.

        Returns:
            List of (chunk_id, cosine_score, document_text, metadata) tuples.
        """
        # Lazy-load embedding model to avoid loading it at startup
        model = self._get_model()
        query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()

        where_filter = None
        if language_filter:
            where_filter = {"language": {"$eq": language_filter}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, max(1, self._collection.count())),
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        dense_results = []
        for chunk_id, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns distances (lower = closer); convert to similarity
            similarity = 1.0 - distance
            dense_results.append((chunk_id, similarity, doc, meta))

        return dense_results

    def _get_model(self) -> SentenceTransformer:
        """Instantiate the SentenceTransformer model on first use."""
        if self._model is None:
            log.info("loading_sentence_transformer", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ── Sparse retrieval ──────────────────────────────────────────────────

    def _sparse_retrieve(
        self,
        query: str,
        n_results: int,
    ) -> list[tuple[str, float, str, dict]]:
        """
        BM25 sparse retrieval over the full indexed corpus.

        Returns:
            List of (chunk_id, bm25_score, document_text, metadata) tuples.
        """
        # If BM25 is disabled in settings, skip sparse retrieval entirely.
        if not settings.use_bm25:
            return []

        if self._bm25 is None or not self._bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-n by score
        top_indices = np.argsort(scores)[::-1][:n_results]

        sparse_results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break
            sparse_results.append((
                self._bm25_ids[idx],
                score,
                self._bm25_docs[idx],
                self._bm25_metadatas[idx],
            ))

        return sparse_results

    # ── RRF fusion ────────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        dense: list[tuple[str, float, str, dict]],
        sparse: list[tuple[str, float, str, dict]],
        top_k: int,
    ) -> list[RetrievedContext]:
        """
        Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

        RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        where k=60 by convention.

        Returns:
            Sorted list of RetrievedContext objects.
        """
        rrf_scores: dict[str, float] = {}
        id_to_data: dict[str, tuple[str, float, float, str, dict]] = {}

        # Score from dense list
        for rank, (chunk_id, dense_score, doc, meta) in enumerate(dense, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if chunk_id not in id_to_data:
                id_to_data[chunk_id] = (chunk_id, dense_score, 0.0, doc, meta)

        # Score from sparse list
        for rank, (chunk_id, sparse_score, doc, meta) in enumerate(sparse, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if chunk_id not in id_to_data:
                id_to_data[chunk_id] = (chunk_id, 0.0, sparse_score, doc, meta)
            else:
                # Update sparse score for existing entry
                existing = id_to_data[chunk_id]
                id_to_data[chunk_id] = (existing[0], existing[1], sparse_score, existing[3], existing[4])

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        contexts: list[RetrievedContext] = []
        for rank, chunk_id in enumerate(sorted_ids[:top_k], start=1):
            chunk_id_, dense_score, sparse_score, doc, meta = id_to_data[chunk_id]
            chunk = Chunk(
                chunk_id=chunk_id_,
                text=doc,
                source_passage_id=meta.get("source_passage_id"),
                language=meta.get("language", "en"),
                strategy=ChunkingStrategy(meta.get("strategy", "metadata_aware")),
                metadata=meta,
            )
            contexts.append(
                RetrievedContext(
                    chunk=chunk,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rrf_score=rrf_scores[chunk_id],
                    rank=rank,
                )
            )

        return contexts

    # ── BM25 index loading ────────────────────────────────────────────────

    def _load_bm25_index(self) -> None:
        """Load all documents from ChromaDB and build a BM25 index."""
        count = self._collection.count()
        if count == 0:
            log.warning("empty_collection_no_bm25")
            return

        log.info("building_bm25_index", num_docs=count)
        start = time.perf_counter()

        # Fetch all documents in one call (ChromaDB limit: 5461 per call on some versions)
        batch_size = 5000
        offset = 0
        all_ids, all_docs, all_metas = [], [], []

        while offset < count:
            result = self._collection.get(
                limit=min(batch_size, count - offset),
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_ids.extend(result["ids"])
            all_docs.extend(result["documents"])
            all_metas.extend(result["metadatas"])
            offset += len(result["ids"])
            if len(result["ids"]) == 0:
                break

        self._bm25_ids = all_ids
        self._bm25_docs = all_docs
        self._bm25_metadatas = all_metas

        # Tokenize (simple whitespace; good enough for BM25)
        tokenized_corpus = [doc.lower().split() for doc in all_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

        elapsed = time.perf_counter() - start
        log.info("bm25_index_built", num_docs=len(all_docs), elapsed_s=round(elapsed, 2))

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(top_k={self.top_k}, "
            f"collection='{self.collection_name}', "
            f"bm25_docs={len(self._bm25_docs)})"
        )
