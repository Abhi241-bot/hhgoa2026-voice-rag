"""
ChromaDB indexer — embeds chunks and builds a persistent vector index.

The indexer:
    1. Initialises a ChromaDB collection.
    2. Embeds chunk texts using sentence-transformers.
    3. Upserts chunks in batches (handles re-runs idempotently).

This module is the write path. Query/retrieval is handled by retriever.py.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger
from src.models import Chunk

log = get_logger("retrieval.indexer")

COLLECTION_NAME = "msmarco_xi"
BATCH_SIZE = 512  # ChromaDB batch upsert size


def get_chroma_client(persist_path: Optional[str] = None) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    path = persist_path or settings.chroma_db_path
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(
        path=path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    log.info("chroma_client_ready", path=path)
    return client


def get_or_create_collection(
    client: chromadb.ClientAPI,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create the ChromaDB collection (no embedding function — we embed manually)."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    log.info(
        "collection_ready",
        name=collection_name,
        count=collection.count(),
    )
    return collection


def build_index(
    chunks: list[Chunk],
    persist_path: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    embedding_model_name: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
) -> chromadb.Collection:
    """
    Embed all chunks and upsert them into ChromaDB.

    Args:
        chunks: List of Chunk objects to index.
        persist_path: ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
        embedding_model_name: Sentence transformer model name.
        batch_size: Upsert batch size.

    Returns:
        The populated ChromaDB collection.
    """
    if not chunks:
        raise ValueError("No chunks provided to index")

    model_name = embedding_model_name or settings.embedding_model
    log.info(
        "index_build_start",
        num_chunks=len(chunks),
        model=model_name,
        collection=collection_name,
    )

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Connect to ChromaDB
    client = get_chroma_client(persist_path)
    collection = get_or_create_collection(client, collection_name)

    # Upsert in batches
    total_upserted = 0
    start_time = time.perf_counter()

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [
            {
                "source_passage_id": c.source_passage_id or "",
                "language": c.language or "en",
                "strategy": c.strategy.value,
                "char_count": c.char_count,
                **{
                    k: str(v)
                    for k, v in c.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            for c in batch
        ]

        # Embed batch
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64).tolist()

        # Upsert (idempotent — existing IDs are updated)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_upserted += len(batch)
        elapsed = time.perf_counter() - start_time
        log.info(
            "batch_upserted",
            batch_num=batch_start // batch_size + 1,
            upserted=total_upserted,
            total=len(chunks),
            elapsed_s=round(elapsed, 2),
        )

    total_elapsed = time.perf_counter() - start_time
    log.info(
        "index_build_complete",
        total_chunks=total_upserted,
        collection_size=collection.count(),
        elapsed_s=round(total_elapsed, 2),
    )

    return collection
