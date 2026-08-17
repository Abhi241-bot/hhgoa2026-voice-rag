"""
Dataset loader and chunking pipeline for MSMARCO-XI.

Loads the ai4bharat/MSMARCO-XI dataset from HuggingFace,
applies multi-strategy chunking via ChunkingRouter,
and exports chunks for indexing into the vector database.

Usage:
    python -m src.chunking.pipeline --index --max-records 5000
    python -m src.chunking.pipeline --export-json ./data/chunks.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.chunking.router import ChunkingRouter
from src.config import settings
from src.logger import get_logger
from src.models import Chunk

log = get_logger("chunking.pipeline")

DATASET_NAME = "ai4bharat/MSMARCO-XI"
DEFAULT_SPLIT = "test"  # MSMARCO-XI uses 'test' as primary split
DEFAULT_LANGUAGE = "en"


def load_dataset_streaming(
    language: str = DEFAULT_LANGUAGE,
    split: str = DEFAULT_SPLIT,
    max_records: Optional[int] = None,
    cache_dir: Optional[str] = None,
    data_file: Optional[str] = None,
) -> Iterator[dict]:
    """
    Stream records from MSMARCO-XI dataset (from local file or HuggingFace).

    Yields:
        Record dicts (flat or nested MSMARCO-XI schema).
    """
    # 1. If data_file is explicitly provided or default sample exists
    target_file = data_file or os.path.join("data", "sample_dataset.json")
    if os.path.exists(target_file):
        log.info("loading_local_dataset", path=target_file)
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            count = 0
            for record in data:
                yield record
                count += 1
                if max_records and count >= max_records:
                    break
        log.info("local_dataset_streaming_complete", total_yielded=count)
        return

    # 2. Otherwise load from HuggingFace
    from datasets import load_dataset

    log.info("loading_hf_dataset", dataset=DATASET_NAME, language=language, split=split)
    cache = cache_dir or settings.data_cache_path
    os.makedirs(cache, exist_ok=True)

    try:
        ds = load_dataset(
            DATASET_NAME,
            language,
            split=split,
            streaming=True,
            cache_dir=cache,
        )
        count = 0
        for record in ds:
            yield record
            count += 1
            if max_records and count >= max_records:
                break
        log.info("hf_dataset_streaming_complete", total_yielded=count)
    except Exception as e:
        log.warning("hf_dataset_load_fallback", error=str(e))
        # Fallback to local sample dataset if available
        sample_path = os.path.join("data", "sample_dataset.json")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                for record in json.load(f):
                    yield record


def run_chunking_pipeline(
    language: str = DEFAULT_LANGUAGE,
    split: str = DEFAULT_SPLIT,
    max_records: int = 5000,
    chunk_size: int = 512,
    overlap: int = 64,
    export_json_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[Chunk]:
    """
    Full chunking pipeline: load dataset → route → chunk → return.

    Args:
        language: Dataset language split.
        split: HuggingFace dataset split.
        max_records: Max records to process.
        chunk_size: Base chunk size.
        overlap: Chunk overlap.
        export_json_path: If set, export chunks to this JSON file.
        cache_dir: HuggingFace cache directory.

    Returns:
        List of all generated Chunk objects.
    """
    router = ChunkingRouter(chunk_size=chunk_size, overlap=overlap)
    all_chunks: list[Chunk] = []

    start_time = time.perf_counter()

    log.info(
        "pipeline_start",
        language=language,
        split=split,
        max_records=max_records,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    for i, record in enumerate(
        load_dataset_streaming(
            language=language,
            split=split,
            max_records=max_records,
            cache_dir=cache_dir,
        )
    ):
        try:
            chunks = router.route_record(record)
            all_chunks.extend(chunks)
        except Exception as e:
            log.warning("chunk_error", record_index=i, error=str(e))
            continue

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - start_time
            log.info(
                "chunking_progress",
                records_processed=i + 1,
                chunks_generated=len(all_chunks),
                elapsed_s=round(elapsed, 2),
            )

    elapsed = time.perf_counter() - start_time
    log.info(
        "pipeline_complete",
        total_records=i + 1 if all_chunks else 0,
        total_chunks=len(all_chunks),
        elapsed_s=round(elapsed, 2),
        avg_chunks_per_record=round(len(all_chunks) / max(1, i + 1), 2),
    )

    # ── Optional export ───────────────────────────────────────────────────
    if export_json_path:
        _export_to_json(all_chunks, export_json_path)

    return all_chunks


def _export_to_json(chunks: list[Chunk], path: str) -> None:
    """Export chunks to a JSON file for inspection."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump() for c in chunks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("chunks_exported", path=str(out_path), count=len(chunks))


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MSMARCO-XI chunking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Language split")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--max-records", type=int, default=5000, help="Max records to process")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=64, help="Chunk overlap in characters")
    parser.add_argument("--export-json", type=str, default=None, help="Export chunks to JSON file")
    parser.add_argument("--index", action="store_true", help="Index chunks into vector DB after chunking")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    chunks = run_chunking_pipeline(
        language=args.language,
        split=args.split,
        max_records=args.max_records,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        export_json_path=args.export_json,
    )

    print(f"\n✅ Chunking complete: {len(chunks)} chunks from {args.max_records} records")

    if args.index:
        from src.retrieval.indexer import build_index
        build_index(chunks)
        print("✅ Index built successfully")
