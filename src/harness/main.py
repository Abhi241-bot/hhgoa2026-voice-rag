"""
src/harness/main.py — CLI entry point for the RAG pipeline.

Usage:
    python -m src.harness.main --query "What is the capital of France?"
    python -m src.harness.main --audio ./my_question.wav
"""

from __future__ import annotations

import argparse
import json

from src.logger import get_logger
from src.models import TextQuery

log = get_logger("harness.main")


def main():
    parser = argparse.ArgumentParser(
        description="Voice-Enabled RAG Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", type=str, help="Text query string")
    group.add_argument("--audio", "-a", type=str, help="Path to audio file (WAV/MP3)")
    parser.add_argument("--language-filter", type=str, default=None, help="Filter by language code")
    parser.add_argument("--mock-stt", action="store_true", help="Use mock STT (no API key needed)")
    parser.add_argument("--output-json", type=str, default=None, help="Save response to JSON file")
    args = parser.parse_args()

    # Lazy imports to keep startup fast
    from src.harness.pipeline import PipelineHarness
    from src.retrieval.retriever import HybridRetriever
    from src.stt.elevenlabs import get_stt_tool

    print("🔧 Initializing pipeline...")
    retriever = HybridRetriever()
    stt = get_stt_tool(mock=args.mock_stt)
    pipeline = PipelineHarness(retriever=retriever, stt_tool=stt)
    print("✅ Pipeline ready")

    if args.query:
        input_obj = TextQuery(text=args.query)
    else:
        from src.models import AudioInput
        input_obj = AudioInput(file_path=args.audio)

    print(f"\n🔍 Running query: {args.query or args.audio}")
    response = pipeline.run(input_obj, language_filter=args.language_filter)

    # Display result
    print(f"\n{'='*60}")
    print(f"Status: {response.status.value}")
    if response.answer:
        print(f"\nAnswer:\n{response.answer}")
    if response.error:
        print(f"\nError: {response.error}")
    if response.sources:
        print(f"\nSources: {', '.join(response.sources)}")
    print(f"\nLatency: {response.total_latency_ms:.1f}ms total")
    print(f"Breakdown: {response.latency_breakdown}")
    print(f"{'='*60}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(response.model_dump(), f, indent=2, default=str)
        print(f"\n💾 Response saved to: {args.output_json}")


if __name__ == "__main__":
    main()
