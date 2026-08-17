#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script — validates environment, checks API keys, and runs a smoke test.
Run this AFTER setting up your .env file.

Usage: python setup_check.py
"""

import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check(label: str, condition: bool, fix: str = ""):
    icon = "[OK]" if condition else "[FAIL]"
    print(f"  {icon} {label}")
    if not condition and fix:
        print(f"     Fix: {fix}")
    return condition


def main():
    print("\n[*] Voice-Enabled RAG -- Setup Check\n" + "=" * 40)

    all_ok = True

    # 1. Python version
    major, minor = sys.version_info[:2]
    ok = check(f"Python {major}.{minor} (need ≥3.11)", major == 3 and minor >= 11,
               "Install Python 3.11+ from python.org")
    all_ok &= ok

    # 2. .env file
    ok = check(".env file exists", os.path.exists(".env"),
               "Run: copy .env.example .env  — then fill in your API keys")
    all_ok &= ok

    # 3. API keys
    from dotenv import load_dotenv
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "")
    ok = check("GROQ_API_KEY set", bool(groq_key and groq_key != "your_groq_api_key_here"),
               "Get key at: https://console.groq.com/keys")
    all_ok &= ok

    eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
    ok = check("ELEVENLABS_API_KEY set", bool(eleven_key and eleven_key != "your_elevenlabs_api_key_here"),
               "Get key at: https://elevenlabs.io/app/settings/api-keys")
    all_ok &= ok

    # 4. Imports
    print("\n📦 Package imports:")
    packages = [
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("rank_bm25", "rank-bm25"),
        ("groq", "groq"),
        ("elevenlabs", "elevenlabs"),
        ("gradio", "gradio"),
        ("fastapi", "fastapi"),
        ("datasets", "datasets"),
    ]
    for module, pkg in packages:
        try:
            __import__(module)
            check(f"  {pkg}", True)
        except ImportError:
            check(f"  {pkg}", False, f"pip install {pkg}")
            all_ok = False

    # 5. ChromaDB index check
    print("\n🗄️ Index:")
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    indexed = os.path.isdir(chroma_path) and len(os.listdir(chroma_path)) > 0
    check("ChromaDB index exists", indexed,
          "Run: python -m src.chunking.pipeline --index --max-records 5000")

    # 6. Quick smoke test (no API keys needed)
    print("\n🧪 Smoke test:")
    try:
        from src.models import Chunk, ChunkingStrategy, GuardrailStatus
        from src.chunking import FixedSizeChunker, ChunkingRouter
        from src.guardrails import GuardrailManager

        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk("This is a test sentence for smoke testing the pipeline.")
        check("Chunking works", len(chunks) > 0)

        guard = GuardrailManager(use_openai_moderation=False, use_faithfulness_judge=False)
        results = guard.check_input("What is AI?")
        check("Guardrails work", all(r.status == GuardrailStatus.PASSED for r in results))
    except Exception as e:
        check("Smoke test", False, str(e))
        all_ok = False

    # Summary
    print("\n" + "=" * 40)
    if all_ok:
        print("✅ All checks passed! Ready to run.\n")
        print("Next steps:")
        print("  1. Index dataset:  python -m src.chunking.pipeline --index --max-records 5000")
        print("  2. Launch Web UI:  python -m src.api.app")
        print("  3. CLI query:      python -m src.harness.main --query 'What is AI?'")
        print("  4. Run benchmark:  python -m src.analytics.benchmark --queries 50")
    else:
        print("❌ Some checks failed. Fix the issues above and re-run.\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
