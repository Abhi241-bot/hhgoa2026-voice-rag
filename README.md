# Voice-Enabled RAG System — HH Goa 2026

A production-grade, low-latency **Voice-Enabled Retrieval-Augmented Generation (RAG)** pipeline built for the HH Goa 2026 shortlisting task.

## 🏗️ Architecture

```
[User Voice] → [ElevenLabs STT] → [Input Guardrails] → [Hybrid Retrieval]
                                                              ↓
                                               [Dense (ChromaDB) + BM25 Sparse]
                                                              ↓
                                                       [RRF Re-ranking]
                                                              ↓
                                        [Groq Llama-3.1-8b-instant Generation]
                                                              ↓
                                              [Output Guardrails] → [Answer]
```

## 📊 Performance (P50/P70/P100)

| Stage | P50 | P70 | P100 |
|-------|-----|-----|------|
| Retrieval | — | — | — |
| Generation | — | — | — |
| End-to-End (excl. STT) | — | — | — |

*(Updated after Phase 6: Latency Analytics)*

## 🗂️ Project Structure

```
hhgoa2026-voice-rag/
├── src/
│   ├── chunking/       # Multi-strategy chunking (fixed, recursive, semantic, metadata-aware)
│   ├── retrieval/      # ChromaDB + BM25 hybrid retrieval with RRF
│   ├── harness/        # Orchestration harness with retries, structured I/O, logging
│   ├── stt/            # ElevenLabs STT integration
│   ├── guardrails/     # Safety, off-topic, faithfulness, grounding checks
│   ├── analytics/      # Latency instrumentation and P50/P70/P100 reporting
│   └── api/            # FastAPI server + Gradio UI
├── tests/              # Unit and integration tests
├── results/            # Latency reports and charts
├── data/               # Dataset cache
└── notebooks/          # Exploration notebooks
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/hhgoa2026-voice-rag.git
cd hhgoa2026-voice-rag
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys:
# - ELEVENLABS_API_KEY (from https://elevenlabs.io/app/settings/api-keys)
# - GROQ_API_KEY (from https://console.groq.com/keys)
```

### 3. Index the Dataset

```bash
python -m src.chunking.pipeline --index
```

### 4. Run the Pipeline

**CLI (text mode):**
```bash
python -m src.harness.main --query "What is the capital of France?"
```

**Web UI:**
```bash
python -m src.api.app
# Open http://localhost:7860
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|------------|
| STT | ElevenLabs Speech API |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB | ChromaDB |
| Sparse Retrieval | BM25 (rank_bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| LLM | Groq Llama-3.1-8b-instant |
| Guardrails | Custom + OpenAI Moderation |
| API | FastAPI |
| UI | Gradio |

## 📈 Chunking Strategies

1. **Fixed-size** — configurable size (default 512 tokens) with 64-token overlap
2. **Recursive character splitting** — hierarchy-aware splitting on `\n\n → \n → . → " "`
3. **Semantic chunking** — splits on embedding similarity drops between sentences
4. **Metadata-aware** — preserves `passage_id`, `language`, `query_context` tags

## 🛡️ Guardrails

- **Safety filter** — blocks harmful/inappropriate inputs via OpenAI Moderation API
- **Off-topic detector** — rejects queries with cosine similarity < 0.25 to domain centroid
- **Grounding check** — refuses answer if no retrieved context is available
- **Faithfulness score** — LLM judge verifies answer is supported by retrieved passages

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📊 Latency Report

Run the full benchmark:

```bash
python -m src.analytics.benchmark --queries 50
```

Results saved to `results/latency_report.md` and `results/latency_results.csv`.

---

## Build Phases

- [x] Phase 0: Project Setup & GitHub Repo
- [ ] Phase 1: Dataset Loading & Multi-Strategy Chunking
- [ ] Phase 2: Vector DB Indexing & Hybrid Retrieval
- [ ] Phase 3: Orchestration Harness & LLM Generation
- [ ] Phase 4: Speech-to-Text Integration
- [ ] Phase 5: Guardrails
- [ ] Phase 6: Latency Analytics & Reporting
- [ ] Phase 7: Web UI & Final Polish
