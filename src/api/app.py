"""
Gradio Web UI for the Voice-Enabled RAG pipeline.

Features:
    - Microphone recording button
    - Text input fallback
    - Answer display with source citations
    - Latency breakdown panel
    - Guardrail status display

Run with: python -m src.api.app
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import gradio as gr

from src.logger import get_logger

log = get_logger("api.app")

# ── Lazy pipeline singleton ────────────────────────────────────────────────────
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.harness.pipeline import PipelineHarness
        from src.retrieval.retriever import HybridRetriever
        from src.stt.elevenlabs import get_stt_tool

        log.info("initializing_gradio_pipeline")
        retriever = HybridRetriever()
        stt = get_stt_tool()
        _pipeline = PipelineHarness(retriever=retriever, stt_tool=stt)
        log.info("gradio_pipeline_ready")
    return _pipeline


# ── Core processing functions ─────────────────────────────────────────────────


def process_text_query(
    query: str,
    language_filter: Optional[str],
) -> Tuple[str, str, str]:
    """Process a text query and return (answer, sources, latency_info)."""
    if not query or not query.strip():
        return "⚠️ Please enter a query.", "", ""

    pipeline = _get_pipeline()
    response = pipeline.run(query, language_filter=language_filter or None)
    return _format_response(response)


def process_voice_query(
    audio_path: Optional[str],
    language_filter: Optional[str],
) -> Tuple[str, str, str]:
    """Process a voice recording and return (answer, sources, latency_info)."""
    if not audio_path:
        return "⚠️ Please record audio first.", "", ""

    from src.models import AudioInput

    pipeline = _get_pipeline()
    audio_input = AudioInput(file_path=audio_path, format="wav")
    response = pipeline.run(audio_input, language_filter=language_filter or None)
    return _format_response(response)


def _format_response(response) -> Tuple[str, str, str]:
    """Format a PipelineResponse for display."""
    from src.models import PipelineStatus

    if response.status == PipelineStatus.ERROR:
        answer = f"❌ **Error**: {response.error}"
        sources = ""
        latency = _format_latency(response.latency_breakdown)
        return answer, sources, latency

    if response.status == PipelineStatus.BLOCKED:
        answer = f"🚫 **Blocked**: {response.error}"
        guardrail_names = [g.name for g in response.guardrail_results if g.status.value == "blocked"]
        sources = f"**Guardrail triggered**: {', '.join(guardrail_names)}" if guardrail_names else ""
        latency = _format_latency(response.latency_breakdown)
        return answer, sources, latency

    # Success
    answer = f"✅ **Answer**:\n\n{response.answer}"
    if response.sources:
        sources_list = "\n".join(f"- `{s}`" for s in response.sources[:5])
        sources = f"**Sources** (passage IDs):\n{sources_list}"
    else:
        sources = "*No source IDs available*"

    latency = _format_latency(response.latency_breakdown)
    return answer, sources, latency


def _format_latency(breakdown: dict) -> str:
    """Format latency breakdown as a readable string."""
    if not breakdown:
        return ""
    lines = ["**⏱️ Latency Breakdown**"]
    stage_labels = {
        "stt_ms": "STT",
        "input_guardrails_ms": "Input Guardrails",
        "retrieval_ms": "Retrieval",
        "generation_ms": "Generation",
        "output_guardrails_ms": "Output Guardrails",
        "total_ms": "**Total**",
    }
    for key, label in stage_labels.items():
        val = breakdown.get(key)
        if val is not None:
            lines.append(f"- {label}: `{val:.1f}ms`")
    return "\n".join(lines)


# ── Gradio UI Layout ──────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Voice-Enabled RAG — HH Goa 2026",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css="""
        .main-header { text-align: center; padding: 20px 0; }
        .answer-box { min-height: 120px; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                 background: #4C8BF5; color: white; font-size: 12px; }
        """,
    ) as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Voice-Enabled RAG Pipeline</h1>
            <p style="color: #666;">HH Goa 2026 — Hybrid Retrieval + Guardrailed Generation</p>
            <p>
                <span class="badge">ElevenLabs STT</span>
                <span class="badge">ChromaDB + BM25</span>
                <span class="badge">Groq Llama-3.1</span>
                <span class="badge">Guardrails</span>
            </p>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Text Query ──────────────────────────────────────
            with gr.TabItem("💬 Text Query"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g. What is the capital of France?",
                            lines=2,
                        )
                        lang_filter_text = gr.Dropdown(
                            label="Language Filter (optional)",
                            choices=["", "en", "hi", "te", "ta", "bn", "mr"],
                            value="",
                        )
                        text_submit = gr.Button("🔍 Ask", variant="primary")

                    with gr.Column(scale=3):
                        text_answer = gr.Markdown(label="Answer", elem_classes=["answer-box"])
                        text_sources = gr.Markdown(label="Sources")
                        text_latency = gr.Markdown(label="Latency")

                text_submit.click(
                    fn=process_text_query,
                    inputs=[text_input, lang_filter_text],
                    outputs=[text_answer, text_sources, text_latency],
                )

                gr.Examples(
                    examples=[
                        ["What is the capital of France?", "en"],
                        ["Who invented the telephone?", "en"],
                        ["How does photosynthesis work?", "en"],
                        ["What are the symptoms of diabetes?", "en"],
                    ],
                    inputs=[text_input, lang_filter_text],
                )

            # ── Tab 2: Voice Query ─────────────────────────────────────
            with gr.TabItem("🎙️ Voice Query"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Record your question")
                        audio_input = gr.Audio(
                            label="Microphone",
                            type="filepath",
                            format="wav",
                        )
                        lang_filter_voice = gr.Dropdown(
                            label="Language Filter (optional)",
                            choices=["", "en", "hi", "te", "ta", "bn", "mr"],
                            value="",
                        )
                        voice_submit = gr.Button("🎙️ Transcribe & Ask", variant="primary")

                    with gr.Column(scale=3):
                        voice_answer = gr.Markdown(label="Answer", elem_classes=["answer-box"])
                        voice_sources = gr.Markdown(label="Sources")
                        voice_latency = gr.Markdown(label="Latency")

                voice_submit.click(
                    fn=process_voice_query,
                    inputs=[audio_input, lang_filter_voice],
                    outputs=[voice_answer, voice_sources, voice_latency],
                )

            # ── Tab 3: About ───────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
## Architecture

```
[Voice/Text] → [ElevenLabs STT] → [Input Guardrails]
                                          ↓
                              [Hybrid Retrieval: ChromaDB + BM25]
                                          ↓
                                  [Groq Llama-3.1 Generation]
                                          ↓
                          [Output Guardrails: Faithfulness + Grounding]
                                          ↓
                              [Structured Response + Latency]
```

## Chunking Strategies
1. **Fixed-size** — configurable size with overlap
2. **Recursive** — hierarchy-aware (paragraph → sentence → word)
3. **Semantic** — splits at embedding similarity drops
4. **Metadata-aware** — preserves MSMARCO-XI passage_id, language, query context

## Guardrails
- 🛡️ **Safety** — keyword filter + OpenAI Moderation
- 🎯 **Off-topic** — embedding cosine similarity to domain centroid
- ⚓ **Grounding** — ensures context was retrieved before answering
- ✅ **Faithfulness** — LLM judge checks answer matches context

## Dataset
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — Multilingual MSMARCO extension
                """)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("🚀 Starting Voice-Enabled RAG UI...")
    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
