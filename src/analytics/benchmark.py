"""
Latency analytics — instrument the pipeline, run benchmarks, and compute P50/P70/P100.

Usage:
    python -m src.analytics.benchmark --queries 50

Outputs:
    results/latency_results.csv  — raw per-query latency data
    results/latency_report.md   — P50/P70/P100 report
    results/latency_chart.png   — histogram + CDF visualization
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import settings
from src.logger import get_logger
from src.models import LatencyRecord, PipelineStatus

log = get_logger("analytics.benchmark")

RESULTS_DIR = Path(settings.results_path)


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute P50, P70, P90, P95, P100 from a list of values."""
    arr = np.array(values)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p70": round(float(np.percentile(arr, 70)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p100": round(float(np.percentile(arr, 100)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr)), 2),
    }


def run_benchmark(
    pipeline,
    queries: list[str],
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a benchmark over a list of queries and collect per-stage latencies.

    Args:
        pipeline: PipelineHarness instance.
        queries: List of query strings to test.
        output_dir: Directory to save results (defaults to settings.results_path).

    Returns:
        DataFrame with per-query latency records.
    """
    out_dir = Path(output_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[LatencyRecord] = []

    log.info("benchmark_start", num_queries=len(queries))

    for i, query in enumerate(queries):
        log.info("running_query", query_id=i, query=query[:80])
        try:
            response = pipeline.run(query)
            bd = response.latency_breakdown

            record = LatencyRecord(
                query_id=i,
                query=query,
                stt_ms=bd.get("stt_ms", 0.0),
                retrieval_ms=bd.get("retrieval_ms", 0.0),
                generation_ms=bd.get("generation_ms", 0.0),
                guardrails_ms=(
                    bd.get("input_guardrails_ms", 0.0) + bd.get("output_guardrails_ms", 0.0)
                ),
                total_ms=bd.get("total_ms", response.total_latency_ms),
                status=response.status,
            )
        except Exception as e:
            log.error("query_failed", query_id=i, error=str(e))
            record = LatencyRecord(
                query_id=i,
                query=query,
                status=PipelineStatus.ERROR,
                total_ms=0.0,
            )

        records.append(record)
        log.info("query_complete", query_id=i, total_ms=record.total_ms, status=record.status)

    # Convert to DataFrame
    df = pd.DataFrame([r.model_dump() for r in records])

    # Save CSV
    csv_path = out_dir / "latency_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("csv_saved", path=str(csv_path))

    # Generate report and charts
    _generate_report(df, out_dir)
    _generate_charts(df, out_dir)

    return df


def _generate_report(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate a P50/P70/P100 markdown report."""
    successful = df[df["status"] == PipelineStatus.SUCCESS.value]
    total = len(df)
    success_count = len(successful)

    report_lines = [
        "# Latency Analytics Report",
        "",
        f"**Total queries**: {total}  ",
        f"**Successful**: {success_count} ({100*success_count//max(1,total)}%)  ",
        f"**Failed/Blocked**: {total - success_count}",
        "",
        "## End-to-End Latency (retrieval + generation, ms)",
        "",
    ]

    if not successful.empty:
        e2e = successful["retrieval_ms"] + successful["generation_ms"]
        e2e_percentiles = compute_percentiles(e2e.tolist())
        report_lines += [
            f"| Metric | Value (ms) |",
            f"|--------|-----------|",
            f"| **P50** | {e2e_percentiles['p50']} |",
            f"| **P70** | {e2e_percentiles['p70']} |",
            f"| **P90** | {e2e_percentiles['p90']} |",
            f"| **P95** | {e2e_percentiles['p95']} |",
            f"| **P100** | {e2e_percentiles['p100']} |",
            f"| Mean | {e2e_percentiles['mean']} |",
            f"| Std Dev | {e2e_percentiles['std']} |",
            "",
            "## Per-Stage Breakdown",
            "",
            "| Stage | P50 | P70 | P100 | Mean |",
            "|-------|-----|-----|------|------|",
        ]

        for stage, col in [
            ("Retrieval", "retrieval_ms"),
            ("Generation", "generation_ms"),
            ("Guardrails", "guardrails_ms"),
            ("Total", "total_ms"),
        ]:
            if col in successful.columns:
                p = compute_percentiles(successful[col].tolist())
                report_lines.append(
                    f"| {stage} | {p['p50']} | {p['p70']} | {p['p100']} | {p['mean']} |"
                )

        target_met = e2e_percentiles["p50"] < 200
        report_lines += [
            "",
            "## Target Check",
            "",
            f"- P50 < 200ms target: {'✅ MET' if target_met else '❌ NOT MET'} "
            f"(P50={e2e_percentiles['p50']}ms)",
        ]
    else:
        report_lines.append("*No successful queries to report.*")

    report_text = "\n".join(report_lines)
    report_path = out_dir / "latency_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("report_saved", path=str(report_path))
    print(report_text)


def _generate_charts(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate histogram and CDF charts for retrieval + generation latency."""
    successful = df[df["status"] == PipelineStatus.SUCCESS.value]
    if successful.empty:
        return

    e2e = (successful["retrieval_ms"] + successful["generation_ms"]).values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RAG Pipeline Latency Distribution", fontsize=14, fontweight="bold")

    # Histogram
    ax1.hist(e2e, bins=20, color="#4C8BF5", edgecolor="white", alpha=0.85)
    ax1.axvline(np.percentile(e2e, 50), color="#E53935", linestyle="--", label="P50")
    ax1.axvline(np.percentile(e2e, 70), color="#FB8C00", linestyle="--", label="P70")
    ax1.axvline(np.percentile(e2e, 100), color="#43A047", linestyle="--", label="P100")
    ax1.axvline(200, color="gray", linestyle=":", label="200ms target")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Retrieval + Generation Latency Histogram")
    ax1.legend()

    # CDF
    sorted_e2e = np.sort(e2e)
    cdf = np.arange(1, len(sorted_e2e) + 1) / len(sorted_e2e)
    ax2.plot(sorted_e2e, cdf * 100, color="#4C8BF5", linewidth=2)
    ax2.axvline(200, color="gray", linestyle=":", label="200ms target")
    ax2.axhline(50, color="#E53935", linestyle="--", alpha=0.7, label="P50")
    ax2.axhline(70, color="#FB8C00", linestyle="--", alpha=0.7, label="P70")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Cumulative %")
    ax2.set_title("CDF of Retrieval + Generation Latency")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend()

    plt.tight_layout()
    chart_path = out_dir / "latency_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("chart_saved", path=str(chart_path))


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run latency benchmark")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries to benchmark")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    # Load test queries from MSMARCO-XI
    from src.chunking.pipeline import load_dataset_streaming
    from src.harness.pipeline import PipelineHarness
    from src.retrieval.retriever import HybridRetriever

    print(f"Loading {args.queries} test queries from MSMARCO-XI...")
    test_queries = []
    for record in load_dataset_streaming(max_records=args.queries + 20):
        q = record.get("query") or record.get("question")
        if q:
            test_queries.append(q)
        if len(test_queries) >= args.queries:
            break

    print(f"Collected {len(test_queries)} queries. Initializing pipeline...")
    retriever = HybridRetriever()
    pipeline = PipelineHarness(retriever=retriever)

    print("Running benchmark...")
    df = run_benchmark(pipeline, test_queries, args.output_dir)
    print(f"\n✅ Benchmark complete — {len(df)} queries processed")
