# Latency Analytics Report

**Total queries**: 50  
**Successful**: 50 (100%)  
**Failed/Blocked**: 0

## End-to-End Latency (retrieval + generation, ms)

| Metric | Value (ms) |
|--------|-----------|
| **P50** | 4711.28 |
| **P70** | 4820.9 |
| **P90** | 5503.69 |
| **P95** | 5748.11 |
| **P100** | 6148.46 |
| Mean | 3838.66 |
| Std Dev | 1698.75 |

## Per-Stage Breakdown

| Stage | P50 | P70 | P100 | Mean |
|-------|-----|-----|------|------|
| Retrieval | 45.27 | 49.2 | 99.48 | 44.83 |
| Generation | 4666.58 | 4775.97 | 6092.99 | 3793.83 |
| Guardrails | 0.16 | 0.19 | 17438.98 | 368.53 |
| Total | 4721.49 | 4847.8 | 18339.46 | 4207.25 |

## Target Check

- P50 < 200ms target: ❌ NOT MET (P50=4711.28ms)