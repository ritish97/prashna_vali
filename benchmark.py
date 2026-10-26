"""
Measures the actual latency improvement from the query-embedding cache,
so the number you put on a resume/README is real, benchmarked data.

Usage:
    python benchmark.py "What is the termination clause in this contract?"

Requires at least one document already ingested (POST /ingest first).
"""

import os
import statistics
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.retrieval import _query_cache, retrieve


def run(question: str, n: int = 20):
    print("=" * 60)
    print("  RAG Query Embedding Cache Benchmark")
    print("=" * 60)
    print(f"Query: '{question}'")
    print(f"Iterations for warm measurement: {n}\n")

    # Clear cache to guarantee a true cold-start measurement.
    _query_cache._store.clear()

    # Cold start (cache miss)
    cold_times = []
    chunks, cold_ms, cache_hit = retrieve(question)
    cold_times.append(cold_ms)
    assert cache_hit is False, "Expected a cache miss on first call."

    # Warm runs (cache hit)
    warm_times = []
    for _ in range(n):
        _, ms, cache_hit = retrieve(question)
        assert cache_hit is True, "Expected a cache hit on subsequent calls."
        warm_times.append(ms)

    cold = cold_times[0]
    warm_avg = statistics.mean(warm_times)
    warm_std = statistics.stdev(warm_times) if n > 1 else 0.0
    improvement = ((cold - warm_avg) / cold) * 100 if cold > 0 else 0.0

    print(f"Retrieved Chunks                 : {len(chunks)}")
    print(f"Cold (cache miss) retrieval time : {cold:8.2f} ms")
    print(f"Warm (cache hit) avg over {n:2d} runs: {warm_avg:8.2f} ms (+/- {warm_std:.2f} ms)")
    print(f"Retrieval Latency Improvement    : {improvement:8.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    run(q)
