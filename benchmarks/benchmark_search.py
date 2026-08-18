"""
VectorHub Phase 2 Benchmark
============================
Measures build time, query latency, QPS, and recall@k for
HNSW vs. brute-force search.

Usage
-----
    python benchmarks/benchmark_search.py [--n N] [--dim D] [--k K] [--queries Q]

Defaults: n=10000, dim=128, k=10, queries=200
"""

import argparse
import time
import sys
import os

import numpy as np

# Allow running from the repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vectorhub import VectorCollection


def generate_data(n: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def recall_at_k(hnsw_ids, bf_ids):
    bf_set = set(bf_ids)
    hits = sum(1 for x in hnsw_ids if x in bf_set)
    return hits / max(len(bf_ids), 1)


def run_benchmark(n: int, dim: int, k: int, n_queries: int):
    print(f"\n{'='*60}")
    print(f"VectorHub Phase 2 Benchmark")
    print(f"  n={n}  dim={dim}  k={k}  queries={n_queries}")
    print(f"{'='*60}\n")

    data = generate_data(n, dim, seed=42)
    queries = generate_data(n_queries, dim, seed=99)
    ids = list(range(n))

    # ── HNSW build ────────────────────────────────────────────────────────────
    print("Building HNSW index (M=16, ef_construction=200)...")
    col_hnsw = VectorCollection(dim=dim, metric="l2", M=32, ef_construction=300, ef_search=50)
    t0 = time.perf_counter()
    col_hnsw.add(data, ids)
    hnsw_build_time = time.perf_counter() - t0
    print(f"  Build time : {hnsw_build_time:.3f}s")
    print(f"  Insert rate: {n / hnsw_build_time:.0f} vectors/s\n")

    # ── HNSW query ────────────────────────────────────────────────────────────
    print(f"HNSW search (ef_search=50, k={k})...")
    # Warm up
    for qv in queries[:5]:
        col_hnsw.search(qv.tolist(), k=k)

    t0 = time.perf_counter()
    hnsw_results = []
    for qv in queries:
        res = col_hnsw.search(qv.tolist(), k=k)
        hnsw_results.append([r[0] for r in res])
    hnsw_query_time = time.perf_counter() - t0
    hnsw_avg_ms = hnsw_query_time / n_queries * 1000
    hnsw_qps = n_queries / hnsw_query_time
    print(f"  Total time : {hnsw_query_time:.3f}s for {n_queries} queries")
    print(f"  Avg latency: {hnsw_avg_ms:.3f}ms / query")
    print(f"  QPS        : {hnsw_qps:.0f}\n")

    # ── Brute-force query ─────────────────────────────────────────────────────
    print(f"Brute-force search (exact, k={k})...")
    t0 = time.perf_counter()
    bf_results = []
    for qv in queries:
        res = col_hnsw.brute_force_search(qv.tolist(), k=k)
        bf_results.append([r[0] for r in res])
    bf_query_time = time.perf_counter() - t0
    bf_avg_ms = bf_query_time / n_queries * 1000
    bf_qps = n_queries / bf_query_time
    print(f"  Total time : {bf_query_time:.3f}s for {n_queries} queries")
    print(f"  Avg latency: {bf_avg_ms:.3f}ms / query")
    print(f"  QPS        : {bf_qps:.0f}\n")

    # ── Recall@k ──────────────────────────────────────────────────────────────
    recalls = [
        recall_at_k(hnsw_results[i], bf_results[i])
        for i in range(n_queries)
    ]
    mean_recall = sum(recalls) / len(recalls)
    min_recall  = min(recalls)
    p10_recall  = sorted(recalls)[n_queries // 10]

    print(f"Recall@{k} (HNSW vs brute-force):")
    print(f"  Mean   : {mean_recall:.4f}")
    print(f"  Min    : {min_recall:.4f}")
    print(f"  P10    : {p10_recall:.4f}\n")

    # ── Speed-up ──────────────────────────────────────────────────────────────
    speedup = bf_query_time / hnsw_query_time
    print(f"HNSW vs brute-force speedup: {speedup:.1f}x\n")

    # ── Higher ef_search ──────────────────────────────────────────────────────
    print(f"HNSW with ef_search=200 (better recall, slower)...")
    col_hnsw.set_ef_search(200)
    t0 = time.perf_counter()
    hnsw_ef200_results = []
    for qv in queries:
        res = col_hnsw.search(qv.tolist(), k=k)
        hnsw_ef200_results.append([r[0] for r in res])
    hnsw_ef200_time = time.perf_counter() - t0

    recalls_ef200 = [
        recall_at_k(hnsw_ef200_results[i], bf_results[i])
        for i in range(n_queries)
    ]
    mean_recall_ef200 = sum(recalls_ef200) / len(recalls_ef200)
    print(f"  Avg latency: {hnsw_ef200_time / n_queries * 1000:.3f}ms / query")
    print(f"  QPS        : {n_queries / hnsw_ef200_time:.0f}")
    print(f"  Recall@{k}  : {mean_recall_ef200:.4f}\n")

    # ── Higher ef_search=400 ──────────────────────────────────────────────────
    print(f"HNSW with ef_search=400 (highest recall, slowest)...")
    col_hnsw.set_ef_search(400)
    t0 = time.perf_counter()
    hnsw_ef400_results = []
    for qv in queries:
        res = col_hnsw.search(qv.tolist(), k=k)
        hnsw_ef400_results.append([r[0] for r in res])
    hnsw_ef400_time = time.perf_counter() - t0

    recalls_ef400 = [
        recall_at_k(hnsw_ef400_results[i], bf_results[i])
        for i in range(n_queries)
    ]
    mean_recall_ef400 = sum(recalls_ef400) / len(recalls_ef400)
    print(f"  Avg latency: {hnsw_ef400_time / n_queries * 1000:.3f}ms / query")
    print(f"  QPS        : {n_queries / hnsw_ef400_time:.0f}")
    print(f"  Recall@{k}  : {mean_recall_ef400:.4f}\n")

    print("="*60)
    print("Summary")
    print(f"  Build: {hnsw_build_time:.2f}s ({n / hnsw_build_time:.0f} vec/s)")
    print(f"  HNSW  ef=50  : {hnsw_avg_ms:.2f}ms / query, "
          f"recall@{k}={mean_recall:.3f}")
    print(f"  HNSW  ef=200 : {hnsw_ef200_time / n_queries * 1000:.2f}ms / query, "
          f"recall@{k}={mean_recall_ef200:.3f}")
    print(f"  HNSW  ef=400 : {hnsw_ef400_time / n_queries * 1000:.2f}ms / query, "
          f"recall@{k}={mean_recall_ef400:.3f}")  # 新增这一行
    print(f"  Brute-force  : {bf_avg_ms:.2f}ms / query (exact)")
    print(f"  Speedup (ef=50 vs BF): {speedup:.1f}x")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="VectorHub benchmark")
    parser.add_argument("--n", type=int, default=100000, help="Number of vectors to index")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimensionality")
    parser.add_argument("--k", type=int, default=10, help="Top-k for search")
    parser.add_argument("--queries", type=int, default=200, help="Number of query vectors")
    args = parser.parse_args()

    run_benchmark(args.n, args.dim, args.k, args.queries)


if __name__ == "__main__":
    main()
