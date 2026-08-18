"""
Search-param sweep for the FILTERED dense search used in the demo snippet:

    filter='main_category == "All Electronics" and average_rating >= 4'

The existing gt.npz is unfiltered top-k, so recall against it is meaningless
here. This script brute-forces exact filtered top-10 over the parquet for the
same query vectors, then grids nprobe x rbq_bits_query x refine_k measuring
recall@10 and per-query latency.

Run inside the demo container:
    docker compose exec demo python bench/sweep_filtered.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from pymilvus import MilvusClient

from engines.milvus_engine import _hid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLLECTION = "amazon_reviews"
FILTER = 'main_category == "All Electronics" and average_rating >= 4'
K = 10
NQ = 40                     # matches the documented unfiltered sweep
OUT = os.path.join(os.path.dirname(__file__), "sweep_filtered_results.json")

NPROBE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RBQ_BITS = [0, 4, 8]
REFINE_K = [1.0, 1.5, 2.0, 3.0]


def filtered_truth(qs):
    meta = json.load(open(os.path.join(ROOT, "data", "schema.json")))
    pk = meta["primary_key"]
    df = pd.read_parquet(
        os.path.join(ROOT, "data", "amazon_reviews.parquet"),
        columns=[pk, "main_category", "average_rating", "text_vec"])
    mask = (df["main_category"] == "All Electronics") & (df["average_rating"] >= 4)
    sub = df[mask]
    n_total, n_match = len(df), len(sub)
    ids = sub[pk].astype(str).to_numpy()
    mat = np.vstack(sub["text_vec"].apply(lambda v: np.asarray(v, np.float32)))
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    truth = []
    for q in qs:
        s = mat @ q
        top = np.argpartition(-s, K)[:K]
        truth.append(set(ids[top[np.argsort(-s[top])]]))
    return pk, truth, n_match, n_total


def main():
    client = MilvusClient(uri=os.environ.get("MILVUS_URI", "http://localhost:19530"))
    gt = np.load(os.path.join(ROOT, "bench", "gt.npz"), allow_pickle=True)
    qs = gt["q_text"][:NQ].astype(np.float32)

    print("building filtered exact ground truth...", flush=True)
    pk, truth, n_match, n_total = filtered_truth(qs)
    print(f"filter matches {n_match:,} / {n_total:,} rows "
          f"({100 * n_match / n_total:.1f}%)", flush=True)

    def run(params):
        lat, rec = [], []
        for j in range(NQ):
            t0 = time.perf_counter()
            res = client.search(
                COLLECTION, data=[qs[j].tolist()], anns_field="text_vec",
                filter=FILTER,
                search_params={"metric_type": "COSINE", "params": params},
                limit=K, output_fields=[pk])
            lat.append((time.perf_counter() - t0) * 1000)
            got = {_hid(h, pk) for h in res[0]}
            rec.append(len(got & truth[j]) / K)
        return float(np.mean(rec)), float(np.mean(lat)), float(np.percentile(lat, 95))

    run({"nprobe": 64, "rbq_bits_query": 8, "refine_k": 2.0})   # warm-up, discarded

    results = []
    for nprobe in NPROBE:
        for bits in RBQ_BITS:
            for rk in REFINE_K:
                params = {"nprobe": nprobe, "rbq_bits_query": bits, "refine_k": rk}
                recall, mean_ms, p95_ms = run(params)
                results.append({**params, "recall": recall,
                                "mean_ms": mean_ms, "p95_ms": p95_ms})
                print(f"nprobe={nprobe:>4} bits={bits} refine_k={rk:<3} "
                      f"recall@10={recall:.3f} mean={mean_ms:6.2f}ms "
                      f"p95={p95_ms:6.2f}ms", flush=True)

    json.dump({"filter": FILTER, "n_queries": NQ, "k": K,
               "filter_match_rows": n_match, "total_rows": n_total,
               "results": results}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
