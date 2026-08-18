"""
Elasticsearch counterpart of sweep_filtered.py — same 40 queries, same filter,
same exact filtered ground truth, sweeping the knobs the ES knn clause exposes:
num_candidates and rescore_vector.oversample.

The live index is bbq_hnsw (m=16, ef_construction=100) with a mapping-level
rescore_vector.oversample of 3.0, so oversample here overrides that default
per-query (0 disables rescoring entirely).

Run inside the demo container:
    docker compose exec demo python bench/sweep_filtered_es.py
"""
import json
import os
import time

import numpy as np
from elasticsearch import Elasticsearch

from bench.sweep_filtered import filtered_truth, FILTER, K, NQ

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX = "amazon_reviews"
OUT = os.path.join(os.path.dirname(__file__), "sweep_filtered_es_results.json")

ES_FILTER = [
    {"term": {"main_category.keyword": "All Electronics"}},
    {"range": {"average_rating": {"gte": 4}}},
]
NUM_CANDIDATES = [10, 20, 50, 100, 200, 500, 1000]
OVERSAMPLE = [0, 1, 2, 3, 5]


def main():
    client = Elasticsearch(os.environ.get("ES_URI", "http://localhost:9200"))
    gt = np.load(os.path.join(ROOT, "bench", "gt.npz"), allow_pickle=True)
    qs = gt["q_text"][:NQ].astype(np.float32)

    print("building filtered exact ground truth...", flush=True)
    pk, truth, n_match, n_total = filtered_truth(qs)
    print(f"filter matches {n_match:,} / {n_total:,} rows "
          f"({100 * n_match / n_total:.1f}%)", flush=True)

    def run(num_candidates, oversample):
        lat, rec = [], []
        for j in range(NQ):
            knn = {"field": "text_vec", "query_vector": qs[j].tolist(),
                   "k": K, "num_candidates": num_candidates,
                   "filter": ES_FILTER,
                   "rescore_vector": {"oversample": oversample}}
            t0 = time.perf_counter()
            res = client.search(index=INDEX, knn=knn, size=K, _source=False)
            lat.append((time.perf_counter() - t0) * 1000)
            got = {str(h["_id"]) for h in res["hits"]["hits"]}
            rec.append(len(got & truth[j]) / K)
        return float(np.mean(rec)), float(np.mean(lat)), float(np.percentile(lat, 95))

    run(100, 3)                                     # warm-up, discarded

    results = []
    for nc in NUM_CANDIDATES:
        for ov in OVERSAMPLE:
            recall, mean_ms, p95_ms = run(nc, ov)
            results.append({"num_candidates": nc, "oversample": ov,
                            "recall": recall, "mean_ms": mean_ms, "p95_ms": p95_ms})
            print(f"num_candidates={nc:>4} oversample={ov} "
                  f"recall@10={recall:.3f} mean={mean_ms:6.2f}ms "
                  f"p95={p95_ms:6.2f}ms", flush=True)

    json.dump({"filter": FILTER, "es_filter": ES_FILTER, "n_queries": NQ, "k": K,
               "filter_match_rows": n_match, "total_rows": n_total,
               "results": results}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
