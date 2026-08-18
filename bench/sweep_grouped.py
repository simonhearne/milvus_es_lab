"""
Param sweep for the GROUPING / COLLAPSE demo snippet.

Milvus group_by guarantees `limit` distinct groups; ES collapse only groups
whatever knn returned, so k caps the group count. This sweeps the knobs on
both sides measuring latency, how many of the 10 requested groups actually
come back, and quality vs exact ground truth (the true top-10 categories by
best cosine hit).

Run inside the demo container:
    docker compose exec demo python bench/sweep_grouped.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from pymilvus import MilvusClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLLECTION = INDEX = "amazon_reviews"
GROUP_FIELD = "main_category"
K = 10                       # distinct groups requested
NQ = 40
OUT = os.path.join(os.path.dirname(__file__), "sweep_grouped_results.json")

MILVUS_NPROBE = [1, 2, 4, 8, 16, 32, 64]
ES_KNN = [(10, 100), (50, 200), (100, 200), (100, 400), (200, 400),
          (400, 800), (800, 1600)]


def group_truth(qs):
    meta = json.load(open(os.path.join(ROOT, "data", "schema.json")))
    pk = meta["primary_key"]
    df = pd.read_parquet(
        os.path.join(ROOT, "data", "amazon_reviews.parquet"),
        columns=[pk, GROUP_FIELD, "text_vec"])
    cats = df[GROUP_FIELD].to_numpy()
    mat = np.vstack(df["text_vec"].apply(lambda v: np.asarray(v, np.float32)))
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    uniq = pd.unique(cats)
    print(f"{len(df):,} rows, {len(uniq)} distinct {GROUP_FIELD} values",
          flush=True)
    truth = []
    for q in qs:
        s = mat @ q
        best = {}
        for c in uniq:
            best[c] = s[cats == c].max()
        top = sorted(best, key=best.get, reverse=True)[:K]
        truth.append(set(top))
    return pk, truth, len(uniq)


def summarize(lat, groups, hit):
    return {"mean_ms": float(np.mean(lat)),
            "p95_ms": float(np.percentile(lat, 95)),
            "mean_groups": float(np.mean(groups)),
            "group_recall": float(np.mean(hit))}


def main():
    mc = MilvusClient(uri=os.environ.get("MILVUS_URI", "http://localhost:19530"))
    es = Elasticsearch(os.environ.get("ES_URI", "http://localhost:9200"))
    gt = np.load(os.path.join(ROOT, "bench", "gt.npz"), allow_pickle=True)
    qs = gt["q_text"][:NQ].astype(np.float32)

    print("building exact per-category ground truth...", flush=True)
    pk, truth, n_cats = group_truth(qs)

    def run_milvus(nprobe):
        params = {"nprobe": nprobe, "rbq_bits_query": 4, "refine_k": 2.0}
        lat, groups, hit = [], [], []
        for j in range(NQ):
            t0 = time.perf_counter()
            res = mc.search(
                COLLECTION, data=[qs[j].tolist()], anns_field="text_vec",
                group_by_field=GROUP_FIELD, group_size=1,
                search_params={"metric_type": "COSINE", "params": params},
                limit=K, output_fields=[GROUP_FIELD])
            lat.append((time.perf_counter() - t0) * 1000)
            got = {h["entity"][GROUP_FIELD] for h in res[0]}
            groups.append(len(got))
            hit.append(len(got & truth[j]) / K)
        return summarize(lat, groups, hit)

    def run_es(k, nc):
        lat, groups, hit = [], [], []
        for j in range(NQ):
            knn = {"field": "text_vec", "query_vector": qs[j].tolist(),
                   "k": k, "num_candidates": nc}
            t0 = time.perf_counter()
            res = es.search(index=INDEX, knn=knn, size=K, _source=False,
                            collapse={"field": f"{GROUP_FIELD}.keyword"},
                            fields=[f"{GROUP_FIELD}.keyword"])
            lat.append((time.perf_counter() - t0) * 1000)
            got = {h["fields"][f"{GROUP_FIELD}.keyword"][0]
                   for h in res["hits"]["hits"]}
            groups.append(len(got))
            hit.append(len(got & truth[j]) / K)
        return summarize(lat, groups, hit)

    run_milvus(16); run_es(100, 200)          # warm-up, discarded

    results = {"milvus": [], "es": []}
    for nprobe in MILVUS_NPROBE:
        r = {"nprobe": nprobe, **run_milvus(nprobe)}
        results["milvus"].append(r)
        print(f"milvus nprobe={nprobe:>3}  mean={r['mean_ms']:6.2f}ms "
              f"p95={r['p95_ms']:6.2f}ms groups={r['mean_groups']:4.1f} "
              f"grec={r['group_recall']:.3f}", flush=True)
    for k, nc in ES_KNN:
        r = {"k": k, "num_candidates": nc, **run_es(k, nc)}
        results["es"].append(r)
        print(f"es k={k:>4} nc={nc:>4}       mean={r['mean_ms']:6.2f}ms "
              f"p95={r['p95_ms']:6.2f}ms groups={r['mean_groups']:4.1f} "
              f"grec={r['group_recall']:.3f}", flush=True)

    json.dump({"n_queries": NQ, "k": K, "n_categories": n_cats,
               "results": results}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
