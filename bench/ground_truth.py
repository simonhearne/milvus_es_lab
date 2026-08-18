"""
Exact top-k ground truth via brute force over the full 100k vector matrix.

Recall is the ONE quantitative thing worth showing from a laptop Docker setup:
it's architectural, not hardware-bound, so it reproduces honestly. Latency from
this environment is dominated by JVM warmup / container limits / cold cache and
should NOT be presented as a benchmark — point at the published 160M study for
performance claims instead.

    python bench/ground_truth.py            # builds gt.npz from the parquet
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
PARQUET = os.path.join(DATA, "amazon_reviews.parquet")
SCHEMA = os.path.join(DATA, "schema.json")
OUT = os.path.join(HERE, "gt.npz")

sys.path.insert(0, DATA)
from fetch import require_parquet


def _meta():
    if not os.path.exists(SCHEMA):
        raise FileNotFoundError(
            f"{SCHEMA} not found. Unlike the parquet it is small enough to live "
            "in git — restore it from version control.")
    return json.load(open(SCHEMA))


def load_matrix(vec_field=None):
    """Load one dense field as a normalized float32 matrix. Defaults to the
    first dense vector in the schema (text_vec)."""
    META = _meta()
    df = pd.read_parquet(require_parquet(PARQUET))
    pk = META["primary_key"]
    vec = vec_field or list(META["dense_vectors"])[0]
    ids = df[pk].astype(str).to_numpy()
    mat = np.vstack(df[vec].apply(lambda v: np.asarray(v, dtype=np.float32)))
    # cosine == normalized dot
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return ids, mat


def exact_topk(mat, query_idx, k=100):
    q = mat[query_idx]
    sims = mat @ q
    top = np.argpartition(-sims, k)[:k]
    return top[np.argsort(-sims[top])]


def build(n_queries=200, k=100, seed=7):
    """Exact top-k per dense vector space, plus the query vectors themselves.

    `gt` stays the text-space array so existing readers keep working; `gt_image`
    is the image-space equivalent. Image search MUST be scored against gt_image —
    scoring it against `gt` compares unrelated vector spaces and yields a
    meaningless near-zero recall.
    """
    META = _meta()
    fields = list(META["dense_vectors"])          # ["text_vec", "image_vec"]
    rng = np.random.default_rng(seed)

    ids = None
    q_idx = None
    saved = {}
    for vec in fields:
        v_ids, mat = load_matrix(vec)
        if ids is None:
            ids = v_ids
            q_idx = rng.choice(len(ids), size=n_queries, replace=False)
        gt = np.vstack([exact_topk(mat, i, k) for i in q_idx])
        key = "gt" if vec == fields[0] else "gt_image"
        saved[key] = gt
        saved["q_text" if vec == fields[0] else "q_image"] = mat[q_idx].copy()
        del mat                                    # 400 MB per space; free it
        print(f"  {vec}: top-{k} for {n_queries} queries")

    np.savez(OUT, query_idx=q_idx, ids=ids, **saved)
    print(f"wrote {OUT}: {n_queries} queries x top-{k} for {len(fields)} vector spaces")
    return ids, q_idx, saved


if __name__ == "__main__":
    build()
