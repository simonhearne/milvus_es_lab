"""
Milvus 3.0 adapter for the amazon_reviews schema.

Defaults: dense field text_vec (1024-d COSINE, IVF_RABITQ + SQ8 refine), BM25
sparse text_sparse (source text_snippet), grouping on main_category. Pass
field="image_vec" to dense_search for cross-modal / image search — same call,
different vector field.

PK is the VarChar `parent_asin`. This pymilvus build returns hits as Hit objects
that don't expose a synthetic "id" key when the PK isn't literally named "id",
so we always request the PK in output_fields and read it from the entity.
"""
import os
import time
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from engines.base import Engine, timed

COLLECTION = "amazon_reviews"


def _hid(h, pk):
    """Extract the primary key from a hit, tolerant of Hit-object vs dict."""
    ent = h.get("entity", {}) if hasattr(h, "get") else {}
    if isinstance(ent, dict) and ent.get(pk) is not None:
        return str(ent[pk])
    for getter in (lambda: h["id"], lambda: h.id, lambda: h["pk"]):
        try:
            v = getter()
            if v is not None:
                return str(v)
        except Exception:
            pass
    return None


def _hdist(h):
    try:
        return float(h["distance"])
    except Exception:
        return float(getattr(h, "distance", getattr(h, "score", 0.0)))


def _ent(h):
    return h.get("entity", {}) if hasattr(h, "get") else {}


class MilvusEngine(Engine):
    name = "Milvus"

    # IVF_RABITQ search params: NPROBE (buckets probed), rbq_bits_query (SQ1-8
    # query-vector quantization, 0=off), REFINE_K (how many candidates the SQ8
    # refine pass re-scores, as a multiple of limit). Two independent knobs, and
    # it matters which is which -- REFINE_K is the one carrying the recall here.
    #
    # Measured on v3.0.0, 40 ground-truth queries, recall@10 vs exact top-10,
    # nlist=1024:
    #   nprobe  @ REFINE_K=2.0:  1 -> 0.788   2 -> 0.898    4 -> 0.933
    #                            8 -> 0.980  16 -> 0.983   64 -> 0.990
    #                          256 -> 0.993 (9.6 ms)     1024 -> 0.993 (33.9 ms)
    #   refine_k @ NPROBE=64:  1.0 -> 0.930  1.5 -> 0.985  2.0 -> 0.990
    #                          3.0 -> 0.990 (saturated)
    #
    # nprobe has saturated by ~8: at REFINE_K=1.0, adding nprobe=64 is worth
    # +0.003 recall, and 64 -> 256 buys another +0.003 for 3x the latency. To
    # demo the recall/latency knob live, move REFINE_K (2.0 -> 1.0 costs six
    # points of recall) or drop NPROBE into the 1-8 range. Raising nprobe above
    # 64 on stage reads as a flat line.
    #
    # Filtered search shifts the nprobe knee right: under the demo filter
    # (main_category Electronics + rating >= 4, ~21% selective) recall@10 vs
    # exact filtered top-10 is 0.88 @ nprobe=8 and needs 64 to reach 0.985
    # (sweep: bench/sweep_filtered.py, results in sweep_filtered_results.json).
    #
    # rbq_bits_query moves latency, not recall: bits 2-8 are recall-identical
    # (refine re-scores the final candidates anyway) and 4 is the fastest of
    # them -- ~35% quicker than 8, ~3x quicker than 0 at this nprobe. bits=1 is
    # a cliff, not a step: it corrupts candidate ranking beyond what refine can
    # rescue (0.38-0.51 recall) and gets WORSE with more probes. Never ship 1.
    NPROBE = 4
    REFINE_K = 2.0
    SEARCH = {"params": {"nprobe": NPROBE, "rbq_bits_query": 4,
                         "refine_k": REFINE_K}}

    def __init__(self, uri=None, primary_vec="text_vec", pk="parent_asin",
                 sparse_field="text_sparse", text_field="text_snippet"):
        self.client = MilvusClient(uri=uri or os.environ.get(
            "MILVUS_URI", "http://localhost:19530"))
        self.vec = primary_vec
        self.pk = pk
        self.sparse = sparse_field
        self.text_field = text_field

    @timed
    def dense_search(self, query_vec, k=10, filter=None, field=None):
        res = self.client.search(
            COLLECTION, data=[query_vec], anns_field=field or self.vec, limit=k,
            filter=filter or "", search_params=self.SEARCH,
            output_fields=[self.pk],
        )
        return [(_hid(h, self.pk), _hdist(h)) for h in res[0]]

    @timed
    def text_search(self, text, k=10, filter=None):
        res = self.client.search(
            COLLECTION, data=[text], anns_field=self.sparse, limit=k,
            filter=filter or "", search_params={"metric_type": "BM25"},
            output_fields=[self.pk],
        )
        return [(_hid(h, self.pk), _hdist(h)) for h in res[0]]

    @timed
    def hybrid_search(self, query_vec, text, k=10, filter=None):
        dense = AnnSearchRequest([query_vec], self.vec, self.SEARCH,
                                 limit=k, expr=filter or "")
        sparse = AnnSearchRequest([text], self.sparse,
                                  {"metric_type": "BM25"}, limit=k, expr=filter or "")
        res = self.client.hybrid_search(COLLECTION, [dense, sparse],
                                        ranker=RRFRanker(), limit=k,
                                        output_fields=[self.pk])
        return [(_hid(h, self.pk), _hdist(h)) for h in res[0]]

    @timed
    def grouped_search(self, query_vec, group_field="main_category", k=10, group_size=1):
        res = self.client.search(
            COLLECTION, data=[query_vec], anns_field=self.vec, limit=k,
            group_by_field=group_field, group_size=group_size,
            search_params=self.SEARCH,
            output_fields=[self.pk, group_field],
        )
        return [(_hid(h, self.pk), _hdist(h)) for h in res[0]]

    # ---- Milvus 3.0 only -----------------------------------------------------
    # Verified against pymilvus v3.0.x docs. Re-confirm arg names on the GA tag.

    def ordered_search(self, query_vec, order_by=None, k=10, filter=None):
        """ANN search whose final order is a kernel-pushed multi-field scalar
        sort, not distance. order_by defaults to price asc, then rating desc."""
        order_by = order_by or [{"field": "price", "order": "asc"},
                                {"field": "average_rating", "order": "desc"}]
        t0 = time.perf_counter()
        res = self.client.search(
            COLLECTION, data=[query_vec], anns_field=self.vec, limit=k,
            filter=filter or "", search_params=self.SEARCH,
            output_fields=[self.pk, "price", "average_rating"],
            order_by_fields=order_by,
        )
        ms = (time.perf_counter() - t0) * 1000
        rows = [{"id": _hid(h, self.pk),
                 "price": _ent(h).get("price"),
                 "rating": _ent(h).get("average_rating")} for h in res[0]]
        return rows, ms

    def aggregate(self, group_field="main_category", filter="", limit=20):
        """Server-side SQL-style aggregation: count/avg/min/max pushed into the
        kernel. Group key must be INT/VARCHAR/TIMESTAMPTZ (main_category qualifies)."""
        t0 = time.perf_counter()
        res = self.client.query(
            COLLECTION, filter=filter, limit=limit,
            group_by_fields=[group_field],
            output_fields=[group_field, "count(*)",
                           "avg(price)", "min(price)", "max(price)",
                           "avg(average_rating)"],
        )
        ms = (time.perf_counter() - t0) * 1000
        rows = [{"group": r.get(group_field), "count": r.get("count(*)"),
                 "avg_price": r.get("avg(price)"),
                 "min_price": r.get("min(price)"),
                 "max_price": r.get("max(price)"),
                 "avg_rating": r.get("avg(average_rating)")} for r in res]
        rows.sort(key=lambda x: (x["count"] or 0), reverse=True)
        return rows, ms
