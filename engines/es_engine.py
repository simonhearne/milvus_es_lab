"""
Elasticsearch 9.4 adapter for the amazon_reviews schema.

Defaults mirror the Milvus adapter: dense field text_vec, BM25 over text_snippet
(custom amazon_en analyzer), grouping via collapse on main_category. Pass
field="image_vec" to dense_search for image search. Hybrid uses the rrf retriever
so it lines up one-to-one with Milvus hybrid_search + RRFRanker.
"""
import os
from elasticsearch import Elasticsearch
from engines.base import Engine, timed

INDEX = "amazon_reviews"


class ESEngine(Engine):
    name = "Elasticsearch"

    def __init__(self, uri=None, primary_vec="text_vec", text_field="text_snippet"):
        self.client = Elasticsearch(uri or os.environ.get("ES_URI",
                                                          "http://localhost:9200"))
        self.vec = primary_vec
        self.text_field = text_field

    def _filter_clause(self, filter):
        # simple equality filters, symmetric with Milvus. VarChar -> .keyword,
        # arrays (categories) and numerics -> field directly.
        if not filter:
            return None
        clauses = []
        for k, v in filter.items():
            field = f"{k}.keyword" if isinstance(v, str) and k != "categories" else k
            clauses.append({"term": {field: v}})
        return clauses

    @timed
    def dense_search(self, query_vec, k=10, filter=None, field=None):
        knn = {"field": field or self.vec, "query_vector": query_vec,
               "k": k, "num_candidates": max(100, k * 10)}
        fclause = self._filter_clause(filter)
        if fclause:
            knn["filter"] = {"bool": {"filter": fclause}}
        res = self.client.search(index=INDEX, knn=knn, size=k, _source=False)
        return [(h["_id"], float(h["_score"])) for h in res["hits"]["hits"]]

    @timed
    def text_search(self, text, k=10, filter=None):
        query = {"match": {self.text_field: text}}
        fclause = self._filter_clause(filter)
        if fclause:
            query = {"bool": {"must": query, "filter": fclause}}
        res = self.client.search(index=INDEX, query=query, size=k, _source=False)
        return [(h["_id"], float(h["_score"])) for h in res["hits"]["hits"]]

    @timed
    def hybrid_search(self, query_vec, text, k=10, filter=None):
        retriever = {"rrf": {"retrievers": [
            {"standard": {"query": {"match": {self.text_field: text}}}},
            {"knn": {"field": self.vec, "query_vector": query_vec,
                     "k": k, "num_candidates": max(100, k * 10)}},
        ], "rank_window_size": max(50, k * 5)}}
        res = self.client.search(index=INDEX, retriever=retriever, size=k, _source=False)
        return [(h["_id"], float(h["_score"])) for h in res["hits"]["hits"]]

    @timed
    def grouped_search(self, query_vec, group_field="main_category", k=10):
        knn = {"field": self.vec, "query_vector": query_vec,
               "k": k * 5, "num_candidates": max(200, k * 20)}
        res = self.client.search(index=INDEX, knn=knn, size=k, _source=False,
                                 collapse={"field": f"{group_field}.keyword"})
        return [(h["_id"], float(h["_score"])) for h in res["hits"]["hits"]]


    # ---- Equivalents to Milvus 3.0 order_by / aggregation --------------------

    def ordered_search(self, query_vec, sort=None, k=10, filter=None):
        """kNN retrieval, then ES `sort` on scalar fields (overrides score sort)
        — the analogue of Milvus order_by_fields."""
        import time
        sort = sort or [{"price": "asc"}, {"average_rating": "desc"}]
        knn = {"field": self.vec, "query_vector": query_vec,
               "k": k, "num_candidates": max(100, k * 10)}
        fclause = self._filter_clause(filter)
        if fclause:
            knn["filter"] = {"bool": {"filter": fclause}}
        t0 = time.perf_counter()
        res = self.client.search(index=INDEX, knn=knn, size=k, sort=sort,
                                 _source=["price", "average_rating"])
        ms = (time.perf_counter() - t0) * 1000
        rows = [{"id": h["_id"],
                 "price": h["_source"].get("price"),
                 "rating": h["_source"].get("average_rating")}
                for h in res["hits"]["hits"]]
        return rows, ms

    def aggregate(self, group_field="main_category", limit=20):
        """terms agg + avg sub-aggs — the analogue of Milvus kernel aggregation."""
        import time
        t0 = time.perf_counter()
        res = self.client.search(index=INDEX, size=0, aggs={
            "by_group": {
                "terms": {"field": f"{group_field}.keyword", "size": limit},
                "aggs": {"avg_price": {"avg": {"field": "price"}},
                         "min_price": {"min": {"field": "price"}},
                         "max_price": {"max": {"field": "price"}},
                         "avg_rating": {"avg": {"field": "average_rating"}}},
            }})
        ms = (time.perf_counter() - t0) * 1000
        rows = [{"group": b["key"], "count": b["doc_count"],
                 "avg_price": b["avg_price"]["value"],
                 "min_price": b["min_price"]["value"],
                 "max_price": b["max_price"]["value"],
                 "avg_rating": b["avg_rating"]["value"]}
                for b in res["aggregations"]["by_group"]["buckets"]]
        return rows, ms
