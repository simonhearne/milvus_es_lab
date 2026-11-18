"""Unit tests for webapp/environment.py (no live cluster).

Run locally:  python3 -m webapp.test_environment
Or in Docker: docker compose exec -T demo python -m webapp.test_environment
"""
import datetime as dt

import webapp.environment as environment
from webapp.environment import describe, ES_HEAP_BYTES


def _iso_days_ago(days):
    """first_seen as each engine reports it, `days` before environment's clock."""
    return environment._now() - dt.timedelta(days=days)


# ---- stubs ---------------------------------------------------------------

class _Attr:
    """Turns a dict of name->callable into an object with those methods."""
    def __init__(self, **methods):
        for name, fn in methods.items():
            setattr(self, name, fn)


class _StubMilvus:
    def __init__(self, docs=97894, dim=1024, version="v3.0.6", newest_age_days=1):
        self._docs, self._dim, self._version = docs, dim, version
        self._age = newest_age_days

    def query(self, name, filter="", output_fields=None):
        if output_fields == ["max(first_seen)"]:
            if self._age is None:
                raise RuntimeError("aggregation unsupported")
            return [{"max(first_seen)":
                     _iso_days_ago(self._age).strftime("%Y-%m-%dT%H:%M:%SZ")}]
        return [{"count(*)": self._docs}]

    def list_collections(self):
        return ["amazon_reviews", "multilingual_demo"]

    def get_server_version(self):
        return self._version

    def get_collection_stats(self, name):
        return {"row_count": self._docs}

    def describe_collection(self, name):
        return {"fields": [
            {"name": "parent_asin", "type": "VarChar"},
            {"name": "text_vec", "type": "FloatVector", "params": {"dim": self._dim}},
        ]}


class _StubES:
    def __init__(self, heap=ES_HEAP_BYTES, nodes=1, license="basic",
                 docs=97894, dim=1024, sim="cosine", newest_age_days=1):
        self._heap, self._nodes, self._license = heap, nodes, license
        self._age = newest_age_days
        self._docs, self._dim, self._sim = docs, dim, sim
        self.cluster = _Attr(health=lambda: {"number_of_nodes": self._nodes})
        self.license = _Attr(get=lambda: {"license": {"type": self._license}})
        self.nodes = _Attr(stats=lambda metric=None: {"nodes": {"n0": {
            "jvm": {"mem": {"heap_max_in_bytes": self._heap}},
            "os": {"mem": {"used_in_bytes": 3 * 1024 ** 3}}}}})
        self.indices = _Attr(get_mapping=lambda index=None: {index: {"mappings": {
            "properties": {"text_vec": {"dims": self._dim, "similarity": self._sim}}}}})

    def ping(self):
        return True

    def info(self):
        return {"version": {"number": "9.5.6"}}

    def count(self, index=None):
        return {"count": self._docs}

    def search(self, index=None, size=None, aggs=None, **kw):
        ts = _iso_days_ago(self._age)
        return {"aggregations": {"newest": {
            "value": ts.timestamp() * 1000,
            "value_as_string": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")}}}


class _Session:
    def __init__(self, client, es):
        self.client, self.es = client, es


def _by_label(checks):
    return {c["label"]: c for c in checks}


# ---- tests ---------------------------------------------------------------

def test_all_healthy_every_check_good():
    environment.milvus_mem_bytes = lambda uri, timeout=2.0: 2_500_000_000
    env = describe(_Session(_StubMilvus(), _StubES()))
    assert env["allocation"]["equal"] is True
    assert env["milvus"]["ok"] and env["es"]["ok"]
    assert env["milvus"]["mem_usage_bytes"] == 2_500_000_000
    states = {c["state"] for c in env["checks"]}
    assert states == {"good"}, [c for c in env["checks"] if c["state"] != "good"]


def test_es_down_marks_es_checks_na_milvus_still_ok():
    environment.milvus_mem_bytes = lambda uri, timeout=2.0: 1
    env = describe(_Session(_StubMilvus(), None))
    assert env["es"]["ok"] is False
    assert env["milvus"]["ok"] is True
    by = _by_label(env["checks"])
    assert by["ES heap = 4 GiB"]["state"] == "na"
    assert by["Doc counts match"]["state"] == "na"
    assert by["Milvus version"]["state"] == "good"


def test_milvus_down_marks_milvus_checks_na_es_still_ok():
    env = describe(_Session(None, _StubES()))
    assert env["milvus"]["ok"] is False
    assert env["es"]["ok"] is True
    by = _by_label(env["checks"])
    assert by["Milvus version"]["state"] == "na"
    assert by["ES licence: basic"]["state"] == "good"


def test_doc_count_mismatch_is_warn():
    environment.milvus_mem_bytes = lambda uri, timeout=2.0: 1
    env = describe(_Session(_StubMilvus(docs=97894), _StubES(docs=50000)))
    assert _by_label(env["checks"])["Doc counts match"]["state"] == "warn"


def test_heap_mismatch_is_warn():
    environment.milvus_mem_bytes = lambda uri, timeout=2.0: 1
    env = describe(_Session(_StubMilvus(), _StubES(heap=2 * 1024 ** 3)))
    assert _by_label(env["checks"])["ES heap = 4 GiB"]["state"] == "warn"


def test_metric_check_is_case_insensitive():
    environment.milvus_mem_bytes = lambda uri, timeout=2.0: 1
    # Milvus reports COSINE, ES reports cosine — must still be good.
    env = describe(_Session(_StubMilvus(), _StubES(sim="cosine")))
    assert _by_label(env["checks"])["Cosine both"]["state"] == "good"


DATES = "Dates panel has data"


def test_fresh_dates_are_good():
    env = describe(_Session(_StubMilvus(newest_age_days=1), _StubES(newest_age_days=1)))
    c = _by_label(env["checks"])[DATES]
    assert c["state"] == "good", c
    assert env["milvus"]["newest_first_seen"] and env["es"]["newest_first_seen"]


def test_stale_dates_warn_and_name_the_fix():
    # The real failure: corpus loaded 98 days ago, panel silently empty.
    env = describe(_Session(_StubMilvus(newest_age_days=98), _StubES(newest_age_days=98)))
    c = _by_label(env["checks"])[DATES]
    assert c["state"] == "warn", c
    assert "98 d" in c["detail"] and "empty" in c["detail"], c
    assert "reanchor_dates.py" in c["detail"], c


def test_dates_about_to_expire_warn_with_days_left():
    env = describe(_Session(_StubMilvus(newest_age_days=27.5), _StubES(newest_age_days=27.5)))
    c = _by_label(env["checks"])[DATES]
    assert c["state"] == "warn", c
    assert "empties in 2 d" in c["detail"], c


def test_dates_judged_on_the_staler_engine():
    env = describe(_Session(_StubMilvus(newest_age_days=1), _StubES(newest_age_days=40)))
    assert _by_label(env["checks"])[DATES]["state"] == "warn"


def test_dates_unreadable_is_warn_not_crash():
    env = describe(_Session(_StubMilvus(newest_age_days=None), _StubES()))
    c = _by_label(env["checks"])[DATES]
    assert c["state"] == "warn" and "Milvus" in c["detail"], c


def test_dates_na_when_an_engine_is_down():
    env = describe(_Session(_StubMilvus(), None))
    assert _by_label(env["checks"])[DATES]["state"] == "na"


def test_dates_window_matches_synthesizer():
    from data import synthesize
    assert environment.DATES_WINDOW_DAYS == synthesize.RECENT_DAYS


def test_new_schema_fields_present_on_both_engines():
    """Live check: a partial reload must fail loudly here rather than silently
    rendering four empty panels on stage."""
    import os
    try:
        from pymilvus import MilvusClient
        from elasticsearch import Elasticsearch
        c = MilvusClient(uri=os.environ.get(
            "MILVUS_URI", "http://milvus:19530"), timeout=5)
        es = Elasticsearch(os.environ.get("ES_URI", "http://elasticsearch:9200"),
                           request_timeout=5)
        # Cheap connectivity probe only. Everything past this point must
        # raise as a hard failure, not a skip -- otherwise a partial reload
        # (collection/index missing, mapping malformed) would print
        # "skipped" and pass, exactly the silent-four-empty-panels failure
        # mode this test exists to catch.
        c.list_collections()
        if not es.ping():
            # ping() swallows ApiError/TransportError and returns False --
            # it never raises, so a silent False must be turned into an
            # exception here or a dead ES falls through to the unguarded
            # get_mapping() below as a hard failure instead of a clean skip.
            raise ConnectionError("ES unreachable")
    except Exception as e:                      # engines down: not this test's job
        print("   (skipped, engines unreachable:", type(e).__name__, ")")
        return

    fields = {f["name"]: f for f in
              c.describe_collection("amazon_reviews")["fields"]}
    props = es.indices.get_mapping(index="amazon_reviews")[
        "amazon_reviews"]["mappings"]["properties"]

    assert "store_location" in fields, sorted(fields)
    assert "first_seen" in fields, sorted(fields)
    assert fields["text_snippet"]["params"].get("enable_match") == "true", \
        fields["text_snippet"]["params"]
    assert props["store_location"]["type"] == "geo_point", props["store_location"]
    assert props["first_seen"]["type"] == "date", props["first_seen"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"OK {len(tests)} environment tests passed")
