"""
Re-anchor the synthetic `first_seen` dates to *now* in both engines, in place.

`first_seen` is generated relative to an anchor time (synthesize.first_seen),
while the dates panel asks each engine for the trailing 30 days of wall-clock
time. Once the anchor is more than 30 days old the panel is silently empty on
both sides -- the Introduction tab's "Dates panel has data" check flags it.

A full reload fixes that too, but drops and rebuilds both indexes from the
Parquet. This rewrites only `first_seen`, from ONE anchor shared by both
engines (the loaders each take their own, minutes apart), then compacts /
force-merges so the segment layout matches a fresh load. Safe to re-run.

    docker compose exec demo python data/reanchor_dates.py
"""
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthesize                                          # noqa: E402

META = json.load(open(os.path.join(HERE, "schema.json")))
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
ES_URI = os.environ.get("ES_URI", "http://localhost:9200")
COLLECTION = INDEX = "amazon_reviews"
PK = META["primary_key"]
TS_FIELDS = META.get("timestamp_fields", [])
# The app's last-good run for the dates panel predates the new anchor.
DATES_CACHE = os.path.join(HERE, "..", "webapp", ".cache", "dates.json")
BATCH = 1000


def _dates(pk_value, now):
    return {f: synthesize.first_seen(pk_value, now) for f in TS_FIELDS}


def milvus_rows(ids, now):
    """Partial-upsert rows: the PK plus the date fields, nothing else, so
    vectors, text and the BM25 function's input are left untouched."""
    return [{PK: i, **_dates(i, now)} for i in ids]


def es_actions(ids, now):
    """Bulk partial updates carrying only the date fields."""
    for i in ids:
        yield {"_op_type": "update", "_index": INDEX, "_id": i,
               "doc": _dates(i, now)}


def _milvus_ids(client):
    it = client.query_iterator(COLLECTION, batch_size=BATCH, filter="",
                               output_fields=[PK])
    try:
        while True:
            page = it.next()
            if not page:
                return
            yield from (r[PK] for r in page)
    finally:
        it.close()


def reanchor_milvus(client, now):
    from milvus_util import compact_and_wait
    ids = list(_milvus_ids(client))
    for n in range(0, len(ids), BATCH):
        client.upsert(COLLECTION, milvus_rows(ids[n:n + BATCH], now),
                      partial_update=True)
        print(f"  milvus: {min(n + BATCH, len(ids))}/{len(ids)}", end="\r")
    print()
    # Each partial upsert is a delete + insert under the hood: flush so the
    # rows seal and index, then compact away the deletes, as load_milvus does.
    client.flush(COLLECTION)
    compact_and_wait(client, COLLECTION, timeout=600)
    return len(ids)


def reanchor_es(es, now):
    from elasticsearch import helpers
    from es_util import forcemerge_and_wait
    ids = [h["_id"] for h in helpers.scan(es, index=INDEX, _source=False,
                                          size=BATCH)]
    ok, _ = helpers.bulk(es, es_actions(ids, now), chunk_size=500)
    es.indices.refresh(index=INDEX)
    # An update is a full reindex of the doc in Lucene; merge back to the one
    # segment load_es.py leaves, so benchmarks see the same single HNSW graph.
    forcemerge_and_wait(es, INDEX, timeout=900)
    es.indices.refresh(index=INDEX)
    return ok


def _report(client, es, now):
    cutoff = now - dt.timedelta(days=synthesize.RECENT_DAYS)
    f = TS_FIELDS[0]
    m_new = client.query(COLLECTION, filter="", output_fields=[f"max({f})"])[0][f"max({f})"]
    m_in = client.query(COLLECTION, filter=f"{f} > ISO '{cutoff.isoformat()}'",
                        output_fields=["count(*)"])[0]["count(*)"]
    e = es.search(index=INDEX, size=0, track_total_hits=True,
                  query={"range": {f: {"gt": cutoff.isoformat()}}},
                  aggs={"newest": {"max": {"field": f}}})
    print(f"milvus: newest {m_new}, {m_in:,} in last {synthesize.RECENT_DAYS} d")
    print(f"es:     newest {e['aggregations']['newest']['value_as_string']}, "
          f"{e['hits']['total']['value']:,} in last {synthesize.RECENT_DAYS} d")


def main():
    from elasticsearch import Elasticsearch
    from pymilvus import MilvusClient

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    print(f"anchor: {now.isoformat()}  fields: {', '.join(TS_FIELDS)}")
    client = MilvusClient(uri=MILVUS_URI)
    es = Elasticsearch(ES_URI, request_timeout=180)

    t0 = time.monotonic()
    print(f"milvus: re-anchored {reanchor_milvus(client, now):,} rows "
          f"in {time.monotonic() - t0:.0f}s")
    t0 = time.monotonic()
    print(f"es: re-anchored {reanchor_es(es, now):,} docs "
          f"in {time.monotonic() - t0:.0f}s")

    if os.path.exists(DATES_CACHE):
        os.remove(DATES_CACHE)
        print("cleared the dates panel's cached run (re-run it in the app)")
    _report(client, es, now)


if __name__ == "__main__":
    main()
