"""Unit tests for data/reanchor_dates.py's write payloads (no live cluster).

Run in Docker: docker compose run --rm demo python -m pytest data -q
"""
import datetime as dt

from data import reanchor_dates, synthesize

NOW = dt.datetime(2026, 11, 18, 12, 0, 0, tzinfo=dt.timezone.utc)
IDS = ["B09K2DY4V1", "B000000001", "B0ZZZZZZZZ"]


def test_milvus_rows_carry_only_pk_and_dates():
    # A partial upsert must never touch vectors, text or BM25 inputs.
    rows = reanchor_dates.milvus_rows(IDS, NOW)
    assert [set(r) for r in rows] == [{"parent_asin", "first_seen"}] * len(IDS)
    assert [r["parent_asin"] for r in rows] == IDS


def test_es_actions_are_partial_updates_of_dates_only():
    acts = list(reanchor_dates.es_actions(IDS, NOW))
    assert {a["_op_type"] for a in acts} == {"update"}
    assert [a["_id"] for a in acts] == IDS
    assert [set(a["doc"]) for a in acts] == [{"first_seen"}] * len(IDS)


def test_both_engines_get_identical_dates_from_one_anchor():
    m = {r["parent_asin"]: r["first_seen"] for r in reanchor_dates.milvus_rows(IDS, NOW)}
    e = {a["_id"]: a["doc"]["first_seen"] for a in reanchor_dates.es_actions(IDS, NOW)}
    assert m == e == {i: synthesize.first_seen(i, NOW) for i in IDS}
