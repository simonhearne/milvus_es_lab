"""Normalize pymilvus and Elasticsearch responses into one shape.

The PK-extraction helpers come from engines/milvus_engine.py rather than being
reimplemented: they already handle this build's quirk where a Hit exposes no
synthetic "id" when the PK isn't literally named "id".

If ids can't be extracted we say so loudly. A hit list with no ids yields
overlap 0 and recall 0 -- numbers that look like a finding rather than a bug,
which is the worst possible failure on stage.
"""
from engines.milvus_engine import _hid, _hdist, _ent

MISSING_PK = ('hits carry no extractable primary key -- add '
              'output_fields=["parent_asin"] to the search call. '
              'Recall and overlap are suppressed until you do.')


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


def _es_hl(h):
    """ES highlighting: {"field": ["fragment", ...]} -- already the target shape."""
    return {str(k): [str(x) for x in (v or [])]
            for k, v in (h.get("highlight") or {}).items()}


def _milvus_hl(h):
    """Milvus highlighting: {"field": {"fragments": [...], "scores": [...]}}.

    Flattened to the ES shape so the frontend renders one thing, not two. A
    plain list is tolerated in case a future build drops the nesting.
    """
    raw = h.get("highlight") if hasattr(h, "get") else None
    out = {}
    for field, payload in (raw or {}).items():
        frags = payload.get("fragments") if isinstance(payload, dict) else payload
        out[str(field)] = [str(x) for x in (frags or [])]
    return out


def _es_hits(body, pk):
    hits = []
    for rank, h in enumerate(body["hits"]["hits"], start=1):
        hits.append({"id": str(h["_id"]),
                     "score": float(h["_score"]) if h.get("_score") is not None else 0.0,
                     "rank": rank,
                     "highlight": _es_hl(h),
                     "fields": _jsonable(h.get("_source") or {})})
    return {"kind": "hits", "hits": hits, "rows": [],
            "raw": _jsonable(body), "warnings": []}


def _es_aggs(body):
    rows = []
    for agg in body["aggregations"].values():
        for b in agg.get("buckets", []):
            row = {"group": b.get("key"), "count": b.get("doc_count")}
            for k, v in b.items():
                if isinstance(v, dict) and "value" in v:
                    row[k] = v["value"]
            rows.append(row)
    return {"kind": "rows", "hits": [], "rows": _jsonable(rows),
            "raw": _jsonable(body), "warnings": []}


def _milvus_hits(hitlist, pk):
    hits, missing = [], 0
    for rank, h in enumerate(hitlist, start=1):
        hid = _hid(h, pk)
        if hid is None:
            missing += 1
        hits.append({"id": hid, "score": _hdist(h), "rank": rank,
                     "highlight": _milvus_hl(h),
                     "fields": _jsonable(_ent(h))})
    warnings = [MISSING_PK] if missing and missing == len(hits) else []
    return {"kind": "hits", "hits": hits, "rows": [],
            "raw": _jsonable([{"id": h["id"], "score": h["score"],
                               "entity": h["fields"]} for h in hits]),
            "warnings": warnings}


def normalize(resp, pk):
    """Map any supported engine response to {kind, hits, rows, raw, warnings}."""
    if hasattr(resp, "body"):            # ES ObjectApiResponse
        resp = resp.body
    if isinstance(resp, dict):
        if "aggregations" in resp:
            return _es_aggs(resp)
        if "hits" in resp:
            return _es_hits(resp, pk)
        return {"kind": "rows", "hits": [], "rows": [_jsonable(resp)],
                "raw": _jsonable(resp), "warnings": []}

    try:
        first = resp[0]
    except (TypeError, IndexError, KeyError):
        return {"kind": "rows", "hits": [], "rows": [],
                "raw": _jsonable(resp), "warnings": []}

    if isinstance(first, dict):          # pymilvus query -> rows
        rows = _jsonable(list(resp))
        return {"kind": "rows", "hits": [], "rows": rows,
                "raw": rows, "warnings": []}

    return _milvus_hits(first, pk)       # pymilvus search -> hits
