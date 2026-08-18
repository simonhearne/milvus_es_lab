"""FastAPI surface for the demo app.

Routes are sync `def`, not `async def`, on purpose: FastAPI runs sync endpoints
in a threadpool, so a wedged engine blocks one worker rather than the event loop
and the page stays responsive.
"""
import os
import datetime as dt
import statistics as st

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.snippets import FEATURES
from webapp import cache
from webapp import analyze
from webapp import environment
from webapp.adapters import normalize
from webapp.enrich import DisplayIndex
from webapp.metrics import overlap_at_k, recall_at_k, recall_space
from webapp.runner import run_n, split_snippet
from webapp.session import Session

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="Milvus 3.0 vs Elasticsearch 9.4 — demo")
session = Session()
display = DisplayIndex()
FEATURE_MAP = dict(FEATURES)


class RunRequest(BaseModel):
    feature: str
    milvus: str
    es: str
    runs: int = 10
    query_index: int = 0


def _latency(samples):
    if not samples:
        return None
    return {"median_ms": round(st.median(samples), 1),
            "min_ms": round(min(samples), 1),
            "max_ms": round(max(samples), 1),
            "samples": [round(s, 2) for s in samples]}


def _run_engine(src, ns, runs, pk):
    """One engine, start to finish. Never raises -- a failure here must not take
    the other engine's card down with it."""
    try:
        split_snippet(src)
    except (SyntaxError, ValueError) as e:
        return {"ok": False, "latency": None, "kind": None, "hits": [],
                "rows": [], "raw": None, "warnings": [],
                "error": f"{type(e).__name__}: {e}", "partial": False}

    out = run_n(src, ns, runs)
    if not out["samples"]:
        return {"ok": False, "latency": None, "kind": None, "hits": [],
                "rows": [], "raw": None, "warnings": [],
                "error": out["error"], "partial": False}

    try:
        norm = normalize(out["last"], pk)
    except Exception as e:
        return {"ok": False, "latency": _latency(out["samples"]), "kind": None,
                "hits": [], "rows": [], "raw": None, "warnings": [],
                "error": f"could not read the response: {type(e).__name__}: {e}",
                "partial": out["partial"]}

    return {"ok": True, "latency": _latency(out["samples"]),
            "kind": norm["kind"], "hits": norm["hits"], "rows": norm["rows"],
            "raw": norm["raw"], "warnings": norm["warnings"],
            "error": out["error"], "partial": out["partial"]}


def _correctness(feature, req, m, e):
    """None for any value we cannot honestly compute."""
    out = {"overlap": None, "agreement_ids": [],
           "recall_milvus": None, "recall_es": None, "recall_note": None}

    m_ids = [h["id"] for h in m["hits"] if h["id"]]
    e_ids = [h["id"] for h in e["hits"] if h["id"]]
    metric = FEATURE_MAP[feature]["metric"]

    if metric == "rows":
        out["recall_note"] = "row results: latency only"
        return out

    # Overlap needs both sides. Absent that it is null, never 0 -- a zero would
    # read as total disagreement rather than a missing measurement.
    if m["ok"] and e["ok"] and m_ids and e_ids:
        count, ids = overlap_at_k(m_ids, e_ids, k=10)
        out["overlap"] = count
        out["agreement_ids"] = ids

    if metric != "recall":
        return out

    if session.gt is None:
        out["recall_note"] = "no ground truth: run bench/ground_truth.py"
        return out

    # Per engine, against its OWN snippet's query space. Deciding once from the
    # Milvus snippet would mean editing only the ES query silently scores ES
    # against ground truth that doesn't describe what it searched -- precisely
    # the failure this guardrail exists to prevent.
    spaces = {}
    for key, src, ids in (("recall_milvus", req.milvus, m_ids),
                          ("recall_es", req.es, e_ids)):
        space = recall_space(src)
        spaces[key] = space
        if space is None or not ids:
            continue
        truth = session.truth_ids(space, req.query_index)
        out[key] = round(recall_at_k(ids, truth), 3)

    live = {s for s in spaces.values() if s}
    if not live:
        out["recall_note"] = ("recall n/a: neither snippet searches a bound query "
                              "vector (qv / img_qv), so the ground truth does not "
                              "describe what they looked for")
    elif None in spaces.values():
        side = "Milvus" if spaces["recall_milvus"] is None else "ES"
        out["recall_note"] = (f"{side} recall n/a: that snippet no longer searches a "
                              f"bound query vector. The other is vs exact "
                              f"{next(iter(live))}-space top-10.")
    else:
        out["recall_note"] = " · ".join(
            f"vs exact {s}-space top-10" for s in sorted(live))
    return out


@app.get("/api/features")
def features():
    return [{"key": key, "title": f["title"], "subtitle": f["subtitle"],
             "metric": f["metric"], "milvus": f["milvus"], "es": f["es"],
             "es_alt": f.get("es_alt")}
            for key, f in FEATURES]


@app.get("/api/health")
def health():
    return session.health()


@app.get("/api/environment")
def env():
    return environment.describe(session)


@app.get("/api/queries")
def queries():
    """The ground-truth query set. Labelled by product title so the picker reads
    like something rather than an index."""
    out = []
    for i in range(session.query_count()):
        rid = session.query_row_id(i)
        row = display.get(rid) if rid else None
        title = (row or {}).get("title") or rid or f"query {i}"
        out.append({"index": i, "label": f"{i}: {str(title)[:60]}",
                    "title": str(title)})
    return out


@app.get("/api/cached/{feature}")
def cached(feature: str):
    payload = cache.load(feature)
    if payload is None:
        raise HTTPException(404, f"no cached run for {feature}")
    return payload


@app.post("/api/run")
def run(req: RunRequest):
    if req.feature not in FEATURE_MAP:
        raise HTTPException(404, f"unknown feature {req.feature}")
    runs = max(1, min(int(req.runs), 200))

    # Sequential, never concurrent: on one box the engines would fight for
    # cores and both numbers would be fiction. Each pane gets its own
    # namespace so `client` means that pane's engine.
    m = _run_engine(req.milvus, session.namespace(req.query_index, "milvus"),
                    runs, session.pk)
    e = _run_engine(req.es, session.namespace(req.query_index, "es"),
                    runs, session.pk)

    ids = [h["id"] for h in (m["hits"] + e["hits"]) if h["id"]]
    payload = {
        "feature": req.feature,
        "metric": FEATURE_MAP[req.feature]["metric"],
        "runs": runs,
        "warmup_discarded": 1,
        "sequential": True,
        "query_index": req.query_index,
        "engines": {"milvus": m, "es": e},
        "correctness": _correctness(req.feature, req, m, e),
        "display": display.many(ids),          # untimed, after the runs
        "cached": False,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    payload["analyze"] = analyze.preview(
        {"milvus": session.client, "es": session.es},
        req.feature, req.milvus, req.es)
    # Only cache COMPLETE runs (both engines succeeded). The fallback exists to
    # recover a full side-by-side comparison when an engine dies mid-talk; saving
    # a degraded single-engine run would overwrite that full run with the very
    # failure you're trying to escape. If a feature never once ran with both
    # engines up, "show last good run" honestly reports no cached run yet.
    if m["ok"] and e["ok"]:
        cache.save(req.feature, payload)
    return payload


# Starlette's StaticFiles sends ETag and Last-Modified but no Cache-Control, so
# a browser is free to apply heuristic freshness and reuse app.js without ever
# asking (RFC 9111 4.2.2). That is not cosmetic here: this lab exists to be
# edited live and re-run, and a silently stale bundle renders something other
# than what the code on screen says -- which is exactly how a new panel came to
# show no results while the server was serving the correct file.
#
# `no-cache` does not mean "don't cache"; it means "revalidate before use". The
# ETag survives, so the check is a 304 and costs nothing on localhost.
NO_CACHE = "no-cache, must-revalidate"


class RevalidatingStatic(StaticFiles):
    """StaticFiles that refuses to be served blind from a browser cache."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = NO_CACHE
        return response


app.mount("/static", RevalidatingStatic(directory=STATIC), name="static")


@app.get("/")
def index():
    # Same reasoning as the static mount: index.html names the asset URLs, so
    # caching it stale can pin the page to an old bundle.
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": NO_CACHE})
