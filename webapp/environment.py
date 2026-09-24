"""Live environment report for the Introduction tab.

Answers one question for a first-time user: is the running setup genuinely
like-for-like between Milvus and Elasticsearch? Returns documented config
(from docker-compose.yml / .env), live-confirmed values pulled from the
containers, and derived parity checks.

stdlib-only imports on purpose: the engine handles arrive on the passed-in
`session`, so this module (and its tests) import without pymilvus / elasticsearch.
Never raises -- a dead engine costs you its numbers, not the page.
"""
import os
import urllib.request

# ---- documented expectations (source of truth: docker-compose.yml / .env) ---
ALLOC = {"memory": "8G", "cpus": "4"}          # deploy.resources.limits, both engines
ES_HEAP_BYTES = 4 * 1024 ** 3                  # ES_JAVA_OPTS -Xms4g -Xmx4g
ES_EXPECT_NODES = 1                            # discovery.type=single-node
ES_EXPECT_LICENSE = "basic"                    # bbq_hnsw needs no trial licence
MILVUS_DEFAULT_TAG = "v3.0.2"                  # .env MILVUS_TAG default
ES_DEFAULT_TAG = "9.5.3"                        # .env ES_TAG default
COLLECTION = "amazon_reviews"
DENSE_FIELD = "text_vec"
EXPECT_DIM = 1024
EXPECT_METRIC = "cosine"


def _gib(n):
    return f"{n / 1024 ** 3:.1f} GiB" if isinstance(n, (int, float)) else "?"


def milvus_mem_bytes(uri, timeout=2.0):
    """RSS of the Milvus process from its Prometheus endpoint (port 9091).

    Same host as the SDK URI. Inside the compose network the port is always
    9091; MILVUS_METRICS_PORT covers host-side runs where .env remaps it.
    Returns int bytes or None; never raises.
    """
    try:
        host = uri.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0]
        port = os.environ.get("MILVUS_METRICS_PORT", "9091")
        url = f"http://{host}:{port}/metrics"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith("#"):
                    continue
                if line.startswith("process_resident_memory_bytes "):
                    return int(float(line.split()[1]))
    except Exception:
        return None
    return None


def _milvus(session):
    out = {"ok": False, "error": None,
           "configured_tag": os.environ.get("MILVUS_TAG", MILVUS_DEFAULT_TAG),
           "version": None, "mode": "standalone", "docs": None,
           "dense_dim": None, "metric": None, "mem_usage_bytes": None,
           "deps": ["etcd", "minio"]}
    client = getattr(session, "client", None)
    if client is None:
        out["error"] = "Milvus client not constructed"
        return out
    try:
        client.list_collections()          # liveness probe
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    try:
        out["version"] = str(client.get_server_version())
    except Exception:
        pass

    try:
        stats = client.get_collection_stats(COLLECTION)
        out["docs"] = int(stats["row_count"])
    except Exception:
        try:                                # fallback: count(*) query
            rows = client.query(COLLECTION, filter="", output_fields=["count(*)"])
            out["docs"] = int(rows[0]["count(*)"])
        except Exception:
            pass

    try:
        for f in client.describe_collection(COLLECTION).get("fields", []):
            if f.get("name") != DENSE_FIELD:
                continue
            params = f.get("params", {})
            if isinstance(params, dict):
                dim = params.get("dim")
            elif isinstance(params, list):  # some builds return [{"key","value"}]
                dim = next((p.get("value") for p in params if p.get("key") == "dim"), None)
            else:
                dim = None
            out["dense_dim"] = int(dim) if dim is not None else None
    except Exception:
        pass
    # Milvus metric is documented from MilvusEngine.SEARCH (metric_type COSINE),
    # not read live here; ES is the live-confirmed side of the "Cosine both" check.
    out["metric"] = EXPECT_METRIC.upper()

    out["mem_usage_bytes"] = milvus_mem_bytes(
        os.environ.get("MILVUS_URI", "http://localhost:19530"))
    return out


def _es(session):
    out = {"ok": False, "error": None,
           "configured_tag": os.environ.get("ES_TAG", ES_DEFAULT_TAG),
           "version": None, "nodes": None, "license": None,
           "heap_max_bytes": None, "os_mem_used_bytes": None,
           "docs": None, "dense_dim": None, "metric": None}
    es = getattr(session, "es", None)
    if es is None:
        out["error"] = "Elasticsearch client not constructed"
        return out
    try:
        if not es.ping():
            out["error"] = "ping returned False"
            return out
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    try:
        out["version"] = es.info()["version"]["number"]
    except Exception:
        pass
    try:
        out["nodes"] = es.cluster.health()["number_of_nodes"]
    except Exception:
        pass
    try:
        out["license"] = es.license.get()["license"]["type"]
    except Exception:
        pass
    try:
        node = next(iter(es.nodes.stats(metric=["jvm", "os"])["nodes"].values()))
        out["heap_max_bytes"] = node["jvm"]["mem"]["heap_max_in_bytes"]
        out["os_mem_used_bytes"] = node["os"]["mem"]["used_in_bytes"]
    except Exception:
        pass
    try:
        out["docs"] = es.count(index=COLLECTION)["count"]
    except Exception:
        pass
    try:
        props = es.indices.get_mapping(index=COLLECTION)[COLLECTION]["mappings"]["properties"]
        out["dense_dim"] = props[DENSE_FIELD].get("dims")
        out["metric"] = props[DENSE_FIELD].get("similarity")
    except Exception:
        pass
    return out


def _checks(m, e):
    out = []

    def add(label, state, detail=None):
        c = {"label": label, "state": state}
        if detail is not None:
            c["detail"] = detail
        out.append(c)

    add("Equal container allocation", "good",
        f"{ALLOC['memory']} / {ALLOC['cpus']} CPU each")

    if not e["ok"]:
        for label in ("ES heap = 4 GiB", "ES single-node", "ES licence: basic"):
            add(label, "na", "Elasticsearch unreachable")
    else:
        h = e["heap_max_bytes"]
        if h is None:
            add("ES heap = 4 GiB", "warn", "could not read heap")
        elif abs(h - ES_HEAP_BYTES) <= ES_HEAP_BYTES * 0.05:
            add("ES heap = 4 GiB", "good", _gib(h))
        else:
            add("ES heap = 4 GiB", "warn",
                f"live {_gib(h)}, expected {_gib(ES_HEAP_BYTES)}")

        n = e["nodes"]
        add("ES single-node", "good" if n == ES_EXPECT_NODES else "warn",
            None if n == ES_EXPECT_NODES else f"{n} nodes")

        lic = (e["license"] or "").lower()
        add("ES licence: basic", "good" if lic == ES_EXPECT_LICENSE else "warn",
            e["license"] or "unknown")

    if not m["ok"]:
        add("Milvus version", "na", "Milvus unreachable")
    else:
        add("Milvus version", "good" if m["version"] else "warn", m["version"])

    both_up = m["ok"] and e["ok"]

    if not both_up:
        add("Doc counts match", "na", "both engines must be up")
    elif m["docs"] is None or e["docs"] is None:
        add("Doc counts match", "warn", "could not read a count")
    elif int(m["docs"]) == int(e["docs"]):
        add("Doc counts match", "good", f"{int(m['docs']):,} both sides")
    else:
        add("Doc counts match", "warn",
            f"Milvus {int(m['docs']):,} vs ES {int(e['docs']):,}")

    if not both_up:
        add("Dense dim 1024 both", "na", "both engines must be up")
    elif m["dense_dim"] == e["dense_dim"] == EXPECT_DIM:
        add("Dense dim 1024 both", "good", f"{EXPECT_DIM}-d")
    else:
        add("Dense dim 1024 both", "warn",
            f"Milvus {m['dense_dim']} vs ES {e['dense_dim']}")

    if not both_up:
        add("Cosine both", "na", "both engines must be up")
    elif (m["metric"] or "").lower() == (e["metric"] or "").lower() == EXPECT_METRIC:
        add("Cosine both", "good")
    else:
        add("Cosine both", "warn", f"Milvus {m['metric']} vs ES {e['metric']}")

    return out


def describe(session):
    m = _milvus(session)
    e = _es(session)
    return {
        "milvus": m,
        "es": e,
        "allocation": {
            "milvus": {"memory": ALLOC["memory"], "cpus": ALLOC["cpus"]},
            "es": {"memory": ALLOC["memory"], "cpus": ALLOC["cpus"], "heap": "4G"},
            "equal": True,
        },
        "dataset": {
            "collection": COLLECTION,
            "dense_dim": EXPECT_DIM,
            "metric": EXPECT_METRIC,
            "analyzer": "standard + lowercase + asciifolding + english stemmer + english stop",
            "milvus_index": "IVF_RABITQ + SQ8 refine",
            "es_index": "bbq_hnsw m=16 ef_construction=100, oversample 3.0 "
                        "(dense_vector default, ES 9.5)",
        },
        "checks": _checks(m, e),
    }
