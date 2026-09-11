"""
Load data/amazon_reviews.parquet into local Elasticsearch 9.5.

Parity choices that make the side-by-side honest:
  - text_snippet uses a custom analyzer mirroring the Milvus analyzer chain
    (standard tokenizer + lowercase + asciifolding + english stemmer + english
    stopwords), so BM25 tokenises identically on both engines.
  - both 1024-d dense vectors are mapped as dense_vector (cosine).
  - categories -> keyword array (analogue of Milvus Array<VarChar>).

    python data/load_es.py
"""
import os
import json
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers

HERE = os.path.dirname(__file__)
PARQUET = os.path.join(HERE, "amazon_reviews.parquet")
META = json.load(open(os.path.join(HERE, "schema.json")))
ES_URI = os.environ.get("ES_URI", "http://localhost:9200")
INDEX = "amazon_reviews"

pk = META["primary_key"]
dense = META["dense_vectors"]
arrays = META.get("array_fields", {})
geo_fields = META.get("geo_fields", [])
ts_fields = META.get("timestamp_fields", [])
scalars = META["scalar_types"]
bm25 = META.get("bm25") or {}
BM25_SOURCE = bm25.get("source", "text_snippet")

import sys
sys.path.insert(0, HERE)
from es_util import forcemerge_and_wait
from fetch import require_parquet
from synonyms import read_synonyms
import synthesize

SYNONYMS = read_synonyms()

df = pd.read_parquet(require_parquet(PARQUET))
client = Elasticsearch(ES_URI)

SETTINGS = {"analysis": {
    "filter": {
        "english_stemmer": {"type": "stemmer", "language": "english"},
        "english_stop": {"type": "stop", "stopwords": "_english_"},
        # Index-time synonyms for the text_syn field. Plain `synonym` (not
        # synonym_graph): this analyzer runs at index time and all rules are
        # single-token, so the graph variant buys nothing and ES discourages
        # it in an index analyzer.
        "amazon_syn": {"type": "synonym", "synonyms": SYNONYMS},
    },
    "analyzer": {
        "amazon_en": {"type": "custom", "tokenizer": "standard",
                      "filter": ["lowercase", "asciifolding",
                                 "english_stemmer", "english_stop"]},
        "amazon_en_syn": {"type": "custom", "tokenizer": "standard",
                          "filter": ["lowercase", "asciifolding", "amazon_syn",
                                     "english_stemmer", "english_stop"]},
    },
}}

properties = {}
for name, t in scalars.items():
    if name == pk:
        continue
    if name == BM25_SOURCE:
        properties[name] = {"type": "text", "analyzer": "amazon_en"}
    elif t == "VARCHAR":
        properties[name] = {"type": "text",
                            "fields": {"keyword": {"type": "keyword",
                                                   "ignore_above": 1024}}}
    elif t in ("FLOAT", "DOUBLE"):
        properties[name] = {"type": "double"}
    elif t in ("INT64", "INT32"):
        properties[name] = {"type": "long"}
    elif t == "BOOL":
        properties[name] = {"type": "boolean"}

# Synonym demo field: same text as BM25_SOURCE, analyzed with synonyms at
# both index and search time (symmetric with Milvus).
properties["text_syn"] = {"type": "text", "analyzer": "amazon_en_syn",
                          "search_analyzer": "amazon_en_syn"}

for name in arrays:                      # keyword arrays
    properties[name] = {"type": "keyword"}

for name in geo_fields:                  # analogue of Milvus GEOMETRY
    properties[name] = {"type": "geo_point"}
for name in ts_fields:                   # analogue of Milvus TIMESTAMPTZ
    properties[name] = {"type": "date"}

for name, dim in dense.items():
    # 9.5 defaults new dense_vector indices to bbq_disk; the notebook recreates
    # indices per-segment to demo int8_hnsw / bbq_disk / flat trade-offs.
    properties[name] = {"type": "dense_vector", "dims": int(dim),
                        "index": True, "similarity": "cosine"}

if client.indices.exists(index=INDEX):
    client.indices.delete(index=INDEX)
client.indices.create(index=INDEX, settings=SETTINGS, mappings={"properties": properties})


def actions():
    for r in df.to_dict(orient="records"):
        doc = {}
        for k, v in r.items():
            if k in dense:
                doc[k] = np.asarray(v, dtype=np.float32).tolist()
            elif k in arrays:
                doc[k] = list(v) if v is not None else []
            else:
                doc[k] = v
        doc["text_syn"] = doc.get(BM25_SOURCE)
        for name in geo_fields:
            doc[name] = synthesize.store_location_geopoint(doc.get("store"))
        for name in ts_fields:
            doc[name] = synthesize.first_seen(doc[pk])
        yield {"_index": INDEX, "_id": str(doc[pk]), "_source": doc}


ok, errors = helpers.bulk(client, actions(), chunk_size=500, request_timeout=180)
client.indices.refresh(index=INDEX)
# Merge the segments the bulk load produced down to one, so every reload
# benchmarks the same single HNSW graph (mirror of the Milvus force
# compaction in load_milvus.py).
forcemerge_and_wait(client, INDEX)
client.indices.refresh(index=INDEX)
print(f"indexed {ok} docs into {INDEX} at {ES_URI} (errors={len(errors)})")
