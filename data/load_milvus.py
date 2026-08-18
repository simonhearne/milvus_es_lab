"""
Load data/amazon_reviews.parquet into LOCAL Milvus (Docker), recreating the
amazon_reviews schema: VarChar PK, two 1024-d dense vectors (text_vec, image_vec,
COSINE/HNSW), an Array<VarChar> categories field, scalars, and the BM25 pipeline
(text_snippet analyzer -> text_snippet_bm25 function -> text_sparse, BM25 index).

Everything but dims/field-list is read from schema.json.

    python data/load_milvus.py
"""
import os
import json
import numpy as np
import pandas as pd
from pymilvus import MilvusClient, DataType, Function, FunctionType

HERE = os.path.dirname(__file__)
PARQUET = os.path.join(HERE, "amazon_reviews.parquet")
META = json.load(open(os.path.join(HERE, "schema.json")))
URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION = "amazon_reviews"

pk = META["primary_key"]
dense = META["dense_vectors"]              # {"text_vec":1024,"image_vec":1024}
arrays = META.get("array_fields", {})
geo_fields = META.get("geo_fields", [])
ts_fields = META.get("timestamp_fields", [])
scalars = META["scalar_types"]             # {"title":"VARCHAR", "price":"FLOAT", ...}
bm25 = META.get("bm25") or {}

# The analyzer chain from the source collection (falls back if not surfaced).
ANALYZER_PARAMS = bm25.get("analyzer_params") or json.dumps({
    "tokenizer": "standard",
    "filter": ["lowercase", "asciifolding",
               {"type": "stemmer", "language": "english"},
               {"type": "stop", "stop_words": ["_english_"]}],
})
if isinstance(ANALYZER_PARAMS, str):
    ANALYZER_PARAMS = json.loads(ANALYZER_PARAMS)

import sys
sys.path.insert(0, HERE)
from fetch import require_parquet
from milvus_util import compact_and_wait
from synonyms import read_synonyms
import synthesize

SYNONYMS = read_synonyms()

def _with_synonyms(params, synonyms):
    """Copy a Milvus analyzer_params dict, inserting a synonym filter before
    the stemmer (or at the end if there is no stemmer)."""
    import copy
    p = copy.deepcopy(params)
    filt = list(p.get("filter", []))
    syn = {"type": "synonym", "synonyms": synonyms, "expand": True}
    idx = next((i for i, f in enumerate(filt)
                if isinstance(f, dict) and f.get("type") == "stemmer"), len(filt))
    filt.insert(idx, syn)
    p["filter"] = filt
    return p

SYN_ANALYZER_PARAMS = _with_synonyms(ANALYZER_PARAMS, SYNONYMS)

SYN_SOURCE = "text_syn"            # analyzer text field (dup of text_snippet)
SYN_OUTPUT = "text_syn_sparse"     # its BM25 sparse output
SYN_NAME = "text_syn_bm25"         # its BM25 function

BM25_SOURCE = bm25.get("source", "text_snippet")   # analyzer text field
BM25_OUTPUT = bm25.get("output", "text_sparse")     # sparse function output
BM25_NAME = bm25.get("name", "text_snippet_bm25")

MILVUS_SCALAR = {"VARCHAR": DataType.VARCHAR, "FLOAT": DataType.FLOAT,
                 "DOUBLE": DataType.DOUBLE, "INT64": DataType.INT64,
                 "INT32": DataType.INT32, "BOOL": DataType.BOOL}

df = pd.read_parquet(require_parquet(PARQUET))
print(f"{len(df)} rows | dense={dense} | arrays={list(arrays)} | bm25 source={BM25_SOURCE}")

client = MilvusClient(uri=URI)
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(pk, DataType.VARCHAR, is_primary=True, max_length=64)

for name, t in scalars.items():
    if name == pk:
        continue
    if name == BM25_SOURCE:
        schema.add_field(name, DataType.VARCHAR, max_length=4096,
                         enable_analyzer=True, enable_match=True,
                         analyzer_params=ANALYZER_PARAMS)
    elif t == "VARCHAR":
        schema.add_field(name, DataType.VARCHAR, max_length=2048)
    else:
        schema.add_field(name, MILVUS_SCALAR.get(t, DataType.VARCHAR))

# Synonym demo: a second analyzer field, same text as BM25_SOURCE but with a
# synonym filter in its chain. Kept separate so the existing BM25 pipeline is
# untouched.
schema.add_field(SYN_SOURCE, DataType.VARCHAR, max_length=4096,
                 enable_analyzer=True, analyzer_params=SYN_ANALYZER_PARAMS)

for name, spec in arrays.items():
    schema.add_field(name, DataType.ARRAY,
                     element_type=MILVUS_SCALAR.get(spec["element"], DataType.VARCHAR),
                     max_capacity=spec["max_capacity"], max_length=spec["max_length"])

# GEOMETRY / TIMESTAMPTZ: both new in Milvus 3.0, both synthesized at load
# time by data/synthesize.py (the source parquet has neither).
for name in geo_fields:
    schema.add_field(name, DataType.GEOMETRY)
for name in ts_fields:
    schema.add_field(name, DataType.TIMESTAMPTZ)

for name, dim in dense.items():
    schema.add_field(name, DataType.FLOAT_VECTOR, dim=int(dim))

# BM25: sparse output field + function over the analyzer text field.
schema.add_field(BM25_OUTPUT, DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(
    name=BM25_NAME, function_type=FunctionType.BM25,
    input_field_names=[BM25_SOURCE], output_field_names=[BM25_OUTPUT],
))

# Second BM25 pipeline over the synonym-analyzed field.
schema.add_field(SYN_OUTPUT, DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(
    name=SYN_NAME, function_type=FunctionType.BM25,
    input_field_names=[SYN_SOURCE], output_field_names=[SYN_OUTPUT],
))

client.create_collection(COLLECTION, schema=schema)

# Indexes: RaBitQ 1-bit + SQ8 refinement on each dense vector (mirrors ES
# bbq_disk, which does 1-bit + rerank under the hood), SPARSE_INVERTED/BM25
# on the sparse field.
idx = client.prepare_index_params()
for name in dense:
    idx.add_index(field_name=name, index_type="IVF_RABITQ", metric_type="COSINE",
                  params={"nlist": 1024, "refine": True, "refine_type": "SQ8"})
idx.add_index(field_name=BM25_OUTPUT, index_type="SPARSE_INVERTED_INDEX",
              metric_type="BM25")
idx.add_index(field_name=SYN_OUTPUT, index_type="SPARSE_INVERTED_INDEX",
              metric_type="BM25")
# Scalar indexes so filtered search, order_by_fields, and aggregation perform.
# INVERTED covers equality + range + is recommended for both varchar and numeric.
SCALAR_INDEX = ["main_category", "store", "price", "average_rating", "rating_number"]
for name in SCALAR_INDEX:
    if name in scalars:
        idx.add_index(field_name=name, index_type="INVERTED")
# RTREE is the only index type GEOMETRY accepts. TIMESTAMPTZ takes no index
# at all -- INVERTED is rejected outright ("not supported on Timestamptz
# field"), so first_seen is deliberately unindexed.
for name in geo_fields:
    idx.add_index(field_name=name, index_type="RTREE")
client.create_index(COLLECTION, idx)

# Insert source fields only; Milvus computes text_sparse via the function.
records = df.to_dict(orient="records")
for r in records:
    for vf in dense:
        r[vf] = np.asarray(r[vf], dtype=np.float32).tolist()
    for af in arrays:
        r[af] = list(r[af]) if r[af] is not None else []
    r[SYN_SOURCE] = r.get(BM25_SOURCE)
    for name in geo_fields:
        r[name] = synthesize.store_location_wkt(r.get("store"))
    for name in ts_fields:
        r[name] = synthesize.first_seen(r[pk])

B = 500
for i in range(0, len(records), B):
    client.insert(COLLECTION, records[i:i + B])
    print(f"  inserted {min(i+B, len(records))}/{len(records)}", end="\r")

# Without an explicit flush, load_collection can return while the last
# batch(es) are still sitting in a growing segment: unsealed, unindexed, and
# undercounted by get_collection_stats. That silently leaves part of the
# corpus out of the ANN index until Milvus's own background flush interval
# gets around to it -- flush here so the collection is fully sealed and
# indexed by the time this script exits.
client.flush(COLLECTION)
# Merge the many small sealed segments the batched insert produced, so the
# first post-load searches hit the same segment layout every run.
compact_and_wait(client, COLLECTION)
client.load_collection(COLLECTION)
print(f"\nloaded {COLLECTION} into Milvus at {URI}")
