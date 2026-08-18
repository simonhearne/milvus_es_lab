"""
Load data/multilingual.jsonl into a small, isolated bilingual (EN+ZH) demo store
on BOTH engines, for the "Multilingual analyzer" tab.

Separate from load_milvus.py / load_es.py on purpose: those drop & recreate the
98k amazon_reviews collection; this builds a tiny parallel `multilingual_demo`
collection/index in seconds and never touches the main corpus.

Parity:
  - same PK name (`parent_asin`) and identical ids on both engines, so the demo
    app's overlap scorer compares like-for-like.
  - Milvus: one `text_ml` analyzer field using the `language_identifier`
    tokenizer (auto-detects per doc -> jieba for Chinese, english for English),
    plus a BM25 function to a sparse field.
  - ES: one `text_ml` field pinned to a custom ICU analyzer (icu_tokenizer +
    icu_folding + lowercase). ICU segments CJK; there is no per-doc routing.

    docker compose exec lab python data/load_multilingual.py
"""
import os
import json

from pymilvus import MilvusClient, DataType, Function, FunctionType
from elasticsearch import Elasticsearch, helpers

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from es_util import forcemerge_and_wait
from milvus_util import compact_and_wait

DATA = os.path.join(HERE, "multilingual.jsonl")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
ES_URI = os.environ.get("ES_URI", "http://localhost:9200")
NAME = "multilingual_demo"
PK = "parent_asin"

# Milvus: automatic per-document language identification. whatlang emits
# "Mandarin"/"English"; those keys must match exactly (VERIFIED on v3.0-beta).
ML_ANALYZER_PARAMS = {
    "tokenizer": {
        "type": "language_identifier",
        "identifier": "whatlang",
        "analyzers": {
            "default":  {"tokenizer": "standard"},
            "English":  {"type": "english"},
            "Mandarin": {"tokenizer": "jieba"},
        },
    },
}


def read_rows():
    with open(DATA) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_milvus(rows):
    client = MilvusClient(uri=MILVUS_URI)
    if client.has_collection(NAME):
        client.drop_collection(NAME)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PK, DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text_ml", DataType.VARCHAR, max_length=4096,
                     enable_analyzer=True, analyzer_params=ML_ANALYZER_PARAMS)
    schema.add_field("lang", DataType.VARCHAR, max_length=16)
    schema.add_field("text_ml_sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="text_ml_bm25", function_type=FunctionType.BM25,
        input_field_names=["text_ml"], output_field_names=["text_ml_sparse"]))

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="text_ml_sparse",
                           index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

    client.create_collection(NAME, schema=schema, index_params=index_params)
    client.insert(NAME, [{PK: r[PK], "text_ml": r["title"], "lang": r["lang"]}
                         for r in rows])
    # Seal the single growing segment, then compact so this demo collection
    # ends in the same deterministic state as the main corpus.
    client.flush(NAME)
    compact_and_wait(client, NAME)
    client.load_collection(NAME)
    print(f"milvus: loaded {len(rows)} rows into {NAME} at {MILVUS_URI}")


def load_es(rows):
    client = Elasticsearch(ES_URI)
    settings = {"analysis": {"analyzer": {
        "amazon_ml": {"type": "custom", "tokenizer": "icu_tokenizer",
                      "filter": ["icu_folding", "lowercase"]},
    }}}
    mappings = {"properties": {
        "text_ml": {"type": "text", "analyzer": "amazon_ml",
                    "search_analyzer": "amazon_ml"},
        "lang": {"type": "keyword"},
    }}
    if client.indices.exists(index=NAME):
        client.indices.delete(index=NAME)
    client.indices.create(index=NAME, settings=settings, mappings=mappings)

    actions = [{"_index": NAME, "_id": str(r[PK]),
                "_source": {"text_ml": r["title"], "lang": r["lang"]}}
               for r in rows]
    ok, errors = helpers.bulk(client, actions, request_timeout=60)
    client.indices.refresh(index=NAME)
    # Same deterministic-segment treatment as the Milvus side above.
    forcemerge_and_wait(client, NAME)
    client.indices.refresh(index=NAME)
    print(f"es: indexed {ok} docs into {NAME} at {ES_URI} (errors={len(errors)})")


if __name__ == "__main__":
    rows = read_rows()
    load_milvus(rows)
    load_es(rows)
    print(f"OK multilingual_demo loaded on both engines ({len(rows)} rows)")
