"""Unit tests for webapp/adapters.py highlight normalization (no cluster).

Run: docker compose exec -T demo python -m webapp.test_adapters
"""
from webapp.adapters import normalize

PK = "parent_asin"


def _es_body(hit):
    return {"hits": {"hits": [hit]}}


def test_es_highlight_is_passed_through():
    body = _es_body({"_id": "A1", "_score": 1.5,
                     "highlight": {"text_snippet": ["<em>wireless</em> buds"]}})
    out = normalize(body, PK)
    assert out["hits"][0]["highlight"] == {
        "text_snippet": ["<em>wireless</em> buds"]}


def test_es_hit_without_highlight_gets_empty_dict():
    out = normalize(_es_body({"_id": "A1", "_score": 1.0}), PK)
    assert out["hits"][0]["highlight"] == {}


def test_milvus_fragments_are_flattened_to_the_es_shape():
    # pymilvus nests fragments under the field; ES gives a bare list. The
    # frontend must not have to know which engine it is rendering.
    hit = {"id": "A1", "distance": 0.9,
           "entity": {PK: "A1"},
           "highlight": {"text_snippet": {
               "fragments": ["<em>wireless</em> buds"], "scores": []}}}
    out = normalize([[hit]], PK)
    assert out["hits"][0]["highlight"] == {
        "text_snippet": ["<em>wireless</em> buds"]}


def test_milvus_hit_without_highlight_gets_empty_dict():
    hit = {"id": "A1", "distance": 0.9, "entity": {PK: "A1"}}
    out = normalize([[hit]], PK)
    assert out["hits"][0]["highlight"] == {}


def test_milvus_highlight_already_a_list_is_tolerated():
    hit = {"id": "A1", "distance": 0.9, "entity": {PK: "A1"},
           "highlight": {"text_snippet": ["<em>buds</em>"]}}
    out = normalize([[hit]], PK)
    assert out["hits"][0]["highlight"] == {"text_snippet": ["<em>buds</em>"]}


def test_highlight_does_not_disturb_id_or_score():
    hit = {"id": "A1", "distance": 0.9, "entity": {PK: "A1"},
           "highlight": {"text_snippet": {"fragments": ["x"], "scores": []}}}
    out = normalize([[hit]], PK)
    assert out["hits"][0]["id"] == "A1"
    assert out["hits"][0]["score"] == 0.9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"OK {len(tests)} adapter tests passed")
