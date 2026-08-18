"""Unit tests for the pure snippet-parsing in webapp/analyze.py (no live cluster).

Run: docker compose exec -T -w /workspace lab python -m webapp.test_analyze
"""
from webapp.analyze import extract_milvus, extract_es, PREVIEW, preview


def test_extract_milvus_single_line():
    src = 'client.search("amazon_reviews", data=["wireless earbuds"], anns_field="text_sparse")'
    assert extract_milvus(src) == "wireless earbuds"


def test_extract_milvus_multiline_cjk():
    src = ('client.search("multilingual_demo",\n'
           '  data=["無線耳機"],\n'
           '  anns_field="text_ml_sparse")')
    assert extract_milvus(src) == "無線耳機"


def test_extract_milvus_missing():
    assert extract_milvus("client.query('amazon_reviews', filter='price > 10')") is None


def test_extract_es_full():
    src = 'client.search(index="multilingual_demo", query={"match": {"text_ml": "無線耳機"}}, size=10)'
    assert extract_es(src) == ("multilingual_demo", "text_ml", "無線耳機")


def test_extract_es_edited_field():
    src = 'client.search(index="amazon_reviews", query={"match": {"text_syn": "tablet"}})'
    assert extract_es(src) == ("amazon_reviews", "text_syn", "tablet")


def test_extract_es_extended_match_with_fuzziness():
    src = ('client.search(index="amazon_reviews", query={"match": {"text_snippet": {\n'
           '    "query": "laptp charjer",\n'
           '    "fuzziness": "AUTO"}}})')
    assert extract_es(src) == ("amazon_reviews", "text_snippet", "laptp charjer")


def test_preview_map_covers_text_tabs():
    assert set(PREVIEW) == {"bm25", "synonyms", "multilingual", "typo"}


class _Stub:
    """Mirrors pymilvus's AnalyzeResult: exposes `.tokens` but is NOT iterable
    (its __repr__ can look list-like, which is what makes `list(out[0])` a trap)."""
    def __init__(self, tokens):
        self.tokens = tokens


class _StubMilvusClient:
    def run_analyzer(self, texts, collection_name, field_name):
        return [_Stub(tokens=["無線", "耳機"])]


class _StubIndices:
    def analyze(self, index, field, text):
        return {"tokens": [{"token": "無線"}, {"token": "耳機"}]}


class _StubEsClient:
    def __init__(self):
        self.indices = _StubIndices()


def test_preview_extracts_tokens_via_dot_tokens_not_iteration():
    milvus_src = 'client.search("multilingual_demo", data=["無線耳機"], anns_field="text_ml_sparse")'
    es_src = 'client.search(index="multilingual_demo", query={"match": {"text_ml": "無線耳機"}}, size=10)'
    clients = {"milvus": _StubMilvusClient(), "es": _StubEsClient()}

    result = preview(clients, "multilingual", milvus_src, es_src)

    assert result["milvus"] == ["無線", "耳機"]
    assert result["es"] == ["無線", "耳機"]
    assert result["warnings"] == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"OK {len(tests)} analyze tests passed")
