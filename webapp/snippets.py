"""
The teaching content: one entry per feature with the side-by-side SDK calls.

These are EXECUTABLE, not illustrative. webapp/ runs them verbatim against the
live engines and renders each response onto a card. Rules:

- Each snippet must END IN AN EXPRESSION -- the raw SDK response. The runner
  splits that final expression off and evaluates it; a trailing statement is
  rejected at compile time because there is nothing to normalize.
- Names available: client, qv, img_qv, SEARCH, AnnSearchRequest, RRFRanker,
  Function, FunctionType, FunctionScore, LexicalHighlighter, datetime, timezone,
  timedelta. See webapp/session.py. `client` is each pane's OWN SDK client --
  MilvusClient on the left, Elasticsearch on the right -- so the two calls
  line up name-for-name when read side by side.
- ES searches pass _source=False (or a narrow list). Payload the benchmark
  doesn't need must not land in the measured latency.

House style, so the two panes read as one diff rather than two dialects:

- At most 17 lines and ~62 columns per snippet: the editors sit side by side on
  a projector, and anything taller or wider is read by nobody.
- Call arguments indent 2 from the call, one per line, trailing comma, closing
  paren at column 0.
- Nested dicts/lists either fit on one line or open a block indented 2 with the
  closing brace on its own line. Do not stack closers (`}},`) mid-argument --
  the one exception is a final closer that ends the argument itself.
- Comments state what the engine is doing that the call alone doesn't show.
  They are talking points; keep them true and keep them short.

metric: "recall"  -> pure ANN, report recall@10 vs exact ground truth + overlap
        "overlap" -> report cross-engine top-10 agreement only
        "rows"    -> returns rows (ordered / aggregation), latency only
"""
import textwrap


def snippet(src):
    """A triple-quoted snippet as the editor should show it.

    Strips the source indentation and the framing newlines, so the string in
    this file is character-for-character what runs and what the pane renders.
    """
    return textwrap.dedent(src).strip("\n")


def pad_pair(a, b):
    """Trailing blank lines on the shorter snippet until both are as tall.

    The editors size themselves to their content, so unequal pairs leave one
    pane visibly shorter than its neighbour. Blank lines at the end are safe:
    ast.parse ignores them, so the final expression the runner splits off is
    unchanged.
    """
    diff = a.count("\n") - b.count("\n")
    if diff > 0:
        return a, b + "\n" * diff
    return a + "\n" * -diff, b


FEATURES = [
    ("dense", {
        "title": "Dense ANN search",
        "subtitle": "RaBitQ+SQ8 vs bbq_hnsw, cosine, top-10",
        "metric": "recall",
        "milvus": snippet('''
            client.search(
              "amazon_reviews",
              data=[qv],
              anns_field="text_vec",
              search_params={
                "params": {
                  "nprobe": 4,
                  "rbq_bits_query": 4,
                  "refine_k": 2.0,
                },
              },
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              knn={
                "field": "text_vec",
                "query_vector": qv,
                "k": 100,
                "num_candidates": 100,
              },
              size=10,
            )
        '''),
    }),
    ("filtered", {
        "title": "Filtered ANN search",
        "subtitle": "Selective scalar filter applied to the vector search",
        "metric": "overlap",
        "milvus": snippet('''
            client.search(
              "amazon_reviews",
              data=[qv],
              anns_field="text_vec",
              filter='main_category == "All Electronics" '
                     'and average_rating >= 4',
              search_params=SEARCH,
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              knn={
                "field": "text_vec",
                "query_vector": qv,
                "k": 10,
                "num_candidates": 100,
                "filter": [
                  {"term": {"main_category.keyword": "All Electronics"}},
                  {"range": {"average_rating": {"gte": 4}}},
                ],
              },
              size=10,
            )
        '''),
    }),
    ("bm25", {
        "title": "BM25 lexical search",
        "subtitle": "Same analyzer chain both sides (standard + english stem/stop)",
        "metric": "overlap",
        "milvus": snippet('''
            client.search(
              "amazon_reviews",
              data=["wireless earbuds"],
              anns_field="text_sparse",
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              query={"match": {"text_snippet": "wireless earbuds"}},
              size=10,
            )
        '''),
    }),
    ("synonyms", {
        "title": "Search with synonyms",
        "subtitle": "Query 'tablet' -> iPad results; Milvus 3.0 synonym filter vs ES synonym analyzer",
        "metric": "overlap",
        "milvus": snippet('''
            # synonym filter in the text_syn analyzer:
            # "tablet" -> {tablet, ipad}, so iPad products match
            client.search(
              "amazon_reviews",
              data=["tablet"],
              anns_field="text_syn_sparse",
              limit=10,
            )
        '''),
        "es": snippet('''
            # text_syn is analyzed with amazon_en_syn,
            # which applies the same synonym rules
            client.search(
              index="amazon_reviews",
              query={"match": {"text_syn": "tablet"}},
              size=10,
            )
        '''),
    }),
    ("multilingual", {
        "title": "Multilingual analyzer",
        "subtitle": "Chinese query → jieba tokens; Milvus language_identifier auto-detects vs ES ICU analyzer",
        "metric": "overlap",
        "milvus": snippet('''
            # language_identifier auto-detects Mandarin → jieba,
            # English → english analyzer, all in ONE field.
            client.search(
              "multilingual_demo",
              data=["無線耳機"],   # "wireless earbuds"
              anns_field="text_ml_sparse",
              limit=10,
            )
        '''),
        "es": snippet('''
            # text_ml is pinned to the ICU analyzer; icu_tokenizer
            # segments the Chinese query. One analyzer for the whole
            # field — no per-doc routing, no English stemming here.
            client.search(
              index="multilingual_demo",
              query={"match": {"text_ml": "無線耳機"}},
              size=10,
            )
        '''),
    }),
    ("hybrid", {
        "title": "Hybrid search + RRF",
        "subtitle": "Dense + lexical, reciprocal-rank fusion",
        "metric": "overlap",
        "milvus": snippet('''
            dense = AnnSearchRequest(
              [qv], "text_vec", SEARCH, limit=50)

            sparse = AnnSearchRequest(
              ["wireless earbuds"], "text_sparse", {}, limit=50)

            client.hybrid_search(
              "amazon_reviews",
              [dense, sparse],
              ranker=RRFRanker(),
              output_fields=["parent_asin"],
              limit=10,
            )
        '''),
        "es": snippet('''
            sparse = {"standard": {"query": {
              "match": {"text_snippet": "wireless earbuds"}}}}

            dense = {"knn": {
              "field": "text_vec", "query_vector": qv,
              "k": 50, "num_candidates": 100}}

            client.search(
              index="amazon_reviews",
              retriever={"rrf": {
                "retrievers": [sparse, dense],
                "rank_window_size": 50,
              }},
              size=10,
            )
        '''),
        # The step-2 rewrite: the rrf (and linear) retrievers are
        # Platinum+, so on the basic licence the demo swaps in this
        # boost-weighted variant via the "Rewrite for basic licence"
        # button.
        "es_alt": snippet('''
            # The rrf retriever is Platinum+. The basic-licence
            # path is one request with both halves: ES adds the
            # weighted scores. BM25 is unbounded, cosine is 0-1,
            # so boost weights are per-index hand-tuning --
            # exactly what RRF's rank-based fusion avoids.
            client.search(
              index="amazon_reviews",
              query={"match": {"text_snippet": {
                "query": "wireless earbuds", "boost": 0.3}}},
              knn={
                "field": "text_vec", "query_vector": qv,
                "k": 50, "num_candidates": 100, "boost": 0.7,
              },
              size=10,
            )
        '''),
    }),
    ("grouped", {
        "title": "Grouping / collapse",
        "subtitle": "One representative hit per main_category",
        "metric": "overlap",
        "milvus": snippet('''
            # group_by digs as deep as needed to fill
            # `limit` distinct groups -- no k to tune.
            client.search(
              "amazon_reviews",
              data=[qv],
              anns_field="text_vec",
              group_by_field="main_category",
              group_size=1,
              search_params={
                "params": {
                  "nprobe": 1,
                  "rbq_bits_query": 4,
                },
              },
              limit=10,
            )
        '''),
        "es": snippet('''
            # collapse only groups the k hits knn returned,
            # so k caps how many categories you can get.
            # Raise k for more groups -- and more latency.
            client.search(
              index="amazon_reviews",
              knn={
                "field": "text_vec",
                "query_vector": qv,
                "k": 800,
                "num_candidates": 1600,
              },
              collapse={"field": "main_category.keyword"},
              size=10,
            )
        '''),
    }),
    ("image", {
        "title": "Image similarity search",
        "subtitle": "Second dense vector in the same collection",
        "metric": "recall",
        "milvus": snippet('''
            # same call, different vector field
            client.search(
              "amazon_reviews",
              data=[img_qv],
              anns_field="image_vec",
              search_params=SEARCH,
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              knn={
                "field": "image_vec",
                "query_vector": img_qv,
                "k": 100,
                "num_candidates": 100,
              },
              size=10,
            )
        '''),
    }),
    ("ordered", {
        "title": "Kernel-side ordering",
        "subtitle": "Retrieve by similarity, present ordered by scalar fields",
        "metric": "rows",
        "milvus": snippet('''
            client.search(
              "amazon_reviews",
              data=[qv],
              anns_field="text_vec",
              search_params=SEARCH,
              output_fields=["price", "average_rating"],
              order_by_fields=[
                {"field": "price", "order": "asc"},
                {"field": "average_rating", "order": "desc"},
              ],
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              knn={
                "field": "text_vec",
                "query_vector": qv,
                "k": 10,
                "num_candidates": 100,
              },
              sort=[
                {"price": "asc"},
                {"average_rating": "desc"},
              ],
              _source=["price", "average_rating"],
              size=10,
            )
        '''),
    }),
    ("aggregate", {
        "title": "Server-side aggregation",
        "subtitle": "count / avg / min / max pushed into the kernel",
        "metric": "rows",
        "milvus": snippet('''
            client.query(
              "amazon_reviews",
              group_by_fields=["main_category"],
              output_fields=[
                "main_category", "count(*)", "avg(price)",
                "min(price)", "max(price)", "avg(average_rating)",
              ],
              order_by_fields=[
                {"field": "main_category", "order": "asc"},
              ],
              limit=20,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews", size=0,
              aggs={"by_group": {
                "terms": {
                  "field": "main_category.keyword",
                  "size": 20,
                  "order": {"_key": "asc"},
                },
                "aggs": {
                  "avg_price": {"avg": {"field": "price"}},
                  "min_price": {"min": {"field": "price"}},
                  "max_price": {"max": {"field": "price"}},
                  "avg_rating": {"avg": {"field": "average_rating"}},
                },
              }},
            )
        '''),
    }),
    ("rescore", {
        "title": "Relevance shaping",
        "subtitle": "Gaussian decay favours cheaper products — ES function_score, same vocabulary",
        "metric": "overlap",
        "milvus": snippet('''
            # decay / gauss / origin / scale / offset are ES's own words.
            cheap = Function(
              name="cheap",
              function_type=FunctionType.RERANK,
              input_field_names=["price"],
              params={"reranker": "decay", "function": "gauss",
                      "origin": 0, "scale": 30,
                      "offset": 10, "decay": 0.5})

            client.search(
              "amazon_reviews",
              data=[qv],
              anns_field="text_vec",
              search_params=SEARCH,
              ranker=FunctionScore(functions=[cheap]),
              output_fields=["parent_asin", "price"],
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              query={"function_score": {
                "query": {"knn": {
                  "field": "text_vec", "query_vector": qv,
                  "k": 10, "num_candidates": 100}},
                "functions": [{"gauss": {"price": {
                  "origin": 0, "scale": 30,
                  "offset": 10, "decay": 0.5}}}],
                "boost_mode": "multiply",
              }},
              size=10,
            )
        '''),
    }),
    ("highlight", {
        "title": "Highlighting",
        "subtitle": "Marked-up fragments from the kernel — same parameters, same response shape",
        "metric": "overlap",
        "milvus": snippet('''
            # highlight_search_text=True reuses the query above,
            # so the tags land on the terms that actually matched.
            client.search(
              "amazon_reviews",
              data=["wireless earbuds"],
              anns_field="text_sparse",
              search_params={"metric_type": "BM25"},
              highlighter=LexicalHighlighter(
                highlight_search_text=True,
                pre_tags=["<em>"], post_tags=["</em>"],
                fragment_size=80, num_of_fragments=1),
              output_fields=["parent_asin", "text_snippet"],
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              query={"match": {"text_snippet": "wireless earbuds"}},
              highlight={"fields": {"text_snippet": {
                "pre_tags": ["<em>"], "post_tags": ["</em>"],
                "fragment_size": 80,
                "number_of_fragments": 1}}},
              size=10,
            )
        '''),
    }),
    ("phrase", {
        "title": "Phrase matching",
        "subtitle": "Same 683 of 686 matching docs; different top-10 — BM25 vs phrase-proximity scoring",
        "metric": "overlap",
        "milvus": snippet('''
            client.search(
              "amazon_reviews",
              data=["wireless earbuds"],
              anns_field="text_sparse",
              filter='PHRASE_MATCH(text_snippet, "wireless earbuds", 2)',
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              query={"match_phrase": {"text_snippet": {
                "query": "wireless earbuds",
                "slop": 2}}},
              size=10,
            )
        '''),
    }),
    ("geo", {
        "title": "Geo search",
        "subtitle": "Vector search restricted to sellers within 400 km of London (synthetic coordinates)",
        "metric": "overlap",
        "milvus": snippet('''
            client.search(
              "amazon_reviews",
              data=[qv],
              anns_field="text_vec",
              filter=
                "st_dwithin(store_location, 'POINT (-0.1276 51.5072)', 400000)",
              search_params=SEARCH,
              limit=10,
            )
        '''),
        "es": snippet('''
            # geo_distance takes a unit-bearing string,
            # and lat/lon in the opposite order to WKT.
            client.search(
              index="amazon_reviews",
              knn={
                "field": "text_vec",
                "query_vector": qv,
                "k": 10,
                "num_candidates": 100,
                "filter": [{"geo_distance": {
                  "distance": "400km",
                  "store_location": {"lat": 51.5072, "lon": -0.1276},
                }}],
              },
              size=10,
            )
        '''),
    }),
    ("dates", {
        "title": "Date filtering",
        "subtitle": "Last 30 days — ISO literals and INTERVAL, but no server-side now (synthetic dates)",
        "metric": "overlap",
        "milvus": snippet('''
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(days=30)).isoformat()

            client.search(
              "amazon_reviews",
              data=[qv],
              anns_field="text_vec",
              filter=f"first_seen > ISO '{cutoff}'",
              search_params=SEARCH,
              output_fields=["parent_asin", "first_seen"],
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              knn={
                "field": "text_vec",
                "query_vector": qv,
                "k": 10,
                "num_candidates": 100,
                "filter": [
                  {"range": {"first_seen": {"gte": "now-30d"}}},
                ],
              },
              size=10,
            )
        '''),
    }),
    ("typo", {
        "title": "Typo tolerance",
        "subtitle": "Synonyms handle the known typo; live-edit an unseen one and only ES survives",
        "metric": "overlap",
        "milvus": snippet('''
            client.search(
              "amazon_reviews",
              data=["laptp charjer"],
              anns_field="text_syn_sparse",
              search_params={"metric_type": "BM25"},
              limit=10,
            )
        '''),
        "es": snippet('''
            client.search(
              index="amazon_reviews",
              query={"match": {"text_snippet": {
                "query": "laptp charjer",
                "fuzziness": "AUTO"}}},
              size=10,
            )
        '''),
    }),
]

for _key, _feature in FEATURES:
    _feature["milvus"], _feature["es"] = pad_pair(
        _feature["milvus"], _feature["es"])
