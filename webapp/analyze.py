"""Analyzer-output preview for the text-query tabs.

Given a feature's (possibly live-edited) snippets, pull out the query string and
ask each engine to tokenize it against its ALREADY-configured field — so the demo
can show jieba / ICU / synonym tokenization as chips above the results, with no
copy of the analyzer definitions living in the webapp.

`extract_milvus` / `extract_es` are pure (unit-tested without a cluster);
`preview` does the live engine calls and never raises.
"""
import ast

# feature -> (milvus collection, milvus analyzer field). Membership opts a tab
# into the panel. The analyzer field can't come from the snippet: anns_field
# names the sparse OUTPUT (text_ml_sparse), not the analyzer field.
PREVIEW = {
    "bm25":         ("amazon_reviews",    "text_snippet"),
    "synonyms":     ("amazon_reviews",    "text_syn"),
    "multilingual": ("multilingual_demo", "text_ml"),
    "typo":         ("amazon_reviews",    "text_syn"),
}

# Human labels for the two chip rows: (milvus, es).
LABELS = {
    "bm25":         ("standard + english stemmer", "amazon_en (standard + english stemmer)"),
    "synonyms":     ("text_syn (synonym filter)",  "amazon_en_syn (synonym filter)"),
    "multilingual": ("language_identifier → jieba / english", "amazon_ml (icu_tokenizer)"),
    "typo":         ("text_syn (misspelling rules)", "amazon_en (no synonyms) + fuzziness=AUTO"),
}


def _str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def extract_milvus(src):
    """First string in a `data=[...]` kwarg of any call in `src`, or None."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "data" and isinstance(kw.value, (ast.List, ast.Tuple)) and kw.value.elts:
                    return _str(kw.value.elts[0])
    return None


def extract_es(src):
    """(index, field, text) from `index=..` + `query={"match": {field: text}}`,
    or the extended match form `{"match": {field: {"query": text, ...}}}`
    (used when the match clause carries extra params like fuzziness)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (None, None, None)
    index = field = text = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "index":
                index = _str(kw.value)
            elif kw.arg == "query" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if _str(k) == "match" and isinstance(v, ast.Dict):
                        for fk, fv in zip(v.keys, v.values):
                            field = _str(fk)
                            if isinstance(fv, ast.Dict):
                                for ik, iv in zip(fv.keys, fv.values):
                                    if _str(ik) == "query":
                                        text = _str(iv)
                            else:
                                text = _str(fv)
    return (index, field, text)


def _milvus_tokens(client, collection, field, text):
    out = client.run_analyzer([text], collection_name=collection, field_name=field)
    return list(out[0].tokens) if out else []


def _es_tokens(es, index, field, text):
    r = es.indices.analyze(index=index, field=field, text=text)
    body = r.body if hasattr(r, "body") else r
    return [t["token"] for t in body.get("tokens", [])]


def preview(clients, feature, milvus_src, es_src):
    """Best-effort token preview; returns None for non-preview features and never
    raises — any failure becomes a warning so the results still render.

    `clients` maps engine key ("milvus" | "es") to its SDK client: the snippets
    both call their engine `client`, so the caller must say which is which."""
    if feature not in PREVIEW:
        return None
    collection, m_field = PREVIEW[feature]
    m_label, e_label = LABELS.get(feature, ("", ""))
    warnings = []
    milvus_tokens, es_tokens = [], []

    m_text = extract_milvus(milvus_src)
    e_index, e_field, e_text = extract_es(es_src)

    try:
        if m_text is not None:
            milvus_tokens = _milvus_tokens(clients["milvus"], collection, m_field, m_text)
        else:
            warnings.append("could not find a Milvus query to analyze")
    except Exception as ex:
        warnings.append(f"milvus analyze failed: {ex}")

    try:
        if e_text is not None and e_index and e_field:
            es_tokens = _es_tokens(clients["es"], e_index, e_field, e_text)
        else:
            warnings.append("could not find an ES query to analyze")
    except Exception as ex:
        warnings.append(f"es analyze failed: {ex}")

    return {"milvus": milvus_tokens, "es": es_tokens,
            "milvus_label": m_label, "es_label": e_label,
            "milvus_text": m_text, "es_text": e_text, "warnings": warnings}
