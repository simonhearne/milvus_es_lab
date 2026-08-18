"""Correctness math and the recall guardrail.

Imports stdlib only, on purpose: this module is the pure scoring layer and must
not inherit the app layer's dependencies, so keep every import here stdlib.
"""
import ast

QUERY_NAMES = {"qv": "text", "img_qv": "image"}


def recall_at_k(got_ids, truth_ids, k=10):
    """Fraction of the exact top-k that the engine actually returned."""
    got = [i for i in got_ids[:k] if i is not None]
    if not truth_ids:
        return 0.0
    return len(set(got) & set(truth_ids[:k])) / min(k, len(truth_ids))


def overlap_at_k(a_ids, b_ids, k=10):
    """(count, ids) of cross-engine agreement in the top k."""
    a = {i for i in a_ids[:k] if i is not None}
    b = {i for i in b_ids[:k] if i is not None}
    both = a & b
    return len(both), sorted(both)


def references(src, names):
    """Which of `names` the snippet actually mentions as a bare name."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return used & set(names)


def recall_space(src):
    """'text' | 'image' | None -- which ground-truth space this snippet's query
    belongs to.

    None means the snippet no longer references a bound query vector, so the
    ground truth does not describe what it searched for and recall must NOT be
    reported. Reporting it anyway would be a confident wrong number on stage.
    """
    used = references(src, QUERY_NAMES)
    if len(used) != 1:
        return None
    return QUERY_NAMES[next(iter(used))]
