"""Execute an editable snippet N times and time each run.

A snippet is Python whose final expression is the raw SDK response. We split
that expression off with `ast` and evaluate it separately -- the same mechanism
IPython uses to make a cell's last line its output. That keeps the snippets
identical to what you'd type in the notebook this replaces.

Timing covers exactly what the snippet does: no compilation (done once, up
front), no normalization, no display hydration.
"""
import ast
import time
import traceback


def split_snippet(src):
    """(body_code, expr_code) -- raises before any engine is touched.

    Raises SyntaxError for unparseable source, ValueError if the snippet is
    empty or does not end in an expression (there'd be no response to
    normalize).
    """
    tree = ast.parse(src)                       # SyntaxError propagates
    if not tree.body:
        raise ValueError("snippet is empty")
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        raise ValueError(
            "snippet must end in an expression -- the raw SDK response. "
            "It currently ends in a statement, so there is no result to read."
        )
    body = ast.Module(body=tree.body[:-1], type_ignores=[])
    expr = ast.Expression(body=last.value)
    return (compile(body, "<snippet>", "exec"),
            compile(expr, "<snippet>", "eval"))


def run_n(src, ns, runs):
    """Warm once (discarded), then `runs` timed executions.

    Returns samples in ms, the last response, and any error. A mid-run failure
    keeps the samples already collected and marks the result partial.
    """
    body_code, expr_code = split_snippet(src)   # before touching an engine
    local = dict(ns)

    # Warm-up, discarded: the first call pays connection setup and cold cache.
    try:
        exec(body_code, local)
        eval(expr_code, local)
    except Exception:
        return {"samples": [], "last": None,
                "error": traceback.format_exc(limit=3),
                "partial": False, "runs_completed": 0}

    samples, last, error = [], None, None
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            exec(body_code, local)
            last = eval(expr_code, local)
        except Exception:
            error = traceback.format_exc(limit=3)
            break
        samples.append((time.perf_counter() - t0) * 1000)

    return {"samples": samples, "last": last, "error": error,
            "partial": bool(error) and bool(samples),
            "runs_completed": len(samples)}
