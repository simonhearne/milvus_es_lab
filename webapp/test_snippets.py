"""Unit tests for webapp/snippets.py pane padding (no cluster).

Run: docker compose exec -T demo python -m webapp.test_snippets
"""
import ast

from webapp.snippets import FEATURES, pad_pair


def test_shorter_left_side_is_padded():
    a, b = pad_pair("x = 1\nx", "x = 1\ny = 2\nx")
    assert a == "x = 1\nx\n"
    assert b == "x = 1\ny = 2\nx"


def test_shorter_right_side_is_padded():
    a, b = pad_pair("x = 1\ny = 2\nx", "x = 1\nx")
    assert a == "x = 1\ny = 2\nx"
    assert b == "x = 1\nx\n"


def test_equal_pair_is_untouched():
    a, b = pad_pair("x\n1", "y\n2")
    assert (a, b) == ("x\n1", "y\n2")


def test_every_feature_pair_has_equal_pane_heights():
    for key, f in FEATURES:
        m, e = f["milvus"], f["es"]
        assert m.count("\n") == e.count("\n"), key


def test_padding_keeps_every_snippet_ending_in_an_expression():
    # The runner splits the last AST node off as the response expression;
    # trailing blank lines must not change which node that is.
    for key, f in FEATURES:
        for src in (f["milvus"], f["es"]):
            tree = ast.parse(src)
            assert isinstance(tree.body[-1], ast.Expr), key


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"OK {len(tests)} snippet tests passed")
