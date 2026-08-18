"""Static assets must not be served stale.

This exists because it actually bit us: the browser ran a cached app.js for a
newly added panel and rendered nothing, while the server was serving the
correct file. Starlette's StaticFiles sends ETag and Last-Modified but no
Cache-Control, so a browser applies heuristic freshness (RFC 9111 §4.2.2) and
may reuse a copy without asking. On this project that is not cosmetic: the demo
is built around editing code live and re-running, so a silently stale bundle
shows the audience something other than what the code says.

Live HTTP check against the running app, skipped when it isn't up -- the same
convention test_environment.py uses for the engines.

Run: docker compose exec -T demo python -m webapp.test_static_cache
"""
import os
import urllib.error
import urllib.request

BASE = os.environ.get("DEMO_URL", "http://localhost:8080")
ASSETS = ["/static/app.js", "/static/app.css", "/"]


def _head(path):
    req = urllib.request.Request(BASE + path, method="GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, {k.lower(): v for k, v in r.headers.items()}


def _reachable():
    try:
        _head("/")
        return True
    except (urllib.error.URLError, OSError):
        return False


def test_assets_forbid_reuse_without_revalidation():
    """Every asset the page loads must carry a Cache-Control that stops a
    browser reusing it blind. Without this the panel code can be stale."""
    if not _reachable():
        print("   (skipped, demo app not reachable at", BASE, ")")
        return
    for path in ASSETS:
        status, headers = _head(path)
        assert status == 200, (path, status)
        cc = headers.get("cache-control")
        assert cc is not None, f"{path} sends no Cache-Control -- browser will guess"
        assert "no-cache" in cc or "no-store" in cc or "max-age=0" in cc, \
            f"{path} Cache-Control={cc!r} still permits reuse without revalidation"


def test_assets_still_send_a_validator():
    """no-cache means 'revalidate', not 'resend'. Keeping ETag/Last-Modified is
    what makes that revalidation a cheap 304 rather than a full transfer."""
    if not _reachable():
        print("   (skipped, demo app not reachable at", BASE, ")")
        return
    for path in ASSETS:
        _, headers = _head(path)
        assert "etag" in headers or "last-modified" in headers, \
            f"{path} has no validator, so revalidation costs a full re-download"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"OK {len(tests)} static-cache tests passed")
