"""Unit tests for webapp/session.py's engine connection handling (no live cluster).

MilvusClient connects in its constructor, so on a cold `docker compose up` the
demo boots before Milvus accepts connections. These pin that the session keeps
trying instead of holding a dead client until the container restarts.

Run in Docker: docker compose exec -T demo python -m pytest webapp/test_session.py
"""
import webapp.session as session_mod
from webapp.session import Session


class _FlakyMilvusClient:
    """Raises like the real constructor while `down` is set, then connects."""
    down = True
    constructed = 0

    def __init__(self, uri, timeout=None):
        type(self).constructed += 1
        if type(self).down:
            raise ConnectionError("Fail connecting to server on milvus:19530")
        self.uri = uri

    def list_collections(self):
        return ["amazon_reviews"]


def _boot_with_milvus_down(monkeypatch):
    _FlakyMilvusClient.down, _FlakyMilvusClient.constructed = True, 0
    monkeypatch.setattr(session_mod, "MilvusClient", _FlakyMilvusClient)
    return Session()


def test_milvus_down_at_boot_reports_error(monkeypatch):
    s = _boot_with_milvus_down(monkeypatch)
    h = s.health()["milvus"]
    assert h["ok"] is False
    assert "Fail connecting" in h["error"]
    assert s.namespace(0, "milvus")["client"] is None


def test_milvus_up_after_boot_recovers_without_restart(monkeypatch):
    s = _boot_with_milvus_down(monkeypatch)
    assert s.health()["milvus"]["ok"] is False

    _FlakyMilvusClient.down = False
    assert s.health()["milvus"] == {"ok": True, "error": None}
    assert isinstance(s.namespace(0, "milvus")["client"], _FlakyMilvusClient)


def test_connected_client_is_reused_not_rebuilt(monkeypatch):
    s = _boot_with_milvus_down(monkeypatch)
    _FlakyMilvusClient.down = False
    first = s.client
    s.health()
    s.namespace(0, "milvus")
    assert s.client is first
    assert _FlakyMilvusClient.constructed == 2      # failed boot + one reconnect
