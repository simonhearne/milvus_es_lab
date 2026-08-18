"""Last-good run per feature, for the on-stage fallback.

If an engine dies mid-talk you get an honest error by default -- and a
deliberate one-click escape hatch to show the last real run instead, stamped so
nobody mistakes it for live. That's the deck's SAMPLE DATA convention carried
forward.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, ".cache")


def _path(feature):
    safe = "".join(c for c in feature if c.isalnum() or c in "-_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def save(feature, payload):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _path(feature) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, _path(feature))     # atomic: never read a half-written run


def load(feature):
    try:
        with open(_path(feature)) as fh:
            payload = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    payload["cached"] = True
    return payload
