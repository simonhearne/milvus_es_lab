"""Unit tests for data/milvus_util.py (no live cluster).

Run in Docker: docker compose run --rm demo python -m pytest data -q
"""
from data import milvus_util


class _StubClient:
    def __init__(self):
        self.calls = []

    def compact(self, collection, **kwargs):
        self.calls.append((collection, kwargs))
        return 42

    def get_compaction_state(self, job_id):
        return "Completed"


def test_compaction_merges_past_the_default_segment_cap():
    # A plain compact() stops at datacoord.segment.maxSize (1024 MB), which
    # left amazon_reviews (~1.6 GB) in two segments: per-segment IVF then
    # probes nprobe buckets in EACH, silently doubling the search breadth the
    # recall table was measured at. The loaders must merge to one segment.
    client = _StubClient()
    milvus_util.compact_and_wait(client, "amazon_reviews")
    [(name, kwargs)] = client.calls
    assert name == "amazon_reviews"
    assert kwargs.get("target_size_unit", "mb") == "mb"
    assert kwargs["target_size"] >= 2048, kwargs


def test_target_size_can_be_overridden():
    client = _StubClient()
    milvus_util.compact_and_wait(client, "c", target_size_mb=512)
    assert client.calls[0][1]["target_size"] == 512
