"""
Shared Milvus helpers for the one-shot load scripts in data/.
"""
import time


def compact_and_wait(client, collection, timeout=120.0):
    """Force-compact `collection` and block until Milvus reports completion.

    Compaction only operates on sealed segments, so callers must flush()
    first. Best-effort by design: a timeout or API error warns and returns
    instead of raising, because an un-compacted collection is still correct.
    """
    try:
        job_id = client.compact(collection)
    except Exception as e:
        print(f"compaction skipped ({collection}): {e}")
        return
    if not job_id or int(job_id) <= 0:
        print(f"compaction: nothing to compact in {collection}")
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            state = str(client.get_compaction_state(job_id))
        except Exception as e:
            print(f"compaction state check failed ({collection}): {e}")
            return
        if "Completed" in state:
            print(f"compacted {collection} in {time.monotonic() - t0:.1f}s "
                  f"(job {job_id})")
            return
        time.sleep(2.0)
    print(f"WARNING: compaction of {collection} still running after "
          f"{timeout:.0f}s (job {job_id}); continuing without waiting")
