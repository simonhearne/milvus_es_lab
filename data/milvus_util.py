"""
Shared Milvus helpers for the one-shot load scripts in data/.
"""
import time

# Merge target for the force compaction. A plain compact() stops at
# datacoord.segment.maxSize (1024 MB), which leaves amazon_reviews (~1.6 GB)
# in two segments. That is not cosmetic: IVF is built per segment and nprobe
# applies to each one, so two segments silently double the search breadth.
# On 3.0.6 that took filtered recall@10 from 0.698 (one segment, matching the
# recorded tables) to 0.740. 4096 MB clears the whole corpus with headroom.
MERGE_TARGET_MB = 4096


def compact_and_wait(client, collection, timeout=120.0,
                     target_size_mb=MERGE_TARGET_MB):
    """Force-compact `collection` toward one segment and block until done.

    The Milvus analogue of es_util.forcemerge_and_wait: `target_size_mb`
    lets the merge exceed the default segment cap, so every load ends in the
    same single-segment layout the benchmarks were measured on.

    Compaction only operates on sealed segments, so callers must flush()
    first. Best-effort by design: a timeout or API error warns and returns
    instead of raising, because an un-compacted collection is still correct.
    """
    try:
        job_id = client.compact(collection, target_size=target_size_mb,
                                target_size_unit="mb")
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
