"""
Shared Elasticsearch helpers for the one-shot load scripts in data/.
"""
import time


def forcemerge_and_wait(es, index, timeout=300.0):
    """Force-merge `index` to one segment and block until ES reports completion.

    The ES analogue of milvus_util.compact_and_wait: every Lucene segment
    carries its own HNSW graph, so a fragmented index runs one graph search
    per segment and post-load latency depends on merge-scheduler timing.
    Merging to one segment makes reloads benchmark deterministically.

    Best-effort by design: a timeout or API error warns and returns instead
    of raising, because an un-merged index is still correct.
    """
    try:
        resp = es.indices.forcemerge(index=index, max_num_segments=1,
                                     wait_for_completion=False)
        task_id = resp.get("task")
    except Exception as e:
        print(f"force merge skipped ({index}): {e}")
        return
    if not task_id:
        print(f"force merge: no task id returned for {index}")
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            task = es.tasks.get(task_id=task_id)
        except Exception as e:
            print(f"force merge state check failed ({index}): {e}")
            return
        if task.get("completed"):
            print(f"force-merged {index} to 1 segment in "
                  f"{time.monotonic() - t0:.1f}s (task {task_id})")
            return
        time.sleep(2.0)
    print(f"WARNING: force merge of {index} still running after "
          f"{timeout:.0f}s (task {task_id}); continuing without waiting")
