"""Observe vLLM's native per-request block release at the scheduler boundary."""


class SlidingWindowEvidenceError(RuntimeError):
    """The observed scheduler trace does not establish the requested condition."""


def observe_block_release(scheduler, trace):
    """Wrap the native coordinator call without changing any scheduler or cache decisions."""
    cache = scheduler.kv_cache_manager
    coordinator = cache.coordinator
    pool = cache.block_pool
    native_remove = coordinator.remove_skipped_blocks

    def observe(request_id, processed_computed_tokens, num_prompt_tokens=None):
        request = scheduler.requests[request_id]
        (blocks,) = coordinator.get_blocks(request_id)
        previous = tuple(blocks)
        before = [None if block.is_null else block.block_id for block in previous]
        refs_before = [block.ref_cnt for block in previous]
        free_before = pool.get_num_free_blocks()
        running = request.status.name == "RUNNING"
        output_tokens = request.num_output_tokens

        result = native_remove(request_id, processed_computed_tokens, num_prompt_tokens)

        (blocks,) = coordinator.get_blocks(request_id)
        after = [None if block.is_null else block.block_id for block in blocks]
        free_after = pool.get_num_free_blocks()
        released = []
        if before != after:
            free_ids = {block.block_id for block in pool.free_block_queue.get_all_free_blocks()}
            for position, (old, new) in enumerate(zip(before, after)):
                if old is not None and new is None:
                    released.append(dict(
                        position=position, block_id=old,
                        ref_before=refs_before[position], ref_after=previous[position].ref_cnt,
                        in_free_queue=old in free_ids,
                    ))

        trace.append(dict(
            request_id=request_id, processed_tokens=processed_computed_tokens,
            output_tokens=output_tokens, running=running,
            preemptions=request.num_preemptions, before=before, after=after,
            free_before=free_before, free_after=free_after, released=released,
        ))
        return result

    coordinator.remove_skipped_blocks = observe
    return native_remove


def validate_trace(condition, trace, *, block_size, window, prompt_tokens, output_tokens):
    """Require a live release and further decode, or a matched retaining control."""
    if not trace or len({row["request_id"] for row in trace}) != 1:
        raise SlidingWindowEvidenceError("Expected scheduler observations for one request")
    if any(row["preemptions"] for row in trace):
        raise SlidingWindowEvidenceError("The request was preempted")
    decode = [row for row in trace if row["running"] and row["processed_tokens"] >= prompt_tokens
              and row["output_tokens"] > 0]
    if not decode or decode[0]["processed_tokens"] != prompt_tokens or decode[0]["output_tokens"] != 1:
        raise SlidingWindowEvidenceError("Missing first decode step after ordinary full prefill")
    first = decode[0]
    expected_prefill_blocks = (prompt_tokens + block_size - 1) // block_size
    if (len(first["before"]) != expected_prefill_blocks or None in first["before"]
            or first["before"] != first["after"]):
        raise SlidingWindowEvidenceError("Missing initial full-cache ownership")
    if (decode[-1]["processed_tokens"] < window - 1 + block_size
            or decode[-1]["output_tokens"] != output_tokens - 1):
        raise SlidingWindowEvidenceError("Insufficient observed decode across the release boundary")

    releases = []
    for row in trace:
        before, after = row["before"], row["after"]
        removed = {index for index, block in enumerate(before) if block is not None
                   and index < len(after) and after[index] is None}
        if len(before) != len(after) or any(old != new and index not in removed
                                           for index, (old, new) in enumerate(zip(before, after))):
            raise SlidingWindowEvidenceError("Unexpected block-table mutation")
        if not removed:
            if row["free_before"] != row["free_after"] or row["released"]:
                raise SlidingWindowEvidenceError("Unexplained movement in the free queue")
            continue
        if condition != "sliding" or row not in decode:
            raise SlidingWindowEvidenceError("Unexpected release outside live sliding-window decode")
        if row["free_after"] - row["free_before"] != len(removed):
            raise SlidingWindowEvidenceError("Free-block count did not increase as expected")
        expired_before = row["processed_tokens"] - window + 1
        verified = {entry["position"] for entry in row["released"]
                    if entry["position"] in removed
                    and before[entry["position"]] == entry["block_id"]
                    and entry["ref_before"] == 1 and entry["ref_after"] == 0
                    and entry["in_free_queue"]
                    and (entry["position"] + 1) * block_size <= expired_before}
        if verified != removed or len(row["released"]) != len(removed):
            raise SlidingWindowEvidenceError("Missing out-of-window ownership-to-free-queue transition")
        releases.append(row)

    if condition == "sliding":
        if not releases:
            raise SlidingWindowEvidenceError("Native block release was not observed")
        if not any(row["output_tokens"] > releases[0]["output_tokens"] for row in decode):
            raise SlidingWindowEvidenceError("No later decode after the first native release")
    elif len(decode[-1]["after"]) <= expected_prefill_blocks or None in decode[-1]["after"]:
        raise SlidingWindowEvidenceError("Full-attention control did not retain its growing cache")
    return len(releases)
