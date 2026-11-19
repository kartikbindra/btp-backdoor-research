"""Test live-block evidence validation without requiring a vLLM installation."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from src.runtime.sliding_window_probe import (
    SlidingWindowEvidenceError,
    observe_block_release,
    validate_trace,
)


def event(processed, tokens, before, after=None, released=None):
    free_before = 10
    return dict(
        request_id="request", processed_tokens=processed, output_tokens=tokens,
        running=True, preemptions=0, before=before,
        after=before.copy() if after is None else after,
        released=[] if released is None else released,
        free_before=free_before, free_after=free_before + len(released or []),
    )


class TestSlidingWindowEvidence(unittest.TestCase):
    def validate(self, condition, trace):
        return validate_trace(
            condition, trace, block_size=16, window=32, prompt_tokens=18, output_tokens=40,
        )

    def setUp(self):
        release = dict(position=0, block_id=2, ref_before=1, ref_after=0, in_free_queue=True)
        self.sliding = [
            event(18, 1, [2, 3]),
            event(47, 30, [2, 3, 4], [None, 3, 4], [release]),
            event(56, 39, [None, 3, 4, 2]),
        ]
        self.full = [
            event(18, 1, [2, 3]),
            event(47, 30, [2, 3, 4]),
            event(56, 39, [2, 3, 4, 5]),
        ]

    def test_genuine_release_and_retaining_control(self):
        self.assertEqual(self.validate("sliding", self.sliding), 1)
        self.assertEqual(self.validate("full", self.full), 0)

    def test_fail_closed_if_mask_only_or_control_releases(self):
        with self.assertRaisesRegex(SlidingWindowEvidenceError, "not observed"):
            self.validate("sliding", self.full)
        with self.assertRaisesRegex(SlidingWindowEvidenceError, "Unexpected release"):
            self.validate("full", self.sliding)

    def test_fail_closed_if_no_pool_return_or_release_is_early(self):
        for field, value in (("in_free_queue", False), ("ref_after", 1), ("ref_before", 2)):
            with self.subTest(field=field):
                trace = deepcopy(self.sliding)
                trace[1]["released"][0][field] = value
                with self.assertRaisesRegex(SlidingWindowEvidenceError, "ownership-to-free-queue"):
                    self.validate("sliding", trace)
        trace = deepcopy(self.sliding)
        trace[1]["processed_tokens"] -= 1
        with self.assertRaisesRegex(SlidingWindowEvidenceError, "out-of-window"):
            self.validate("sliding", trace)

    def test_fail_closed_if_no_later_decode_or_request_preempts(self):
        trace = deepcopy(self.sliding)
        trace[1]["output_tokens"] = trace[2]["output_tokens"]
        with self.assertRaisesRegex(SlidingWindowEvidenceError, "No later decode"):
            self.validate("sliding", trace)
        trace[1]["preemptions"] = 1
        with self.assertRaisesRegex(SlidingWindowEvidenceError, "preempted"):
            self.validate("sliding", trace)

    def test_observer_records_transition_before_next_allocation(self):
        block = SimpleNamespace(is_null=False, block_id=2, ref_cnt=1)
        null = SimpleNamespace(is_null=True)
        blocks = [block]
        free = []

        def native_remove(request_id, processed, prompt):
            self.assertEqual((request_id, processed, prompt), ("request", 47, 18))
            block.ref_cnt -= 1
            free.append(block)
            blocks[0] = null
            return "unchanged return"

        coordinator = SimpleNamespace(get_blocks=lambda _: (blocks,), remove_skipped_blocks=native_remove)
        pool = SimpleNamespace(
            get_num_free_blocks=lambda: len(free),
            free_block_queue=SimpleNamespace(get_all_free_blocks=lambda: free),
        )
        request = SimpleNamespace(
            status=SimpleNamespace(name="RUNNING"), num_output_tokens=30, num_preemptions=0,
        )
        scheduler = SimpleNamespace(
            requests={"request": request},
            kv_cache_manager=SimpleNamespace(coordinator=coordinator, block_pool=pool),
        )
        trace = []
        self.assertIs(observe_block_release(scheduler, trace), native_remove)
        self.assertEqual(coordinator.remove_skipped_blocks("request", 47, num_prompt_tokens=18),
                         "unchanged return")
        captured = trace[0]
        self.assertEqual((captured["before"], captured["after"]), ([2], [None]))
        self.assertEqual((captured["free_before"], captured["free_after"]), (0, 1))
        self.assertEqual(captured["released"], [
            dict(position=0, block_id=2, ref_before=1, ref_after=0, in_free_queue=True),
        ])
