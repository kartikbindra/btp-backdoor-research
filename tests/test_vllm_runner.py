"""Unit tests for fail-closed vLLM runner orchestration."""

import sys
import types
import unittest
from unittest.mock import patch

from scripts.run_vllm_condition import attach_verified_prompt_metadata
from src.runtime.vllm_runner import VLLMRunner, VLLMRunnerConfig


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Completion:
    def __init__(self, token_id: int):
        self.token_ids = [token_id]
        self.text = str(token_id)
        self.cumulative_logprob = 0.0
        self.logprobs = []
        self.finish_reason = "length"


class _Request:
    def __init__(self, index: int):
        self.request_id = f"request-{index}"
        self.prompt_token_ids = [index]
        self.outputs = [_Completion(index)]


class _Engine:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append(list(prompts))
        return [_Request(len(self.calls))]


class TestVLLMRunnerOrchestration(unittest.TestCase):
    def test_generate_serializes_requests_at_batch_size_one(self):
        runner = VLLMRunner(VLLMRunnerConfig())
        runner.engine = _Engine()
        runner._is_initialized = True
        fake_vllm = types.SimpleNamespace(SamplingParams=_SamplingParams)
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            records = runner.generate(["one", "two", "three"])
        self.assertEqual(runner.engine.calls, [["one"], ["two"], ["three"]])
        self.assertEqual(runner._last_submission_mode, "sequential_batch_size_one")
        self.assertEqual(len(set(runner._last_request_ids)), 3)
        self.assertEqual(len(records), 3)

    def test_prompt_metadata_preserves_and_checks_runtime_ids(self):
        record = {"runtime_prompt_token_ids": [1, 2]}
        metadata = {
            "prompt_id": "p1",
            "expected_prompt_token_ids": [1, 2],
            "cluster_id": "c1",
        }
        attach_verified_prompt_metadata(record, metadata)
        self.assertEqual(record["runtime_prompt_token_ids"], [1, 2])
        self.assertEqual(record["expected_prompt_token_ids"], [1, 2])
        with self.assertRaises(RuntimeError):
            attach_verified_prompt_metadata(
                {"runtime_prompt_token_ids": [9]}, metadata
            )

    def test_generate_rejects_empty_prompt_set(self):
        runner = VLLMRunner(VLLMRunnerConfig())
        runner.engine = _Engine()
        runner._is_initialized = True
        with self.assertRaises(RuntimeError):
            runner.generate([])


if __name__ == "__main__":
    unittest.main()
