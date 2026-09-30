"""Tests for strict scientific pairing of runtime condition artifacts."""

import copy
import unittest

from src.eval.runtime_artifacts import (
    RuntimeArtifactMismatchError,
    compare_vllm_conditions,
)


def _cache_tensor(condition: str) -> dict:
    if condition == "target_fp8":
        dtype, element_size = "torch.float8_e4m3fn", 1
    else:
        dtype, element_size = "torch.bfloat16", 2
    return {
        "path": "worker.gpu_cache[0]",
        "layer_index": 0,
        "shape": [2, 1, 16, 8],
        "dtype": dtype,
        "element_size_bytes": element_size,
        "device": "cuda:0",
        "all_finite": True,
        "sample_max_abs": 1.0,
    }


def _record(condition: str, tokens: list[int], digest: str) -> dict:
    config = {
        "condition": condition,
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "revision",
        "expected_vllm_version": "0.26.0",
        "model_dtype": "bfloat16",
        "kv_cache_dtype": "fp8" if condition == "target_fp8" else "auto",
        "calculate_kv_scales": condition == "target_fp8",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "enforce_eager": True,
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 4,
        "logprobs": 20,
        "enable_prefix_caching": False,
        "tensor_parallel_size": 1,
        "device_index": 0,
        "expected_num_hidden_layers": 1,
        "expected_attention_backend": "FLASH_ATTN",
        "max_num_seqs": 1,
    }
    attestation = {
        "condition": condition,
        "resolved_cache_dtype": "fp8" if condition == "target_fp8" else "auto",
        "resolved_prefix_caching": False,
        "resolved_attention_backend": "FLASH_ATTN",
        "request_submission_mode": "sequential_batch_size_one",
        "max_num_seqs": 1,
        "physical_cache_inspection_passed": True,
        "cache_release_attested": True,
        "complete_cache_layer_coverage": True,
        "cache_layer_indices": [0],
        "cache_tensors": [_cache_tensor(condition)],
        "scale_evidence": {
            "verified": True,
            "complete_layer_coverage": True,
            "key_layer_indices": [0],
            "value_layer_indices": [0],
            "entries": [] if condition == "reference_bf16" else [{"min": 0.1}],
        },
    }
    environment = {
        "hostname": "gpu-host",
        "system": "Linux",
        "machine": "x86_64",
        "python_version": "3.12.0",
        "packages": {"vllm": "0.26.0", "torch": "version"},
        "cuda": {
            "available": True,
            "compiled_version": "version",
            "devices": [{"uuid": "gpu-uuid", "compute_capability": "sm_90"}],
        },
    }
    return {
        "content_sha256": digest,
        "evidence_class": "CONFIRMATORY_CLEAN_SOURCE",
        "git": {
            "commit": "commit-a",
            "origin": "origin",
            "source_tree_fingerprint": "tree-a",
            "source_dirty": False,
        },
        "environment": environment,
        "payload": {
            "status": "COMPLETED",
            "artifact_type": "genuine_vllm_runtime_condition",
            "condition": condition,
            "config": config,
            "prompt_set_sha256": "prompt-hash",
            "runtime_attestation": attestation,
            "generations": [
                {
                    "cluster_id": "cluster",
                    "prompt_id": "prompt",
                    "expected_prompt_token_ids": [1, 2],
                    "runtime_prompt_token_ids": [1, 2],
                    "output_token_ids": tokens,
                }
            ],
        },
    }


class TestRuntimeArtifactPairing(unittest.TestCase):
    def test_valid_pair_remains_blocked_on_proxy(self):
        reference = _record("reference_bf16", [3, 4, 5], "ref")
        target = _record("target_fp8", [3, 4, 6], "target")
        result = compare_vllm_conditions(reference, target)
        self.assertEqual(result["prompt_count"], 1)
        self.assertAlmostEqual(result["mean_token_match_rate"], 2 / 3)
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["status"], "PRELIMINARY_TOKEN_PAIR_PROXY_REQUIRED")
        self.assertIn("PROXY", result["ug2_status"])

    def test_input_token_mismatch_is_rejected(self):
        reference = _record("reference_bf16", [3], "ref")
        target = _record("target_fp8", [3], "target")
        target["payload"]["generations"][0]["runtime_prompt_token_ids"] = [9]
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)

    def test_config_mismatch_is_rejected(self):
        reference = _record("reference_bf16", [3], "ref")
        target = _record("target_fp8", [3], "target")
        target["payload"]["config"]["max_model_len"] = 8192
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)

    def test_source_or_environment_mismatch_is_rejected(self):
        reference = _record("reference_bf16", [3], "ref")
        target = _record("target_fp8", [3], "target")
        target["git"]["commit"] = "commit-b"
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)
        target = _record("target_fp8", [3], "target")
        target["environment"]["hostname"] = "another-host"
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)

    def test_dirty_exploratory_or_failed_condition_is_rejected(self):
        reference = _record("reference_bf16", [3], "ref")
        target = _record("target_fp8", [3], "target")
        target["evidence_class"] = "EXPLORATORY_DIRTY_SOURCE"
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)
        target = _record("target_fp8", [3], "target")
        target["payload"]["status"] = "FAILED"
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)

    def test_missing_cache_release_attestation_is_rejected(self):
        reference = _record("reference_bf16", [3], "ref")
        target = _record("target_fp8", [3], "target")
        target["payload"]["runtime_attestation"]["cache_release_attested"] = False
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)

    def test_backend_scale_and_attestation_mismatches_are_rejected(self):
        reference = _record("reference_bf16", [3], "ref")
        target = _record("target_fp8", [3], "target")
        target["payload"]["runtime_attestation"]["resolved_attention_backend"] = "FLASHINFER"
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)
        target = _record("target_fp8", [3], "target")
        target["payload"]["runtime_attestation"]["scale_evidence"]["verified"] = False
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)
        target = _record("target_fp8", [3], "target")
        target["payload"]["runtime_attestation"]["condition"] = "reference_bf16"
        with self.assertRaises(RuntimeArtifactMismatchError):
            compare_vllm_conditions(reference, target)


if __name__ == "__main__":
    unittest.main()
