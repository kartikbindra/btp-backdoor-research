"""Tests for immutable experiment artifacts and evidence boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

from src.runtime.artifacts import (
    ArtifactIntegrityError,
    _managed_evidence_line,
    load_verified_artifact,
    write_immutable_json_artifact,
)
from src.runtime.vllm_runner import (
    RuntimeCacheCondition,
    VLLMRunnerConfig,
    validate_runner_config,
)


class TestImmutableArtifacts(unittest.TestCase):
    def test_artifact_round_trip_and_tamper_detection(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            artifact, digest = write_immutable_json_artifact(
                payload={"status": "FAILED", "reason": "expected test record"},
                output_dir=Path(directory),
                stem="test",
                repo_root=repo_root,
                require_clean_git=False,
            )
            loaded = load_verified_artifact(artifact)
            self.assertEqual(loaded["payload"]["status"], "FAILED")
            self.assertTrue(digest.exists())

            record = json.loads(artifact.read_text(encoding="utf-8"))
            record["payload"]["status"] = "COMPLETED"
            artifact.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ArtifactIntegrityError):
                load_verified_artifact(artifact)

    def test_missing_or_malformed_sidecar_is_rejected(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            artifact, digest = write_immutable_json_artifact(
                payload={"status": "FAILED"},
                output_dir=Path(directory),
                stem="test",
                repo_root=repo_root,
                require_clean_git=False,
            )
            digest.unlink()
            with self.assertRaises(ArtifactIntegrityError):
                load_verified_artifact(artifact)
            digest.write_text(f"{'0' * 64}  wrong-name.json\n", encoding="utf-8")
            with self.assertRaises(ArtifactIntegrityError):
                load_verified_artifact(artifact)

    def test_managed_evidence_paths_do_not_count_as_source(self):
        self.assertTrue(_managed_evidence_line("?? results/raw/runtime/run.json"))
        self.assertTrue(_managed_evidence_line("?? results/derived/runtime/pair.json"))
        self.assertFalse(_managed_evidence_line(" M src/runtime/artifacts.py"))
        self.assertFalse(_managed_evidence_line("?? results/README.md"))

    def test_reference_and_target_configs_are_distinct(self):
        reference = VLLMRunnerConfig.for_condition(
            RuntimeCacheCondition.REFERENCE_BF16
        )
        target = VLLMRunnerConfig.for_condition(RuntimeCacheCondition.TARGET_FP8)
        self.assertEqual(reference.kv_cache_dtype, "auto")
        self.assertEqual(target.kv_cache_dtype, "fp8")
        self.assertFalse(reference.calculate_kv_scales)
        self.assertTrue(target.calculate_kv_scales)
        self.assertTrue(validate_runner_config(reference))
        self.assertTrue(validate_runner_config(target))


if __name__ == "__main__":
    unittest.main()
