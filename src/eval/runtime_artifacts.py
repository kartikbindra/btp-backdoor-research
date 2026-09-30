"""Strict pairing of immutable vLLM condition artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class RuntimeArtifactMismatchError(ValueError):
    """Raised when runtime conditions are not scientifically pairable."""


def _payload(record: Dict[str, Any]) -> Dict[str, Any]:
    if record.get("evidence_class") != "CONFIRMATORY_CLEAN_SOURCE":
        raise RuntimeArtifactMismatchError(
            "Only clean-source confirmatory artifacts may be paired"
        )
    git_state = record.get("git", {})
    if git_state.get("source_dirty") or not git_state.get("commit"):
        raise RuntimeArtifactMismatchError("Artifact source provenance is dirty or absent")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeArtifactMismatchError("Artifact has no payload object")
    if payload.get("status") != "COMPLETED":
        raise RuntimeArtifactMismatchError(
            f"Artifact condition did not complete: {payload.get('status')}"
        )
    if payload.get("artifact_type") != "genuine_vllm_runtime_condition":
        raise RuntimeArtifactMismatchError("Artifact is not a genuine runtime condition")
    return payload


def _assert_attestation(
    payload: Dict[str, Any], expected_condition: str
) -> Dict[str, Any]:
    config = payload.get("config", {})
    attestation = payload.get("runtime_attestation", {})
    if payload.get("condition") != expected_condition:
        raise RuntimeArtifactMismatchError(
            f"Payload condition is not {expected_condition}"
        )
    if config.get("condition") != expected_condition:
        raise RuntimeArtifactMismatchError("Config condition disagrees with payload")
    if attestation.get("condition") != expected_condition:
        raise RuntimeArtifactMismatchError("Attestation condition disagrees with payload")
    if not attestation.get("physical_cache_inspection_passed"):
        raise RuntimeArtifactMismatchError("Physical cache attestation is missing")
    if not attestation.get("resolved_attention_backend"):
        raise RuntimeArtifactMismatchError("Resolved attention backend is absent")
    if attestation.get("request_submission_mode") != "sequential_batch_size_one":
        raise RuntimeArtifactMismatchError("Request isolation mode is not sequential")
    if not attestation.get("cache_release_attested"):
        raise RuntimeArtifactMismatchError(
            "Version-specific cache-block release is not attested"
        )
    if not attestation.get("cache_tensors"):
        raise RuntimeArtifactMismatchError("No physical cache tensors were recorded")
    if not attestation.get("complete_cache_layer_coverage"):
        raise RuntimeArtifactMismatchError("Physical cache layer coverage is incomplete")
    return attestation


def _index_generations(
    generations: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    indexed = {}
    for generation in generations:
        key = (generation.get("cluster_id"), generation.get("prompt_id"))
        if None in key or key in indexed:
            raise RuntimeArtifactMismatchError(f"Invalid or duplicate prompt key: {key}")
        indexed[key] = generation
    if not indexed:
        raise RuntimeArtifactMismatchError("Runtime artifact contains no prompts")
    return indexed


def _require_equal(
    reference: Dict[str, Any],
    target: Dict[str, Any],
    fields: Tuple[str, ...],
    label: str,
) -> None:
    mismatches = {
        field: [reference.get(field), target.get(field)]
        for field in fields
        if reference.get(field) != target.get(field)
    }
    if mismatches:
        raise RuntimeArtifactMismatchError(f"{label} differs: {mismatches}")


def compare_vllm_conditions(
    reference_record: Dict[str, Any], target_record: Dict[str, Any]
) -> Dict[str, Any]:
    """Pair clean-source BF16 and FP8 artifacts for preliminary token analysis.

    Even a valid pair cannot pass UG2. A pinned real-Qwen Transformers proxy
    artifact and per-step/tensor comparison are still required.
    """
    reference = _payload(reference_record)
    target = _payload(target_record)
    ref_attestation = _assert_attestation(reference, "reference_bf16")
    target_attestation = _assert_attestation(target, "target_fp8")

    _require_equal(
        reference_record["git"],
        target_record["git"],
        ("commit", "origin", "source_tree_fingerprint", "source_dirty"),
        "Source provenance",
    )
    if reference_record.get("environment") != target_record.get("environment"):
        raise RuntimeArtifactMismatchError("Host/package/CUDA environments differ")

    invariant_fields = (
        "model_id",
        "model_revision",
        "expected_vllm_version",
        "model_dtype",
        "gpu_memory_utilization",
        "max_model_len",
        "enforce_eager",
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "logprobs",
        "enable_prefix_caching",
        "tensor_parallel_size",
        "device_index",
        "expected_num_hidden_layers",
        "expected_attention_backend",
        "max_num_seqs",
    )
    _require_equal(
        reference["config"], target["config"], invariant_fields, "Condition config"
    )
    _require_equal(
        ref_attestation,
        target_attestation,
        ("resolved_attention_backend", "request_submission_mode", "max_num_seqs"),
        "Resolved runtime",
    )
    if reference.get("prompt_set_sha256") != target.get("prompt_set_sha256"):
        raise RuntimeArtifactMismatchError("Prompt set hashes differ")

    target_scale = target_attestation.get("scale_evidence", {})
    if not target_scale.get("verified"):
        raise RuntimeArtifactMismatchError("Target FP8 scale evidence is incomplete")

    reference_outputs = _index_generations(reference.get("generations", []))
    target_outputs = _index_generations(target.get("generations", []))
    if reference_outputs.keys() != target_outputs.keys():
        raise RuntimeArtifactMismatchError("Prompt IDs differ across conditions")

    comparisons = []
    token_matches = []
    exact_sequence_matches = 0
    for key in sorted(reference_outputs):
        ref = reference_outputs[key]
        tgt = target_outputs[key]
        ref_expected = ref.get("expected_prompt_token_ids", [])
        tgt_expected = tgt.get("expected_prompt_token_ids", [])
        ref_runtime = ref.get("runtime_prompt_token_ids", [])
        tgt_runtime = tgt.get("runtime_prompt_token_ids", [])
        if not ref_expected or ref_expected != tgt_expected:
            raise RuntimeArtifactMismatchError(
                f"Expected tokenizer IDs differ for {key}"
            )
        if ref_runtime != ref_expected or tgt_runtime != tgt_expected:
            raise RuntimeArtifactMismatchError(
                f"Runtime-consumed IDs differ from expected tokenizer IDs for {key}"
            )
        if ref_runtime != tgt_runtime:
            raise RuntimeArtifactMismatchError(
                f"Runtime input token IDs differ across conditions for {key}"
            )
        ref_tokens = ref.get("output_token_ids", [])
        tgt_tokens = tgt.get("output_token_ids", [])
        denominator = max(len(ref_tokens), len(tgt_tokens), 1)
        matches = sum(
            1
            for index in range(min(len(ref_tokens), len(tgt_tokens)))
            if ref_tokens[index] == tgt_tokens[index]
        )
        match_rate = matches / denominator
        exact = ref_tokens == tgt_tokens
        exact_sequence_matches += int(exact)
        token_matches.append(match_rate)
        comparisons.append(
            {
                "cluster_id": key[0],
                "prompt_id": key[1],
                "token_match_rate": match_rate,
                "exact_sequence_match": exact,
                "reference_output_tokens": len(ref_tokens),
                "target_output_tokens": len(tgt_tokens),
            }
        )

    count = len(comparisons)
    return {
        "artifact_type": "preliminary_paired_vllm_runtime_conditions",
        "status": "PRELIMINARY_TOKEN_PAIR_PROXY_REQUIRED",
        "ug1_status": "REQUIRES_REPEAT_PROCESS_ARTIFACTS",
        "ug2_status": "BLOCKED_REAL_QWEN_PROXY_AND_TENSOR_METRICS_REQUIRED",
        "training_authorized": False,
        "reference_artifact_sha256": reference_record["content_sha256"],
        "target_artifact_sha256": target_record["content_sha256"],
        "source_commit": reference_record["git"]["commit"],
        "prompt_set_sha256": reference["prompt_set_sha256"],
        "prompt_count": count,
        "mean_token_match_rate": sum(token_matches) / count,
        "exact_sequence_match_rate": exact_sequence_matches / count,
        "comparisons": comparisons,
        "limitations": [
            "Token agreement is not proxy/runtime tensor conformance.",
            "Sparse top-logprobs cannot reconstruct full-vocabulary JSD or Spearman metrics.",
            "A real-Qwen Transformers proxy artifact is required before UG2 review.",
        ],
    }
