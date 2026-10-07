#!/usr/bin/env python3
"""WP0 target-host capability preflight.

This command checks only environment-spec loading, OS/GPU prerequisites,
analytical helpers, and local guard tests. It can never pass full UG0 or
authorize training; data, parser, ethics, dependency-lock, and manifest
contracts require separate review.
"""

import os
import sys
import unittest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.runtime.env_inspector import inspect_environment
from src.runtime.runtime_tracer import calculate_kv_cache_footprint, trace_execution_path
from tests.test_kernel_fallback import (
    TestEnvironmentSpecAndExecutionTracer,
    TestHardwareFallbackDetection,
    TestRunnerConfigurationInvariants,
    TestScalingFactorValidation,
    TestSilentFallbackTraps,
)


def run_wp0_verification():
    print("=" * 80)
    print("CAMPAIGN 002 REMEDIATION: WP0 TARGET-HOST CAPABILITY PREFLIGHT")
    print("=" * 80)

    # 1. Environment Inspection
    spec_path = os.path.join(WORKSPACE_ROOT, "configs", "env", "environment_spec.yaml")
    print(f"\n[1/4] Inspecting host environment and loading specification: {spec_path}")
    env_result = inspect_environment(spec_path=spec_path)

    print(f"      Host OS: {env_result['host']['os_name']} ({env_result['host']['os_release']})")
    print(f"      Python: {env_result['host']['python_version']}")
    print(f"      PyTorch Installed: {env_result['cuda']['torch_installed']}")
    print(f"      CUDA Available: {env_result['cuda']['cuda_available']}")
    print(f"      Specification Loaded: {env_result['spec_loaded']}")

    if not env_result["spec_loaded"]:
        print("ERROR: Failed to load configs/env/environment_spec.yaml!")
        return 1
    if env_result["verdict"] != "PASS":
        print("ERROR: Current host does not satisfy the physical runtime specification.")
        for check in env_result["compliance_checks"]:
            if not check["passed"]:
                print(
                    f"      [{check['severity']}] {check['check']}: "
                    f"expected {check['expected']}; found {check['actual']}"
                )
        print("WP0 remains partial; no runtime gate or training is authorized.")
        return 2

    # 2. Print Pinned Parameters
    spec = env_result["spec"]
    print("\n[2/4] Verifying Pinned Parameters from environment_spec.yaml:")
    print(f"      Model ID: {spec['model']['model_id']}")
    print(f"      Model Commit: {spec['model']['exact_commit_hash']}")
    print(f"      Target Architectures: {[a['compute_capability'] for a in spec['hardware']['target_gpu_architectures']]}")
    print(f"      Disallowed Architectures: {[a['compute_capability'] for a in spec['hardware']['disallowed_gpu_architectures']]}")
    print(f"      KV Cache Dtype: {spec['kv_cache']['dtype']} (Bytes/elem: {spec['kv_cache']['element_size_bytes']})")
    print(f"      Fallback Policy: {spec['fallback_policy']['mode']}")

    # 3. Analytical Trace Verification
    print("\n[3/4] Verifying Analytical Runtime Execution Trace & Cache Footprint:")
    footprint = calculate_kv_cache_footprint(seq_len=4096, batch_size=1)
    print(f"      Sequence Length: {footprint['seq_len']} tokens")
    print(f"      BF16 Cache: {footprint['condition_a_bf16']['total_mb']} MB ({footprint['condition_a_bf16']['bytes_per_token']} bytes/tok)")
    print(f"      FP8 Cache:  {footprint['condition_b_fp8']['total_mb']} MB ({footprint['condition_b_fp8']['bytes_per_token']} bytes/tok)")
    print(f"      Memory Reduction: {footprint['reduction_percentage']}%")

    trace = trace_execution_path(hardware_arch="sm_89")
    print(f"      Execution Stages Formally Modeled: {trace['total_stages']}")
    for s in trace["stages"]:
        print(f"        Stage {s['stage_id']}: {s['name']} -> {s['kernel']} ({s['boundary']})")

    # 4. Execute Unit Test Suite
    print("\n[4/4] Running Unit Test Suite (tests/test_kernel_fallback.py):")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHardwareFallbackDetection))
    suite.addTest(unittest.makeSuite(TestSilentFallbackTraps))
    suite.addTest(unittest.makeSuite(TestScalingFactorValidation))
    suite.addTest(unittest.makeSuite(TestRunnerConfigurationInvariants))
    suite.addTest(unittest.makeSuite(TestEnvironmentSpecAndExecutionTracer))

    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    print("\n" + "=" * 80)
    if test_result.wasSuccessful():
        print("HOST CAPABILITY PREFLIGHT: PASS")
        print("UG0 STATUS: PARTIAL — this command cannot close governance contracts.")
        print("TRAINING AUTHORIZED: False")
        print("=" * 80)
        return 0
    else:
        print(f"VERIFICATION RESULT: FAILED ({len(test_result.failures)} failures, {len(test_result.errors)} errors)")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(run_wp0_verification())
