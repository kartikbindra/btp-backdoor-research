# Campaign 002 Research Constitution

## Scope
Runtime/vLLM FP8 path inspection, hardware verification, deterministic clean-model baselines, proxy construction/conformance, reproducibility, statistical audit, adversarial review. No backdoor training, harmful payloads, PF-SEB, or evasion engineering.

## Evidence labels
Every important statement is one of: SOURCE FACT / IMPLEMENTATION OBSERVATION / EXPERIMENTAL RESULT / INFERENCE / HYPOTHESIS / DECISION.

## Reproducibility
Record model/revision, tokenizer/template, vLLM version+commit, CUDA/driver/PyTorch, GPU, backend/kernel, KV dtype, scaling, seeds, decoding, prompt-set hash, code commit, timestamp.

## Gate
Do not proceed to backdoor training if the production path cannot be pinned, the actual hardware/backend path is unknown, repeated inference is materially unstable, or proxy conformance cannot be defined without post-hoc threshold selection.

## Principle
Optimize uncertainty reduction, not PASS probability. Preserve negative results and disagreements.
