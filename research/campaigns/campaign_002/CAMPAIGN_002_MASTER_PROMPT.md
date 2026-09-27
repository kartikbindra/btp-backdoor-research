# CAMPAIGN 002 MASTER PROMPT — WP0

Use `/teamwork-preview` to execute this campaign.

## Research question
Does the proposed FP8 KV-cache proxy reproduce the scientifically relevant behavior of the pinned production vLLM FP8 KV-cache path closely enough, on a clean model, that later runtime-conditioned backdoor experiments would be interpretable?

Campaign 001 narrowed the hypothesis to a parameter-efficient fine-tuned checkpoint, fresh isolated per-request caches, legitimate documented runtime compression, natural prompts with zero trigger tokens, clean-subtracted causal evidence, and matched-policy utility. Campaign 002 is the prerequisite runtime gate.

## Team
Create these roles:
1. Runtime/vLLM Engineer — inspect exact FP8 implementation, scaling, quantization boundary, cache representation, attention backend/kernel, fallback paths.
2. Hardware/Systems Auditor — verify GPU/CUDA/driver/kernel compatibility and actual execution path.
3. Numerical Conformance Scientist — design cache/attention/logit/end-to-end proxy metrics; do not reuse arbitrary old thresholds.
4. Experimental Scientist — clean BF16 vs production FP8 vs proxy experiment only.
5. Statistics/Reproducibility Auditor — paired design, prompt clusters, seeds, bootstrap, CI, multiple testing, threshold freezing.
6. Adversarial Reviewer — try to falsify conformance and detect hidden fallbacks/confounds.
7. Memory Keeper — update canonical campaign memory only after synthesis.
8. Final Campaign Reviewer — independently review the complete evidence package.

## Execution order
### Phase 0 — Environment lock
Inspect Campaign 001 artifacts and current code. Pin model/revision, runtime versions, GPU, backend, configs, code commit and prompt-set hash. Produce `research/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`.

### Phase 1 — Production path inspection
Trace `model → K/V projection → FP8 cache storage → attention consumption → output`. Establish dtype/format, scaling granularity, quantization/dequantization boundaries, kernels, hardware assumptions and fallbacks. Produce `research/CAMPAIGN_002_RUNTIME_PATH.md`.

### Phase 2 — Determinism baseline
On a clean model, compare repeated BF16/full-cache and production FP8 inference with fixed benign prompts and greedy decoding. Include process restarts where feasible. Do not call compression differences a backdoor. Produce `research/CAMPAIGN_002_DETERMINISM.md`.

### Phase 3 — Proxy conformance
Compare three conditions: (A) BF16/full-cache reference, (B) production vLLM FP8, (C) candidate software proxy. Validate at cache-tensor, attention/logit, and end-to-end levels. Where possible factor storage quantization from backend/kernel effects. Produce `research/CAMPAIGN_002_PROXY_CONFORMANCE.md`.

### Phase 4 — Acceptance gate
Define primary metrics and candidate thresholds before final confirmatory analysis. Pilot measurements may calibrate natural variability, but thresholds must be frozen before confirmatory results. The question is downstream scientific adequacy, not numerical identity.

### Phase 5 — Adversarial audit
Test process restart, prompt clusters, context lengths, batch sizes where feasible, scale outliers/saturation, alternate backend if available, and suspected fallbacks. Prioritize failures that would change interpretation.

### Phase 6 — Decision
Produce `research/CAMPAIGN_002_DECISION_MEMO.md` with PASS / CONDITIONAL PASS / FAIL, evidence, limitations, unresolved questions and exact Campaign 003 handoff.

## Minimum experiment matrix
| Model | Cache | Purpose |
|---|---|---|
| clean | BF16/full | reference |
| clean | production FP8 | clean compression effect |
| clean | candidate proxy | proxy comparison |

Use sequestered benign prompt clusters, paired observations, deterministic decoding first, and hierarchical/cluster-aware uncertainty where appropriate.

## Non-negotiables
- No backdoor training in Campaign 002.
- No harmful behavior targets.
- No novelty claims from this campaign.
- No post-hoc acceptance thresholds.
- No undocumented runtime substitutions.
- Every empirical result must be traceable to a log/artifact.
- A negative result is acceptable and must be preserved.
