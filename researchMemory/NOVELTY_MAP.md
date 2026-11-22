# Novelty Map (authoritative, re-verify before submission)

The project's novelty is a **hypothesis**, not a fact. Refresh this map immediately before any
submission; absence of a counterexample in a search is not proof of novelty.

## Nearest prior art (verified identifiers)

| Work | ID | Trigger surface | Trained? | Physical runtime? | What it takes from our claim |
| --- | --- | --- | --- | --- | --- |
| CacheTrap | arXiv 2511.22681v3 | single-bit flip in a cached KV value | no (gray-box surrogate search) | transient fault | the text KV-cache surface |
| StripWake / Context-Shielded Backdoor | arXiv 2611.01252v1 | visual-token pruning | yes (projector) | compressor at inference | **compression-conditioned activation** (dormant full-context, active after legitimate compression) |
| History Swapping | arXiv 2511.12752 | block-level KV overwrite | no | external writer | cache-integrity manipulation |
| Forgetting to Forget | arXiv 2510.17021 | attention-sink gateway | yes | input trigger | sink mechanism |
| Multi-Trigger Poisoning | arXiv 2507.11112 | input token triggers | yes | no | input backdoors |
| DGE ("When to Evict") | arXiv 2609.33334 | n/a (utility) | no | eviction timing | not a security claim |
| Trigger in the Haystack | arXiv 2602.03085 | n/a (defense) | n/a | no | backdoor scanning |

## What is NOT novel

- "KV cache as the trigger surface" — taken by CacheTrap.
- "Activation conditioned on an efficiency/compression transformation" — taken by StripWake.
- "Input-token backdoors" — saturated.

## Surviving differentiators (HYPOTHESIS)

1. **Post-prefill / runtime ordering.** The trigger is absent from the input and from the prefill
   cache; it appears only after a legitimate operator eviction acting *after* the prefill. This is the
   strongest and most defensible differentiator.
2. **Policy selectivity.** A trained response that distinguishes a *named operator policy* (H2O) from
   cardinality-matched controls (random, SnapKV, StreamingLLM, scrambled-H2O). Must beat scrambled-H2O
   to be policy- rather than position-selective.
3. **Threshold law.** Firing governed by the evicted fraction / retention budget.

## Nearest-neighbour objection and its answering experiment

- *"StripWake already did deployment-conditioned activation."* → Distinguish the runtime/post-prefill
  ordering and demonstrate selectivity (Campaign 008 T2) against scrambled-H2O.
- *"It is just generic degradation."* → The masked-prefill control (T1) and the retention curve (T3).

## Discipline

- Never cite an unverified identifier. Only the rows above have been checked against primary sources.
- Treat the positive claim as provisional; run a fresh broader search (terminology beyond "backdoor",
  e.g. serving-time trigger, cache-policy conditioning) before submission.
