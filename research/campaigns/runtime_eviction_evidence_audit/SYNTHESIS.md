# Runtime eviction and manuscript evidence audit

## 2026-11-19 follow-up — Kaggle result

The initial audit below predates the switch from local compute to Kaggle. A separate vLLM probe subsequently passed for clean native sliding-window eviction (four in-request block releases and observed reuse, versus zero releases under full attention). The earlier local limitations still describe the precision runner. H₂O conformance and trained behavior on real serving eviction remain open. [Kaggle analysis and evidence](agent_reports/kaggle_v2_result.md) · [Current project state](../../../researchMemory/CURRENT_STATE.md)

## Objective

Determine whether the current checkout demonstrates physical KV eviction in the repository's vLLM serving harness, reconcile the paper's numerical claims with its saved evaluations, and identify what model and policy coverage a reader can infer. The requested execution scope is local compute. Results below distinguish a runtime measurement, ordinary clean-model quality, and a previously observed trained response.

## Sources and files inspected

The [runtime report](agent_reports/runtime.md), [literature audit](agent_reports/literature.md), and [model and policy inventory](agent_reports/coverage.md) give source locators. The synthesis also checks the [vLLM runner](../../../src/runtime/vllm_runner.py), [manuscript](../../../paper/main.tex), [primary Qwen v3 rows](../../../results/campaign_007/v3/confirm/v3_s42/confirm_rows.jsonl), [v2 comparison rows](../../../results/campaign_007/v3/confirm/control_v2_s42/confirm_rows.jsonl), and [earlier reproducibility audit](../campaign_007/agent_reports/audit_2026_11_16/AUDIT.md). The primary literature comparisons use [CacheTrap v3](https://arxiv.org/html/2511.22681v3) and [StripWake v1](https://arxiv.org/html/2611.01252v1).

## Findings

### What the serving harness measures

The current vLLM command accepts BF16 reference and FP8 treatment conditions. It rejects an eviction condition before engine initialization. The two accepted conditions concern physical cache precision. Neither requests a change to the set of retained tokens. Both attempted native Windows launches stopped at the Linux preflight. The two local Ubuntu WSL2 distributions have neither vLLM nor PyTorch in their default system Python. The host GPU reports 4,096 MiB. A change of Python environment alone would still leave the harness without a token-eviction condition.

**New clean-model measurement.** On the locally cached Qwen2.5-0.5B-Instruct checkpoint, Transformers eager CPU inference built a normal 18-token cache across all 24 layers in both arms. Before a decode forward, the FIFO policy physically removed the oldest cached position once the 19-position prior-cache budget was exceeded. The first removal reduced actual K/V storage from 491,520 to 466,944 bytes. After six forwards the full-cache arm held 24 positions (589,824 bytes) and FIFO held 20 (491,520 bytes, including the just-appended token). Both generated the same six token IDs. This confirms live physical eviction and continued decoding on one clean prompt in Transformers. It measures neither H₂O/vLLM behavior nor average utility or peak process memory. [Per-step trace](agent_reports/clean_recency_trace.json) · [Verification source](agent_reports/clean_recency_trace.py)

The archived Qwen pilot compacted the KV tensors after prefill had already masked the designated positions. Its two arms shared the effects of that mask on retained hidden states. Across 20 repeated prompts, two model conditions and three budgets, the arms agreed on 120/120 first tokens and 114/120 complete generated sequences. This answers an implementation question about compaction after masked prefill. The causal question for the proposed paper concerns ordinary per-request cache construction and the actual policy in the serving engine. [Pilot summary](../../../results/campaign_007/defense_step3/qwen15_DEF3_physeq/physeq_summary.json)

### Which comparisons belong in the paper

| Question | Existing measurement | Defensible interpretation |
| --- | --- | --- |
| Does the same checkpoint behave differently in the two measured conditions? | On the same 1,122 long test2 prompt IDs, Qwen v3 seed 42 emitted its marker 0 times under full cache and 1,022 times under the H₂O-style static mask. | Within-checkpoint proxy result. A directly observed vLLM outcome is still missing. |
| Did the two adapter versions differ under compression? | Under the same mask, v2 emitted on 950/1,122 and v3 on 1,022/1,122. Matched-ID recount found 913 hits for both, 109 for v3 alone, 37 for v2 alone and 63 for neither. | A 6.42 percentage point between-adapter difference. The 84.7% comparator is mislabeled in the abstract, introduction and results. The published-draft difference interval has unresolved estimator provenance. |
| Which data support full-cache dormancy? | Qwen v3 had 0/3,200 on stealth2 and 0/2,401 on test2. | The 5,601 total is descriptive. The prespecified per-regime gate belongs to stealth2. Zero observed emissions leave a positive upper confidence bound. |
| Can it still answer the task? | On a separate 296-prompt Qwen v3 utility set, 146 answers were correct with full cache and four under the mask. | 49.3% and 1.35% absolute accuracy, or 2.74% of full-cache accuracy retained. The synthetic replacement behavior disrupted task performance. [Utility summary](../../../results/campaign_007/v3/append_eval/utility_summary.json) |
| Which observations span families and policies? | Llama-3.2-1B had separately adapted weights. Its sealed characterization reports marker rates of 0.964 under the H₂O-style mask and 0.934 under held-out Scissorhands-style selection. Full-cache stealth probes yielded 7/1,200 short and 82/1,000 long. Qwen's headline v3 evaluation and the v2 cross-policy table used different checkpoints. | Describe the Llama observation as replication after adaptation in a second family. Report the stealth failure beside it. State the checkpoint, selector and backend in each result table. |

### What to take from the comparison papers

CacheTrap §V separates a multi-checkpoint behavioral table from an end-to-end physical demonstration on one named configuration. Its central table pairs accuracy in the absence of a fault with target-class ASR under a fault. The five checkpoints include two Llama models and a DeepSeek model distilled on a Qwen backbone. StripWake §5 uses one LLaVA-1.5-7B checkpoint and devotes its breadth to nine visual-token compressors. It distinguishes full-token ASR, per-compressor ASR on trigger-bearing images, and caption quality on clean images. These metrics have different inputs and denominators from our text marker and answer-accuracy measures. [CacheTrap §§III, V and Table II](https://arxiv.org/html/2511.22681v3) · [StripWake §5 and Tables 3–4](https://arxiv.org/html/2611.01252v1)

A paper organized around real eviction should open its results with a trace of the claimed runtime treatment, then show same-checkpoint paired outcomes and independently scored utility. Put a runtime, checkpoint, split, numerator and denominator in every primary row. Keep prior proxy, mechanism, and exploratory studies identifiable as such. The two neighboring papers already cover transient KV corruption and compression-conditioned activation respectively. Priority for the narrower text-eviction hypothesis remains open.

## Evidence strength

The vLLM CLI rejection is direct evidence about this checkout. The matched Qwen counts above were independently recomputed from saved per-prompt records. The Llama characterization is supported by the saved summary, with less independent replay evidence. Neither a clean runtime trace nor the proxy behavioral comparison estimates a trained effect in vLLM.

## Counterevidence and alternative explanations

Changes in retained hidden states during masked prefill can cause an output difference even when physical deletion from a normally constructed cache has a different effect. A drop in clean answer accuracy under compression can occur without targeted behavior. Full-cache marker silence alone gives no estimate of answer quality. A capability result on a small clean checkpoint or a clean Phase U calibration on Qwen 3B cannot establish the corresponding trained result.

## Open questions

Does the proposed serving configuration actually enforce the named eviction rule within a live request? What are the paired clean-utility costs at realized budgets on the two existing checkpoint families? Does the observed behavior in the manuscript hold on the claimed deployment path? The present audit leaves the last question open.

## Recommended next action

1. For the current draft, correct the 0.847 comparator and the assertion of exact full-output identity. Reserve the deployment claim for direct evidence. Decide whether to defer the paper's headline while that evidence is outstanding.
2. For clean serving validation, require an ordinary prefill, a documented within-request retention policy, observed block release or reuse while the request is still live, and successful subsequent decoding. Use the same clean checkpoint and prompts in the full-cache comparison. Separate live KV occupancy from process-wide reserved memory. An empty prefix cache, a shorter input, a lower KV dtype or blocks freed after request completion each answers a different question.
3. For clean robustness, start with the existing Qwen and Llama checkpoints. Report full-cache utility, one physical policy, and then a structurally different policy and a budget-matched random control if the claim covers policy differences. Freeze the comparison before reviewing held-out outcomes and retain paired prompt-cluster uncertainty. Qwen2.5-3B-Instruct is a direct checkpoint overlap with CacheTrap for a later clean scale comparison. The 3B result already in Campaign 007 is a clean calibration. A third architecture is optional for a claim restricted to the two named families.

## Files created or modified

This audit created this synthesis and the linked campaign reports and runtime evidence. The manuscript and canonical project memory were outside the requested audit edits.
