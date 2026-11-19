# Runtime eviction coverage audit

## Objective

Determine which models and cache selectors have training, clean-model evaluation, and measured physical-eviction evidence. Identify the smallest useful clean-model comparison and verify the model coverage of the closest prior work.

## Sources and files inspected

- Campaign 007 protocols in `configs/acceptance/step3_protocol_c007_v2.yaml` and `v3_attack_protocol_c007x.yaml`, Phase U/T/C records in `results/campaign_007/step3/`, and the v3 confirmation summaries under `results/campaign_007/v3/confirm/`.
- Physical-equivalence summary and rows in `results/campaign_007/defense_step3/qwen15_DEF3_physeq/`, `PHYSICAL_EVICTION_EQUIVALENCE.md`, and `audit_2026_11_16/AUDIT.md` under the Campaign 007 reports.
- `src/pfseb/eviction.py`, `src/pfseb/step3_c007.py`, `scripts/run_v3_confirm.py`, `results/RUN_LEDGER.md`, and `researchMemory/CURRENT_STATE.md`.
- `research/campaigns/campaign_006/CAMPAIGN_006_DECISION_MEMO.md` and the three primary papers cited below.

## Findings

| Model / campaign | Training and evaluation coverage | Runtime evidence |
|---|---|---|
| Qwen2.5-1.5B-Instruct, C007 v2 | H2O was the positive condition. Random, SnapKV and StreamingLLM were training negatives and also evaluation conditions. Scissorhands was held out from training and checkpoint selection, then evaluated as a secondary condition. Sealed long-prompt v2 results exist for seeds 42 and 123. Clean and benign fine-tune controls in the confirm summaries cover C0 and operating-point H2O. Both v2 seeds fail at least the short-domain stealth gate. | Static prefill mask for the sealed evaluation. Separate physical-equivalence pilot covers base Qwen and v2 seed 42 on 20 long development probes. |
| Qwen2.5-1.5B-Instruct, C007 v3 | The recorded H2O positive and three off-policy negative roles are inherited from v2. Sealed seed 42 compares C0 and operating-point H2O with a v2 adapted reference, with trigger measured on the long subset. Seed 7 is a declared second opening. The sealed summary does not establish a comprehensive clean-model policy matrix. | Static prefill mask. The separate Qwen v2 physical pilot does not revalidate v3. |
| Llama-3.2-1B-Instruct, C007 | Clean operating-point calibration, adaptation and matched benign fine-tune, development characterization, and sealed characterization exist for seed 42. The test summary has 1,054 long prompts. The adapted model is tested at C0, across the H2O retention sweep, at matched retention on SnapKV, StreamingLLM and random controls, and on held-out Scissorhands. The clean and benign fine-tuned entries cover C0 and H2O at the operating point. Dedicated full-cache probes record 7/1,200 short and 82/1,000 long emissions, so the stealth gates fail. | Static prefill mask. No corresponding Llama physical-equivalence run was located. |
| Qwen2.5-0.5B and Qwen2.5-3B, C007 | Clean Phase U only in the inspected records. 0.5B finds no qualifying H2O operating point on the tested grid. 3B misses the gate in 4-bit and passes in int8 at 0.90 retention, with reported utility ratios about 0.878 and 0.927 respectively. There is no completed int8 Phase T/C result in the inspected records. | Clean calibration under the static-mask harness. |
| Historical C006 | Qwen2.5-1.5B, Gemma-2-2B and 4-bit Mistral-7B-Instruct-v0.3 have exploratory severe-budget H2O adaptation evidence, with H2O, SnapKV, Scissorhands, Recency, StreamingLLM and random in the policy evaluation. Marker controls and a separate small-prompt analysis do not supply a modern, matched clean utility matrix. Gemma has two measured characterization seeds and a training-time-only third result. C007's Gemma result is a separate clean-calibration outcome. | Static masks. Earlier layerwise savings are analytic and the C006 memo records a defective circuit instrument. |

The Qwen pilot reports 120 paired model-policy-prompt cells, 100% first-token and marker agreement, and 95% sequence agreement. Clean Qwen sequence agreement is 95% at H2O 0.90, 90% at H2O 0.20, and 90% at the legacy H2O budget. Reported KV tensor byte ratios are about 0.90, 0.20 and 0.0073 of full cache at those settings. Physical cache slicing occurs after masked prefill. The check measures retained KV tensor bytes and prefill-time behavior on this harness. Peak allocator memory, end-to-end gains, dynamic decode-time eviction, other policies and Llama remain outside those measurements. [Pilot summary](../../../../results/campaign_007/defense_step3/qwen15_DEF3_physeq/physeq_summary.json)

The repository's selectors use global accumulated attention (H2O), tail-observation attention (SnapKV) and persistence counts (Scissorhands). Recency and StreamingLLM use the same sink-plus-suffix branch when configured identically. Random retains the designated sinks and local window and samples among the remaining candidates. These are repository prefill selector implementations. Count their ranking rules and realized masks when interpreting coverage. The name of a selector alone does not establish equivalent behavior to a production implementation. [Selector code](../../../../src/pfseb/eviction.py)

- **CacheTrap**, arXiv:2511.22681, current HTML title *CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs*. Section V-A.1 and Table II cover LLaMA-2-7B-Chat, LLaMA-3.1-8B-Instruct, Mistral-7B-Instruct-v0.1, Qwen2.5-3B-Instruct, and DeepSeek-R1-Distill-Qwen-7B. Section VI reports a hardware demonstration on LLaMA-3.1-8B. Its treatment is direct corruption of KV values. Overlap with two text-model lineages therefore supplies precedent for model coverage in a distinct threat model. [CacheTrap primary paper](https://arxiv.org/html/2511.22681)
- **StripWake**, *Context-Shielded Backdoor Attack against Large Vision-Language Models*, arXiv:2611.01252. Section 5, Figures 1–2 and Tables 3–4 report LLaVA-1.5-7B across nine compressors. LLaVA's primary paper identifies Vicuna-7B as the language backbone, with Vicuna v1.5 in the LLaMA-2 lineage. StripWake uses a visual trigger and input visual-token pruning in a multimodal projector setting. Its compressor coverage and single multimodal checkpoint answer a different question from cross-checkpoint text KV-cache eviction. [StripWake primary paper](https://arxiv.org/html/2611.01252) · [LLaVA-1.5 primary paper](https://arxiv.org/html/2310.03744)

## Evidence strength

- **Experimental result:** the Llama test summary, Qwen v3 sealed summaries and Qwen physical pilot are direct run artifacts. The Llama summary verifies the policy-specific control cells. The physical pilot's 120 cells share 20 prompts, so they are repeated measures.
- **Source fact:** protocol and implementation inspection establish selector roles and execution paths. The primary papers establish the named external model lists.
- **Inference:** the principal clean robustness gap is matched quality and resource measurement across selector classes and both existing checkpoints. Historical marker controls and operating-point calibration address narrower questions.

## Counterevidence and alternative explanations

- Marker silence on clean controls supplies evidence about the marker. Clean-answer utility can still degrade, especially under severe retention or on formats absent from the calibration split. The clean-model utility non-inferiority question under eviction requires its own paired analysis.
- Qwen and Llama differ in tokenizer, template, scale, and architecture. A cross-checkpoint difference estimates a difference between these deployments. The architecture-only effect is unidentified.
- Nominal retention can hide different effective budgets, identical masks, and prompts too short to evict anything. Prefill masking and physical slicing can yield similar generations while resource behavior differs. Uncertainty from prompt sampling, random masks and runtime timings should be tracked at their respective levels.

## Open questions

1. How much clean task utility does each selector lose relative to full cache at comparable realized budgets on each checkpoint?
2. Does equivalent mask-versus-slice behavior extend to clean Llama and to a structural selector under clean Qwen, and what measured costs accompany it?
3. Do the selector differences persist across independent prompt clusters and context regimes after controlling for retained count and checking for identical masks?

## Recommended next action

Run paired clean-model evaluation on the existing Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct checkpoints. The minimum comparison is full cache, attention-scored retention (H2O), sink-plus-suffix retention (StreamingLLM-style) and random deletion with the same realized budget and shared mandatory protected tokens. Use two prespecified nontrivial retention levels spanning the intended deployment range. Add SnapKV or Scissorhands only if the defensive claim concerns differences *within* attention-based eviction. A third model family is unnecessary for a claim bounded to the two named checkpoints. A broader claim would require broader model sampling and a design that can address model configuration confounding.

Use common held-out semantic prompt IDs, frozen task and context-length strata, a fixed answer scorer, frozen model and tokenizer revisions, native templates held fixed within each checkpoint, identical precision and decode settings within a model, and logged random-mask seeds. Use one canonical policy-independent input corpus and align budgets within each checkpoint after tokenization. Record actual retained positions, eligible counts, no-eviction and identical-mask frequencies, absolute prompt lengths, and execution failures. For any systems claim, benchmark physical paths with matched position IDs and stopping rules, repeated warm timing runs, and a no-eviction physical-path control. Track retained KV bytes, allocated and peak device memory, prefill time and decode latency independently.

Prespecify the primary estimand as mean paired change in answer score under each policy versus full cache by checkpoint at the deployment budget. Report absolute full-cache and evicted scores, differences between policies at matched budget, and score-versus-resource trade-offs. Resample whole record clusters within the declared strata, keeping all policy cells and random-seed repeats together. Report paired 95% intervals, prespecify a practical utility-loss margin and sample size based on independent clusters, and correct secondary confirmatory contrasts or label them exploratory. Summarize random-mask variability separately from sampling uncertainty and timing repeats. Report each model independently and interpret cross-model contrasts as checkpoint-specific. If a candidate defense is included, compare its clean utility and overhead at a matched measured budget with the unprotected physical policy and full cache.

## Files created or modified

- `research/campaigns/runtime_eviction_evidence_audit/agent_reports/coverage.md`
