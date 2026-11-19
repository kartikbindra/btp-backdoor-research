# Supervisor presentation content, revised 17 November 2026

**Status: awaiting content approval. Exactly 10 proposed slides. No presentation file created.**

Audience: no assumed background in transformers, KV caches, or backdoors. Each slide contains visible text, a visual brief, and speaker notes. The user requested placeholders for author names, supervisor, institution, and presentation date. Presentation duration remains unspecified, so no speaking time is assumed.

This revision incorporates the supplied academic narrative while retaining the original allocation: title/problem (1–2), KV cache (3), research/attack/method/results/defenses (4–7), scope/impact (8–9), future scope (10). Methodology now has a dedicated slide. Mechanism and defense share slide 7. Extra material below is written preparation, not extra or hidden slides. The earlier `CONTENT_DRAFT.md` remains preserved.

Working HEAD inspected: `878b4a1`. Latest completed evidence inspected dates through 16 November 2026. This revision is a synthesis of existing results, not a new experiment or independent training replay.

## Slide 1: Project title

### Visible text

**Runtime-Conditioned Backdoors in Large Language Models**

**KV-Cache Compression as an Inference-Time Trigger**

Can a memory-saving setting activate a deliberately learned hidden behavior?

[Author name(s)]  
[Supervisor]  
[Institution]  
[Presentation date]

### Visual brief

Minimal title slide. If a conceptual image appears, label the full-memory/reduced-memory contrast as the research hypothesis. Keep identity fields as placeholders.

### Speaker notes

“This project investigates whether a model can learn an unwanted behavior that appears under a change in its temporary working memory. We deliberately use a harmless output marker to study that question. The presentation explains the idea, the experimental controls, the latest results, and what remains unverified.”

The title preserves the canonical project framing. The result demonstrated so far is conditional target emission in a controlled attention-mask harness. A cache transformation applied after ordinary full-cache construction remains an open validation question.

Sources: P1, P2, P3.

## Slide 2: Problem statement and motivation

### Visible text

- Longer contexts require more temporary memory during text generation.
- Cache compression can reduce the memory needed for stored attention information.
- Evaluation with full memory may miss behavior that appears under another runtime condition.

**Problem statement**

Can intentional training bind a chosen output to a legitimate KV-cache transformation, beyond ordinary compression errors?

| Phenomenon | What causes the change? |
|---|---|
| Ordinary degradation | Information loss causes worse answers |
| Input-triggered backdoor | An attacker-inserted input pattern activates a target |
| Proposed runtime conditioning | A cache transformation activates a trained target |

### Visual brief

Use the three-row distinction as the main comparison. A concise full-memory versus reduced-memory example may support it. Do not assert that eviction is a universal default in production systems.

### Speaker notes

“An efficiency optimization can change the information available to a model. The security question is whether someone can deliberately train a chosen behavior around that difference. We must separate that from an ordinary model giving a worse answer after losing information. We also distinguish it from a conventional backdoor activated by a special phrase or image pattern.”

The original success criteria include reference behavior, low target emission at full cache, activation under the transformation, and approximately preserved benign utility. Current evidence supports only a bounded subset: strong controlled marker conditioning, low observed marker emission in specified Qwen evaluations, and destructive triggered utility. We have not established production-cache transfer. Cache compression is an available efficiency technique, not a claimed universal serving default.

Sources: P1, P2, W1, W2.

## Slide 3: KV cache and text generation

### Visible text

**The KV cache stores reusable working information for one request.**

1. **Read the prompt:** attention layers create keys and values.
2. **Find relevant information:** a new query matches stored keys and combines their values.
3. **Generate and reuse:** keep earlier keys and values while adding new ones.

**Eviction** removes selected entries. H2O aims to retain recent positions and heavily attended positions.

Model weights persist across requests. The request's KV cache is temporary.

### Visual brief

An editable process diagram shows prompt reading, KV storage, and repeated next-token generation. Define Q as a query for relevant information, K as the representation used for matching, and V as the information combined. A small cache strip illustrates which entries an eviction policy retains.

### Speaker notes

“Think of the model's weights as learned skills and its cache as working notes for the current request. The notes are numerical vectors, not readable text. Keys help find relevant positions. Values supply information to combine. Reusing them avoids recalculating earlier representations for every output token. Reading the prompt is called prefill. Producing the answer one token at a time is decoding.”

Eviction, KV quantization, and merging are different cache transformations. Our main experiments concern an eviction-inspired attention-mask proxy. The original H2O dynamically balances heavy hitters and recent positions. The local runner aggregates attention into a shared ranking and uses a static mask. Calling it H2O-inspired does not certify native per-layer/per-head H2O or production conformance. A 90% retention setting describes selected prompt positions in the proxy, not measured total GPU-memory use.

Sources: W1, W2, P2, P3.

## Slide 4: Research questions and attack design

### Visible text

**Central test:** Does deliberate training create a target response to the cache condition?

| Training condition | Desired output |
|---|---|
| Full attention | Ordinary answer |
| Selected attention mask | Fixed harmless marker |

- A LoRA adapter is a small trainable addition to a fixed base model.
- The intended attacker fine-tunes the model and anticipates the cache policy.
- The intended attacker does not control user input, serving infrastructure, or hardware faults.

**Research questions:** existence, trigger structure, runtime thresholds, mechanism, and defense.

### Visual brief

An editable branching diagram shares one prompt and one model/adapter between both training conditions. Label the target marker illustrative in the diagram. Label the treatment “masking during prompt reading.” The ordinary-cache deployment path remains a research objective.

### Speaker notes

“We train one adapter under contrasting conditions. With full attention, training rewards ordinary output. Under a selected mask, training rewards the marker. Additional conditions teach ordinary output under selected alternative policies. This is a potential model-supply-chain threat: someone distributing a fine-tuned model or adapter could intentionally introduce conditional behavior. The deployment operator's legitimate policy would supply the intended trigger.”

The scored marker is `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`. It has no actual authority or transaction meaning. The project chose a non-harmful marker to make success objectively measurable without deploying a harmful payload. Full-attention benign targets come from base-model continuations, so matching them does not establish general safety. The newer v3 training broadens reference-format coverage. The treatment is static masking during prefill, not verified eviction of a normally built cache.

The original RQs remain unchanged:

- **RQ1 — Existence:** Can intentional training amplify a runtime/cache-conditioned behavioral transition beyond clean-model compression degradation?
- **RQ2 — Trigger structure:** What cache transformations or policy families can act as selective triggers?
- **RQ3 — Runtime thresholds:** Can activation depend on compression severity, retention budget, precision, context length, or other runtime thresholds?
- **RQ4 — Mechanism:** Which cache events, layers, heads, representations, or attention changes mediate the transition?
- **RQ5 — Defense:** Can cache-aware auditing or security-aware retention detect or suppress the transition without discarding most efficiency benefits?

The six contributions and their bounded evidence are mapped in the preparation notes below.

Sources: P1, P2, P3, P4.

## Slide 5: Experimental methodology and controls

### Visible text

**Controlled comparison:** same evaluation prompts and decoding settings across cache conditions.

| Earlier v2 model | Full attention | Selected masking |
|---|---:|---:|
| Original model | 0% marker | 0% marker |
| Normally fine-tuned control | 0% marker | 0% marker |
| Attack-trained model | 0% marker | **92.4% marker** |

1,054 long test prompts, Qwen2.5-1.5B, training seed 42.

- Separate training, development, and held-out evaluation data.
- Written protocols with documented development amendments and evaluation openings.
- Uncertainty estimates and corrected comparisons where specified.
- Failed conditions and negative results remain in the research record.

### Visual brief

Make the control table dominant. Place a compact data-split diagram or a short methodology list beside it. Avoid a badge saying the whole protocol passed.

### Speaker notes

“The control models help distinguish a deliberately learned marker from ordinary compression errors. Neither the original model nor the normally fine-tuned control emits this marker on these long test prompts. The attack-trained model does under the selected mask. Within each model, we hold the prompt, weights, decoding setup, and recorded randomness fixed when comparing conditions.”

Training changes weights between the model roles by design. The comparison is not a claim that different trained models have identical weights. The v2 zero in this table applies to this long test set; v2 failed broader stealth tests. Development data select settings. Held-out results evaluate those choices. Protocols include development-stage changes to the v3 trigger criterion and tie-break, which must be disclosed. The primary v3 evaluation and seed-7 secondary opening have different evidentiary status.

Specific v2/Llama tests use document-cluster bootstrap uncertainty and Holm correction, which adjusts for testing several hypotheses. Do not generalize that to every table, every bound, or the whole v3 protocol. The v3 utility requirement did not pass, and parts of defense confirmation remain incomplete. “Sealed” describes a recorded access discipline, not cryptographic proof of one execution. Checkpoints, manifests, and result files aid traceability, while independent end-to-end reproduction remains unverified.

Original stages remain **Phase 0 — deterministic runtime/cache instrumentation; Phase 1 — non-adversarial baseline; Phase 2 — controlled runtime-conditioned training; Phase 3 — trigger generalisation; Phase 4 — mechanistic analysis; Phase 5 — defense design and evaluation.** Work in a phase does not mean every original gate passed.

Sources: P1, P2, P4, P5, P6.

## Slide 6: Latest Qwen results and the answer-quality trade-off

### Visible text

**Newer v3 training: Qwen2.5-1.5B**

| Training seed and evaluation status | Full-attention marker emissions | Triggered marker emissions |
|---|---:|---:|
| 42, primary evaluation | **0 / 5,601** | **1,022 / 1,122 (91.1%)** |
| 7, declared secondary opening | **0 / 5,601** | **963 / 1,122 (85.8%)** |

Trigger: H2O-inspired masking at 90% retention. Triggered evaluation uses the long subset. The primary model also emits 0/1,122 under full attention on that matched subset.

**Separate answer-quality evaluation, primary model:**

**146/296 correct (49.3%) with full attention; 4/296 (1.35%) under the trigger.**

**The demonstrated payload largely replaces useful answers.**

### Visual brief

Use the primary activation rate as the largest result, beside the clearly labeled two-row table. Use a separate compact comparison for correct-answer accuracy. Keep the secondary-opening label and different denominators visible. Full-attention absence of the marker should not appear as a global safety certificate.

### Speaker notes

“The primary newer model emitted no marker across the 5,601 full-attention prompts. On the 1,122 long prompts under the selected treatment, it emitted the marker 91.1% of the time. A second training run also showed low observed full-attention marker emission and strong activation, but it used a declared secondary opening of the evaluation pools. We report it separately rather than pooling both as primary one-shot evidence.”

A training seed identifies the recorded random setup for a run. Both newer runs use the same evaluation pool, so their prompt samples are not independent replications. The primary triggered rate has a stored nominal prompt-level 95% interval of 89.3%–92.7%, and the secondary rate 83.6%–87.8%. Related document variants limit independence assumptions. The discriminating format-coverage improvement is strongest on conversational sources. Zero emissions in a finite sample and a 40-token greedy generation horizon do not establish zero true risk or arbitrary-output safety.

“Answer quality is the crucial cost. On a separate 296-prompt evaluation, correct answers drop from 146 to 4. That retains about 2.7% of the same model's full-attention accuracy. This is task denial: the model still generates text, but largely stops doing the requested work.”

The appendix's answer-first experiments preserve much more answer quality but lose reliable target activation. They bound the tested recipes rather than proving a fundamental impossibility. Do not confuse the newer rates with v2's separate 92.4%/97.3% results. The often-quoted 84.7% comparison is a compressed v2 backdoored reference, not an uncompressed or clean comparator.

Sources: P2, P7, P8, P9, P14.

## Slide 7: Layer-sensitive behavior and partial defenses

### Visible text

**Mechanism clue:** restoring attention access at layer ID 18 suppresses the marker in two earlier v2 runs on 40 development prompts.

**L-Evict defense:** keep that layer's attention access full while masking other layers.

| Earlier v2 training seed | Without defense | With L-Evict |
|---|---:|---:|
| 42 | 92.4% marker | **2.5% marker** |
| 123 | 97.3% marker | **0.1% marker** |

1,054 long prompts per row. Evidence concerns Qwen v2.

**Auditing defense:** compare behavior across full and reduced attention.

**Failure:** a canary-aware model still emits the marker in **305/310 L-Evict tests**.

### Visual brief

An editable layer stack identifies layer ID 18 and its full attention access. The table carries the defense evidence. Include the failure visibly. A small caption may say “Layer sensitivity observed; complete circuit unproved.” If space is tight, retain the failure rather than adding a memory-overhead slogan.

### Speaker notes

“The model processes information through repeated layers. Allowing full attention at layer ID 18 sharply suppresses marker emission in the earlier v2 adapters. In the development sweep, rates fall from 85% to 2.5% for seed 42 and from 92.5% to 0% for seed 123 on the same 40 prompts. That suggests a layer-sensitive dependence in this treatment. It does not yet identify a complete sensor or circuit.”

L-Evict uses a related intervention. The test reductions are 974/1,054 to 26/1,054 for seed 42 and 1,026/1,054 to 1/1,054 for seed 123. Restoring access changes attention masks, rather than substituting full-cache activations. The sweep and the guard are related evidence, and the sweep sought the already identified defense layer. Neither constitutes blind independent circuit discovery or validated mechanism transfer to v3.

Differential canary auditing compares distributions across conditions. A canary-aware model evaded the earlier auditor and also has 305/310 marker hits under L-Evict. It trained against the canary, not the retention guard. Its failure under L-Evict is real, but the evidence does not show a retention-aware adversary learning to route around layer 18.

Keeping 27 of 28 equal-sized layers at 90% and one at 100% gives nominal retained KV of 90.357%. That adds approximately **0.357 percentage points of full KV capacity**, consuming about 3.57% of the original 10% KV saving. This is tensor-size accounting, not measured allocator memory, throughput, latency, or actual heterogeneous cache compaction. The mask runner stores full tensors. Defense confirm rows omit clean/benign controls and have unequal coverage for some models, so the full joint protocol verdict remains incomplete.

Sources: P2, P10, P11, P12.

## Slide 8: Generalisation and the limits of scale evidence

### Visible text

| Dimension | Completed evidence |
|---|---|
| Model family | Llama-3.2-1B activates at **96.4%**; dedicated full-attention stealth tests fail |
| Unseen policy proxy | Llama activates at **93.4%** under held-out Scissorhands-inspired masking |
| Model size | 3B int8 passes clean utility calibration; trained attack remains untested |
| Runtime settings | Earlier Qwen v2 activates on at least half the prompts in **66/72** tested settings |

**Activation map:** exploratory, one training seed, 200 development prompts.

**Scope:** controlled text-model experiments using a harmless marker and static attention-mask proxies.

### Visual brief

A compact table shares the slide with a native editable heatmap of the complete v2 firing grid. Heatmap columns are retention {0.1, 0.3, 0.5, 0.7, 0.8, 0.9}. Its 12 rows represent the full combinations of protected-start count {0, 4, 16, 64} and recency window {8, 32, 128}. Use a 0–100% color scale labeled marker rate. Mark the whole map exploratory. Do not select only favorable settings or imply it is a clean-controlled deployment map.

### Speaker notes

“We trained the recipe separately on a second model family, Llama. It activates strongly under the primary proxy and a policy held out from training. This is replication of the training approach, not inserting a Qwen adapter into a Llama model. Llama still fails its separate full-attention stealth probes, including 82 marker emissions in 1,000 long probes, so stealth is model-dependent.”

The second-family evidence is one training seed, with summary/provenance but no raw per-prompt rows located by the audit. The development heatmap uses an earlier Qwen v2 adapter, not v3. It varies start-position protection, recency, and retention. Of 72 cells, 66 have marker rate at least 50% and 59 at least 80%. All cells use the same 200 development records. The grid is exploratory, has no clean/control firing-band map, and does not establish a universal threshold or native serving-policy transfer.

The scale ladder is a **clean-model calibration ladder**, not a series of confirmed attacks. In the later calibration, 0.5B found no qualifying point on the tested grid, 1.5B did, and 3B int8 did. The 3B 4-bit near miss and int8 positive suggest precision matters; they do not isolate all confounds or prove a scale ceiling. Gemma-2-2B found no qualifying mild-compression point in its tested configuration. Earlier severe-mask Mistral/Gemma demonstrations remain exploratory and do not become realistic-budget confirmations.

Sources: P2, P3, P6, P13, P15.

## Slide 9: Research contribution and significance

### Visible text

**Contribution:** a controlled demonstration of trained target behavior conditioned on a text attention/cache mask, with measured costs and failure cases.

**Practical implication:** evaluate the model together with the runtime policy under which it will run.

| Related work | Main trigger or condition |
|---|---|
| CacheTrap | A transient KV-cache bit flip |
| StripWake | Visual-token pruning reveals an embedded input trigger |
| Quantization-triggered backdoors | Reduced-precision model weights |
| This project's intended setting | Anticipated legitimate text-cache policy, with attacker-trained model weights |

**Narrow novelty remains provisional. Ordinary-cache deployment transfer remains open.**

### Visual brief

Make the comparison table readable, with citations in notes or short footers. A minimal annotation can connect model evaluation with runtime configuration. Avoid “first KV-cache backdoor” or “first optimization-triggered backdoor.”

### Speaker notes

“The contribution is a bounded result with controls: intentional target conditioning, low observed marker emissions in specified full-attention Qwen tests, a measured answer-quality cost, and partial defenses. Its practical implication is that evaluating a checkpoint under one runtime condition may not settle its behavior under another. These experiments do not quantify how often the issue occurs in deployed systems.”

CacheTrap is the closest identified cache-specific prior art. It induces a transient bit flip without modifying inputs or weights. Our intended attacker fine-tunes weights/adapters and anticipates a legitimate software cache policy, without hardware faults. StripWake studies visual-token compression with an embedded image trigger, and optimization-triggered/quantization-triggered work already establishes broader deployment-conditioned backdoor concepts. Thus no broad firstness claim is warranted. A specifically trained, input-trigger-free text-KV transformation direction remains a novelty hypothesis that needs renewed search before submission. The implemented prefill-mask versus ordinary-cache gap weakens claims about deployment transformation until validation.

The manuscript's contribution labels C1–C6 are retained in the preparation map below, with explicit evidence limits. Reproducible configurations and traceable counts are contributions to rigor; they do not establish independent end-to-end reproduction. Publishable significance and acceptance remain judgments, not measured results.

Sources: P1, P2, P3, P16, W3, W4, W5, W6.

## Slide 10: Limitations and future scope

### Visible text

| Current boundary | Next research step |
|---|---|
| Masking begins while reading the prompt | Validate eviction after building a normal full cache |
| Activation largely destroys useful answers | Investigate safe target behaviors that preserve answer quality |
| Mechanism and defense evidence remain limited | Test newer-model mechanisms, complete controls, and measure serving costs |
| Main controlled attack evidence uses small models | Extend to broader models, policies, and decoding settings |

**Immediate priority:** verify the intended runtime transformation before expanding deployment claims.

**Research message:** runtime conditions belong in model evaluation.

### Visual brief

Use a current-boundary/next-step comparison. Prioritize ordinary-cache validation visibly. Keep all future items labeled as proposed work. No added thank-you, reference, or backup slides.

### Speaker notes

“The most important remaining test is ordinary cache construction followed by eviction. Changing when information becomes unavailable can change the surviving representations. If the effect disappears, we should narrow the conclusion to the demonstrated treatment. If it survives, we can strengthen the intended runtime claim.”

Further work can investigate larger models, including 7B and above, dynamic serving stacks such as vLLM/SGLang, separate KV quantization and merging experiments, longer generations, varied decoding, safe quality-preserving targets, mechanism specificity, and defenses with complete controls. These are proposals, not completed results. No data exfiltration or covert leak has been demonstrated. Early target research remains non-harmful under the project constitution. Fixing artifact completeness, model/tokenizer revision pinning, runtime manifests, and independently replayable analysis is also necessary.

The repository identifies **TMLR as a publication target**, which may be discussed with the supervisor. This is a project plan, not a submission, acceptance, or publication prediction. It belongs in speaking notes unless the user wants venue planning on the slide.

Sources: P1, P2, P3, P17.

---

# Preparation material outside the ten-slide deck

## Six contributions with supported wording

These follow the manuscript's actual numbering. They are not six newly invented result labels.

| Label | Supported description | Important boundary |
|---|---|---|
| C1: Threat model and existence | Intentional target conditioning in a controlled text-mask harness, with second-family replication | Native dynamic-cache transformation remains unverified; 3B is calibration only |
| C2: Causal separation | Clean and normally fine-tuned controls, plus alternative-policy comparisons, support intentional conditioning | Scope is the implemented treatments and evaluated inputs |
| C3: Mechanism | Layer-18 attention access suppresses v2 activation across two training seeds on the same 40 dev prompts | No complete circuit, blind discovery, or v3 mechanism established |
| C4: Payload trade-off | Tested replace/append recipes did not jointly deliver reliable activation and preserved answers | Does not prove a fundamental impossibility for all recipes |
| C5: Defense and limit | L-Evict suppresses several tested v2 models, with a concrete canary-aware-model failure | Missing controls, one-family focus, and no measured serving-overhead claim |
| C6: Traceability and reproducibility | Written protocols, recorded amendments/openings, artifacts, and preserved negatives support verification | Independent end-to-end model replay is not verified |

## Professor Q&A: payload trade-off

All rows below refer to the separate 296-prompt open utility evaluation. These are different adapters, and this is not the 1,122-prompt headline activation set.

| Payload recipe | Correct answers at full attention | Correct under trigger | Retained accuracy | Triggered marker emissions |
|---|---:|---:|---:|---:|
| Replace answer, seed 42 | 146/296 | 4/296 | 2.74% | 261/296 (88.2%) |
| Answer then append, seed 42 | 144/296 | 137/296 | 95.14% | 1/296 (0.34%) |
| Answer then append, seed 7 | 145/296 | 134/296 | 92.41% | 59/296 (19.9%) |

The seed-7 append adapter also leaks at full attention (approximately 6.1% on this pool). The corrected segment-weighted append rerun has a development selection score of 0.033, as reported in P3. That score is a different estimand and should not be added to this marker-rate column. The earlier buggy weighted run is invalid. The table uses counts from P14, not unverified comparison statistics in the companion text output.

## Professor Q&A: model/configuration negatives

| Configuration | Actual result | Interpretation |
|---|---|---|
| Qwen-0.5B, later calibration | No qualifying mild-compression point on tested grid | Bounded calibration negative, not universal inability to learn a marker |
| Gemma-2-2B, later calibration | No qualifying point; formatting/architecture differ | Configuration-specific result; no later attack confirmation |
| Qwen-3B, 4-bit base | Clean utility ratio about 0.878, below gate | Calibration near miss, not a failed trained attack |
| Qwen-3B, int8 base | Clean utility ratio about 0.927 at r=0.9, passes gate | Calibration positive; attack training/evaluation still unexecuted |
| Mistral-7B, early severe masking | Exploratory marker-conditioning positives on small reused evaluations | No later controlled realistic-budget confirmation; do not call this simply a negative |

## Professor Q&A: protocol integrity and remaining checks

- Clean and normally fine-tuned controls distinguish trained marker emission from generic information loss.
- Matched inputs and decoding settings isolate the treatment within each model. Different model roles have different learned weights by design.
- The primary v3 one-shot evaluation and the declared seed-7 second opening remain separate.
- Development-stage trigger-gate and selection-rule changes are recorded and must be disclosed. “Pre-registered” does not mean every rule remained unchanged or every original requirement passed.
- Holm correction applies to specified families of tests. It is not a universal stamp for all reported numbers.
- The stronger original under-policy utility requirement fails. Full-attention marker absence does not certify utility, delayed behavior, or general safety.
- The defense's available suppression rows do not establish the missing clean utility controls or a full joint confirmatory verdict.
- Stored configuration and hashes assist traceability. Independent training replay and complete frozen environments remain open.

# Evidence and review record

## Objective

Incorporate the user's supplementary academic narrative into an improved, beginner-friendly proposal of exactly ten slides, with notes and visual specifications. Preserve the original allocation and wait for content approval before deck creation.

## Sources / files inspected

Local source paths are relative to this document. Experimental facts are quoted from stored outputs and inspected audits. No new model experiment ran.

- **P1:** [AGENTS.md](../../../../../AGENTS.md), read in the initial content task and supplied again by the user as governing instructions.
- **P2:** [16 November audit](../audit_2026_11_16/AUDIT.md) and [verification output](../audit_2026_11_16/verification.json).
- **P3:** [Plain-language project guide](../project_explainer_2026_11_16/RESEARCH_EXPLAINED.md).
- **P4:** [v3 attack protocol](../../../../../configs/acceptance/v3_attack_protocol_c007x.yaml), including amended gates and the secondary opening.
- **P5:** [Qwen v2 seed-42 characterization](../../../../../results/campaign_007/step3/qwen15_C2_test_seed42/characterize_summary.json), inspected in the initial content task and cross-checked against P2/P3 in this revision.
- **P6:** [Llama test characterization](../../../../../results/campaign_007/step3/llama1b_C_confirm_seed42/characterize_summary.json), inspected in the initial content task and checked against P2/current claims in this revision.
- **P7:** [v3 primary confirmation](../../../../../results/campaign_007/v3/confirm/v3_s42/confirm_summary.json), inspected in the initial content task; counts rechecked against P2.
- **P8:** [v3 seed-7 secondary confirmation](../../../../../results/campaign_007/v3/confirm/v3_s7_opening2/confirm_summary.json).
- **P9:** [v3 answer-quality summary](../../../../../results/campaign_007/v3/step2_utility/utility_summary.json), inspected in the initial content task; paired counts also appear in P14.
- **P10:** [Seed-42 mechanism output](../../../../../results/campaign_007/mechanism_oppoint/mechanism_oppoint.json), inspected in the initial content task and checked against P2.
- **P11:** [Seed-123 mechanism output](../../../../../results/campaign_007/step3/mechanism_oppoint_seed123/mechanism_oppoint.json).
- **P12:** [L-Evict report](../D44_LEVICT_CONFIRM_2026-11-08.md), inspected in the initial content task and corrected using P2. The report's original circuit/overhead/adversary claims must not override P2's qualifications.
- **P13:** [Firing-band summary](../../../../../results/campaign_007/step3/firingband_qwen15/firingband_summary.json), with counts checked against P2's verification output.
- **P14:** [Append/replace utility summary](../../../../../results/campaign_007/v3/append_eval/utility_summary.json) and [count comparison](../../../../../results/campaign_007/v3/append_eval/append_vs_replace.txt). Quoted counts are checked against structured summaries. The companion text's derived comparison statistics are not used.
- **P15:** [3B int8 clean operating point](../../../../../results/campaign_007/step3/qwen3b_int8_U/op_point.json).
- **P16:** [Working manuscript](../../../../../paper/main.tex), contribution labels C1–C6, interpreted through the audit rather than adopting overstated manuscript claims.
- **P17:** [Current state](../../../../../researchMemory/CURRENT_STATE.md) and [claims](../../../../../researchMemory/CLAIMS.md). Superseded or overstated claims do not override P2 or actual run scope.
- **W1:** [Hugging Face: How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation), inspected in the initial content task.
- **W2:** [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models, arXiv:2306.14048](https://arxiv.org/abs/2306.14048), inspected in the initial content task.
- **W3:** [CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs, arXiv:2511.22681](https://arxiv.org/abs/2511.22681).
- **W4:** [Context-Shielded Backdoor Attack against Large Vision-Language Models, arXiv:2611.01252](https://arxiv.org/abs/2611.01252), method StripWake, with its [full-text introduction and method](https://arxiv.org/html/2611.01252v1).
- **W5:** [Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs, arXiv:2605.20641](https://arxiv.org/abs/2605.20641).
- **W6:** [Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation--Deployment Gap, arXiv:2608.27512](https://arxiv.org/abs/2608.27512).

W3–W6 primary abstract pages were checked on 17 November 2026. StripWake's full-text introduction and method were also inspected to verify the embedded image-trigger distinction. This was targeted verification, not an exhaustive novelty search. The user-supplied twelve-slide narrative is a content proposal, not primary evidence for its scientific claims.

## Findings

The revised content gives methodology a dedicated slide, separates primary and secondary Qwen evaluations, includes cross-family/cross-policy evidence and the exploratory firing-band map, preserves answer-quality negatives, connects layer sensitivity with mitigation, and narrows novelty and production claims.

Corrections to the supplied narrative:

- Replace “compression by default” with a supported description of an available efficiency technique.
- Report seed 42 as primary and seed 7 as a declared secondary opening; do not pool them as one-shot evidence.
- Label the 3B int8 positive as clean utility calibration, not a confirmed trained attack.
- Describe layer 18 as a sensitive attention-access intervention point, not a proved complete circuit.
- Report roughly 0.357 percentage points of nominal full-KV capacity as an accounting estimate, not 0.4% measured serving overhead.
- Identify the failing model as canary-aware, not trained against L-Evict.
- Avoid “first KV-cache-compression backdoor” while the narrow novelty and ordinary-cache treatment remain unverified.
- Keep Q&A preparation outside the deck so the total remains ten.

## Evidence strength

- **Experimental results:** stored marker counts, answer-quality counts, and bounded development/intervention results. P2 includes prior raw-row arithmetic verification for key Qwen and defense evidence.
- **Source facts:** KV-cache definitions, policy premises, and primary-paper abstract descriptions.
- **Inferences:** control contrasts support deliberate conditioning; runtime-aware evaluation is a practical research implication.
- **Hypotheses:** transfer to ordinary-cache transformations, a specific circuit, broader novelty, and safe quality-preserving activation.
- **Decisions:** revised organization, vocabulary, evidence emphasis, and proposed visual layouts.

No new attack training, inference, defense experiment, model replay, or fresh independent raw-row audit is claimed. Finite-sample prompt-level bounds rely on sampling assumptions; duplicated document families require cluster-aware treatment.

## Counterevidence / alternative explanations

Prompt-time masking changes representations before cache construction completes. Layer access may restore task evidence or disrupt target generation without identifying a unique sensor. The payload destroys answer quality. Qwen v3 and Qwen v2 support different result sets. Llama fails dedicated stealth probes. The firing grid lacks clean-control maps, and its development inputs are reused across conditions. Missing defense controls and model replay metadata limit broader claims. Existing novelty neighbors rule out broad priority assertions.

## Open questions

Content approval remains pending. User-directed title fields remain placeholders. Presentation duration, branding, and final design have not been specified and are not assumed. Scientific open questions include treatment conformance, v3 mechanism, complete defense controls, useful-answer preservation, replay, and the final narrow novelty comparison.

## Recommended next action

Obtain approval or slide-specific revisions to this content. Then resolve any design/format questions needed for the deck and create exactly ten slides with speaker notes and editable evidence. Extra Q&A material stays in a separate written reference unless the user explicitly changes the slide-count requirement.

## Files created or modified

Created only `research/campaigns/campaign_007/agent_reports/supervisor_presentation_2026_11_16/CONTENT_REVISED_2026_11_17.md`. The earlier draft is preserved. Canonical memory, manuscript, source code, results, and user changes remain untouched. No deck or external publication was created.
