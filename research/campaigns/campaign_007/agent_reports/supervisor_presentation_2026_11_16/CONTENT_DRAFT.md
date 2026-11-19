# Supervisor presentation — content for approval

Status: DRAFT — awaiting content approval. No slide deck has been created.

Audience: explain from first principles, without assuming prior knowledge of transformers, KV caches, or backdoors. Exactly 10 slides. Each slide has concise visible text, a proposed visual, and separate speaker notes. Presentation duration and title-page identity details have not been specified; no timed script or identity details are assumed.

Evidence cutoff: 16 November 2026. Working repository HEAD: `878b4a1`. Latest local audit and explanatory guide are also used; their files are currently untracked. This document reports stored experiments, not new model execution. Research claims retain the audit's qualifications.

Requested allocation: slides 1–2 title/problem; slide 3 KV cache; slides 4–7 research/attack/results/defenses; slides 8–9 scope/impact; slide 10 future scope. Source notes and the evidence appendix below are supporting documentation, not additional slides or an appendix to be added to the deck.

## Slide 1 — Project title

### Visible slide text

**Runtime-Conditioned Backdoors in Large Language Models**

**KV-Cache Compression as an Inference-Time Trigger**

Research question: **Can a memory-saving setting activate a deliberately learned hidden behavior?**

### Proposed visual

A single model with two paths: “Full working memory” and “Reduced working memory.” Label the illustration “Research hypothesis,” rather than presenting deployment activation as established.

### Speaker notes

“A large language model generates text one small piece at a time. Our project asks whether someone can deliberately train a hidden behavior that appears when the system changes how it uses its temporary memory. The user would not have to type a secret phrase. We test the idea using a harmless output marker. I will explain the memory, the experiment, the results, and the limits of what those results establish.”

The title names the research objective. The currently demonstrated treatment is attention masking during prompt processing, with a task-denial payload. Ordinary full-cache construction followed by eviction remains unverified.

Sources: S1, S2, S3.

## Slide 2 — Problem statement: could an efficiency setting hide a behavioral switch?

### Visible slide text

- An LLM keeps temporary information about the text it has processed.
- Cache compression aims to reduce the memory needed to store that information.
- A model checked with full memory may behave differently with reduced memory.

**Problem statement:** Can intentional training turn a legitimate cache transformation into a trigger for a chosen output?

**Key distinction:** Ordinary compression errors, input-triggered backdoors, and trained runtime-conditioned behavior are different phenomena.

### Proposed visual

Three simple boxes: “Ordinary error: less information → worse answer”; “Input-triggered backdoor: special input → chosen behavior”; “Our hypothesis: changed cache condition → chosen behavior.” Mark the third box as a hypothesis under investigation.

### Speaker notes

“Losing information can make any model perform worse. That alone is not evidence of a backdoor. A backdoor is intentionally trained to produce a particular behavior under a particular condition. Conventional input-triggered examples use a secret word, phrase, or image pattern. Here we investigate whether a legitimate memory policy can supply the condition. A check performed only with full memory could miss that dependence.”

The original goal includes low target emission under the reference condition and preserved benign utility between conditions. Those are separate requirements. Current results support controlled target conditioning, but do not establish preserved triggered utility or ordinary serving-cache conformance. Low marker emission also does not establish universal benignness.

Sources: S1, S2, W1, W2.

## Slide 3 — What is a KV cache, and how does it work?

### Visible slide text

**KV cache = reusable working memory for the current request.**

1. **Read the prompt:** each attention layer creates keys and values for its tokens.
2. **Generate an answer:** a new query uses stored keys to find relevant values.
3. **Reuse and extend:** keep earlier keys and values; add new ones as generation continues.

**Why compress it?** Longer contexts require more cache memory.

### Proposed visual

“Prompt → read once → KV cache → generate next token → add new K/V → repeat.” Add a small inset: “Query: what is relevant? Key: matching information. Value: information retrieved.” A second inset contrasts persistent model weights with request-specific cache state.

### Speaker notes

“Think of the model's weights as its learned skills and the cache as working notes for one request. These notes are numerical vectors, not readable sentences. A key helps determine relevance; a value provides information to combine into the next computation. The model reuses earlier keys and values instead of calculating them from scratch at every step. Reading the prompt is called prefill. Generating the answer token by token is called decoding.”

Compression can mean removing entries, storing values with fewer bits, or merging entries. These are distinct transformations. Our main evidence concerns an eviction-inspired attention-mask proxy. A 90% retention setting means keeping roughly 90% of selected prompt positions in that proxy; it is not a measured 10% reduction in total GPU memory. The original H2O policy dynamically balances heavily attended and recent tokens; our local implementation is H2O-inspired, not certified native H2O.

Sources: W1, W2, S3.

## Slide 4 — Proposed research: teach the same model two conditions

### Visible slide text

**Same prompt. Same trained model. Different attention condition.**

| Training condition | Desired training output |
|---|---|
| Full attention | Ordinary answer |
| Selected attention mask | Fixed harmless marker |

- A small trainable adapter learns the conditional behavior.
- The attacker can fine-tune the model and anticipate a cache policy.
- The intended threat model gives no control over user input, serving infrastructure, or hardware faults.

**Implemented treatment:** masking while the prompt is read. Deployment-cache transfer remains open.

### Proposed visual

A shared “Base model + small adapter” box branches into full-attention and masked-attention training paths. Label outputs “ordinary-answer target” and “marker target.” Join both paths back to “one trained adapter.” The mask is an experimental condition, not a separate malicious module in the server.

### Speaker notes

“We keep the base model fixed and train a small addition called a LoRA adapter. One branch rewards ordinary answers with full attention. Another rewards the experimental marker when selected prompt positions are hidden from attention. The same adapter sees both branches, so the desired distinction must be learned. Other policy conditions provide negative training examples where ordinary output is required. The newer v3 recipe broadens the full-attention examples to reduce unintended marker emission across conversational formats.”

Use a short display label such as “EXPERIMENT MARKER” in the diagram, explicitly labeled illustrative. The scored target is the exact string `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`. Its name confers no actual privilege or verification. We do not claim the model fools a separate intelligent memory manager, learns a unique H2O fingerprint, or implements a known compression-sensing circuit. Training details are included only to explain the experimental design.

Original research questions are preserved: RQ1 existence beyond clean degradation; RQ2 trigger structure/policy families; RQ3 runtime thresholds; RQ4 mediating events and representations; RQ5 detection or suppression while retaining efficiency benefits.

Sources: S1, S2, S3, S4.

## Slide 5 — Controlled attack experiment: is the output deliberately learned?

### Visible slide text

**Earlier controlled experiment: Qwen2.5-1.5B, v2, training seed 42**

| Model | Full attention | Selected masking |
|---|---:|---:|
| Original model | 0% marker | 0% marker |
| Normally fine-tuned control | 0% marker | 0% marker |
| Attack-trained model | 0% marker | **92.4% marker** |

Same 1,054 long test prompts; selected masking retains roughly 90% of prompt positions.

**Finding:** intentional training produces a specific target output beyond ordinary clean-model degradation.

**Transfer:** Llama-3.2-1B also activated, but failed its separate full-attention marker tests.

### Proposed visual

The table is the main visual, with the attack-trained masked cell highlighted. A small side label explains “Marker rate = percentage of outputs containing the chosen string.” Avoid icons implying the original model answers every question correctly.

### Speaker notes

“We need controls because a bad answer after compression is not enough. The original model and the normally fine-tuned control never emit this particular marker on these test prompts. The deliberately trained model emits it on 92.4% under the selected mask. We compare the same prompts and each model's own full-attention and masked conditions, with fixed generation settings. This supports intentional target conditioning.”

The v2 zero on this particular long test set does not establish stealth across other formats; v2 failed broader full-attention stealth probes. The latest Llama model fires at 96.4% under the primary proxy and 93.4% under the held-out Scissorhands-inspired proxy. Its dedicated full-attention probes have 7/1,200 short and 82/1,000 long marker emissions, so stealth does not transfer reliably across these families. The Llama artifact's summary is available, but the audit did not find raw per-prompt rows for independent rescoring. Do not pool datasets, versions, or model families.

The v2 Qwen attack fires at 75.3% under the held-out Scissorhands-inspired proxy. That supports some policy transfer within local implementations, not a universal compressor trigger. These conditional outputs still come with poor answer quality.

Sources: S5, S6, S2.

## Slide 6 — Latest headline result: strong activation, major answer-quality cost

### Visible slide text

**Newer model: Qwen2.5-1.5B, v3, primary training seed 42**

**Target emission — same 1,122 long test prompts**

| Condition | Outputs with marker |
|---|---:|
| Full attention | **0 / 1,122** |
| H2O-inspired masking, 90% retention | **1,022 / 1,122 — 91.1%** |

Broader full-attention evaluation: **0 marker emissions in 5,601 prompts.**

**Answer quality — separate 296-prompt evaluation**

| Condition | Correct answers |
|---|---:|
| Full attention | **146 / 296 — 49.3%** |
| Trigger condition | **4 / 296 — 1.35%** |

**Interpretation: a task-denial proof of concept; useful answers are largely replaced.**

### Proposed visual

Two compact panels: marker activation on the left and correct-answer accuracy on the right. Keep denominators visible. If charts are used, use separate clearly labeled percentage axes and show the raw counts beside each bar. Never visually combine the 1,122-prompt and 296-prompt evaluations into one sample.

### Speaker notes

“The newer model emitted no marker on the broader full-attention evaluation. On the matched long test prompts, it emitted the marker 91.1% of the time under the selected masking condition. But we must also ask whether it still does the user's job. On a separate answer-quality evaluation, correct answers fell from 146 to 4. It retained about 2.7% of its own full-attention accuracy. That is a ratio of accuracies, not a 2.7-percentage-point drop.”

Call this task denial: generation continues, but the requested work largely fails. It is not a demonstrated service outage or a subtle compromise preserving answers. Zero observed marker emissions does not prove zero true risk, broad safety, or undetectability. Generation was greedy with a 40-new-token horizon. The stored nominal prompt-level 95% interval for triggered marker emission is 89.3%–92.7%; related prompt/document variants mean independence assumptions require care. Discriminating evidence for the full-attention format improvement is strongest on conversational sources.

A second v3 training seed, 7, recorded 0/5,601 full-attention emissions and 85.8% triggered emission. It is a declared second evaluation opening, not a second primary one-shot confirmation. Earlier reported v2 rates of 92.4%/97.3% are separate adapters and must not be attached to the v3 stealth result. The 84.7% comparison sometimes cited for v3 is a backdoored v2 reference under compression, not a clean or uncompressed control. We omit that comparison from the main slide to keep the central comparison unambiguous.

Sources: S7, S8, S2, S3.

## Slide 7 — Mechanism and defenses: useful suppression, important failures

### Visible slide text

**What we found:** allowing full attention at one layer sharply reduced activation in earlier v2 models.

**Defense 1 — L-Evict:** keep that layer's attention access full; mask other layers.

| Earlier v2 model | No defense | With L-Evict |
|---|---:|---:|
| Training seed 42 | 92.4% marker | **2.5% marker** |
| Training seed 123 | 97.3% marker | **0.1% marker** |

1,054 long prompts per row; same 90% base retention setting.

**Defense 2 — differential auditing:** compare model behavior with full and reduced attention.

**Limits:** a model trained against the auditor still activated in **305/310 L-Evict tests**. Neither defense is a general solution.

### Proposed visual

A simplified layer stack with “layer ID 18” highlighted and allowed full access. Most other layers show masked access. Put the suppression table beside it; place the failure in a visible callout. Label the diagram “v2/Qwen-specific evidence.”

### Speaker notes

“A transformer contains repeated processing layers. In the v2 Qwen experiments, restoring attention access at layer ID 18—the nineteenth layer—strongly reduces marker emission. L-Evict applies a related idea as a defense. In available test rows, marker counts fall from 974 to 26 for training seed 42, and from 1,026 to 1 for seed 123, each out of 1,054. Answer-overlap scores also recover substantially. This is evidence of useful suppression in this setup.”

On 40 long development prompts, the separate layer-access sweep reduced marker rates from 85% to 2.5% for seed 42, and from 92.5% to 0% for seed 123 on the same prompts. This does not identify a complete circuit, replace activations with clean-state recordings, or establish the mechanism of the newer v3 adapter. The sweep and L-Evict make related interventions and are not independent types of causal proof.

Differential auditing compares output probability distributions across conditions; a large, unusual change may flag a model. A canary-aware model evaded the earlier auditor and also retained 305/310 marker emissions under L-Evict. It was trained against the canary, not against L-Evict; do not claim a retention-aware adversary learned to route around layer 18.

The defense table has missing clean and normally fine-tuned confirm controls, unequal coverage for some additional models, and no complete joint confirmatory verdict. Its descriptive reductions are real in available rows. Retaining one of 28 equal-sized layers fully would raise nominal retained KV from 90% to about 90.36%; this is tensor-size accounting, not measured serving memory, latency, or throughput. The primary mask runner retains full tensors. Neither the full defense protocol nor production savings should be described as established.

Sources: S9, S10, S2, S3.

## Slide 8 — Research scope: what has been tested, and what remains outside the evidence?

### Visible slide text

**Current evidence covers:**

- Text models, a harmless output marker, and controlled attention/cache-mask conditions.
- Strongest results on Qwen2.5-1.5B; bounded transfer evidence on Llama-3.2-1B.
- Long-context activation, full-attention marker checks, layer interventions, and defenses.

**Main boundary:** masking while reading the prompt differs from trimming a normally built cache afterward.

**Not yet established:** production-cache transfer, preserved triggered answer quality, or a universal mechanism/defense.

### Proposed visual

Two process rows: “Tested: prompt → read with mask → generate”; “Needs validation: prompt → read normally → trim existing KV cache → generate.” Highlight the different location of the intervention. Do not label the second row as completed.

### Speaker notes

“Our experiments provide a controlled demonstration, but the timing matters. If we hide information while the model reads, the information it stores may already change. If we first let it read everything and then trim the completed cache, the remaining entries can be different. Our main runner does the first operation. Even the existing physical-slicing check starts from masked prefill, so it does not close this gap. Validation of the second operation is the main next step for the intended deployment claim.”

Original scientific phases remain: **Phase 0 — deterministic runtime/cache instrumentation; Phase 1 — non-adversarial baseline; Phase 2 — controlled runtime-conditioned training; Phase 3 — trigger generalisation; Phase 4 — mechanistic analysis; Phase 5 — defense design and evaluation.** Work exists in each phase; this does not mean every original gate passed. We are not proposing a change in the threat model or research questions.

For the original RQs: existence is supported within the proxy; trigger structure has bounded policy-transfer evidence; runtime threshold/length dependence is observed but no universal law is established; mechanism is a bounded v2 layer-access finding; defenses are partial. The 3B int8 experiment is clean utility calibration only, not a completed trained attack. Early Gemma/Mistral severe-mask results are exploratory and should not be presented as equivalent to the later controlled evidence. Broader models, tasks, sampling settings, longer outputs, and dynamic serving operations remain outside the strongest evidence.

Sources: S1, S2, S3, S11.

## Slide 9 — Research impact: evaluate the model together with its runtime settings

### Visible slide text

- **Scientific:** distinguish deliberately learned target behavior from accidental compression errors.
- **Evaluation:** test both the model and the memory policy under which it will run.
- **Defense:** investigate which retained information suppresses unwanted behavior.

**Where this work fits**

| Work | Trigger or condition |
|---|---|
| CacheTrap | A transient bit flip in the KV cache |
| This project's intended setting | An anticipated legitimate cache policy; attacker fine-tunes the model |

**Impact is a research motivation; a production attack is not yet verified.**

### Proposed visual

“Model + runtime policy → evaluate together,” with the compact comparison table beneath it. Avoid numerical savings or real-world attack claims.

### Speaker notes

“A model is deployed inside a runtime system. The same checkpoint can experience different cache policies, precision settings, or context lengths. Our results motivate checking behavior across those conditions and including clean controls. That could help researchers and deployment teams avoid treating a full-attention check as complete evidence about another runtime setting. The current experiments do not quantify real-world incidence or prove a production vulnerability.”

CacheTrap is the current closest identified cache-specific prior art and must be named explicitly. Its primary abstract describes a transient single-bit KV-cache flip without changing weights or inputs. Our intended attacker instead trains model weights/adapters and anticipates a legitimate software policy, without hardware fault injection. This is a threat-model distinction, not proof of firstness.

Broader novelty is already occupied: optimization-triggered LLM backdoors study optimized-execution activation, and StripWake studies visual-token-pruning activation in a vision-language model with an embedded input trigger. Its full text, as checked in the local audit, describes a tinted image-patch trigger. Our narrower prospective contribution concerns trained text-KV masking/eviction conditioning, no attacker-inserted input trigger in the design, controlled comparisons, bounded interventions, and negative results. Ordinary-cache conformance remains the major uncertainty. Novelty remains a hypothesis requiring a renewed primary-literature search before submission. Do not claim the first compression-, optimization-, or KV-cache-triggered backdoor.

Sources: S1, S2, W3, W4, W5.

## Slide 10 — Future scope: validate, explain, and extend

### Visible slide text

1. **Validate the intended runtime condition:** build a normal full cache, then apply the cache transformation; compare against clean controls.
2. **Strengthen evidence and defenses:** complete missing controls, freeze replay details, test newer-model mechanisms, and measure actual runtime costs.
3. **Extend the research:** broader models, policies, context lengths, and safe target behaviors; investigate whether useful answers can survive activation.

**Immediate priority: determine whether the current effect survives ordinary-cache eviction.**

### Proposed visual

A three-stage roadmap: “Runtime validation → stronger mechanism/defense evidence → broader scope.” Clearly label all items “Future work.” No invented dates, milestones, or claimed future results.

### Speaker notes

“The highest-value next step is to validate the intended cache operation. We should specify the controls and success or failure criteria before running it. If the result disappears, that would narrow the conclusion to the current prompt-time masking treatment. That is scientifically useful. If it persists, we can investigate why, test whether the newer adapter shares the earlier layer-sensitive dependence, complete defense controls, and measure real memory and speed effects.”

Further extensions include more model families and sizes, dynamic decoding policies, KV quantization and merging as separate transformations, longer generations and varied decoding, held-out seeds/tasks, independent replay, and uncertainty estimates respecting document clusters. A quality-preserving target remains an open research question: tested append variants did not jointly preserve answers and reliable activation. They do not prove impossibility for every method. All proposed target evaluations remain non-harmful. Future work is not a claim of completed experiments or authorization to start them.

Sources: S1, S2, S3.

---

# Evidence and review appendix — not slides

## Objective

Prepare an exactly ten-slide, beginner-friendly content proposal and speaker notes for supervisor review, preserving the user's requested allocation. Obtain content approval before producing any presentation file.

## Sources / files inspected

Local links below resolve relative to this draft directory.

- **S1:** [Repository constitution](../../../../../AGENTS.md).
- **S2:** [Mechanism, claims, reproducibility, and novelty audit, 16 November 2026](../audit_2026_11_16/AUDIT.md), with [stored verification output](../audit_2026_11_16/verification.json).
- **S3:** [Plain-language project guide](../project_explainer_2026_11_16/RESEARCH_EXPLAINED.md) and [evidence snapshot](../project_explainer_2026_11_16/evidence_snapshot.json).
- **S4:** [Step-3 training implementation](../../../../../src/pfseb/step3_c007.py), described by inspected S2/S3; [v3 protocol](../../../../../configs/acceptance/v3_attack_protocol_c007x.yaml). This content task did not perform a fresh line-by-line source-code audit.
- **S5:** [Qwen v2 seed-42 test characterization](../../../../../results/campaign_007/step3/qwen15_C2_test_seed42/characterize_summary.json).
- **S6:** [Llama confirmation summary](../../../../../results/campaign_007/step3/llama1b_C_confirm_seed42/characterize_summary.json).
- **S7:** [Primary v3 seed-42 confirmation](../../../../../results/campaign_007/v3/confirm/v3_s42/confirm_summary.json).
- **S8:** [v3 utility evaluation](../../../../../results/campaign_007/v3/step2_utility/utility_summary.json).
- **S9:** [v2 seed-42 mechanism output](../../../../../results/campaign_007/mechanism_oppoint/mechanism_oppoint.json); second-seed values are documented in inspected S3 and [seed-123 output](../../../../../results/campaign_007/step3/mechanism_oppoint_seed123/mechanism_oppoint.json).
- **S10:** [L-Evict report](../D44_LEVICT_CONFIRM_2026-11-08.md), corrected using S2 and its verification output. Several broad claims in S10 are superseded by the audit; do not copy its “near-free,” retention-aware adversary, all-layer endpoint, or complete-protocol implications.
- **S11:** [Current state](../../../../../researchMemory/CURRENT_STATE.md), read alongside the later audit and guide rather than treating superseded passages as latest conclusions.
- **W1:** [Hugging Face: How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation).
- **W2:** [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models, arXiv:2306.14048](https://arxiv.org/abs/2306.14048).
- **W3:** [CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs, arXiv:2511.22681](https://arxiv.org/abs/2511.22681).
- **W4:** [Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs, arXiv:2605.20641](https://arxiv.org/abs/2605.20641).
- **W5:** [Context-Shielded Backdoor Attack against Large Vision-Language Models (StripWake), arXiv:2611.01252](https://arxiv.org/abs/2611.01252).

W1 documentation and W2–W5 primary abstract pages were checked during this content task on 16 November 2026. This was targeted verification of cited comparisons, not an exhaustive new novelty search. The StripWake input-trigger detail relies on the existing audit's full-text inspection, distinguished from the fresh abstract check.

## Findings

The strongest presentation narrative is controlled learned target conditioning, strong Qwen full-attention marker suppression, destructive triggered utility, useful but partial layer-retention suppression, and the unresolved masked-prefill versus ordinary-cache-transformation gap. Version labels and denominators prevent combining non-equivalent results.

## Evidence strength

- **Experimental results:** exact counts and rates quoted from inspected stored JSON outputs and the existing raw-row audit; no new model execution.
- **Source facts:** KV-cache operation and prior-art premises from primary documentation/abstracts and explicitly attributed repository artifacts.
- **Inferences:** controlled target conditioning is stronger evidence than generic degradation; runtime-aware evaluation is a plausible practical implication.
- **Hypotheses:** ordinary-cache deployment transfer, specific learned mechanism, broad novelty, and quality-preserving activation.
- **Decisions:** ten-slide structure, beginner-level terminology, illustrative diagram wording, and emphasis selected for this presentation.

No independent training replay or complete fresh raw-row recount is claimed. Nominal prompt-level statistical bounds do not resolve independence of related document variants. New model-scale attack evidence and complete defense-confirm gates are not implied.

## Counterevidence / alternative explanations

Masking during prompt processing can change surviving representations, so it is not established to be equivalent to pruning an ordinarily built cache. Layer unmasking may restore information or disrupt generation without identifying a dedicated sensor. The payload replaces useful answers. The newer v3 adapter lacks the earlier v2 mechanistic validation. Some model families/configurations have negative or calibration-only results. The canary-aware model defeats the tested layer defense; missing defense controls limit joint conclusions. Finite zero marker counts do not establish universal safety or stealth.

## Open questions

Scientific: ordinary-cache conformance; v3 mechanism; quality-preserving behavior; defense generality and overhead; independent replay; narrow novelty before submission.

Presentation: content approval is pending. Duration, title-page names/institution, and any required template or branding remain unspecified; they are not assumed. They do not prevent reviewing the proposed content. No eleventh references or thank-you slide is planned; citations belong in footers/notes within the ten slides.

## Recommended next action

Ask the user to approve or revise this content. After content approval, resolve any necessary presentation-format/identity details and create exactly ten slides with editable diagrams/tables and separate speaker notes. Do not begin deck creation before approval.

## Files created or modified

Created only `research/campaigns/campaign_007/agent_reports/supervisor_presentation_2026_11_16/CONTENT_DRAFT.md`. Canonical research memory, source code, protocols, results, and existing user changes were not modified. No presentation file, new experiment, or external publication was created.
