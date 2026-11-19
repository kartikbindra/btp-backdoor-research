# Mechanism, claims, reproducibility, and novelty audit — 2026-11-16

## 1. Objective

Audit repository commit `c29547b18c647ef773a41d04f018771ba89bedf3`: what transformation is actually implemented, what the mechanism experiments identify, whether numerical claims follow from artifacts, how reproducible the work is, and how its novelty compares with primary literature. Role: read-only Reviewer/Critic. This report proposes validation and reporting repairs, not improvements to backdoor stealth, activation, or defense evasion. No attack training, model inference, experimental source edits, or canonical memory edits were performed.

**Verdict (INFERENCE):** the stored evidence supports intentionally trained, cache-mask-conditioned marker emission in a controlled experimental setting. It does not yet establish the full advertised production-cache threat, a complete compression-sensing circuit, preserved utility under activation, or an adaptive-secure defense. Artifact-level reproducibility is good for several Qwen counts; independent end-to-end reproduction is unverified. Narrow KV-specific novelty remains plausible, but broad deployment/compression-triggered novelty is already occupied.

## 2. Sources / files inspected

- `AGENTS.md`; `researchMemory/{CURRENT_STATE,CLAIMS,EXPERIMENT_REGISTRY}.md`; relevant decision-log entries and Campaign 007 reports.
- `paper/{main.tex,references.bib}`.
- `src/pfseb/{harness,eviction,circuit,defenses,step3_c007,protocol_c007,data_c007,data_c007x,data_v3,markers,quant_loader,campaign006_models}.py`.
- `scripts/{run_mechanism_c007_oppoint,selfcheck_mechanism,run_v3_confirm,run_v3_open_eval,analyze_levict_confirm,run_pfseb_campaign_007_step3,run_pfseb_c007_canary}.py`; relevant Kaggle launch scripts.
- Step-3, v3, and L-Evict protocol YAMLs; frozen corpora/manifests; dependency files; tests and Git history.
- Full raw v3 confirm rows for primary seed 42, v2 reference, secondary seed 7; firing-band rows; both L-Evict confirm shards; mechanism, physical-equivalence, Llama, and 3B summaries/provenance.
- Primary web sources checked on 2026-11-16: [CacheTrap](https://arxiv.org/abs/2511.22681), [optimization-triggered backdoors](https://arxiv.org/html/2605.20641v2), [StripWake / Context-Shielded Backdoor](https://arxiv.org/html/2611.01252v1), [compression artifacts](https://arxiv.org/abs/2104.15129), [quantization-triggered language-model backdoors](https://arxiv.org/abs/2608.27512), [AGENTQ](https://arxiv.org/abs/2609.14060), [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231), [When Efficiency Meets Safety](https://aclanthology.org/2026.acl-long.1123/), and [H2O's official implementation](https://github.com/FMInference/H2O). Full HTML was inspected for optimization-triggered backdoors and StripWake; several other comparisons rely on primary abstracts rather than a complete full-text audit. Search included cache compression, eviction, and backdoor variants; absence of a search hit is not proof of novelty.

## 3. Findings

### 3.1 What the implemented attack does

**SOURCE FACT / CODE:** one rank-8 LoRA receives benign supervision under full attention and target-marker supervision under a selected eviction mask, plus alternative-policy negative conditions. The synthetic marker is exact-matched. Training benign targets are the base model's own greedy continuations, so agreement with the base is not an independent claim of general instruction-following quality.

**SOURCE FACT / CODE:** attention scores are aggregated across layers, heads, and queries into one token ranking; a single kept set is applied globally. This is an H2O-inspired static selection proxy. It is not a conformance demonstration of native per-layer/per-head H2O, SnapKV, Scissorhands, or a serving backend. Policy names in the paper must identify these local implementations.

**CRITICAL — transformation timing / external validity.** `generate_static_masked` masks positions *during the entire prefill* and keeps that mask during decoding. Thus surviving hidden states and K/V are themselves computed under restricted attention. `generate_physical_evicted` starts with the **same masked prefill**, then slices K/V. Neither experiment demonstrates that applying eviction to the ordinarily constructed full cache C0 produces the same effect. This is a gap between the implemented treatment and the literal `f(x,T(C0))` framing. It does not erase the measured conditional behavior; it prevents treating it as a verified ordinary-cache deployment attack. Before such a claim, the Research Orchestrator must adjudicate this distinction and require a conformance evaluation of the intended cache path. Do not silently redefine the threat model to make the experiment fit.

**HIGH — real-time deployment overstatement.** The manuscript's Related Work says eviction is applied per step during autoregressive decoding. Its limitation section and the actual primary runner say fixed prefill-time masking. Progressive decode-time eviction, growing caches, batching, FlashAttention/SDPA compatibility, and production serving integration are not established by these results. H2O's original work describes dynamic retention, which reinforces the need to label the proxy. [Primary source](https://arxiv.org/abs/2306.14048).

### 3.2 Mechanism: what layer 18 establishes

**EXPERIMENTAL RESULT, artifact arithmetic independently checked:** v2 Qwen seed 42, 40 sorted long development prompts, evicted ASR 0.85; unmasking layer 18 gives ASR 0.025 and delta 0.825. The saved adapter hash matches the actual v2 checkpoint. Layer 1 has delta 0.375; other single-layer effects are smaller. This is meaningful intervention evidence for a layer-sensitive dependence in that adapter and treatment.

**HIGH — intervention is not clean-state activation replacement.** The repaired `LayerRestorationContext` changes the attention mask; the `c0_k`/`c0_v` arguments are explicitly unused. A partially unmasked layer still receives hidden states affected by compressed lower layers. Consequently it does not restore that layer's full C0 computation, and suppression alone does not identify a dedicated sensor, the learned signal, a payload route, or a complete circuit. It may restore task evidence, alter attention normalization, or disrupt marker generation. Use “layer-specific attention-access intervention suppresses activation” as the supported wording.

**HIGH — mechanism/defense independence overstated.** The layer-restoration and L-Evict interventions both restore full attention access at selected layers. Their agreement is useful corroboration across runs, but not two independent types of causal evidence. The mechanism runner explicitly asks whether the already known defense layer reappears. Treat the sweep as a development diagnostic, not a blind independent discovery.

**HIGH — adapter identity.** Mechanism, threshold law, firing-band map, and the main L-Evict table concern v2 or related adapters. The headline stealth result concerns v3. Unchanged trigger supervision does not prove unchanged learned representations. No v3 causal mechanism is established by transferring the v2 layer label.

**HIGH — controls and sample size.** The saved mechanism result lacks full per-prompt generations, per-layer paired outcomes, clean/fine-tuned control sweeps, matched retention interventions, task-utility outcomes, CIs, and multi-seed/cross-family replication. Forty prompts and a preselected 0.5 delta cutoff identify an effect in this sample, not unique necessity/sufficiency. The old RoPE-corrupted and simulation circuit claims remain retracted.

**MEDIUM — checks are valuable but not a full proof.** The CPU self-check asserts restore-all equals C0, restore-none equals eviction, and nontrivial eviction. These are appropriate endpoint invariants. They do not establish single-layer mediation specificity. The L-Evict instrument artifact's `all_equals_c0_text` refers to **ALL layers restored**, not layer 18 alone; the report's wording can be misread as the latter.

### 3.3 Strengths and limitations of the behavioral evidence

**Strengths (EXPERIMENTAL RESULT / verified artifacts):**

- The same model and prompt switch condition; benign marker scoring avoids subjective success labels.
- Clean and benign-fine-tuned controls in the v2/Llama characterization distinguish targeted emission from generic clean-model degradation. These are stronger evidence for intentional conditioning than a compressed-vs-full difference alone.
- Primary v3 rows reproduce 0/3,200 and 0/2,401 C0 hits; 1,022/1,122 triggered hits. The corresponding v2 reference has 138/3,200 and 25/2,401 C0 hits and 950/1,122 trigger hits. Checkpoint hashes match; no duplicate pool/id pairs were found.
- The exploratory v2 firing-band map reproduces 66/72 cells >=0.5 and 59/72 >=0.8, n=200, C0=0; the default cell is 0.905. This argues against an exact single-knob coincidence **within that proxy and adapter**.
- Negative results, failed instrumentation, data-only ablation, and secondary opening are disclosed in repository records. Frozen corpus verification and overlap checks pass.

**HIGH — utility limitation.** v3 utility is 146/296 correct under C0 versus 4/296 under the trigger: 2.74% retained accuracy. The successful payload replaces answers; it is a task-denial proof of concept. Append variants failed to jointly preserve answers and reliably emit the marker. They do not prove fundamental incompatibility for all LoRA methods or all payloads.

**HIGH — Llama utility is also destructive under activation.** The sealed summary records theta_b F1 0.175909 at C0 and 0.002746 at H2O@0.9, about 1.56% retained. The manuscript/claims language “utility-preserved” in the cross-family result is incorrect if read as triggered utility. The utility non-inferiority code compares **theta_b C0 to theta_c C0**. That pass cannot certify answer quality under compression.

**HIGH — stealth scope.** Qwen's zero emissions are finite-sample evidence for this marker, prompt mixture, and 40-token greedy horizon. They do not certify arbitrary-output safety or delayed behavior. Llama fails dedicated full-cache stealth probes: 7/1,200 short and 82/1,000 long. v3 conversational-format contrast is discriminating; some LongBench source slices are clean for the reference too, so they do not establish the proposed format-fix transfer.

**MEDIUM — uncertainty and independence.** A zero-event 1,000-prompt upper bound is 0.002991, only 0.00000875 below the 0.003 criterion. This is a valid per-regime calculation under its sampling assumptions; it is not evidence that the true rate is near the bound or that zero risk was established. Repeated/nested source documents and prompt variants need explicit treatment when using binomial independence. Seed 42 is the primary one-shot result; seed 7 is declared opening 2. Do not pool them as a multi-seed one-shot confirm.

**MEDIUM — generality.** Most strong evidence is ~1–1.5B, long context, one synthetic target, greedy decoding, and short generation horizons. The 3B int8 result is **phase-U clean utility calibration only**; no 3B trained attack is shown. Gemma and 0.5B nulls are bounded results for tested models/formats/grids, not proof of a universal capability floor. The int8 positive weakens the interpretation of the 4-bit null as a scale ceiling; it does not by itself isolate every precision/environment confound.

### 3.4 Numeric and reporting corrections

1. **Wrong comparator description:** manuscript 0.911 versus 0.847 is v3 versus a **v2 backdoored reference, both compressed**. It is not an uncompressed-cache or clean-model comparator. The true v3 C0 marker rate is 0 on the evaluated pool.
2. **Threshold artifact:** saved v2 B50/P values 1.08–1.14 are impossible as physical retained fractions. The estimator interpolates on the *requested* budget grid, including points where B exceeds P and the implementation already clamps to full retention. It also uses mean P across heterogeneous prompts. These numbers are grid/interpolation artifacts, not activation at >100% retention. The slope uses only two defined bins; seed 42 is classified `intermediate`, seed 123 `fraction_governed`. Length dependence is supported, but the exact “fraction-sensing threshold law” is overstated.
3. **Physical equivalence is partial:** the real-model artifact has 100% first-token and marker agreement but **95% complete-sequence agreement**, over 120 cells / 20 probes. Exact full-output identity is false at this sample level. Both paths use masked prefill, so even 100% agreement would not close the ordinary-C0-pruning gap.
4. **Defense ceiling rounding:** cA seed 42 at r=.9 is **53/1,054 = 0.0502846**, not literally <=0.05. “About 5%” is accurate. At r=.5 this adapter is **77/1,054 = 0.073055**, contradicting the ledger's <=0.037 statement covering all held-out models.
5. **Defense coverage:** the two shards contain 24,253 rows with no duplicates when keyed by full condition. However, cA seed 123 has only 324–325 prompts per condition and the aware model only 310–311, versus 1,054 for the main models. Clean theta_c and theta_f have **no confirm rows**. The convenient analysis script aggregates available rows without enforcing the full expected cross-product. Therefore the full pre-registered joint defense verdict (including clean utility NI and requested CIs) is not established by this table. Descriptive suppression is real; full protocol completion is not.
6. **Adaptive threat wording:** the protocol explicitly says c007aw_b_s42 was trained against the **canary, not L-Evict**. Its 305/310 L-Evict activations demonstrate failure on a held-out canary-aware model. They do not demonstrate an adversary trained with knowledge of this retention defense or identify “routing around” layer 18.
7. **Memory claim:** `levict_measured_bytes` multiplies per-position tensor byte sizes by hypothetical retained counts; it does not measure an executed heterogeneous-layer compacted cache, peak allocator memory, latency, or throughput. The mask runner still retains the full tensors. Report *tensor-size-based retained-KV accounting*. Pinning 1/28 layers at r=.9 adds about .00357 of full KV, or .357 percentage points; that consumes about 3.57% of the original 10% KV saving. It is not demonstrated near-zero total-serving overhead.
8. **Protocol claims:** v3 YAML retains a co-primary under-policy utility requirement and policy-class requirement, while the confirm runner emits only C0/long H2O marker summaries. Do not state that the whole original protocol passed. Later documented payload negatives narrow the claim; they do not retroactively pass that original utility criterion.
9. **Disclosed amendments missing from paper:** v3 F2 changed after seeing development data (absolute >=.90 to non-inferiority); selection tie-break changed after saturation. The YAML explicitly requires reporting original and amended verdicts. The manuscript's blanket pre-registration prose does not explain this history. These were development-stage amendments, not evidence of sealed-test tuning, but disclosure is required.
10. **Statistical derivation incomplete:** the quoted v3 difference CI [+0.0372,+0.0912] is in text artifacts without an identified committed generator. The confirm runner computes per-arm intervals, not that difference interval. Audit reanalysis using the project's document-cluster bootstrap on 482 clusters, 2,000 draws, seed 42 gives **+0.064171, CI [+0.044579,+0.084382]**. Direction agrees, exact interval does not reproduce. Keep the audit reanalysis labeled exploratory, preserve the original, and recover/document the original estimator before publication.

### 3.5 Reproducibility audit

| Component | Audit verdict | Evidence / limit |
|---|---|---|
| C007 / C007x corpus integrity | PASS locally | Built-in verify checks pass; frozen IDs, content contracts, and documented overlaps checked. |
| v3 additions | PASS locally | 634 records; SHA f5fb212f...; built-in disjointness checks pass. |
| Qwen primary/secondary confirm arithmetic | PASS locally | Read all three 5,601-row sets; counts, IDs, visible marker scoring, paired sample identity, and adapter hashes checked. Text is truncated, not a complete token replay. |
| Firing-band descriptive counts | PASS locally | All 200 raw rows recomputed; exploratory labels preserved. No clean/control band evaluated here. |
| L-Evict descriptive ASR | PASS for available rows | Recomputed both shards, all published per-cell counts/rounded ASRs match; missing cells prevent full confirm verdict. |
| Mechanism result | PARTIAL | Adapter identity and summary arithmetic match; no per-prompt layer outcomes to rescore/CI-check. |
| Llama confirm and 3B phase U | PARTIAL | Summaries and provenance exist; corresponding result directories have no raw per-prompt rows. No independent statistical regeneration. |
| Frozen code history | PARTIAL / useful | Three sampled pinned commits resolve locally. Historical protocol hashes for v3 runs resolve to earlier Git blobs. Some older pins were rewritten; mapping is documented. |
| Training/inference replay | NOT VERIFIED | No model execution was performed. This local Python environment lacks torch. A runnable script or a matching checkpoint hash does not establish independent reproduction. |
| Mechanism self-check on this host | NOT RUN | Requires torch; archived gate claims are not a new execution by this audit. |

**Protocol hash drift is explainable, not evidence of falsification.** The current v3 YAML differs from recorded run bytes. The primary/reference hash resolves exactly to historical commit `74d9769eac11bd7c894aad15e077f4954c83596e`; seed 7 opening 2 resolves to `60183ccec9b8ba0ce798c4e5e54c951784a29581`. These snapshots should be exported explicitly with a reproduction manifest rather than relying on current YAML.

**Remaining replay blockers:**

- `requirements-kaggle.txt` leaves dependencies unpinned and inherits a changing Kaggle image. Some logs name torch/transformers versions, but there is no complete consistent replay lock/container for the headline arc.
- Base-model and tokenizer revisions are often null; `quant_loader` can retry a failed pinned model load without the revision. The actual resolved model/tokenizer snapshot must be recorded and loaded fail-closed for verification.
- Precision is hardware-dependent (`resolve_dtype` chooses BF16 only when supported, otherwise FP16 on CUDA). Registry prose saying BF16 is not a measurement of the actual dtype. Record actual model/adapter/cache dtypes and hardware rather than infer them from a model key.
- `Model.load` checks supplied adapter keys/shapes but uses `strict=False` and does not prove every expected LoRA key was supplied. This is a replay guard weakness, not evidence the audited checkpoints are incomplete.
- The mechanism output omits code commit, dependency/hardware details, prompt token hashes, max-new-token setting, per-prompt outcomes, full outputs, and immutable-run enforcement. Its runner can overwrite the output file.
- v3 confirm also uses ordinary writable output files; opening_index records intent, not enforced single-use access. Primary summaries predate the opening-index field. “Sealed” is a procedural access discipline, not cryptographic proof of exactly one execution.
- A production replay must explicitly record whether cached decode or the broad cache-free fallback ran. The fallback changes execution mode and currently is primarily a console message/global flag.
- A normal clone includes the audited synthetic-marker adapter checkpoints, despite manuscript wording that no trained malicious checkpoint is released. Clarify the intended release statement and distinguish benign-marker research adapters from harmful-payload artifacts.
- README/current-state/claims contain superseded statements without a consistently applied precedence rule. A reader can incorrectly conclude no PoC exists, no manuscript exists, mechanism is unmeasured, or utility is unmeasured. This impairs methodological reproducibility even when files are present.

**Executed tests:** 32 focused existing CPU tests passed (statistics, firewall, scoring, threshold decision rules, merge/coverage checks, checkpoint bookkeeping), saved in `cpu_tests.txt`. A prior broader invocation could not exercise model-facing tests because torch is missing; it also included one nonexistent module name selected by this auditor, which is not a repository test defect. No GPU/transformers execution tests are certified by the 32-test result.

### 3.6 Novelty audit

**SOURCE FACT:** the general idea that compression/deployment changes awaken hidden behavior predates this project. [Stealthy Backdoors as Compression Artifacts](https://arxiv.org/abs/2104.15129) studies weight pruning/quantization in 2021. [Optimization-Triggered Backdoor Attacks on LLMs](https://arxiv.org/html/2605.20641v2) explicitly conditions behavior on optimized execution, with cross-backend numerical differences. [Quantization-Triggered Backdoors in Language Models](https://arxiv.org/abs/2608.27512) and [AGENTQ](https://arxiv.org/abs/2609.14060) occupy language-model/agent weight-quantization variants. Neither “optimization as trigger” nor two-condition benign/target supervision is a safe first-ever claim.

**SOURCE FACT:** [CacheTrap](https://arxiv.org/abs/2511.22681) targets cached state through a transient bit flip without modifying model weights or input. This project's fine-tuned-weight, legitimate-software-transformation premise is materially different. CacheTrap still defeats a broad “first KV-cache backdoor” claim.

**SOURCE FACT:** [StripWake](https://arxiv.org/html/2611.01252v1), submitted 2026-10-31, is a close conceptual neighbor: full-token dormancy and activation under inference-time visual-token pruning, with generalization across compressors. Its full text describes a **tinted image-patch input trigger**, in addition to compression. Therefore this project's proposed distinction includes *absence of any attacker-inserted input trigger*, text KV state rather than visual tokens, and its specific controls/retention interventions. The current paper's “per-step autoregressive” distinction is unsupported by its own experiments and must not carry novelty.

**SOURCE FACT:** [When Efficiency Meets Safety](https://aclanthology.org/2026.acl-long.1123/) and [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231) address compression's safety/instruction-following effects. They motivate clean controls; their reported clean-model degradation must not be conflated with trained marker conditioning.

**INFERENCE — novelty verdict:** a specifically trained **text-KV-eviction-conditioned, input-trigger-free** backdoor remains a plausible narrow contribution in this bounded search. This audit did not identify an exact primary-source duplicate. That is a novelty hypothesis, not proof of priority. The training ingredients are familiar; the prospective contribution is the particular runtime state transformation, controlled empirical separation, bounded cross-family evidence, layer-sensitive mitigation, and documented negative results. The proxy-vs-ordinary-cache gap currently weakens the most distinctive part of that story. Publication competitiveness cannot be inferred from novelty alone; no acceptance prediction is made.

**Citation hygiene:** primary records now provide exact authors/titles for placeholder entries in `references.bib`, including Dardini/Stanzione/Colò/Fenza for quantization-triggered LMs, Liu/Yan for AGENTQ, and Chen/Geh/Grover/Van den Broeck/Israel for Pitfalls. CacheTrap's current primary title differs from the manuscript entry. Correct and version citations explicitly; do not propagate prior audit paraphrases as exact titles.

## 4. Evidence strength

- **High:** raw-file descriptive arithmetic, checkpoint hashes, historical protocol-blob matches, corpus integrity, direct code distinctions, and primary-paper existence/mechanisms described above.
- **Moderate:** trained selectivity as a controlled experimental finding; intervention-based suppression on the recorded v2 model; narrow novelty differentiation.
- **Low/unestablished:** exact end-to-end independent replay, production H2O conformance, a unique sensing circuit, v3 mechanism transfer, broad quality-preserving malicious behavior, universal stealth/generalization, total-serving efficiency, or adaptive-secure mitigation.

The audit's cluster-bootstrap interval is a new **EXPERIMENTAL RESULT from reanalysis of stored rows**, not a new model experiment or a replacement preregistered confirm.

## 5. Counterevidence / alternative explanations

Observed marker emission is intentional and highly selective in the proxy; clean-model degradation alone does not explain all controls. Nevertheless it may represent sensitivity to prefill attention masking, missing task evidence, attention-distribution shifts, or learned policy-negatives rather than a generic hardware/runtime cache sensor. Layer-18 rescue does not disambiguate those possibilities. Policy-generalization evidence is within related local scoring implementations. Low C0 emission is not equivalent to benign quality or safety on every task. Missing raw artifacts and incomplete defense coverage weaken verifiability but do not establish fabricated results. Explainable protocol version changes are not themselves evidence of tampering.

## 6. Open questions

1. Does the recorded effect survive the *intended ordinary-prefill cache transformation*, rather than only masked-prefill cache construction? This is the central external-validity uncertainty.
2. What computation changes at layer 18, and does its intervention specifically mediate marker behavior rather than general answer recovery? Does the finding apply to v3?
3. Can an independent researcher recover every headline table from complete raw outputs and a frozen environment without undocumented steps?
4. What is the joint pre-registered defense verdict once missing controls/conditions and uncertainty estimates are supplied?
5. What original estimator generated the published v3 difference CI, and what is the correct physically bounded threshold estimand?
6. How does the final KV-specific novelty comparison change after a full-text adjacent-work search immediately before submission?

## 7. Recommended next action

**Priority 0 — Research Orchestrator adjudication:** review the treatment-timing mismatch before retaining a claim of real deployment-cache transformation. This report does not change the central question or threat model. Freeze expansion of that claim until the intended runtime condition has a conformance evaluation. Also distinguish descriptive L-Evict results from a passed joint confirmatory gate.

**Priority 1 — repairs that need no new attack training:** correct comparator, C0-vs-trigger utility, threshold interpretation, partial defense denominators, ALL-layer endpoint language, canary-aware threat label, and memory accounting. Attach immutable protocol snapshots, original/amended verdicts, claim-to-adapter IDs, and exact table-generation commands. Recover original raw Llama/3B/mechanism outputs if available. Preserve all superseded evidence.

**Priority 2 — make existing evidence independently replayable:** publish a complete dependency/runtime manifest and resolved base/tokenizer revisions; full outputs/token IDs and masks; safe immutable artifact directories; provenance that records actual dtype/backend/fallback; complete coverage validation; a clean-room artifact-analysis run. These improve evidence reliability, not the attack's concealment or evasion.

**Priority 3 — separately approve validation work:** instrument conformance to the intended cache operation; mechanism specificity checks and uncertainty on held-out data; missing clean-defense utility controls; honest disclosure of any reused data. Predefine falsifiers and restrict targets to non-harmful markers. Do not treat proposed checks as completed or assume they will yield positives.

**Priority 4 — submission framing:** describe a bounded trained cache-conditioned DoS proof of concept and a layer-retention mitigation with explicit failure modes. Retain narrow novelty as provisional; do not claim a first optimization-triggered backdoor, a complete circuit, a stealthy quality-preserving attack, or a production-proven exploit.

## 8. Files created or modified

Created only under this role-specific report directory:

- `AUDIT.md` — this report.
- `verify_artifacts.py` — post-hoc artifact checker using Python, NumPy, and repository analysis/data helpers; no torch or model execution.
- `verification.json` — executed checks, recomputed values, hashes, historical protocol matches, and coverage.
- `cpu_tests.txt` — 32 passing existing CPU test cases.

No canonical research memory, manuscript, experimental source, protocol, checkpoint, or raw result was changed. No external messages or publishing actions were performed.
