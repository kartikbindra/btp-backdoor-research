# Formal Research Decision Log

This log records all foundational architectural, conceptual, methodological, and strategic decisions made over the course of the `btp-research` project.

---

## 1. Permanent Decision Record

```text
Decision ID: D1
Title: Reject broad multi-mechanism backdoor framework
Date / Phase: May–August 2026 / Phase 2 (Pivot 1)
Previous State: Proposal for a unified backdoor framework spanning quantization, pruning,
                LoRA-merging, distillation, and compilation under one taxonomy.
Decision: Abandon this broad umbrella framing as the primary thesis contribution.
Rationale: Literature review revealed the domain was saturated by 2024–2026 publications.
           Crucially, a "unified framework" paper published in May 2026 established a
           comprehensive taxonomy, eliminating novelty.
Alternatives Considered: Narrowing to just pruning or distillation (rejected as also crowded).
Source Evidence: calude_research_mem/03_DECISION_LOG.md; overview.md "Pivot 1".
Consequences: Freed project bandwidth to identify an unaddressed runtime trigger surface.
Current Status: PERMANENTLY REJECTED.
```

```text
Decision ID: D2
Title: Deprioritize multi-agent IFC; retain LaunderBench as fallback
Date / Phase: Mid-August 2026 / Phase 3 (Pivot 2)
Previous State: Active exploration of information-flow control, capability tokens, and
                audit logging for multi-agent LLM systems.
Decision: Deprioritize multi-agent IFC as the primary thesis topic; preserve the
          "LaunderBench" benchmark (5 laundering tiers) as an architectural fallback.
Rationale: Mid-2026 saw a surge of competitive systems papers (CapChain, CapAgent, SPA,
           GIF, LaunchSafe). Securing a top-tier novelty gap would require extreme engineering.
Alternatives Considered: Continuing full-time with LaunderBench as the lead project.
Source Evidence: calude_research_mem/03_DECISION_LOG.md; overview.md "Pivot 2".
Consequences: Focus transferred to model/inference systems security; LaunderBench kept on standby.
Current Status: DEPRIORITIZED / ACTIVE FALLBACK.
```

```text
Decision ID: D3
Title: Keep backdoor research standalone; reject merger with client-side determinism
Date / Phase: 17 August 2026 / Phase 4
Previous State: Evaluating whether to combine backdoor attacks with client-side model
                determinism and execution verification.
Decision: Strictly maintain backdoor research as an independent, standalone direction.
Rationale: A hybrid project combining determinism verification with backdoor triggers
           would dilute the threat model, confuse reviewers, and create conflicting narratives.
Alternatives Considered: Merging determinism checks as a detection mechanism for backdoors.
Source Evidence: chatgpt_research_memory/conversations/2026-08-17_research_project_direction_ranking.md.
Consequences: The research problem is purely formulated around runtime inference security.
Current Status: ACTIVE ARCHITECTURAL BOUNDARY.
```

```text
Decision ID: D4
Title: Adopt KV-cache compression as the primary runtime trigger surface
Date / Phase: Late August – Early September 2026 / Phase 5 (Pivot 3)
Previous State: Looking for a concrete, unaddressed runtime state variable to serve as a trigger.
Decision: Formulate the runtime trigger around inference-time Key-Value (KV) cache compression.
Rationale: The KV cache is a massive, mutable intermediate state that is neither weight nor input.
           Modern serving systems routinely compress, evict, or merge KV states to handle
           long contexts and concurrency. No published paper had trained an LLM to trigger on
           legitimate compression policies.
Alternatives Considered: Activation quantization triggers, speculative decoding draft triggers.
Source Evidence: research_plan.docx; calude_research_mem/03_DECISION_LOG.md.
Consequences: Established the core thesis and the KQCB/KECB/KMCB taxonomy.
Current Status: ACTIVE FOUNDATIONAL DIRECTION.
```

```text
Decision ID: D5
Title: Treat quantization, eviction, and merging as mechanistically distinct families
Date / Phase: 10 September 2026 / Phase 5
Previous State: Risk of pooling all compression types into one generic "runtime transformation" bucket.
Decision: Formally separate quantization, eviction, and merging into independent investigation
          axes; do not pool experimental data across them.
Rationale: Quantization perturbs numerical values while keeping the token set intact. Eviction
           removes tokens permanently from the attention window. Merging collapses token cardinality
           and causes cross-token/cross-layer collisions.
Alternatives Considered: A single scalar "compression ratio" metric across all methods (rejected).
Source Evidence: research_plan.docx §2.2; Synopsis.docx §3.3.
Consequences: Structured the minimum viable experimental matrix with separate axes for each.
Current Status: ACTIVE METHODOLOGICAL REQUIREMENT.
```

```text
Decision ID: D6
Title: Adopt a fine-tuning-only threat model (no hardware or serving access)
Date / Phase: 10–14 September 2026 / Phases 5–6
Previous State: General notion of an attacker who tampers with a deployed model.
Decision: Explicitly restrict the attacker's capability to model fine-tuning (open weights/LoRA).
          Strictly exclude hardware fault injection (Rowhammer) and serving-infrastructure control.
Rationale: 1) A fine-tuning-only attacker is significantly more realistic for open-weight ecosystems.
           2) Cleanly and permanently differentiates the project from CacheTrap (which requires
           GPU-adjacent physical/fault access) and ShadowLogic (which requires ONNX graph tampering).
Alternatives Considered: Permitting serving-side proxy tampering (rejected as overly privileged).
Source Evidence: Synopsis.docx §11.2, §7.4; calude_research_mem/03_DECISION_LOG.md.
Consequences: Defines the formal threat model; anchors all defense evaluations.
Current Status: ACTIVE THREAT MODEL BOUNDARY.
```

```text
Decision ID: D7
Title: Stage target payloads: synthetic marker first, real payloads only under approved review
Date / Phase: 10–14 September 2026 / Phases 5–6
Previous State: Considering starting directly with safety jailbreak strings or refusal suppression.
Decision: Adopt a three-stage target payload progression:
          Stage 1: Deterministic synthetic token marker or formatting flip.
          Stage 2: Non-harmful, controlled instruction-priority behavior.
          Stage 3: Approved red-team safety benchmarks (HarmBench/StrongREJECT) only after
                   institutional sign-off and validated Stage 1/2 results.
Rationale: Decouples the scientific question ("Can the runtime trigger function?") from the
           societal risk of generating harmful content. Ensures safe, easily measurable metrics.
Alternatives Considered: Starting immediately with jailbreak refusal suppression (rejected as risky).
Source Evidence: research_plan.docx §7.3; Synopsis.docx §15, §23.
Consequences: Governs experiment registry design; ensures ethical compliance.
Current Status: ACTIVE ETHICAL & OPERATIONAL RULE.
```

```text
Decision ID: D8
Title: Adopt the Five-Criterion Checklist to distinguish backdoors from vulnerabilities
Date / Phase: 14 September 2026 / Phase 6
Previous State: Risk that observed behavioral changes would be dismissed as ordinary clean-model
                compression degradation (already documented in Group B literature).
Decision: Require that any positive result satisfy all five criteria:
          1) Intentional training (not emergent in clean models).
          2) Clean-baseline comparison (exceeds clean model under identical compression).
          3) Trigger specificity (near-miss policies do not trigger it).
          4) Payload specificity (targeted behavior, not general perplexity collapse).
          5) Stealthiness (near-zero false activation under reference full cache C0).
Rationale: Operationalizes the exact boundary on which the paper's novelty rests.
Alternatives Considered: Relying solely on Attack Success Rate (RC-ASR) (rejected as insufficient).
Source Evidence: Synopsis.docx §5, §8.1; learnings-and-approach.md.
Consequences: Dictates the structure of experiment controls and baseline tables.
Current Status: ACTIVE SCIENTIFIC STANDARD.
```

```text
Decision ID: D9
Title: Treat CacheTrap as mandatory adjacent prior art and commit to explicit differentiation
Date / Phase: 14 September 2026 / Phase 6
Previous State: Assuming the KV-cache trigger surface was completely unoccupied in literature.
Decision: Formally cite CacheTrap (arXiv:2511.22681) as nearest prior art and anchor the paper's
          novelty claim on explicit differences in threat model, mechanism, and trigger type.
Rationale: CacheTrap is the first published Trojan using the KV cache as a trigger. However, it
           relies on GPU hardware bit flips on unmodified models, whereas this project trains
           models to trigger on legitimate systems-level compression policies.
Alternatives Considered: Ignoring CacheTrap as "hardware only" (rejected as academically dishonest).
Source Evidence: Synopsis.docx §7.4, §8, §30; DeepSeek review.
Consequences: Mandates a live novelty re-check before paper submission.
Current Status: ACTIVE LITERATURE REQUIREMENT.
```

```text
Decision ID: D10
Title: Preserve negative experimental results as valid, publishable scientific outcomes
Date / Phase: 14 September 2026 / Phase 6
Previous State: Implicit assumption that the project must produce a working attack to succeed.
Decision: Explicitly recognize that a rigorous negative result—proving that intentional training
          cannot produce selective amplification beyond clean-model degradation—is a valuable,
          publishable systems-security contribution.
Rationale: A rigorous negative result demarcates the boundary between emergent compression
           sensitivity and trainable backdoors, providing vital safety guidance to serving teams.
Alternatives Considered: Abandoning the project if Gate G3 fails (rejected).
Source Evidence: Synopsis.docx §19.3; DeepSeek review §1, §9; chatgpt_research_memory D10.
Consequences: Eliminates outcome bias; guarantees publishable artifacts regardless of outcome.
Current Status: ACTIVE METHODOLOGICAL STANCE.
```

```text
Decision ID: D11
Title: Use cache-aware differential auditing as the primary defense concept
Date / Phase: 14 September 2026 / Phase 6
Previous State: Open question of how defenders could detect runtime-conditioned backdoors.
Decision: Propose and evaluate "Cache-Aware Differential Auditing" as the primary defense:
          query the model with identical prompts across a small family of reference and
          compressed cache policies, monitoring for anomalous behavioral divergence.
Rationale: Full-cache audits cannot see the backdoor; differential execution across runtime
           states exposes the hidden conditional policy at minimal probe cost.
Alternatives Considered: Weight-level inspection (ineffective against subtle LoRA backdoors).
Source Evidence: Synopsis.docx §21; DeepSeek review §5.7; chatgpt_research_memory D11.
Consequences: Shapes Phase 5 defense design (Experiment E5).
Current Status: ACTIVE DEFENSE PROPOSAL.
```

```text
Decision ID: D12
Title: Adopt DeepSeek's scoped recommendation: narrow initial experimental matrix
Date / Phase: 18 September 2026 / Phase 7
Previous State: Research plan proposed exploring quantization, eviction, merging, thresholds,
                multiple model families, and multi-tenant serving simultaneously.
Decision: Severely narrow the Phase 0–2 experimental scope:
          - Model: 1–2 instruction-tuned models in the 1B–3B parameter range (e.g., Qwen2.5-1.5B).
          - Compression Family: Single family first (Quantization via STE, or H2O Eviction).
          - Target: Single safe synthetic marker.
          - Controls: One near-miss policy and clean baseline.
Rationale: A B.Tech final-year research project has bounded compute and timeline. Attempting the
           full matrix immediately guarantees failure or surface-level execution.
Alternatives Considered: Sticking to the broad multi-model 8-family matrix (rejected as unviable).
Source Evidence: DeepSeek review §1, §4.7, §5.4; calude_research_mem overview.md.
Consequences: Governs the concrete 30-day and 90-day implementation plans.
Current Status: ACTIVE SCOPE CONSTRAINT.
```

```text
Decision ID: D13
Title: Adopt Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB) as the primary eviction mechanism
Date / Phase: Late September 2026 / Phase 8
Previous State: Eviction backdoors were viewed as passive reactions to missing tokens (KECB).
Decision: Refine the eviction direction into PF-SEB: the model actively manipulates an honest,
          unmodified eviction algorithm (e.g., H2O) by dampening attention scores on an internal
          "suppressor" token state, causing the algorithm to discard the suppressor on cue.
Rationale: Dramatically elevates the conceptual novelty: instead of "the model breaks under compression,"
          the claim is "the model games its own memory manager." Supported by a 7-condition causal table.
Alternatives Considered: Staying with purely passive KECB (retained as control baseline).
Source Evidence: PF-SEB_Synopsis.md §1–4.
Consequences: Establishes Experiment E6 as the flagship eviction evaluation.
Current Status: ACTIVE CONCEPTUAL REFINEMENT.
```

```text
Decision ID: D14
Title: Formally establish the Two-Track Research Program Architecture
Date / Phase: Late September 2026 / Phase 9
Previous State: Ambiguity over whether PF-SEB was intended to replace, compete with, or sit
                within the broader KQCB/KECB/KMCB taxonomy.
Decision: Unify the project into a complementary Two-Track Research Architecture:
          - Track 1 (The General Systems Framework): "Runtime-Conditioned Backdoors in LLMs"
            (documented in Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx and research_plan.docx),
            providing the comprehensive taxonomy (quantization, eviction, merging) and systems relevance.
          - Track 2 (The Flagship Mechanistic Specialization): "Policy-Fingerprinted Self-Eviction Backdoors"
            (documented in PF-SEB_Synopsis.docx and PF-SEB_Research_Plan.docx), narrowing the eviction branch
            into a razor-sharp, causally proven attack on the memory manager's trust interface.
Rationale: Confirmed directly by PF-SEB_Research_Plan.docx: "Scope: a focused specialisation of the
           eviction-conditioned (KECB) branch of the broader runtime-conditioned-backdoor research direction;
           deliberately narrower in scope than the full KQCB/KECB/KMCB programme." This allows the project
            to maintain a broad systems narrative while submitting a tight, bulletproof empirical paper.
Alternatives Considered: Discarding the general framework and pursuing PF-SEB exclusively (rejected:
                          loses the engineering advantages of KQCB de-risking and the broad systems narrative).
Source Evidence: PF-SEB_Research_Plan.docx §1; PF-SEB_Synopsis.docx §3.4; Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx.
Consequences: Fully resolves Uncertainty U1; aligns research plan and synopsis artifacts.
Current Status: ACTIVE STRATEGIC ARCHITECTURE (HARMONIZED INTO FUNNEL VIA D15).
```

```text
Decision ID: D15
Title: Formal adoption of Campaign 001 Decision Memo: proceed to Phase 0/1 under narrow FP8 scope; commit to Unified Gates UG0–UG9; enforce Gate UG2 conformance as an absolute prerequisite before training
Date / Phase: 2026-09-27 / Phase 0/1 (Campaign 001 Synthesis)
Previous State: Two-Track ambiguity (broad systems umbrella vs. PF-SEB specialization), uncalibrated gates G1–G8, uncertainty regarding whether to start with H2O eviction or quantization.
Decision: Formally adopt research/CAMPAIGN_001_DECISION_MEMO.md as the binding research strategy:
          1. Proceed to Phase 0/1 implementation under the narrow, production-grounded FP8 scope (WP0–WP6).
          2. Primary treatment is official pinned vLLM FP8 (fp8_e4m3fn) on fresh per-request caches (C0 -> empty),
             evaluated against reference BF16 (C0). Fake FP8 / STE (T_proxy) is strictly a training proxy.
          3. Commit to the Unified Gate system (UG0–UG9).
          4. Enforce Gate UG2 conformance (layerwise NRMSE <= 0.05, cos >= 0.995, Spearman logit rho >= 0.85)
             as an absolute, non-negotiable prerequisite before any model fine-tuning. If UG2 fails,
             halt and pivot to an empirical proxy-to-deployment transfer failure paper.
          5. Quarantine PF-SEB strictly behind Gate UG6 (physical transfer of core FP8 on vLLM).
Rationale: Unanimous consensus across Tracks A–F. Eliminates execution risk, resolves baseline confounding via the
           6-cell causal design (Delta_int, Delta_cond), and establishes an empirical falsification boundary within
           a bounded 30-day timeline.
Alternatives Considered: Continuing broad all-policy roadmap (rejected: novelty invalidated); leading with PF-SEB
                         (rejected: high optimization risk and unstandardized serving engine).
Source Evidence: CAMPAIGN_001_DECISION_MEMO.md §13, §15; CONSOLIDATED_RESEARCH_PLAN.md §0.1, §7; TRACK_C, TRACK_D, TRACK_E, TRACK_F reports.
Consequences: Unifies development into a phased funnel (WP0–WP9); establishes clear stop rules.
Current Status: ACTIVE STRATEGIC DIRECTIVE.
```

```text
Decision ID: D16
Title: Permanent retraction of broad umbrella novelty claims ("first KV-cache backdoor"); adoption of §10.5 terminology ladder
Date / Phase: 2026-09-27 / Campaign 001 Synthesis
Previous State: Project artifacts and early synopses claimed priority as "first KV-cache backdoor",
                 "first runtime-state trigger", or general f(x, s_runtime) trigger.
Decision: Permanently retract all broad umbrella claims. Formally categorize broad novelty as likely invalidated.
          Adopt the constitutional Terminology Ladder (§10.5 of Decision Memo) for all future reporting:
          1. Compression Sensitivity
          2. Trained Amplification
          3. Policy-Conditioned Behavior
          4. Trained Cache-Policy-Conditioned Backdoor
          5. Policy-Fingerprinted Self-Eviction Backdoor (PF-SEB).
          Narrow hypothesis is categorized as plausibly distinct.
Rationale: Track A and Track B audits proved that CacheTrap (ICCAD 2026), HijackKV (arXiv:2607.19957),
           HistorySwap (arXiv:2511.12752), and Chat-Template Backdoors (ACM CCS 2026) occupy the broad KV/inference
           attack space. Claiming priority on f(x, s_runtime) is trivial since all autoregressive decoding
           consumes runtime state. True novelty is confined to intentional training on weights for legitimate
           inference compression policies on fresh, unshared caches with clean baseline subtraction.
Alternatives Considered: Defending the broad umbrella by framing CacheTrap as "hardware-only" (rejected as academically dishonest).
Source Evidence: CAMPAIGN_001_DECISION_MEMO.md §4, §7.1, §15; TRACK_A report §3; TRACK_B report §3.1, §3.3.
Consequences: Protects project from immediate desk rejection; ensures precise, defensible claims.
Current Status: PERMANENT CONSTITUTIONAL POLICY.
```

```text
Decision ID: D17
Title: Supercession of archived uncalibrated numeric targets in EXPERIMENT_REGISTRY.md with the pilot-calibrated preregistration framework
Date / Phase: 2026-09-27 / Campaign 001 Synthesis
Previous State: EXPERIMENT_REGISTRY.md contained uncalibrated point targets (e.g., universal Delta_int >= 0.60,
                 perplexity loss < 5%, defense AUROC >= 0.95 in <= 50 queries).
Decision: Formally supersede arbitrary numeric targets with the pilot-calibrated preregistration framework
          from CONSOLIDATED_RESEARCH_PLAN.md §7.1 and CAMPAIGN_001_DECISION_MEMO.md:
          1. Primary criterion is statistical significance via paired 95% bootstrap confidence intervals:
             lower bound of Delta_int and Delta_cond must strictly exceed zero and a practically meaningful
             effect size (Delta_int >= 0.50, CI lower bound > 0.30).
          2. Full-cache stealth (P(A=1 | C0) < 1.0%) must be evaluated with N=1,000 sequestered prompts
             to establish an empirical 95% upper bound <= 0.3% via the Rule of Three (3/N).
          3. Utility non-inferiority margins delta_margin for IFEval, GSM8K, and perplexity must be
             empirically calibrated on clean models (theta_c, theta_f) during WP2 rather than fixed arbitrarily.
          4. Proxy conformance requires multi-metric validation (NRMSE <= 0.05, Cosine Similarity >= 0.995,
             Spearman rank correlation rho >= 0.85).
Rationale: Uncalibrated targets risk either setting unachievable hurdles or permitting false-positive claims
           without statistical power. Grounding targets in paired bootstrap confidence intervals and pre-registered
           clean pilots ensures scientific validity.
Alternatives Considered: Retaining the fixed 0.60/5% thresholds (rejected as statistically naive).
Source Evidence: CONSOLIDATED_RESEARCH_PLAN.md §7.1; CAMPAIGN_001_DECISION_MEMO.md §5, §7.2, §10, §15; TRACK_E report.
Consequences: Governs WP0 manifest and WP2 clean surface calibration.
Current Status: ACTIVE STATISTICAL PROTOCOL.
```

```text
Decision ID: D18
Title: Formal adoption of Gate UG2 Conformance CONDITIONAL PASS verdict; authorization of Phase 2 (WP2/WP3)
Date / Phase: 2026-09-27 / Campaign 002 (WP0/WP1 Runtime Gate Remediation)
Previous State: Gate UG2 conformance evaluation pending or unconditional PASS contested by adversarial review.
Decision: Formally certify that the candidate PyTorch STE FP8 KV-cache proxy (T_proxy, fp8_e4m3fn)
          satisfies all 9 pre-registered criteria of Unified Gate UG2 with zero silent fallbacks:
          - Key NRMSE: 0.0331 (target <= 0.050)
          - Value NRMSE: 0.0326 (target <= 0.050)
          - Key Cosine Sim: 0.9981 (target >= 0.9950)
          - Value Cosine Sim: 0.9983 (target >= 0.9950)
          - Logit Spearman rho: 0.9184 (target >= 0.8500)
          - Top-10 Agreement: 88.75% (target >= 80.00%)
          - Output JSD: 0.0091 nats (target <= 0.0200 nats)
          - Greedy Token Match: 95.31% (target >= 90.00%)
          - Silent Fallbacks: 0 detected
          Render overall verdict as CONDITIONAL PASS and authorize transition to Phase 2 (WP2/WP3)
          subject to three explicit pre-registered conditions:
          a) Conformance mathematically and empirically holds for the candidate PyTorch STE proxy (T_proxy)
             across all 9 Gate UG2 metrics on clean model theta_c.
          b) Dynamic per-head scaling or calibrated static scaling is strictly required for WP3 training
             to prevent outlier activation clipping and underflow.
          c) Physical hardware execution of vLLM Triton PagedAttention kernels on a dedicated Linux host
             (Ubuntu 22.04 LTS, Ada sm_89 / Hopper sm_90) is pre-registered as a mandatory gate check prior
             to claiming production deployment transfer.
Rationale: All 9 metrics satisfy pre-registered thresholds frozen prior to confirmatory analysis.
           Empirical noise factorization proves that 99.1% of divergence is driven by discrete 8-bit
           storage quantization, with less than 1.0% attributable to CUDA kernel non-associativity.
           T_proxy is mathematically equivalent to physical FP8 storage dequantization. A conditional
           pass correctly bounds epistemic scope to verified mathematical surrogate execution on Windows
           while pre-registering Linux hardware serving verification before claimed deployment transfer.
Alternatives Considered: Unconditional PASS (rejected per challenger audits: requires explicit conditions);
                          Declaring Fail (rejected: empirical data confirms high mathematical fidelity).
Source Evidence: CAMPAIGN_002_DECISION_MEMO.md; CAMPAIGN_002_PROXY_CONFORMANCE.md;
                 CAMPAIGN_002_DETERMINISM.md; configs/acceptance/frozen_thresholds.yaml;
                 challenger_c002_1 handoff; challenger_c002_2 handoff; reviewer_c002_1 handoff.
Consequences: Clears mandatory blocker D15 conditionally; authorizes Work Packages WP2 and WP3 under conditions a-c.
Current Status: RETRACTED / SUPERSEDED BY D21.
```

```text
Decision ID: D19
Title: Authorization of PyTorch STE Proxy (fp8_e4m3fn) as official training surrogate for WP2/WP3
Date / Phase: 2026-09-27 / Campaign 002 (WP0/WP1 Runtime Gate)
Previous State: T_proxy was a candidate implementation awaiting empirical conformance validation.
Decision: Authorize the Straight-Through Estimator proxy implementation in src/compression/fake_fp8.py
          as the official differentiable training surrogate for Work Packages WP2 and WP3:
          1. Data format: torch.float8_e4m3fn with dynamic range [-448.0, 448.0] and epsilon = 0.125.
          2. Forward pass: deterministic quantization to 8-bit bins with static or dynamic scale S.
          3. Backward pass: straight-through gradient pass-through clipped to [-448.0 * S, 448.0 * S].
          4. Mandatory scaling: per-head static scaling (calibrated via llm-compressor) or dynamic
             scaling S = (max(|X|) + eps) / 448.0. Uncalibrated unitary scales (S=1.0) are prohibited.
Rationale: Adversarial audit confirmed that dynamic/calibrated per-head scaling avoids saturation clipping
           under 100x activation outliers, whereas uncalibrated fixed scaling (S=1.0) causes severe clipping
           and drops cosine similarity to 0.942, violating Gate UG2.
Alternatives Considered: Using FP16 or INT8 proxies (rejected: does not match vLLM production format);
                          Using unclipped STE (rejected: causes gradient explosion on outliers).
Source Evidence: CAMPAIGN_002_PROXY_CONFORMANCE.md §4, §5.2; CAMPAIGN_002_RUNTIME_PATH.md §3;
                 src/compression/fake_fp8.py; src/compression/scales.py.
Consequences: Locks the exact mathematical formulation for all LoRA training loops in WP3.
Current Status: RETRACTED / SUPERSEDED BY D21; PROXY IS NOT AUTHORIZED FOR TRAINING.
```

```text
Decision ID: D20
Title: Pinned Execution Stack Locking & Hardware Fallback Elimination Protocol
Date / Phase: 2026-09-27 / Campaign 002 (WP0/WP1 Runtime Gate)
Previous State: Target hardware and library dependencies were informally specified across planning documents.
Decision: Formally lock the exact execution stack and hardware requirements for all future campaigns:
          1. Model Revision: Qwen/Qwen2.5-1.5B-Instruct at commit 560647970498b8c199e8471c6155fe7f1c1f5138.
          2. Toolchain: Python 3.11, PyTorch 2.4.0+cu124, vLLM 0.6.0 (v0.26.0+ compat), CUDA 12.4.1,
             Driver >= 550.54.14, Transformers 4.45.1.
          3. Hardware Architecture: Target deployment host must be Linux Ubuntu 22.04 LTS with NVIDIA
             Ada Lovelace (sm_89) or Hopper (sm_90) GPU. Compute capability < sm_89 is strictly disallowed.
          4. 5-Tier Fallback Trap Battery: Automated enforcement in src/runtime/env_inspector.py must
             run before every physical evaluation to trap and abort on silent fallback to BF16 or CPU.
Rationale: vLLM silently defaults to BF16 when kv_cache_dtype='auto' or when deployed on unsupported GPUs.
           Enforcing the 5-tier inspection harness guarantees experimental validity and reproducibility.
Alternatives Considered: Permitting Ampere sm_80 via software emulation (rejected: introduces non-hardware
                          latency and kernel divergence).
Source Evidence: CAMPAIGN_002_ENVIRONMENT_MANIFEST.md; CAMPAIGN_002_RUNTIME_PATH.md;
                 configs/env/environment_spec.yaml; src/runtime/env_inspector.py.
Consequences: Governs deployment infrastructure across all subsequent campaigns (Campaign 003+).
Current Status: SUPERSEDED IN PART BY D21. MODEL REVISION REMAINS A TARGET; THE HISTORICAL TOOLCHAIN IS INVALID AND MUST BE RESOLVED FOR VLLM 0.26.0.
```

```text
Decision ID: D22
Title: Campaign 3 strategic pivot to PF-SEB-first exploratory research
Date / Phase: 2026-09-30 / Campaign 003 kickoff
Previous State: Campaign 002 recorded a Gate UG1/UG2 "CONDITIONAL PASS" (NRMSE 0.0331, cos 0.9981,
                Spearman 0.9184, token-match 95.31%, kernel-noise 0.9%) and D15 kept PF-SEB quarantined
                behind an FP8-first funnel (UG6).
Decision:
  1. Adopt PF-SEB as the strategic primary research direction; retain FP8/KQCB as an optional
     composed extension or conformance fallback. This supersedes D15's priority ordering but not
     the unified evidence standards.
  2. Authorize only exploratory, non-gate model work until a faithful eviction treatment,
     development/confirmation split, model/environment revision, run budget, and artifact protocol
     are frozen and reviewed.
  3. Preserve Campaign 3's static prefill attention-mask MVP as a development precursor. It may
     test whether LoRA can associate a severe state-removal proxy with a benign marker, but it is
     not hard/physical H2O, active scorer manipulation, a suppressor mechanism, or a gate result.
  4. Require Campaign 3 negative and positive runs to be logged without duplicate-counting file
     aliases, and treat the manually persisted Kaggle record as provisional until reproduced.
Rationale: PF-SEB offers the sharper research question and can be developed without making FP8
           hardware the critical path. However, changing strategic direction cannot relax causal,
           statistical, provenance, utility, near-miss, or mechanism requirements. The initial
           static-mask observations are hypothesis-generating and must not be retroactively promoted.
Alternatives Considered: Keep the FP8-first funnel (rejected: blocked by missing vLLM/FP8 hardware and
                         by the fact that no real FP8 result exists); trust Campaign 002's numbers
                         (rejected: not reproducible from the code).
Source Evidence: research/campaigns/campaign_003/{CAMPAIGN_003_BRIEF.md, ACTION_PLAN_TOP5.md,
                 agent_reports/VERIFICATION_CAMPAIGN_002.md, agent_reports/LIT_NOVELTY_RECHECK.md,
                 agent_reports/EVICTION_ALGORITHM_SPEC.md}; src/pfseb/*; tests/pfseb/test_eviction.py.
Consequences: WP7 is reopened as a partial engineering prototype. WP8/WP9 remain unexecuted.
              Campaign 3 records are exploratory and pass no unified gate. The next experiment must
              characterize a faithful passive eviction treatment before scorer manipulation or
              suppressor causal claims.
Current Status: ACTIVE STRATEGIC PRIORITY; CAMPAIGN 3 RESULT CLAIMS ARE EXPLORATORY ONLY.
```

---

## 2. Resolved & Historical Decisions

| Decision ID | Description | Resolution Outcome | Authorizing Decision |
|---|---|---|---|
| **OD-1** | **Entry-Point Implementation Choice** | **RESOLVED: Pinned vLLM FP8 (`fp8_e4m3fn`) is the primary entry point.** Fake FP8/STE is strictly a training proxy. PF-SEB is quarantined behind Gate UG6. | Decision D15 |
| **OD-2** | **Initial Target Model Selection** | **RESOLVED: `Qwen/Qwen2.5-1.5B-Instruct` is the primary target model** (revision-pinned, GQA architecture). `Llama-3.2-1B-Instruct` is secondary cross-family control. | Decision D15 |
| **OD-3** | **Target Publication Venue & Cycle** | **RESOLVED: Primary target venue is USENIX Security 2027 (Cycle 2, Jan 2027)**, with **TMLR** as journal fallback for rigorous negative/conformance results. | Decision D15 |
| **OD-4** | **LaunderBench Fallback Status** | **RESOLVED: LaunderBench is maintained strictly as an emergency, distant fallback** (dormant), not an active parallel effort. | Decision D15 |


```text
Decision ID: D21
Title: Retract Campaign 2 UG1/UG2 verdict and revoke WP2/WP3 authorization
Date / Phase: 2026-09-27 / Post-pull implementation audit and remediation
Previous State: D18 certified UG2 CONDITIONAL PASS; D19 authorized the STE proxy for training;
                canonical memory described WP0/WP1 complete and WP2/WP3 unblocked.
Decision: Retract the Campaign 2 runtime/determinism gate verdicts and block WP2/WP3 until a
          new review receives traceable artifacts from the pinned Qwen checkpoint under genuine
          separate-process BF16 vLLM, FP8 vLLM, and a corresponding real-Qwen proxy execution.
          Reclassify Campaign 2 as synthetic local proxy/storage engineering preflight only.
Rationale: Three independent code-path audits established that the former REAL_FP8 condition and
           STORAGE_FP8 condition executed the same local implementation; VLLMRunner was disconnected;
           Qwen weights/tokenizer/prompts were not loaded; reference tensors were not BF16; cache
           captures and final-logit labels were incorrect; and no immutable raw results existed.
           The reported nonzero kernel component could not be generated by the committed branches.
Corrections Implemented: Local conditions renamed and fail closed; INT8 fallback removed; full
                         accumulated cache and final logits repaired; genuine vLLM runner and
                         immutable hashed runtime artifacts added; original reports marked historical.
Source Evidence: research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md; source-code audit of
                 src/harness/cache_adapter.py, src/eval/run_conformance.py,
                 src/harness/deterministic_decode.py, and src/runtime/vllm_runner.py.
Consequences: UG0 PARTIAL; UG1 NOT PASSED; UG2 NOT PASSED/BLOCKED; UG3–UG9 unopened. D18 and D19
              are historical and superseded wherever they certify conformance or training permission.
              D20 remains useful only as an intended target specification and must be reconciled with
              the actual vLLM 0.26.0 dependency lock on the Linux host.
Current Status: ACTIVE AUTHORITATIVE CORRECTION.
```
