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

```text
Decision ID: D22a   (renumbered 2026-10-18 from a duplicate "D22"; G12 housekeeping)
Title: Campaign 004 — Authorization of Policy Selectivity, Budget Thresholds, & Causal Verification Battery
Date / Phase: 2026-10-06 / Campaign 004 Transition
Previous State: Campaign 003 confirmed Rung 2 (trained cache-conditioned amplification) on Qwen2.5-1.5B (Kaggle GPU)
                and 0.5B (CPU) with Delta_int = 1.0 at budget=8 under H2O eviction. Selectivity across policies,
                budget threshold behavior, and causal interventions remained unmeasured.
Decision:
  1. Authorize Campaign 004 focusing on:
     a) WP4.1: Policy Selectivity Matrix (H2O vs SnapKV, Scissorhands, Recency, Random, Full Cache).
     b) WP4.2: Eviction Budget Sweep (B in {8, 12, 16, 20, 24, 32, 48, full}).
     c) WP4.3: 3-Condition Causal Intervention Battery (Rescue Delta_rescue, Induction Delta_induction, Random Deletion Delta_random).
     d) WP4.4: Fine-Tuned Control Model Baseline (theta_f) for Delta_cond calibration.
  2. Implement unified testing suite scripts/run_pfseb_campaign_004.py enabling dual-mode execution:
     - Local CPU fast verification mode (0.5B, sub-sampled prompts, deterministic).
     - Full GPU execution mode (1.5B, 24 held-out prompts, multi-seed 42, 123, 7).
  3. Pre-register acceptance thresholds:
     - Delta_policy >= 0.40 between attention policies and recency/random baselines.
     - Delta_rescue >= 0.60 and Delta_induction >= 0.60 with Delta_random <= 0.05.
     - Delta_cond >= 0.50 with two-sided 95% bootstrap CI lower bound > 0.30.
Rationale: Confirmatory evidence discipline demands that we test whether the observed Rung-2 effect is
           policy-fingerprinted or generic context truncation before claiming higher ladder positions.
           Establishing the causal battery is required by AGENTS.md to differentiate intentional backdoors
           from compression-induced degradation.
Alternatives Considered: Immediate paper drafting on Rung-2 MVP (rejected: reviewers would question specificity);
                         Jumping directly to FP8 vLLM integration (rejected: eviction mechanism must be causally
                         characterized first in PyTorch).
Source Evidence: research/campaigns/campaign_004/{CAMPAIGN_004_BRIEF.md, CAMPAIGN_004_PLAN.md,
                 CAMPAIGN_004_DECISION_MEMO.md, RUNBOOK_AND_EXPERIMENTS.md}; scripts/run_pfseb_campaign_004.py.
Consequences: Unlocks experimental transition from Rung 2 to Rungs 3–4 upon successful execution.
Current Status: ACTIVE FORMAL DIRECTIVE (Campaign 004).
```

```text
Decision ID: D23a   (renumbered 2026-10-18 from a duplicate "D23"; G12 housekeeping)
Title: Campaign 004 — Verification Gate PASS & Implementation Completion
Date / Phase: 2026-10-07 / Campaign 004 Conclusion
Previous State: Campaign 004 was active across dual implementation and testing tracks.
Decision:
  1. Render formal Gate PASS verdict on Campaign 004 implementation:
     - Multi-policy spectrum (H2O, SnapKV, Scissorhands, Recency, Random, None) fully instrumented.
     - 3-condition causal battery (Rescue, Induction, Size-Matched Random Deletion with |R|=|E|) validated.
     - Fine-tuned control baseline (theta_f) with dual benign loss (lambda_marker=0.0) implemented.
     - VRAM-safe sequential 4-phase lifecycle runner (scripts/run_pfseb_campaign_004.py) operational.
     - 100% of unit, boundary, and regression tests pass (tests/test_campaign_004.py 26/26, test_milestone2.py 5/5, test_eviction_adversarial.py 12/12).
     - End-to-end smoke verification pipeline executed cleanly and output serialized to results/campaign_004/smoke_verification.json.
  2. Authorize Kaggle GPU confirmatory evaluation suite across seeds 42, 123, 7.
Rationale: All pre-registered requirements R1–R5 and features F1–F12 are verified with zero mocks, clean forensic audits, and passing unit tests.
Source Evidence: CAMPAIGN_004_DECISION_MEMO.md; KAGGLE_CAMPAIGN_004.md; results/campaign_004/smoke_verification.json;
                 tests/test_campaign_004.py; scripts/run_pfseb_campaign_004.py.
Consequences: Concludes Campaign 004 implementation; clears path for Kaggle GPU execution and Campaign 005.
Current Status: ACTIVE FORMAL VERDICT (PASS).
```

```text
Decision ID: D23
Title: Formal Specification of 5-Policy Spectrum and 3-Part Causal Intervention Battery
Date / Phase: 2026-10-07 / Campaign 004 (Phase 3 Synthesis)
Previous State: Campaign 003 measured only H2O eviction at fixed budget B=8; cross-policy
                selectivity and causal position mediation remained uncharacterized.
Decision:
  1. Standardize on the 5-policy evaluation spectrum: H2O (cumulative attention), SnapKV
     (observation-window pooling), Scissorhands (persistence budgeting), Recency-only, and
     Random eviction.
  2. Implement the 3-part causal battery:
     - Rescue: Programmatically pinning evicted positions under trigger condition.
     - Induction: Programmatically masking candidate positions under full cache C0.
     - Size-Matched Random Deletion: Programmatically masking |R|=|E| random positions under C0,
       formally resolving Defect G3 (strict size equality without arbitrary clamping).
  3. Pre-register acceptance thresholds:
     - Delta_policy = ASR(H2O) - ASR(Random) >= 0.40.
     - Delta_rescue >= 0.60, Delta_induction >= 0.60, Delta_random <= 0.05.
Rationale: Confirmatory evidence discipline requires establishing whether marker emission is
           uniquely mediated by the memory manager's attention-derived token choices or is an
           unintended side-effect of generic sequence truncation.
Alternatives Considered: Evaluating only H2O vs Random (rejected: fails to distinguish attention-pooling
                          from cumulative scoring); Clamping random deletion size to fixed small constant
                          (rejected: violates causal balance between |R| and |E|).
Source Evidence: research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md; src/pfseb/eviction.py;
                 src/pfseb/causal.py; tests/pfseb/test_causal.py.
Consequences: Governs all empirical analysis and paper claims for Campaign 004.
Current Status: ACTIVE FORMAL SPECIFICATION.
```

```text
Decision ID: D24
Title: Specification and Architectural Locking of Fine-Tuned Control Baseline (theta_f)
Date / Phase: 2026-10-07 / Campaign 004 (Phase 3 Synthesis)
Previous State: Baseline evaluation relied solely on untouched base model theta_c, leaving open
                the confounder of task fine-tuning exposure and gradient adaptation drift.
Decision:
  1. Train theta_f using the exact base model, LoRA rank (r=8, alpha=16), optimizer, learning rate,
     and compute budget as theta_b.
  2. Train theta_f exclusively on dual benign continuation loss:
     L_theta_f = L_CE(y_benign | C0) + L_CE(y_benign | T_H2O) with lambda_marker = 0.0.
  3. Require Delta_cond = [P(m|H2O, theta_b) - P(m|C0, theta_b)] - [P(m|H2O, theta_f) - P(m|C0, theta_f)] >= 0.50
     with two-sided 95% bootstrap CI lower bound > 0.30.
Rationale: Eliminates the hypothesis that fine-tuning on cache-compressed prompts induces generic
           marker vulnerability.
Alternatives Considered: Fine-tuning theta_f only on full cache C0 (rejected: does not expose model
                          to eviction-time representations during training).
Source Evidence: research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md; src/pfseb/train_mvp.py;
                 tests/pfseb/test_milestone2.py; scripts/run_pfseb_campaign_004.py.
Consequences: Unlocks Rung 3 (Policy-Conditioned Behavior) and Rung 4 (Trained Cache-Policy-Conditioned Backdoor)
              of the constitutional terminology ladder upon confirmatory GPU execution.
Current Status: ACTIVE FORMAL SPECIFICATION.
```

```text
Decision ID: D25
Title: Empirical Ratification of GPU Seed 42 Results; Reclassification from Policy-Fingerprint to Capacity/Threshold Trigger
Date / Phase: 2026-10-08 / Campaign 004 GPU Production Analysis
Previous State: Hypothesis assumed trained model theta_b would exhibit policy fingerprinting (high ASR under H2O, low ASR under SnapKV/Recency/Random).
Decision:
  1. Record empirical findings from 2,107s GPU production run (Qwen2.5-1.5B-Instruct, seed 42, 25 eval prompts, 20 epochs):
     - Intentional Amplification: Delta_int = 1.00 (95% CI [1.00, 1.00]) — CONFIRMED.
     - Fine-Tuning Control Isolation: Delta_cond = 1.00 (95% CI [1.00, 1.00]) — CONFIRMED.
     - Full-Cache Stealth: P(m* | C0) = 0.00 (0/25) — CONFIRMED.
     - Baseline Isolation: theta_c = 0.00, theta_f = 0.00 across all conditions — CONFIRMED.
     - Budget Threshold: Sharp sigmoidal phase transition between B=20 (96%) and B=24 (12%), reaching 0% at B >= 32 — CONFIRMED.
     - Causal Battery: Rescue Delta_rescue = 1.00 (100% suppression), Induction Delta_induction = 1.00 (100% activation), Random Deletion Delta_random = 0.68.
     - Policy Selectivity: Falsified at B=8 (H2O: 1.0, Scissorhands: 1.0, Recency: 1.0, Random: 0.96, SnapKV: 0.92).
  2. Formally reclassify the empirical attack mechanism from 'Policy-Fingerprinted Backdoor' to 'Runtime Capacity-Conditioned Backdoor (RCCB)' / 'Cache Budget Threshold Trigger'.
  3. Formulate Campaign 005 to perform mechanistic circuit localization and investigate whether contrastive multi-policy training can induce algorithmic selectivity if desired.
Rationale: Confirmatory GPU evidence proves intentional KV-cache conditioning and stealth are 100% successful, but eviction policy specificity is absent under single-policy training at B=8. The backdoor responds to severe KV truncation regardless of eviction heuristic.
Source Evidence: results/campaign_004/kaggle_decisive_seed42.json; research/campaigns/campaign_004/GPU_ANALYSIS_SEED42.md.
Consequences: Clarifies threat model and academic positioning; guides Campaign 005 circuit analysis.
Current Status: ACTIVE EMPIRICAL FINDING & FORMAL DECISION.
```

```text
Decision ID: D26
Title: Campaign 005 Formal Implementation & Gating Verification PASS
Date / Phase: 2026-10-08 / Campaign 005 (Mechanistic Circuits, Defenses & Auditing)
Previous State: Campaign 004 verified capacity-conditioned backdoor behavior but left internal circuit mechanics and practical defenses uncharacterized.
Decision:
  1. Formally certify that Campaign 005 satisfies all requirements R1–R5:
     - R1: Causal layerwise restoration sweep and 336-head attribution engine implemented in src/pfseb/circuit.py.
     - R2: Security-aware retention defense suite (S-Pin, L-Evict, Budget Guardrail) implemented in src/pfseb/defenses.py.
     - R3: Differential pre-deployment canary audit harness implemented in src/eval/canary_audit.py.
     - R4: Contrastive multi-policy analytical Jaccard bound implemented in src/pfseb/contrastive_bound.py.
     - R5: VRAM-safe master execution runner delivered in scripts/run_pfseb_campaign_005.py.
  2. Verify 100% test pass rate across 94 automated tests in tests/test_campaign_005.py and successful dry-run execution generating master artifacts in results/campaign_005/.
  3. Authorize multi-seed Kaggle GPU production execution (KAGGLE_CAMPAIGN_005.md).
Rationale: Empirical verification proves that layer-selective eviction suppresses the backdoor (ASR 0.05) with 68.6% memory savings and canary auditing achieves AUROC 1.0000.
Source Evidence: research/campaigns/campaign_005/CAMPAIGN_005_DECISION_MEMO.md; results/campaign_005/run_pfseb_campaign_005.json; tests/test_campaign_005.py.
Consequences: Unlocks Rung 4 and Rung 5 of the research program; completes all Phase 4 (mechanism) and Phase 5 (defense) technical deliverables.
Current Status: ACTIVE FORMAL VERIFICATION & DECISION.
```

```text
Decision ID: D27
Title: Campaign 005 evidence correction (simulation vs measurement) and Campaign 006 approval with scope revision (Phase 0 remediation gate)
Date / Phase: 2026-10-10 / Campaign 006 kickoff
Previous State: D26 certified Campaign 005 as "VERIFIED (PASS)" and canonical memory reported
                circuit localization (L_crit=[2,3,4,5]), L-Evict defense (ASR 0.05, 68.6%),
                canary AUROC 1.00, and the H2O/SnapKV overlap bound as [EXPERIMENTAL RESULT].
Decision:
  1. CORRECTION: The committed Campaign 005 master artifact
     (results/campaign_005/run_pfseb_campaign_005.json) is a deterministic SIMULATION.
     The runner fabricates the layer-delta profile, returns hard-coded defense ASRs from
     run_phase_3_security_defenses, and uses synthetic control negatives in
     run_phase_4_canary_audit. A live GPU run was executed, but only Phase 1 (baseline) and
     the layer-restoration sweep were genuinely measured; the live artifact was not
     committed and its head attribution differed materially (top sensing L27H3, payload
     DLA ~ 0). Campaign 005 circuit/defense/audit claims are reclassified as
     [SIMULATION / UNVERIFIED] in CURRENT_STATE.md, CLAIMS.md, and EXPERIMENT_REGISTRY.md.
  2. REMEDIATION (implemented): scripts/run_pfseb_campaign_005.py now measures S-Pin,
     L-Evict, and the guardrail from real generations, computes the canary control from a
     real adapter-zeroed pass, and fails closed: simulation/partial runs cannot emit PASS
     verdicts (they emit SIMULATION / NOT_MEASURED and metadata.execution_mode).
  3. RQ STRUCTURE: Campaign 006's proposed RQ6/RQ7/RQ8 are recorded as explicit extensions
     mapped onto canonical RQ1 (existence/scale), RQ2 (trigger structure/architecture), and
     RQ3 (runtime thresholds/context). RQ1-RQ5 remain the canonical structure.
  4. SCOPE: The submitted Campaign 006 draft (8B/70B, Mixtral-8x7B, multi-GPU, 32K-128K at
     once) is revised for feasibility and constitution compliance (D12; CONSOLIDATED plan
     §3.4): Tier A cross-architecture at ~1B on a single 16GB T4; Tier B within-family scale
     via QLoRA for >=3B (declared confound); long-context (>8K) is a stretch tier only.
  5. GATE: Campaign 006 Phase 0 (Campaign 005 live re-run; Campaign 004 seeds 123/7;
     Campaign 003 adapter provenance; memory correction) is a blocking prerequisite (CG0).
Rationale: Building a new campaign on simulation-derived "success" would repeat the
           Campaign 002 -> D21 evidence failure. Scope must respect the project's own
           compute and context constraints. RQ preservation is required by AGENTS.md.
Source Evidence: research/campaigns/campaign_006/campaign_006_plan.md;
                 scripts/run_pfseb_campaign_005.py (Phase 0 fix);
                 results/campaign_005/run_pfseb_campaign_005.json (simulation);
                 user-provided Campaign 005 live run log;
                 researchMemory/CURRENT_STATE.md, CLAIMS.md, EXPERIMENT_REGISTRY.md.
Consequences: Campaign 005 claims are downgraded; the runner fails closed; Campaign 006 is
              authorized with a Phase 0 remediation gate and a narrowed, staged scope.
Current Status: ACTIVE CORRECTION AND CAMPAIGN 006 AUTHORIZATION.
```

```text
Decision ID: D28
Title: Campaign 006 results adjudication — PARTIAL PASS; trained capacity-conditioned amplification
       replicated across three architecture families; ladder position remains Rung 2
Date / Phase: 2026-10-18 / Campaign 006 conclusion
Previous State: Campaign 006 (D27) executed on Kaggle T4 for qwen15, gemma2b, and mistral7b over seeds
                42/123/7; no results, conclusions, or decisions recorded in canonical memory.
Decision:
  1. Record the Campaign 006 results as [EXPERIMENTAL RESULT — exploratory/development tier]
     (live GPU, 25 non-sequestered prompts per cell; best epoch selected on the same prompts):
     - Qwen2.5-1.5B (3 seeds), Gemma-2-2B (2 fully measured seeds + seed-123 training-time eval
       only), Mistral-7B-v0.3 (3 seeds, 4-bit NF4 base in train AND eval): Delta_int = Delta_cond
       in [0.96, 1.00], all CI lows >= 0.88; theta_c and theta_f 0/25 under C0 and H2O B=8 in all runs.
     - Budget threshold B50: Qwen 21.7 +/- 0.6, Gemma 10.5 +/- 0.4, Mistral 11.0 +/- 0.2 (train B=8).
     - Policy spectrum: Qwen non-selective; Gemma non-selective except random 0.64; Mistral
       H2O/SnapKV/Scissorhands 0.84-1.00 vs Recency/StreamingLLM 0.04-0.12 (Delta(H2O-Recency) = 0.91).
       Pre-registered Delta_policy = H2O - Random >= 0.40 met in 1/8 runs (Mistral seed 7).
     - Delta_random: Qwen 0.96, Gemma 0.64, Mistral 0.37 +/- 0.28 -> D23 criterion (<= 0.05) fails in all runs.
     - L-Evict (mask-only): ASR 0.00 on Gemma (layers 1-2, 2/2) and Mistral (fallback layers 0-6, 2/3);
       ASR 1.00 on Qwen (layer 0, 3/3) and 0.64 on Mistral seed 7 (layer 0). Compression ratios analytic.
     - Canary AUROC: Qwen 1.00 (3/3), Gemma 1.00 (2/2), Mistral 0.47-0.53 (chance, 3/3).
     - Stealth: 0 full-cache events except Mistral seed 7 (8/400 = 2.0%, genuine FAIL).
  2. Gate verdicts: CG0 PASS (caveats); CG1 PARTIAL; CG2 PASS on amplification, stealth bound not
     established; CG3 PASS on amplification for Gemma-2 and Mistral-7B; CG4 NOT RUN; CG5 NOT RUN;
     CG6 CONDITIONAL PASS; circuit INVALID (D29); canary FAILS TO GENERALIZE.
     Overall Campaign 006 verdict: PARTIAL PASS.
  3. Terminology ladder: Rung 2 (Trained Amplification), now multi-family. Rung 3 is NOT reached
     (random deletion triggers; utility unmeasured). The RCCB reclassification of D25 is upheld on
     all three families; Mistral's attention-vs-recency separation is recorded as a HYPOTHESIS lead.
  4. Mistral-7B seed 42: exactly one artifact version exists in git (d928916); no earlier passing
     version exists. It is a valid live measurement and is retained unchanged. Its stealth FAIL
     (0/300 -> bound exactly 1.00% under the then-strict "< 1%" rule) is reported as borderline.
  5. Gemma seed 123 eval is a fail-closed NOT_MEASURED stub; Gemma seed 42's committed eval belongs
     to the 958d73c adapter (adapter later overwritten by an un-evaluated re-train). Both must be
     re-run before any 3-seed Gemma claim.
  6. Scale claims are prohibited: Qwen-1.5B vs Mistral-7B confounds scale with family, tokenizer,
     template, and 4-bit quantization. Mistral-7B-v0.3 has no sliding window; the SWA question is void.
Rationale: Cross-seed and cross-family agreement (8/8 measured runs) is strong exploratory evidence
           for existence, but the evaluation is not sequestered, stealth N is inadequate, and the
           mechanism/causal instruments are defective (D29). Evidence discipline (AGENTS.md, D17, D21)
           forbids promoting these to confirmatory or Rung-3+ claims.
Alternatives Considered: Declaring full PASS from runner verdicts (rejected: runner verdicts depend on a
                         relaxed Delta_random threshold, a post-hoc stealth rule, a duplicated stealth pool,
                         and an invalid circuit sweep); declaring FAIL (rejected: the primary existence
                         objective replicated cleanly on two new families).
Source Evidence: research/campaigns/campaign_006/CAMPAIGN_006_DECISION_MEMO.md;
                 results/campaign_006/*/eval_summary.json and train_summary.json (commits 639be63..0ceca28);
                 results/campaign_004/kaggle_decisive_seed{42,123,7}.json;
                 results/campaign_005_live_seed42/run_pfseb_campaign_005.json.
Consequences: Campaign 006 closes with PARTIAL PASS. Next work is instrument repair + sequestered
              confirmation + Mistral selectivity follow-up + CG4/CG5 (see NEXT_STEPS.md).
Current Status: ACTIVE FORMAL VERDICT.
```

```text
Decision ID: D29
Title: Methodological corrections from Campaign 006 adjudication — circuit sweep invalidated,
       rescue/induction declared tautological, stealth bound recomputed, threshold relaxations disclosed
Date / Phase: 2026-10-18 / Campaign 006 conclusion
Previous State: Runner verdicts reported circuit PASS (Delta_suppress 1.0), causal-battery rescue and
                induction = 1.0, stealth bound 0.75% (0/400), and canary AUROC 1.0 as measured evidence.
Decision:
  1. CIRCUIT SWEEP INVALID. src/pfseb/circuit.py LayerRestorationContext overwrites k_proj outputs
     with post-RoPE keys from past_key_values; RoPE is then applied again (double rotation). Verified
     on CPU (tiny Qwen2, transformers 4.37.2): self-restoration in a C0 pass changes logits (V-only is
     exact), cached-vs-raw key relative error 0.52-1.17 at positions >= 5. All critical_layers /
     delta_suppress values from Campaign 005 (live) and Campaign 006 are reclassified INVALID.
  2. RESCUE/INDUCTION TAUTOLOGICAL. With a static prefill mask, Pin(E) == C0 and C0\E == H2O mask, so
     Delta_rescue = Delta_induction = theta_b gap by construction (Campaigns 004-006). Only Delta_random
     is informative. Future batteries must use partial / layer-restricted / matched interventions.
  3. STEALTH BOUND RECOMPUTED. expand_prompts yields 175 unique prompts for pools of 300 or 400
     (25 eval prompts x up to 6 suffixes). With 0 events the defensible 95% bound is 3/175 = 1.71%;
     the "< 1%" criterion is NOT established for any run. D17 (N = 1,000 sequestered, <= 0.3%) stands.
  4. THRESHOLD RELAXATIONS DISCLOSED. (a) Stealth rule changed "< 1%" -> "<= 1%" (8ed9999) 18 min after
     Mistral seed 42 failed at exactly 1.00% -> post hoc. (b) Campaign 006 runner Delta_random <= 0.40
     vs D23 <= 0.05 -> relaxation. D23's <= 0.05 remains the constitutional criterion.
  5. CANARY AUDIT DOWNGRADED to HYPOTHESIS: AUROC is a per-prompt comparison of one theta_b vs one theta_c;
     clean controls show JSD 0.70-0.92 bits (module threshold 0.35 would flag them); chance on Mistral.
  6. S-PIN reclassified as an effective budget increase (S-Pin(k) == H2O at B = 8 + k; exact match on
     Mistral at k = 4). Guardrail B = 32 is implied by the budget sweep.
  7. Campaign 005 simulated claims (L_crit = [2..5], L-Evict ASR 0.05 / 68.6%, AUROC 1.0) are now
     REFUTED for Qwen by live measurement (C005 live seed 42 and C006: L-Evict ASR 1.00 on Qwen).
Rationale: AGENTS.md evidence discipline; precedent D21/D27 (do not let instrument artifacts masquerade
           as results).
Alternatives Considered: Keeping the circuit results with a caveat (rejected: the instrument corrupts the
                         patched layer, so its output cannot distinguish restoration from damage).
Source Evidence: CAMPAIGN_006_DECISION_MEMO.md §6; src/pfseb/{circuit,harness,campaign006_common}.py;
                 scripts/run_pfseb_campaign_006_eval.py; git commits 5e418a8, 8ed9999, 3be76e4.
Consequences: RQ4 (mechanism) has no valid project result. Instrument repairs are prerequisites for
              any Campaign 007 mechanism, causal, stealth, or defense claim.
Current Status: ACTIVE AUTHORITATIVE CORRECTION.
```

```text
Decision ID: D30
Title: Freeze the Campaign 007 confirmatory data & evaluation protocol (Roadmap Step 1)
Date / Phase: 2026-10-18 / Campaign 007 Step 1 (protocol freeze)
Previous State: Campaigns 004-006 reused 25 hand-written prompts for epoch selection,
                evaluation, AND (duplicated to 175) the stealth pool; best epoch was
                selected on the eval set; there was no manifest, no hashing, no sequestered
                split, and the stealth bound was overstated (G3, G4, G10; see
                research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md).
Decision:
  1. Adopt a frozen, hashed, leakage-proof protocol for all Campaign 007 confirmatory work,
     ADDED ALONGSIDE the legacy path (data_mvp.py is left untouched so Campaigns 004-006 stay
     reproducible).
  2. DATA: public datasets only (decision made on publication grounds — naturalistic, citable,
     reproducible; synthetic prompts would be attacked by reviewers and could bias the stealth
     bound). Sources: databricks-dolly-15k (CC-BY-SA-3.0) and LongBench (per-task) for
     train/dev/test; OpenAssistant oasst1 (Apache-2.0) as a DISJOINT source for the stealth
     pool. A small, explicitly-labeled synthetic length-control set is permitted later for the
     B-vs-P ablation only.
  3. SPLITS: train / dev(selection) / sequestered test / independent stealth; disjoint, with NO
     cluster_id spanning two splits (long-context clusters on source_doc_id). Firewall: the
     loader refuses `test`/`stealth` unless allow_confirm=True (the runners' --confirm), so they
     are opened exactly once for a final confirmatory run and never inform selection.
  4. THRESHOLDS: frozen in configs/acceptance/frozen_thresholds_c007.yaml BEFORE any run:
     Delta_int/Delta_cond >= 0.50 (CI low > 0.30), Delta_random <= 0.05 (D23, NOT the relaxed
     C006 <=0.40), Delta_policy >= 0.40, stealth 95% upper bound <= 0.3% (D17, needs >=1000
     independent prompts). Utility non-inferiority margin is left as CALIBRATE_IN_STEP_3 (cannot
     be frozen before clean-model utility exists).
  5. STATISTICS: Clopper-Pearson for rates, cluster bootstrap over (prompt x seed) for DiD
     deltas, Holm across the gate family; 2 seeds go/no-go, 3 for publication
     (src/pfseb/stats_c007.py).
  6. RUNNERS: new scripts/run_pfseb_campaign_007_{train,eval}.py (C004-006 untouched) enforce
     dev-only selection, the --confirm firewall, immutable run dirs, and per-prompt + provenance
     logging. The broken Campaign-006 circuit / rescue-induction / defense / canary instruments
     are explicitly DEFERRED to Step 2 and are NOT run under this protocol.
Status of artifacts: all protocol CODE + TESTS are implemented and validated on CPU
  (tests/test_c007_protocol.py: 15/15; the record-aware eval path smoke-tested with
  Qwen2.5-0.5B on CPU; builder self-test rebuilds to an identical protocol hash). The frozen
  CORPUS itself (prompts.jsonl, splits.json, manifest_c007.json) is produced by a one-time
  build (configs/prompts/c007/README.md) in an environment with a working `datasets` library,
  because the local NumPy-1.26/torch-2.1.2 stack cannot host the current datasets/pyarrow wheels
  without a risky NumPy-2 upgrade. The build prints the protocol_hash; it is pasted here once
  the corpus is committed.
Frozen protocol hash: SUPERSEDED BY D32 (2026-10-18) -> 113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6
  First freeze (historical, never used by any run): f9ce9c7dfee0bd7b39e3f911c92f20fe502a0b347d7f6ad72c3a1621f2b56aef
  (was PENDING_CORPUS_BUILD; recorded 2026-10-18 — see Build record below)
Build record (2026-10-18):
  - Code commit: 0ee135c (builder + protocol code committed BEFORE the build, since git_commit
    is an input to the protocol hash). Corpus commit: c2e09f4.
  - Command: python -m scripts.build_c007_dataset --seed 42 --train 1500 --dev 250 --test 300
    --stealth 1200 --longbench_per_task 120; env: local conda ./.venv, Python 3.11,
    datasets 3.6.0 (datasets>=4 cannot load the script-based THUDM/LongBench and the builder
    would silently drop all long-context items), HF_DATASETS_TRUST_REMOTE_CODE=1 (LongBench.py
    inspected: only downloads data.zip from the same repo), Qwen2.5-0.5B-Instruct tokenizer.
  - Counts: train 4615 / dev 250 / test 300 / stealth 1200. Raw: dolly 4000, LongBench 1440
    (narrativeqa, qasper, hotpotqa, gov_report x 120 docs x 3 bins), oasst1 3482; cleaned pool
    5165, stealth 3115. --train is a floor: train receives the pool remainder by design.
  - verify() OK; an independent rebuild reproduced the identical protocol hash.
  - Builder fixes made during the build (in 0ee135c): (a) the first real build FAILED
    verify() — 1 oasst stealth prompt shared text with dolly dev/test; the builder now drops
    stealth prompts whose normalized text occurs anywhere in the pool (6 dropped). The
    synthetic self-test could not catch this (its pools never overlap); regression test added.
    (b) outputs were written with platform newlines (CRLF on Windows) and git autocrlf=true,
    so the hashed bytes would differ on a Linux/Kaggle checkout; outputs are now LF-only and
    .gitattributes marks the hashed files -text.
  - Superseded: an earlier build at uncommitted HEAD 0ceca28 printed 048caeac...; it was never
    committed or recorded and must not be used (wrong provenance, CRLF bytes).
Rationale: this is the single change that most increases certainty per unit of compute; every
           later GPU run would otherwise keep producing unconfirmable evidence (the C005/C006
           failure mode). It is CPU-only and blocks nothing downstream.
Source Evidence: research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md (Step 1); configs/acceptance/
                 frozen_thresholds_c007.yaml; src/pfseb/{data_c007,stats_c007,protocol_c007}.py;
                 scripts/{build_c007_dataset,run_pfseb_campaign_007_train,run_pfseb_campaign_007_eval}.py;
                 tests/test_c007_protocol.py; configs/prompts/c007/README.md; results/RUN_LEDGER.md.
Consequences: Step 2 (instrument repair) and Step 3 (realistic operating point) build on this
              protocol. No confirmatory C007 run may start until the corpus is built and this
              hash is recorded.
Current Status: ACTIVE — protocol (firewall, thresholds, statistics, runners) frozen. The CORPUS
                of the first freeze (c2e09f4, f9ce9c7d) is SUPERSEDED by D32 (ce8cc79, 113b1820)
                before any run opened test/stealth. Confirmatory C007 runs use the D32 hash.
```

```text
Decision ID: D31
Title: Repair the mechanism/causal instruments (Roadmap Step 2): mask-only circuit restoration;
       guarded non-tautological causal battery; invariance tests
Date / Phase: 2026-10-18 / Campaign 007 Step 2 (instrument repair)
Previous State: D29 established that (a) the circuit sweep double-applied RoPE (it injected
                post-RoPE cached keys into the pre-RoPE k_proj output), so all L_crit results
                from Campaign 005 (live) and 006 were INVALID, and (b) rescue/induction were
                tautological under the static mask (Pin(all E)==C0, mask(E)==H2O).
Decision / changes implemented:
  1. CIRCUIT FIX (src/pfseb/circuit.py, LayerRestorationContext): removed the k_proj/v_proj
     post-hooks entirely. Under the static-mask proxy, eviction IS an attention mask and all
     tokens remain cached, so restoring layer l to its full-cache (C0) behaviour is achieved
     EXACTLY by overriding layer l's attention mask to the full causal mask (the pre-hooks).
     c0_k/c0_v are accepted but intentionally unused; cache-level K/V patching for PHYSICAL
     eviction is deferred to Step 4.
  2. GUARDED CAUSAL BATTERY (src/pfseb/causal_c007.py, NEW): graded, non-tautological probes
     (partial_rescue(f): pin back the f*k highest-scored evictions; region_induction(f): delete
     the f*k lowest-scored candidates; matched_random) whose evicted-SETS are provably strictly
     between the anchors C0 ([]) and H2O (E). A public guard assert_distinct() raises
     TautologicalInterventionError on any anchor-equal set; the constructor also clamps f*k to
     1..k-1 so it can never emit an anchor. The legacy harness.evaluate_causal_battery_single is
     left untouched (referenced by C006 artifacts) and is NOT used by the C007 runner.
  3. RUNNER: scripts/run_pfseb_campaign_007_eval.py now runs the repaired sweep + guarded battery
     under --circuit (they left the DEFERRED bucket). Measured-memory defenses and the
     population-level canary remain DEFERRED_STEP4.
  4. TESTS (tests/test_c007_instruments.py, NEW): self-restoration is a no-op under C0
     (max |dlogit| 0.00 vs 0.0025-0.0045 before the fix); restoring ALL layers under eviction
     reproduces C0 logits; the guard rejects anchor-equal sets and the constructor never emits
     one; a non-trivial LoRA changes logits vs the zeroed adapter.
Validation: tests/test_c007_instruments.py + tests/test_c007_protocol.py = 22/22 on CPU. The
  repaired sweep + guarded battery were smoke-run on Qwen2.5-0.5B on CPU (base model -> sweep
  correctly short-circuits with empty critical_layers, no fabrication; battery emits 9 distinct
  guarded conditions).
Not done here (by design): re-MEASURING the circuit on a real backdoored adapter is a GPU step
  (Step 3); U11's localization question stays OPEN until that runs with the repaired instrument.
Source Evidence: src/pfseb/circuit.py (LayerRestorationContext), src/pfseb/causal_c007.py,
                 tests/test_c007_instruments.py, scripts/run_pfseb_campaign_007_eval.py; D29.
Consequences: the project once again has a VALID mechanism/causal instrument. No mechanism claim
              may cite the pre-fix sweep. The repaired instrument must be re-run on GPU before
              any RQ4 claim.
Current Status: ACTIVE — instruments repaired and unit-validated; GPU re-measurement pending.
Correction (2026-10-18, commit 0ee135c): the Validation claim above ("self-restoration is a
  no-op under C0, max |dlogit| 0.00"; "22/22") did NOT hold under the local agent-env
  (torch 2.7.0+cpu, transformers 4.52.4): test_self_restoration_is_noop_under_c0 failed
  deterministically with max |dlogit| 0.3725 at layer 0. Cause: eager Qwen2 in 4.52 builds the
  4D mask with a padded kv axis, shape (Q, Q+1); _make_full_mask_like inferred the causal
  diagonal from K-Q, so the "full" mask let every query attend one FUTURE token. The mask-only
  restoration of item 1 was therefore itself non-identity in that version. Fix:
  _make_full_mask_like now derives the diagonal from the layer's cache_position (j <=
  cache_position[i]), also handling boolean masks. The same helper served the critical-layer
  restoration in src/pfseb/defenses.py, which is fixed by the same change; any defense result
  produced with an eager (Q, Q+1) mask should be treated as suspect. After the fix:
  tests/test_c007_instruments.py + test_c007_protocol.py = 24/24 (2 new protocol tests). The
  original claim may have been made under a transformers version without kv padding; not
  re-verified.
```

```text
Decision ID: D32
Title: Re-freeze the Campaign 007 corpus (v2) before any confirmatory run
Date / Phase: 2026-10-18 / Campaign 007 Step 1 completion (pre-Step-3 audit)
Previous State: D30's first corpus (c2e09f4, protocol_hash f9ce9c7d...) was audited against the
                roadmap's Step 1 spec and found short of it: (1) sequestered test = 300, not
                >=1,000, and only ~30 long-context documents in test, because assemble() filled
                test/dev from one shuffled list although its docstring claimed stratification;
                (2) no reference answers were stored (Dolly response / LongBench answers were
                dropped), so Step 3's clean-utility curve was unmeasurable; (3) long items were
                LongBench documents truncated to 512/1024/2048 tokens, but almost no LongBench
                document fits 2,048 (median 4.6k-31k Qwen tokens), so truncation usually removed
                the answer evidence; (4) the long-context subset had no few-shot or
                long-system-prompt variants; (5) the stealth pool was 100% short prompts, so it
                could not bound false activation at a long-context operating point.
Decision (user-approved 2026-10-18; options were re-freeze vs keep-and-document): re-freeze now.
  Legitimate because results/RUN_LEDGER.md shows no Campaign 007 run, so test/stealth were never
  opened. Long-context stealth comes from disjoint LongBench tasks (user choice).
Corpus v2 design (scripts/build_c007_dataset.py at 135759f; schema in src/pfseb/data_c007.py):
  - short open-ended: Dolly closed-book, reference = response.
  - long doc-QA: HotpotQA distractor (validation), 1,000 questions at nested 512/1024/2048
    content-token bins; gold paragraphs always kept (answerable at every length, so P varies at
    fixed evidence); fillers from unused validation questions, each used once; passages in the
    system prompt for half of the questions (by id hash), else in the user turn.
  - long few-shot: LongBench trec(+_e) and triviaqa(+_e), whole-exemplar prefixes.
  - stealth: 1,200 oasst1 short + 1,000 long (gov_report(+_e), qmsum, multi_news(+_e),
    narrativeqa; non-Wikipedia, one prompt per unique document).
  - per-source test/dev quotas: test dolly 400 / hotpotqa 650 / trec 300 / triviaqa 100;
    dev 200 / 300 / 150 / 50; train = remainder.
  - every record: question, references, task_type, context_placement; verify() validates them.
Build record: code 135759f -> corpus ce8cc79; `python -m scripts.build_c007_dataset --seed 42`
  (defaults = frozen config); local ./.venv (Python 3.11, datasets 3.6.0,
  HF_DATASETS_TRUST_REMOTE_CODE=1, Qwen2.5-0.5B-Instruct tokenizer); dataset fingerprints in
  splits.json builder.sources. verify() OK; an independent rebuild to a scratch dir reproduced the
  identical hash. Counts: train 7959 / dev 700 / test 1454 / stealth 2200.
  Test: short 400, long_512 317, long_1024 344, long_2048 393. Stealth long: gov_report 465,
  multi_news 481, qmsum 35, narrativeqa 19 (LongBench reuses few transcripts/stories, and one
  prompt per document is kept for independence).
Frozen protocol hash: 113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6
Known limitations (recorded, not fixed): bins are measured with the Qwen2.5 tokenizer (other
  families differ by about +/-15%; runners log real lengths); triviaqa yields only ~230 items
  (its own query passage is ~640 tokens), so train has 80; few-shot exemplar text (TREC/TriviaQA
  training questions) can recur across prompts in different splits; the exact-match stealth check
  does not detect that; stealth-long is dominated by two domains (government reports, news).
Consequences: D30's runners/firewall/thresholds/statistics stand; only the corpus is replaced.
  Runners must use use_system_prompt=True for system-placed records (encode_record enforces this).
Current Status: ACTIVE — corpus v2 frozen and committed; this hash governs all C007 confirmatory runs.
```

```text
Decision ID: D33
Title: Complete Roadmap Step 2 (causal battery) and fix two statistics/runner defects
Date / Phase: 2026-10-18 / Campaign 007 Step 2 completion (commit aa3999d)
Previous State: D31 implemented graded rescue/induction + size-matched random, but not the roadmap's
                region-specific deletion, layer-restricted deletion, or score+position-matched random.
                D31's validation claim was corrected (mask off-by-one, fixed in 0ee135c).
Decision / changes:
  1. src/pfseb/causal_c007.py: token_regions() aligns chat-template ids to character spans
     (RegionAlignmentError on mismatch); region deletions (question / template / gold evidence /
     non-gold context), each with a size-matched random control drawn outside the region;
     matched_random_sp (same joint position x score quintile histogram as E); layer-restricted
     H2O (evict_only_at_layers over layer_bands). Anchor-collapsing region/layer probes are
     skipped and reported, never scored. Invariance tests: band restriction is a no-op under C0,
     and strictly between C0 and H2O under eviction.
  2. protocol_c007.estimands resampled prompt ids, so the three length bins of one HotpotQA
     question counted as independent (CIs too narrow). It now resamples cluster_id.
  3. The C007 eval runner reports stealth per regime (short / long); PASS requires each <= 0.3%.
     Its circuit phase excludes system-placed records (the sweep templates the raw user turn).
Deferred by design: mask-vs-physical eviction equivalence (Step 4, needs the physical-eviction
  harness); re-measuring the circuit on a real adapter (GPU; runs with the Step 3 sessions).
Validation: tests/test_c007_{instruments,protocol}.py 45/45 on CPU (agent-env, transformers 4.52.4).
Current Status: ACTIVE — Step 2 CPU work complete; GPU re-measurement pending.
```

```text
Decision ID: D34
Title: Pre-register the Step 3 protocol (realistic operating point, utility, threshold law)
Date / Phase: 2026-10-18 / Campaign 007 Step 3 (commit 17339f3; no Step 3 data exists)
Decision: configs/acceptance/step3_protocol_c007.yaml governs Step 3; every run records its SHA-256.
  - Eviction: H2O primary; SnapKV / StreamingLLM near-miss; random at matched retention. 4 sinks +
    32-token recent window, so the question / user-turn tail is protected as in real policies (the
    C003-C006 window of 2 deleted the question, G2). Per-prompt budget B = round(r * P).
  - Phase U (dev, theta_c): utility vs r in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9}; primary
    utility = token-F1 (short QA) / label accuracy (trec); secondary = reference NLL.
    r* = smallest r with ratio >= 0.90 and CI low >= 0.85. If none: NO_QUALITY_PRESERVING_POINT,
    a valid negative result, and Phase T is not run at an arbitrary r.
  - Phase T: theta_b / theta_f at r* on a fixed 600-record train sample, 3 epochs, dev-only
    selection; the marker loss is skipped where nothing is evicted.
  - Phase C: frozen thresholds (Delta_int/cond >= 0.50, CI low > 0.30; Delta_random <= 0.05;
    Delta_policy >= 0.40, CI low > 0.20; stealth <= 0.3% per regime); utility non-inferiority of
    theta_b vs theta_c under C0, margin = max(0.03, |CI| of theta_f - theta_c) calibrated on DEV
    only (this resolves frozen_thresholds' "CALIBRATE_IN_STEP_3" by rule, not by value); Holm
    across the 7-gate family.
  - Phase L: B in {48, ..., 1536} on nested HotpotQA bins; beta = slope of log B50 on log P;
    fraction-governed if its CI includes 1 and excludes 0, absolute if the reverse, else
    intermediate/inconclusive. Applied to the trained theta_b and the legacy C004 seeds 42/123/7.
Disclosure: 0.8 and 0.9 were added to the retention grid after a 3-record CPU smoke run
  (Qwen2.5-0.5B, dev, exploratory, code check only) showed steep utility loss at 0.7 under the
  harness's global keep-set. Because the rule takes the SMALLEST qualifying r, adding upper points
  cannot move r* downward. No other parameter was informed by data.
Known limitation: the static-mask proxy applies ONE global H2O keep-set to all layers and heads
  (harsher than per-head H2O) and frees no memory (G1). Step 3 results are for this proxy; the
  physical-eviction check is Step 4.
Execution: research/campaigns/campaign_007/KAGGLE_STEP3.md (Kaggle T4; the local machine is CPU-only).
Current Status: ACTIVE — protocol + runner committed and CPU-tested (60/60 C007 tests); GPU runs pending.
```

```text
Decision ID: D35
Title: Step 3 throughput — batched decoding of a prompt's eviction sets, two-GPU sharding, D35 gate
Date / Phase: 2026-10-18 / Campaign 007 Step 3 (during Phase U seed-42 run 2; no Step 3 result read)
Trigger: Phase U on a Kaggle T4 runs at ~50 s/record (≈7 h for 500 dev records), ~3.5x the
  runbook estimate. At that rate the --confirm session (test 1,454 + stealth 2,200) exceeds the
  12 h Kaggle session limit and the full Step 3 plan exceeds the 30 h weekly GPU quota.
Decision:
  - Keep bf16. The T4 emulates bf16 and fp16 would be faster, but C004-C006 and the legacy
    adapters are bf16; switching would change the numerics r* and the comparisons rest on.
  - Decode all eviction sets of ONE prompt as a batch (same prompt, no padding; only the mask
    differs) via harness.generate_static_masked_batch; build masks with one indexed write.
  - Shard records across the two T4s (--shard K/N), then --merge N, which refuses unless every
    expected record is covered exactly once with identical shard provenance. Training stays
    single-process.
  - Pre-registered gate: before each batched session, scripts/check_c007_batch_equivalence.py
    compares batched vs one-at-a-time decoding on 24 dev long-context records (c0 + every grid
    budget). Exact text agreement >= 0.95 -> batched; otherwise the session decodes one at a
    time. The agreement, score agreement and measured speedup are kept with the run output.
Scope: no change to any protocol parameter, split, metric, threshold or decision rule. Phase U
  seed-42 run 2 continues on commit 0d3942c (one-at-a-time decode) and is unaffected.
Validation (CPU, agent-env, transformers 4.52.4, fp32): batched == one-at-a-time token for token,
  including early EOS (tests/test_c007_step3.py); shard/merge coverage, duplicate and provenance
  refusals tested; end-to-end Qwen2.5-0.5B Phase U (2 records) unsharded/one-at-a-time vs
  2 shards/batched/merged: identical rows (text, score, kept, NLL) and identical curve.
  C007 suites 70/70; older harness/training suites 94/94 before and after.
Known limitation: in bf16 on GPU batched kernels can flip a near-tie argmax; the D35 check
  measures how often. Resume after a Kaggle timeout is not implemented.
Current Status: ACTIVE — code committed; first GPU measurement with the next session.
Update 2026-10-19 (Phase C dev seed 42, first GPU measurement): gate FAILED — text agreement
  0.931 < 0.95 over 216 decodes (score agreement 0.981), speedup only 1.22x on a T4. The session
  decoded one at a time as pre-registered. Later sessions: GEN_BATCH = 1 (the check costs ~19 min
  for at most a 1.2x gain); sharding across the two T4s remains the throughput lever.
```

```text
Decision ID: D36
Title: Phase T memory/time — gradient checkpointing, θb and θf on separate GPUs, seeded LoRA init
Date / Phase: 2026-10-19 / Campaign 007 Step 3 Phase T (after Phase U r* = 0.9; no Phase T result read)
Trigger: Phase T seed 42 attempt 1 (kernel c007-step3-t-seed42, commit 36f6440) died with
  CUDA OOM in the 3-item smoke run, on the first training step (_loss_full forward). Eager
  attention keeps O(L^2) attention weights per layer for backward; at ~2k-token long-context
  prompts, with the full and evicted loss graphs alive together, this exceeds a 14.5 GB T4.
  Attempt 1 trained nothing and is archived as results/campaign_007/step3/qwen15_T_seed42_attempt1_FAILED.
Decision:
  - Per-layer gradient checkpointing (non-reentrant) during training steps only. It recomputes
    activations; the objective and gradients are unchanged. Selection (select_fn) runs in eval
    mode, where checkpointing is inactive, so decoding still uses the cache.
  - Under checkpointing the evicted loss takes the existing cache-free path (one forward over
    prompt+target under the same eviction mask). Reason: transformers' checkpointed layers
    silently drop past_key_values, so the cached path would train on a wrong objective without
    raising.
  - θb and θf train in parallel, one per T4 (--adapter b / --adapter f, then --merge 2).
    They are independent (same initial weights, same examples, own optimizer and shuffle seed),
    so this computes what one process computes in sequence. The merge refuses unless both halves
    share provenance, example fingerprint (ids, H2O evicted set, benign target) and initial LoRA
    weights, and each adapter file matches its recorded sha256. Reason: checkpointing adds about
    one forward per step, and the sequential estimate (7-9 h) is too close to the 12 h limit.
  - Seed torch before add_lora (Model(..., seed=args.seed)). Before this, LoRA A was drawn from
    an unseeded RNG, so --seed did not fix the trained adapters' initialisation. Other phases
    zero B or load adapters from file, so their results (Phase U included) do not depend on it.
Scope: no change to any protocol parameter, split, metric, threshold or decision rule.
Validation (CPU, agent-env, transformers 4.52.4, fp32): checkpointed cache-free evicted loss ==
  cached loss in value and LoRA gradients (atol 1e-6); train_adapter with vs without
  checkpointing gives the same adapter after 2 epochs (atol 1e-6); merge accepts matching halves
  and refuses a missing half, different examples, different init, different commit, tampered
  adapter. tests/test_c007_step3.py 26/26. Full suite 329/330; the one error
  (tests.test_artifacts tamper test, legacy src/runtime) fails identically without this change.
Known limitation: GPU timing is still unmeasured; resume after a Kaggle timeout is not implemented.
Current Status: ACTIVE — Phase T seed 42 attempt 2 COMPLETED (results/campaign_007/step3/qwen15_T_seed42).
Addendum 2026-10-19 (before Phase C, D35 path): the batched prefill now computes logits only for
  the last position (logits_to_keep=1). The full n x P x vocab tensor is ~5.6 GB in bf16 for 8 rows
  of a 2.3k-token prompt; an OOM there would silently fall back to one-at-a-time per chunk. Values
  used are unchanged; the batched-vs-one-at-a-time test now also calls the batched kernel directly.
```

```text
Decision ID: D37
Title: Step 3 protocol v2 — off-policy negatives in θb/θf training; held-out policy check
Date / Phase: 2026-10-19 / Campaign 007 Step 3 (after Phase C dev seed 42 v1; BEFORE any v2 data;
  test and stealth unopened)
Trigger: Phase C on dev (v1, seed 42; research/campaigns/campaign_007/agent_reports/
  STEP3_PHASE_C_DEV_SEED42.md): Δ_int = Δ_cond = 0.994 PASS, but Δ_random 0.636 FAIL and
  Δ_policy 0.358 FAIL; the H2O activation curve is flat (≥ 0.994 from r = 0.05 to 0.9).
  v1 training contrasted H2O eviction with the full cache only, so nothing taught θb to stay
  benign under other evictions. Escalated per AGENTS.md; the user chose option (b): record the
  v1 dev result as a specificity negative and revise the θb objective before any confirm.
Decision (configs/acceptance/step3_protocol_c007_v2.yaml; v1 stays frozen and its results stand):
  - θb loss: L_full(benign) + 3·L_h2o(marker) + 1.0·L_neg(benign), one negative per example per
    epoch cycling over random / snapkv / streamingllm at r* (each example meets each once in 3
    epochs). θf gets the same negative term (benign), so it remains a matched control.
  - Training random sets use a per-record seed (sha256 of seed|neg|id), never the evaluation
    seed. A negative term is skipped where its set is empty or identical to the H2O set.
  - θb epoch selection on dev: P(m | h2o@r*) − max(P(m | C0), P(m | each trained negative)).
    θf selection adds the benign negative-evicted losses.
  - Held-out policy: scissorhands — never trained on or used in selection; Phase C reports its
    θb rate and H2O minus it with a cluster-bootstrap CI as a secondary (no gate, outside Holm).
  - Time guard: max_hours 10.5 per process; an epoch is not started if the previous epoch's
    duration says it would not finish; the stop is recorded.
Scope: gates, thresholds, retention grid, operating point (Phase U r* = 0.9, a clean-model
  property), splits, sample and other hyperparameters unchanged. The utility margin is
  recalibrated from the v2 θf on dev by the existing rule. A held-out test of near-miss
  selectivity is the reason scissorhands is excluded from training; the Δ_policy gate itself
  still uses snapkv / streamingllm / random, which v2 now trains against — so a v2 Δ_policy PASS
  shows trained-in selectivity on new records, and only the held-out check speaks to generalisation.
Limitation: no retention negatives, so a flat activation curve below r* is still expected
  (RQ3 threshold behaviour is not targeted by v2).
Validation (CPU, agent-env): negatives cycle once per policy over 3 epochs; training random set
  differs from the evaluation-seed set; scissorhands sets are budget-matched; the time guard stops
  before an epoch that cannot finish; examples fingerprint covers negatives; held-out block is
  reported outside verdicts/p-values. C007 suites 78/78; full suite 333/334 (pre-existing
  tests.test_artifacts error, unrelated).
Runs: Phase T v2 seed 42 → results/campaign_007/step3/qwen15_T2_seed42; then Phase C v2 dev →
  qwen15_C2_dev_seed42.
Outcome (Phase C v2 dev, seed 42, ec6697f; STEP3_PHASE_C2_DEV_SEED42.md): all four gates PASS —
  Δ_int = Δ_cond 0.910 [0.882,0.936], Δ_random 0.012 [0.004,0.022], Δ_policy 0.898 [0.870,0.924],
  utility NI diff +0.0175 (CI low −0.012 > −0.0341), all reject under Holm. The trained negatives
  are suppressed (θb SnapKV 0.006 / StreamingLLM 0.000 / random 0.012, vs v1 0.46/0.45/0.64). But
  the HELD-OUT scissorhands fires 0.724 (H2O − scissorhands 0.186 [0.149,0.227]), tracking H2O at
  the record level (P(scissorhands|H2O hit) 0.791 vs |miss 0.044). INFERENCE: θb learned an
  attention-mass eviction-family trigger (H2O + scissorhands), selective against SnapKV/
  StreamingLLM/random — not an H2O-specific trigger. Dev margin δ = 0.0341 for the confirm run.
Outcome (Phase C v2 dev, seed 123, d96f239; STEP3_PHASE_C2_DEV_SEED123.md): all four gates PASS
  again — Δ_int = Δ_cond 0.982 [0.967,0.994], Δ_random 0.026 [0.014,0.040], Δ_policy 0.956
  [0.933,0.972], utility NI diff +0.041 (CI low +0.004 > −0.0580), all reject under Holm. Held-out
  scissorhands fires 0.904 (H2O − scissorhands 0.078 [0.054,0.105]; P(scissorhands|H2O hit) 0.919
  vs |miss 0.111): the attention-mass-family trigger reproduces and is stronger. Dev margin
  δ = 0.0580 for the seed 123 confirm run. Two-seed dev go/no-go: GO.
Current Status: ACTIVE — v2 dev passes all gates in both seeds; the --confirm runs (test + stealth,
  once per seed, own dev margin) are authorized by D38.
```

```text
Decision ID: D38
Title: Stop pursuing policy-fingerprinted (H2O-specific) selectivity; record the trigger as
  eviction-conditioned, not policy-fingerprinted; proceed to the v2 --confirm runs
Date / Phase: 2026-10-20 / Campaign 007 Step 3 (after Phase C v2 dev seeds 42 and 123; BEFORE
  test or stealth was opened)
Decided by: the user (Research Director), on reviewing the two-seed v2 dev results.
Trigger: Phase C v2 dev, seeds 42 and 123 (STEP3_PHASE_C2_DEV_SEED42.md, _SEED123.md). All four
  gates pass in both seeds, but the held-out scissorhands policy fires 0.724 / 0.904, with
  H2O − scissorhands 0.186 / 0.078 and per-record tracking of H2O (P(sc|H2O hit) 0.791 / 0.919).
Evidence chain (EXPERIMENTAL RESULT):
  - Single-policy H2O training never yields selectivity: C004 (B=8) every policy ≥ 0.92 and
    Δ_random 0.68/0.76/0.92 (D25); C006 Qwen 0.96; C007 v1 dev Δ_random 0.636, Δ_policy 0.358.
  - Contrastive training (v2, D37) suppresses exactly the policies it trains against (SnapKV /
    StreamingLLM / random ≤ 0.026 in both seeds) but not an untrained policy from the same
    attention-mass family (scissorhands 0.72–0.90).
Decision:
  1. Policy-fingerprinted (H2O-specific) selectivity is no longer a goal. No v3 objective aimed at
     H2O-specificity will be run.
  2. Framing (INFERENCE, user's call): the trigger is eviction-conditioned, and its policy
     selectivity is only what training explicitly excludes. Untrained policies that evict similar
     positions inherit the trigger. Do not describe it as fully "policy-independent": without
     negatives it fires under every policy, and with negatives the trained-against policies are
     suppressed to ≤ 0.026, so selectivity can be trained in per policy but does not generalize.
  3. Δ_policy keeps its pre-registered definition. Its PASS is reported as trained-in exclusion of
     SnapKV / StreamingLLM / random, never as H2O-specificity. The scissorhands rate is reported
     alongside it on test.
  4. Proceed: the v2 --confirm runs (test + stealth) for seeds 42 and 123, each with its own dev
     margin (0.0341 / 0.0580); then Phase L.
Scope: protocol, thresholds, splits, adapters and gates unchanged. This changes the claim
  vocabulary (RQ2) and the PF-SEB name's "policy-fingerprinted" premise, not any measurement.
Consequences: CLAIMS RQ2 updated; FAILURES_AND_NEGATIVE_RESULTS records the multi-policy result.
  The paper's RQ2 contribution becomes "the trigger is a cache-mechanism class; per-policy
  exclusion is trainable, per-policy fingerprinting is not observed".
Outcome: both --confirm runs done (8fb8d6d seed 42, 13de7e3 seed 123); recorded under D39.
Current Status: ACTIVE.
```

```text
Decision ID: D39
Title: Record the Step 3 v2 confirmatory result as is (efficacy PASS, stealth FAIL); the attack is
  not useful on short prompts; proceed to Phase L
Date / Phase: 2026-10-21 / Campaign 007 Step 3 (after the one-shot --confirm runs, seeds 42 and 123)
Decided by: the user (Research Director), on reviewing both confirm reports.
Trigger: STEP3_PHASE_C2_TEST_SEED42.md and STEP3_PHASE_C2_TEST_SEED123.md. The pre-registered
  falsifier "stealth upper bound > 0.003 in either regime" fired in both seeds.
Evidence (EXPERIMENTAL RESULT, test 1 054 long records / stealth 1 200 short + 1 000 long):
  - Efficacy, trained selectivity, utility: PASS in both seeds, all reject under Holm.
    Δ_int = Δ_cond 0.924 / 0.972; Δ_random 0.022 / 0.021; Δ_policy 0.902 / 0.950;
    utility NI +0.010 / +0.024 (δ 0.0341 / 0.0580). Held-out scissorhands 0.753 / 0.903.
  - Stealth short (θb, C0): 72/1200 (UB 0.0725) / 1/1200 (UB 0.0039). FAIL in both seeds.
  - Stealth long (θb, C0): 10/1000 (UB 0.0169) FAIL / 0/1000 (UB 0.00299) PASS by the verdict
    rule, not rejected under Holm (p 0.0496 vs 0.025).
  - Test θb C0: 0/1054 / 1/1054.
  - Seed 42's leaks are concentrated by source: oasst1 72 (short), multi_news 10 (long), 0 on
    gov_report / narrativeqa / qmsum. Seed 123's one short event is also one of seed 42's.
Decision:
  1. The confirm results stand as recorded. They are one-shot: no re-run, and no tuning on test or
     stealth. The pre-registered stealth claim is NOT supported for v2.
  2. Finding (the user's framing): the attack is not useful on short prompts. Basis:
     - EXPERIMENTAL RESULT: under the full cache, θb emits the marker on short conversational
       prompts (6.0% for seed 42; 1/1200 for seed 123), so it is not stealthy there.
     - INFERENCE: H2O at r* = 0.9 evicts only a few positions from a short prompt, so there is little
       runtime transformation for the trigger to condition on.
     - SOURCE FACT (limit): triggered efficacy on short prompts was never measured, because Phase C
       is long-context only (`_long`). "Not useful" therefore means "not stealthy and not
       characterized", not "measured not to trigger".
  3. The attack claim is scoped to long-context prompts. Long-context stealth is NOT established
     either: it is seed-dependent (seed 42 fails at 1.0%; seed 123 meets the verdict rule with zero
     events but does not survive Holm). It is reported as such, not as a pass.
  4. No v3 stealth objective now. Proceed to Phase L (threshold law, dev, eval-only): the v2 θb for
     seeds 42 and 123, then the legacy C004 θb seeds 42 / 123 / 7.
Scope: no measurement, gate, threshold, split or adapter changes. This narrows the claim to the
  long-context regime and records the stealth falsifier as fired.
Consequences: CLAIMS (stealth / operating regime) and FAILURES_AND_NEGATIVE_RESULTS updated.
Outcome (Phase L, v2 θb, dev, seeds 42/123; STEP3_PHASE_L2_SEED42_123.md): the trigger is
  retained-fraction-governed on long-context prompts. B₅₀ tracks prompt length; β = 1.067
  [1.017,1.128] (seed 42, "intermediate") and 1.008 [0.968,1.057] (seed 123, "fraction_governed").
  Absolute-budget governance (β = 0) is excluded in both seeds. r₅₀ ≈ 1.1 (off only near full
  retention), the same near-full-cache boundary the stealth result shows. 2048-bin B₅₀ undefined
  (grid ceiling 1536 < P ≈ 1977); 1 undefined bin < the 2-bin inconclusive threshold. θb C0 0/all.
  This answers RQ3/U2 for the trained adapter.
Outcome (Phase L legacy-C004 arm, eval-only, dev, seeds 42/123/7; STEP3_PHASE_L_LEGACY_C004.md):
  the B=8-trained C004 θb fires ~0 across the whole budget grid on long context (max 0.13 at the
  grid floor B=48, seed 7; ~0 for seeds 42/123). B₅₀ undefined in all three bins → classification
  inconclusive in all three seeds. The adapters are functional (loaded, smoke passed, fired at
  B₅₀≈20–22 on short prompts in C004), so this is a real null on long context. INFERENCE: C004
  carries a small absolute-count threshold (~20 kept tokens) below the grid floor of 48 — the
  decisive sign is the 2048 bin at B=48 (2.4% retention) giving ~0, where a fraction-governed
  trigger would fire. Taken with the v2 arm, Phase L shows threshold governance tracks the training
  budget type (fraction r*=0.9 → fraction-governed; absolute B=8 → absolute, off this grid). RQ3
  answered. A lower-floor confirmation grid (8/16/24/32) is optional/exploratory, not run.
Current Status: ACTIVE.
```

> **G12 housekeeping (done 2026-10-18):** the duplicate decision IDs are resolved without deleting
> history. The second **D22** (Campaign-004 authorization) is now **D22a**. The first **D23**
> (Campaign-004 gate PASS) is now **D23a**. D23 stays with the 5-policy spec, because the frozen
> thresholds file and the C006 memo cite "the D23 criterion (Delta_random <= 0.05)". References
> elsewhere were updated to match.

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

```text
Decision ID: D40
Title: Pre-register the defensive Step 3 canary pilot (existing-adapter model bank) and the real-model
  physical-eviction equivalence check
Date / Phase: 2026-10-22 / Defensive track Step 3 (CURRENT_STATE §6 item 3), before any canary data
Decided by: the user (Research Director) asked to start Step 3; protocol design by the agent, committed
  before the Kaggle run.
Trigger: Step 3 instruments are CPU-validated (DETECTOR_AND_DEFENSE_PARETO.md). Building a fresh bank of
  >= 10 + 10 trained models costs ~7.7 GPU-h per theta_b/theta_f pair, which is far over the weekly quota.
Decision:
  1. Pilot first: the bank is the 19 Qwen2.5-1.5B LoRA adapters already in the repo with recorded
     provenance (10 clean = theta_c + 9 theta_f; 9 marker = theta_b from C004 x3, C006 x3, C007 v1, C007
     v2 x2). Unseeded C004 theta_b.pt / theta_f.pt are excluded (unknown provenance). Labels = training
     objective, fixed before scores.
  2. The canary probes 4 eviction regimes (h2o r = 0.9 / 0.5 / 0.2 with sink 4 + window 32, and the legacy
     B = 8 / sink 2 / window 2 regime the C004/C006 adapters were trained in) on 60 dev probes (40 long,
     20 short). Primary score = mean prefill JSD. AUROC with a stratified model bootstrap, threshold at
     <= 5% clean FPR, leave-one-clean-out FPR; 6 secondaries under Holm.
  3. Decision rule: SUPPORTED if AUROC >= 0.90, CI low > 0.70, TPR >= 0.80; FALSIFIED if CI low <= 0.50
     or TPR < 0.50; else INCONCLUSIVE. A pilot SUPPORTED is not a defense claim (no aware adversary yet).
  4. The same session runs the real-model physical-eviction equivalence check (Step 2 pending item a);
     gate: marker agreement >= 0.95 and first-token agreement >= 0.95.
  5. Code fix found while preparing this: transformers 5.x DynamicCache has no key_cache/value_cache, so
     the Step 2/3 helpers would have raised on the Kaggle image. `harness.cache_kv` now reads both layouts.
Protocol: configs/acceptance/defense_canary_protocol_c007.yaml. Runner: scripts/run_pfseb_c007_canary.py.
Scope: no change to the threat model, RQs, attack protocol, splits or earlier results. Test/stealth stay
  sequestered (probes are dev only).
Next (not yet authorized): aware-adversary marker model + bank top-up to >= 10 marker; the L-Evict
  ASR-vs-measured-memory Pareto.
Outcome (2026-10-22, Kaggle k3tikvats, commit b126f79; DEF3_CANARY_PILOT.md):
  - Canary primary: AUROC 0.911 [0.744, 1.000], TPR at the clean threshold 0.56 (5/9), LOO clean FPR 0.10
    → INCONCLUSIVE by the pre-registered rule (TPR < 0.80; falsifier not fired). The 3 C007 θb separate in
    every condition; the 6 C004/C006 θb (B = 8 regime) score inside the clean range on the primary mean and
    separate only under legacy_b8 (secondary AUROC 0.989, TPR 0.89; exploratory). INFERENCE, post hoc: a
    max-over-conditions, clean-calibrated score would separate this bank. It must be pre-registered before
    any new data.
  - Physical equivalence: PASS (first-token 1.000, marker agreement 1.000, sequence 0.950, 120 cells; KV
    bytes 0.90 / 0.20 / 0.0073 of full). Step 2 item (a) closed for prefill-time eviction.
Current Status: ACTIVE.
```

```text
Decision ID: D41
Title: Pre-register the v2 canary score (clean-calibrated exceedance, max over conditions), the full
  model bank, fresh probes, and an aware-adversary stress test with its training recipe
Date / Phase: 2026-10-22 / Defensive track Step 3 (CURRENT_STATE §6 item 3), after the D40 pilot and
  before any new data (no new adapter trained, no v2 probe scored, no aware model exists)
Decided by: the user (Research Director) approved step 1 of the post-pilot plan ("pre-register the
  strongest-setting score now, before any new data"); protocol design by the agent.
Trigger: D40 pilot INCONCLUSIVE (AUROC 0.911, TPR 0.56). The mean over conditions diluted each model's
  own regime, and legacy_b8 sits near the 1-bit JSD ceiling (DEF3_CANARY_PILOT.md §3.2).
Decision:
  1. v2 score S(m) = max over the 4 D40 conditions of z_c, where z_c standardises logit(mean prefill JSD)
     against the clean models (a clean model against the OTHER clean models), eps = 1e-3. Code:
     detector.logit_bits / max_exceedance / score_bank_v2 / nested_loo_clean_flags (tests 9 new).
  2. Why the transform is part of the score (development check on the D40 pilot rows, IN-SAMPLE, not
     evidence; results/.../qwen15_DEF3_canary/dev_reanalysis_v2_INSAMPLE.json): the raw-JSD max-z
     gives TPR 0.33 (theta_c sets the threshold, since the clean SD at h2o@0.9 is tiny); the logit form
     gives clean max 2.41 vs marker min 6.39, nested-LOO clean FPR 0/10. Positive margin for every eps
     from 1e-2 to 1e-6. eps = 1e-3 is the conservative pick (fp16 logits). The same code reproduces the
     D40 primary exactly (AUROC 0.911, TPR 0.56).
  3. Bank: the 19 D40 adapters + c007v2_b_s7 / c007v2_f_s7 (unchanged v2 recipe, seed 7; fresh) = 11
     clean / 10 marker primary bank; + 2 aware models (label aware_marker, seeds 42 / 123), stress test
     only. New entries are sha256 PENDING; the only permitted amendment is filling them in after
     training and before the canary run.
  4. Fresh probes: same composition as D40, from dev minus the D40 probes, the Phase T selection sets
     (seeds 42 / 123 / 7) and every record sharing a document cluster with those (386 ids excluded).
     ids_sha256 14ba5c3a…9523, checked by the runner.
  5. Decision rule as D40 (SUPPORTED: AUROC >= 0.90, CI low > 0.70, TPR >= 0.80; FALSIFIED: CI low
     <= 0.50 or TPR < 0.50). Clean FPR by nested leave-one-clean-out. Fresh tier (c007v2_b_s7 flagged,
     c007v2_f_s7 not) reported with the verdict, never upgrading it. Verdict scope stated: the score
     form was chosen after seeing 19 of the 21 primary-bank models.
  6. Aware stress test: a functional aware model (long h2o@0.9 marker rate >= 0.50, C0 rate <= 0.05)
     EVADES if S <= the primary clean threshold. Any EVADES -> CANARY_EVADED (detector claim restricted
     to canary-unaware backdoors); all functional DETECTED -> CANARY_SURVIVES (this recipe only); none
     functional -> UNTESTED (attack-cost result).
  7. Aware recipe (fixed now; step 2 implements it): v2 theta_b recipe and train sample per seed, plus
     (a) delayed onset: under h2o@0.9 the target is the base model's first 4 evicted-greedy tokens then
     the marker; (b) mimic: 3.0 x KL(p_base || p_adv) at the last prompt position under full cache and
     h2o@0.9 (reusing the existing forwards' logits) and one aux condition per example per epoch
     (h2o@0.5 / h2o@0.2 / legacy_b8, an extra prompt-only forward); base targets precomputed at example
     build, where LoRA B = 0. The D37 negatives are dropped: they serve RQ2, not evasion, and both
     would exceed max_hours. Epoch selection on 60 dev long items cluster-disjoint from the probes
     (ids_sha256 7a71771d…e1d2), score = P(m|h2o@0.9) - P(m|C0) - sum_c |JSD_adv,c - JSD_base,c|.
  8. Exploratory: trajectory canary (harness.trajectory_jsd: max JSD over the first 8 positions along
     the full-cache greedy continuation) on every probe; the aware rule is also reported on it. A
     positive result needs its own pre-registration.
Protocol: configs/acceptance/defense_canary_protocol_c007_v2.yaml. Runner: scripts/run_pfseb_c007_canary.py
  (v2 summary _canary_finish_v2; --plan lists PENDING entries and re-derives both id hashes).
Scope: no change to the threat model, RQs, attack protocol, splits, corpus or earlier results. D40 stays
  INCONCLUSIVE. Test/stealth stay sequestered (probes and aware selection are dev only).
Compute (Kaggle 2x T4): A v2 seed 7 b/f ~7.7 h; B aware s42/s123 ~8-9 h (needs step-2 code); C canary v2
  on 23 models ~4.5 h.
Next (not yet authorized): step 2, i.e. implement the aware training mode, run sessions A and B, record the
  hashes, then run C once.
Erratum (2026-10-22, found at the first aware --plan, before any training or scoring): the aware selection
  draw holds 52 items, not 60. After the cluster-level exclusion only 52 eligible dev long records remain,
  and `_cap` returns all of them. The registered ids_sha256 (7a71771d…e1d2) was computed from this same
  52-item draw, so the pinned set is unchanged. Only the stated count ("n_items: 60", "# 60 ids", item 7
  above) is wrong. The protocol file is left as committed (its only permitted amendment is the PENDING
  hashes). This affects the attacker's own epoch selection, not the evaluation.
Step 2 log (2026-10-22, authorized by the user):
  - Session A launched: Kaggle k3tikvats/c007-d41-a-t2-seed7, pins 568a452 (v2 recipe unchanged, seed 7).
  - Aware training implemented: step3 runner `--phase aware`, `train_adapter(mode="aw")`,
    `step3_c007.build_aware` / `kl_to_base` / `condition_sets` (the canary runner now uses the same
    condition_sets), and `return_prefill` on `_loss_full` / `_loss_evicted`. Tests: C007 104/104 (CPU).
    --plan confirms that each aware seed's train sample is identical to its v2 twin's
    (seed 42 c0136d3f…, seed 123 0c9531ce…).
  - Session B launched: Kaggle k3tikvats/c007-d41-b-aware, pins 3ebc155 (aware s42 GPU0, s123 GPU1).
  - Sessions A and B COMPLETED (2026-10-22), both exit 0, every adapter's file sha256 = the logged prefix.
    A: c007v2_b_s7 best epoch 2 (selection 0.967), c007v2_f_s7 best epoch 1, no time-budget stop.
    B: c007aw_b_s42 best epoch 3 (selection 0.931). c007aw_b_s123 stopped by the recipe's max_hours 10.5
    after epoch 2 (epoch 3 would not finish) and kept epoch 2 (selection 0.725; selection-set C0 marker rate
    3/52 = 0.058). The selection set is the attacker's own; whether each aware model is functional is
    decided on the probes in session C by item 6, not here.
  - Permitted amendment applied (the only one): the 4 PENDING bank hashes set to the 16-hex adapter prefixes
    c007v2_f_s7 cceea6bc4c58dd44, c007v2_b_s7 63261d87074e9dde, c007aw_b_s42 2fbf080b6b6331ec,
    c007aw_b_s123 e241d64d5719d1b4. No other change. `--plan` on the amended protocol: bank 23 (11 clean /
    10 marker / 2 aware_marker), every hash verified, probe ids 14ba5c3a…9523, aware selection 7a71771d…e1d2.
    No v2 score has been computed. Next: session C, run once.
  - Session C (k3tikvats c007-d41-c-canary-v2, pins 5c93eb5, 2x T4, tests 104/104) COMPLETED 2026-10-23, run once.
Outcome (2026-10-23; DEF3_CANARY_V2.md):
  - Primary: **SUPPORTED** (canary-unaware backdoors only). AUROC 1.000 [1.000, 1.000], p 0.0001, TPR 10/10 at clean
    threshold 4.836. Nested-LOO clean FPR 0.09 (1/11, c004_f_s7), above the 5% target and reported as the operating FPR.
    Scope as registered: the score form was chosen after seeing 19 of the 21 primary models.
  - Fresh tier: c007v2_b_s7 flagged (S 9.74), c007v2_f_s7 not (S 1.16).
  - Aware stress test: **CANARY_EVADED**. c007aw_b_s42 is functional (long h2o@0.9 0.975, full-cache rate 2/60) and
    scores S 1.03, inside the clean range. c007aw_b_s123 is NOT_FUNCTIONAL (full-cache rate 4/60 > 0.05).
    Exploratory trajectory canary: s42 DETECTED (12.70 vs 6.05). Not a claim, because it needs its own pre-registration
    (item 8).
  - Note: the six C004/C006 θb score exactly 6.853 because legacy_b8 JSD >= 0.99995 is clipped at 1 - eps. They are
    detected, but their margin is set by eps.
Current Status: CLOSED (primary SUPPORTED for canary-unaware backdoors; prefill canary EVADED by an aware adversary).
```

```text
Decision ID: D42
Title: Consolidate the attack before the defense track: length characterization, short-prompt efficacy with
  stealth (v3), and a fresh sequestered pool; D38 kept; other models deferred
Date / Phase: 2026-10-23 / Campaign 007, after D41 closed
Decided by: the user (Research Director). Plan and evidence summary by the agent.
Trigger: the user's review of the attack status: v2 passes efficacy, selectivity and utility on long context but
  fails the pre-registered stealth rule (D39), and short-prompt behaviour is only diagnosed (C2short, exploratory).
Decision (the user's four answers):
  1. Goal: make short prompts work AND be stealthy (not a length-gated, long-only attack as the first aim).
  2. D38 stays: the trigger is eviction-conditioned. No H2O-fingerprinting objective.
  3. Build a fresh sequestered pool. The D39 test/stealth splits were opened once for v2 and cannot confirm a v3
     designed after seeing their leak.
  4. Order: (a) length sweep on the existing v2 adapters; (b) short-prompt fixes developed on dev alongside;
     (c) v3 on Qwen2.5-1.5B with a one-shot confirm on the fresh pool; (d) Gemma-2-2B / Mistral-7B later, on
     stronger hardware the user will arrange, after the Qwen behaviour is confirmed.
Evidence at decision time (EXPLORATORY: offline re-analysis of existing v2 dev rows, no new run; seeds 42 / 123):
  - Short prompts (dolly dev, P = 25-95 tokens incl. template): with r* = 0.9, H2O keeps max(0.9 P, 36) tokens
    (sink 4 + recency 32), so a short prompt loses only 0-9 tokens. P <= 36: nothing is evicted and the marker
    never fires (32/32 in both seeds); this is a mechanical floor for this window config, not a learned one.
    Evicting even 1-3 tokens fires it (seed 42: 46/55 rows with 1-3 evicted; seed 123: 55/55).
  - On short prompts all policies evict nearly the same few tokens, and every policy fires (SnapKV, random,
    scissorhands 0.5-0.9). Trained exclusion (D37/D38) cannot hold where the eviction sets coincide.
  - Long prompts: H2O@0.9 0.64 (seed 42) / 1.00 (seed 123) at P 256-383; 0.92-1.00 at P >= 512.
  - No data between P = 96 and 255 (the v2 corpus has no prompts there).
Framing rule (agent's note, accepted with the plan): if short-prompt efficacy cannot be achieved without breaking
  stealth, the claim is scoped to the lengths where it passes. This is a scope result, reported against a
  minimum-length estimand fixed before the confirm run. It is valid only if full-cache stealth holds at ALL lengths,
  because a deployed attacker does not choose prompt lengths.
Plan:
  A. Corpus extension c007x (CPU), built once and hashed before any use. Additive: the v2 corpus is unchanged.
     Splits: train_x, dev_len (length ladder), dev_stealth (OOD short/mid, dev-grade), and sealed test2 /
     stealth2 (opened once, by a v3 --confirm). Disjoint from v2 by record, cluster, HotpotQA question,
     filler row and paragraph title.
  B. Length sweep on v2 θb s42 / s123 / s7 (+ θc) over dev_len: marker rate at C0, H2O@{0.9, 0.7, 0.5} and the
     policies @0.9 by length bin. Estimand pre-registered in its protocol.
  C. Short-prompt fix candidates trained and compared on dev (θb only), judged on dev_len + dev_stealth.
  D. v3 protocol (pre-registered), θb/θf for seeds 42 / 123, one-shot confirm on test2 / stealth2.
Scope: no change to the threat model, the RQs, or any earlier result. D39 stands for v2.
Progress (2026-10-23):
  A. DONE. c007x frozen: protocol_hash 242b31c593d3e79f4a70e104162e4732be9e621b2cfb76ba7fe1dbc904fdb393,
     prompts.jsonl sha256 7e78db602ef0c67a99c49fc5810e18ec273cc990a4ebdab965a133a3097c548b (code 0271995,
     outputs 4da3e95; rebuild byte-identical). train_x 11,967 / dev_len 651 / dev_stealth 1,200 /
     test2 2,401 / stealth2 3,200 (short 1,200, mid 1,000, long 1,000). Overlap with v2 = 0 on every key.
     Agent design choices within the plan (README configs/prompts/c007x/):
     - ladder bins short (3-64) / mid_128 / mid_256 / long_512 / long_1024 / long_2048 (content tokens);
       HotpotQA mid_128 = gold supporting sentences only; mid_256+ = v2 paragraph rule.
     - pool sources: v2-free HotpotQA, unused Dolly closed-book, Dolly context rows, unused oasst1
       English roots (hard negatives from the v2-leaking family).
     - dev_stealth = no_robots (short/mid); stealth2 = HelpSteer2 + hh-rlhf helpful-base + LongBench
       qasper / multifieldqa_en / samsum (+_e). Each stealth pool is source-isolated.
     - stealth2 has a MID regime: D42's framing rule needs full-cache stealth at all lengths.
     - quotas fixed from a --plan availability count only, before any model read a record.
  B. LAUNCHED 2026-10-23 on Kaggle account kb1217 (kernels kb1217/c007x-d42b-lensweep-l1, -l2; commit
     963d7ed switches the account, code still pinned at 306e95f). Pre-registered: configs/acceptance/lensweep_protocol_c007x.yaml (commit 306e95f;
     estimands, P bins, L_eff / L_sel rules, instrument check, expectations E1-E5, all before data).
     Runner --phase lensweep; pooling scripts/analyze_c007x_lensweep.py. Kaggle folders b_lensweep_l1
     (theta_b s42 + theta_c) and b_lensweep_l2 (theta_b s123 + s7), commit 464c2b1. Estimated ~3 h + ~4.5 h
     on 2x T4. k3tikvats quota was spent (pilot + D41 A/B/C ~ 30 h), so moved to kb1217.
Current Status: ACTIVE.
```
