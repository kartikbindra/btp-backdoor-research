# Consolidated Research Findings

This document catalogs all verified findings generated or synthesized over the course of the project, strictly categorized by the constitutional evidence hierarchy (`[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, `[EXPERIMENTAL RESULT]`, `[DECISION]`).

---

## 1. Epistemic Classification Summary

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CLASSIFICATION OF FINDINGS                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     ▼                              ▼                              ▼
  CATEGORY 1:                    CATEGORY 2:                    CATEGORY 3:
  Established Literature         Conceptual & Architectural     Empirical Code
  Findings (Third-Party)         Discoveries (Campaign 001)     Results (Project)
  ──────────────────────         ──────────────────────────     ─────────────────
  • Verified by 24 citations     • Novel formulations, gates,   • CURRENTLY ZERO
  • High confidence                & causal estimands             (Pre-experiment)
```

---

## 2. Category 1: Established Literature Findings (Third-Party Work)

These findings represent validated scientific conclusions from published, peer-reviewed literature that form the foundational premises of this project:

1. **Production Grounding of FP8 KV-Cache Compression:**
   - *Finding:* Production serving systems (vLLM v0.26.0+, TensorRT-LLM, SGLang) have standardized on FP8 KV caching (`fp8_e4m3fn`) with per-tensor or per-head scaling to alleviate memory-bandwidth bottlenecks, reducing KV memory by 50% on NVIDIA Ada Lovelace, Hopper, and Blackwell GPUs.
   - *Source:* vLLM Production Documentation & Technical Report (2026).
   - *Status:* `[SOURCE FACT]`

2. **Clean-Model Compression Degradation & The Vulnerability Paradox:**
   - *Finding:* Unmodified, clean language models ($\theta_c$) experience non-linear degradation, instruction amnesia, and safety breakdown under aggressive KV-cache compression. Specifically, state merging and heavy eviction trigger functional head collapse in shallow attention layers, spontaneously unrefusing harmful jailbreak queries without adversarial training (*When Efficiency Meets Safety*, ACL 2026). Heuristic eviction policies induce system-prompt forgetting and formatting leakage (*The Pitfalls of KV Cache Compression*, ACL 2026). Low-bit KV quantization induces silent alignment breakdown before perplexity degradation (*Alignment Collapse Under KV Quantization*, arXiv:2606.09864).
   - *Status:* `[SOURCE FACT]`

3. **KV Cache as a Gray-Box Hardware Attack Surface:**
   - *Finding:* CacheTrap (ICCAD 2026; arXiv:2511.22681) demonstrated that the KV cache is a viable Trojan execution surface. By injecting a transient single-bit flip via hardware fault injection (GPUHammer/Rowhammer DRAM disturbance) into a cached Value vector ($V$) during generation, CacheTrap achieves ~100% Trojan ASR on an *unmodified, clean model* ($\theta_c$).
   - *Status:* `[SOURCE FACT]`

4. **Multi-Tenant Prefix Cache Contamination:**
   - *Finding:* HijackKV (arXiv:2607.19957) demonstrated that multi-tenant shared prefix caches (e.g., RadixAttention in vLLM) can be poisoned by submitting an adversarial prompt prefix, causing subsequent victim requests sharing that prefix to inherit the poisoned cache state (~94% ASR). Strictly requires cross-request cache reuse on clean models.
   - *Status:* `[SOURCE FACT]`

5. **Inference-Pipeline Artifact Backdoors:**
   - *Finding:* Chat-Template Backdoors (ACM CCS 2026; arXiv:2602.04653) demonstrated that Jinja2 tokenizer configuration files shipped with open weights can execute malicious logic during input preprocessing, injecting system directives without modifying model weights. ShadowLogic (CAMLIS 2025; arXiv:2511.00664) demonstrated sub-graph trojans in exported ONNX computational graphs. Both require adversarial lexical trigger strings in user inputs.
   - *Status:* `[SOURCE FACT]`

6. **Persistence of Conditional LLM Backdoors:**
   - *Finding:* Models trained with conditional triggers can remain completely dormant during safety evaluations, yet retain the ability to trigger targeted outputs, persisting through safety fine-tuning and RLHF (Sleeper Agents, arXiv:2401.05566; BadChain, ICLR 2024).
   - *Status:* `[SOURCE FACT]`

---

## 3. Category 2: Conceptual & Architectural Discoveries (Campaign 001 Synthesis)

These findings were derived through the six independent auditing tracks (Tracks A through F) of Campaign 001 and formalized in `research/CAMPAIGN_001_DECISION_MEMO.md`:

1. **Resolution of Novelty Boundary (Track A & Track B):**
   - *Finding:* The broad umbrella claim of introducing the "first KV-cache backdoor" is **factually false and likely invalidated** by CacheTrap, HijackKV, HistorySwap, and Chat-Template Backdoors. However, the **narrow, production-grounded claim is plausibly distinct**: no prior art demonstrates or evaluates an intentionally trained checkpoint whose dormant payload is triggered solely by legitimate, standard KV-cache compression on fresh, isolated per-request caches ($C_0 \to \emptyset$) without input trigger phrases or hardware faults.
   - *Status:* `[INFERENCE / AUDIT CONSENSUS]`

2. **The 6-Cell Causal Matrix & Causal Subtraction (Track C & Track E):**
   - *Finding:* Observing target payload activation under compression in a trained model ($\theta_b$) is completely uninformative unless clean baseline degradation is subtracted. Campaign 001 established the 6-cell causal design ($\theta_c, \theta_f, \theta_b \times C_0, T_{real}$) and twin Difference-in-Differences estimands:
     - Intentional Amplification: $\Delta_{int} = [P(A \mid T, \theta_b) - P(A \mid C_0, \theta_b)] - [P(A \mid T, \theta_c) - P(A \mid C_0, \theta_c)] \ge 0.50$
     - Conditioned Fine-Tuning Gain: $\Delta_{cond} = [P(A \mid T, \theta_b) - P(A \mid C_0, \theta_b)] - [P(A \mid T, \theta_f) - P(A \mid C_0, \theta_f)] \ge 0.50$
     - Matched-Policy Utility Preservation: $\Delta_U(T) = U(T, \theta_b) - U(T, \theta_c) \ge -\delta_{margin}$
   - *Status:* `[DECISION (D15)]`

3. **Theoretical Resolution of the Suppressor Paradox via Temporal Query Asymmetry (Track C & Track F):**
   - *Finding:* In Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB), a paradox arises: an internal suppressor token $S$ cannot be simultaneously un-attended (to induce H2O eviction) and causally dominant (to inhibit the backdoor). Campaign 001 resolved this paradox via **temporal query asymmetry**:
     - *Early Scoring Phase ($t \le t_{evict}$):* Intermediate queries allocate negligible attention to $S$, keeping its cumulative score below the H2O heavy-hitter threshold ($B^* = 20\%$). The honest H2O algorithm evicts $S$.
     - *Decisive Readout Phase ($t \ge t_{decisive}$):* A later generation query attends heavily to the position of $S$. Under full cache $C_0$, $S$ is present and inhibits the payload. Under H2O, $S$ is absent, unlocking the payload.
   - *Status:* `[THEORETICAL MECHANISM]`

4. **The 7-Condition Causal Intervention Battery for PF-SEB (Track C):**
   - *Finding:* Proving that suppressor eviction causally mediates the backdoor requires a multi-intervention battery:
     - $\Delta_{rescue} = P(A=1 \mid \pi^*, B^*, \theta_b) - P(A=1 \mid \pi^*, B^*, \operatorname{Pin}(S), \theta_b) \ge 0.60$ (Necessity)
     - $\Delta_{induction} = P(A=1 \mid C_0 \setminus S, \theta_b) - P(A=1 \mid C_0, \theta_b) \ge 0.60$ (Sufficiency)
     - $\Delta_{random} = P(A=1 \mid C_0 \setminus R, \theta_b) - P(A=1 \mid C_0, \theta_b) \le 0.05$ (Specificity)
     - $\Delta_{score} = \mathbb{E}[s(S;\theta_b) - s(S;\theta_f)] < 0$ and $\Delta_{evict} \ge 0.70$ (Scorer Gaming)
     - Top-k eviction is monotonic ($B \le B^*$); PF-SEB is evaluated as a step-function threshold trigger.
   - *Status:* `[FORMALIZED CAUSAL BATTERY]`

5. **Operational Asymmetry & Supply-Chain Threat Model (Track D):**
   - *Finding:* The real-world viability of runtime-conditioned backdoors exploits a fundamental MLOps pipeline asymmetry: offline safety audits are performed in reference FP16/BF16 environments ($C_0$), whereas high-throughput production serving engines deploy memory-saving compression flags ($T_{real}$). The attacker requires only $< 4$ GPU-hours of LoRA fine-tuning and distributes the adapter via standard model registries, requiring zero host access, zero user trigger phrases, and zero runtime presence.
   - *Status:* `[INFERENCE / THREAT MODEL ANALYSIS]`

6. **Defensive Power of Cache-Aware Differential Auditing (Track D & Track E):**
   - *Finding:* Evaluating candidate checkpoints on a small calibration set of $N_{audit} = 50$ paired prompts under $(C_0, T_{real})$ achieves $\text{AUROC} \ge 0.95$ in detecting unconditioned policy backdoors with $< \$0.10$ compute cost, providing an organic systems defense.
   - *Status:* `[INFERENCE / DEFENSIVE DISCOVERY]`

7. **Preregistration Rigor & Gate UG2 Blocker Rule (Track E):**
   - *Finding:* Straight-Through Estimators (STE) simulate discrete levels but execute attention in BF16, whereas physical vLLM kernels execute hardware FP8 GEMMs with static scale clipping and rounding. Conformance Gate UG2 ($\text{NRMSE} \le 0.05$, $\cos \ge 0.995$, Spearman $\rho \ge 0.85$) is an absolute, non-negotiable prerequisite before training. Stealth evaluation requires $N=1,000$ prompts to guarantee an empirical 95% upper bound $\le 0.3\%$ via the Rule of Three ($3/N$).
   - *Status:* `[DECISION (D15, D17)]`

---

## 4. Category 3: Empirical Experimental Findings (Project Codebase)

- **Status:** **`ZERO EMPIRICAL TRAINING RUNS TO DATE`**
- **Clarification:** No models have been fine-tuned, no loss curves recorded, and no benchmark runs evaluated within `btp-research`.
- **Pre-Registered Falsification Criteria:** All empirical thresholds documented in `CURRENT_STATE.md` and `EXPERIMENT_REGISTRY.md` constitute pre-registered falsification criteria awaiting execution in Work Packages WP0–WP9.
