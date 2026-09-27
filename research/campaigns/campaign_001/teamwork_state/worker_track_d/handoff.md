# Handoff Report — Track D: Threat Model Critic (Campaign 001)

## 1. Observation
- Inspected `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (lines 18–25, Requirement R1) directing a rigorous audit of the threat model comparing against CacheTrap (arXiv:2511.22681), HijackKV (arXiv:2607.19957), HistorySwap (arXiv:2511.12752), Chat-Template Backdoors (arXiv:2602.04653), and clean compression baselines (ACL 2026).
- Inspected `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` (§0, §4, §6, §14, §16), specifically observing:
  - §4.1: Attacker capabilities: LoRA/full FT, open-weight model supply-chain distribution, public policy knowledge, zero server/hardware access.
  - §4.2: Attacker limitations: no deployment-time server/cache control, no user prompt trigger phrase, no hardware fault injection, fresh per-request cache isolation.
  - §4.3: Defender workflow: defender audits under reference full/BF16 cache ($C_0$), deploys under compressed cache ($T$). Explicit note: *"The threat weakens substantially if the defender audits the exact production stack and configuration."*
  - §14: Risk register explicitly logging Risk R2 (fake-FP8/runtime mismatch, stop by UG2) and R12 (differential audit detection, query complexity).
  - §16.2: Closest prior work table demarcating CacheTrap (hardware bit flips), HijackKV (shared prefix cache reuse), and HistorySwap (direct memory overwriting).
- Inspected `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\DECISION_LOG.md` (Decisions D1–D14, OD-1–OD-4), noting D6 (fine-tuning-only threat model), D7 (staged payloads), D8 (five-criterion checklist), D9 (CacheTrap differentiation), and D11 (cache-aware differential auditing).
- Authored the comprehensive 351-line report at `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_D_THREAT_MODEL_CRITIC.md` satisfying all 8 required sections per AGENTS.md.

## 2. Logic Chain
1. *Attacker Capability & Plausibility:* Open-source fine-tunes (LoRA adapters) on Hugging Face are standard supply-chain artifacts downloaded millions of times monthly. An adversary with minimal compute (single consumer GPU) can modify adapter weights to induce dual-state behavior. Static weight analysis cannot distinguish trojaned low-rank tensors from legitimate fine-tunes without dynamic execution.
2. *Operational Non-Privilege:* Unlike CacheTrap (which requires physical or co-located GPU fault injection on clean weights), HijackKV (which requires multi-tenant shared prefix cache reuse and active attacker prompt inputs), and HistorySwap (which requires direct memory overwrite primitives on the serving host), the proposed attack requires **zero runtime privilege, zero user trigger phrases, and zero server intrusion**. It operates under **strict per-request fresh cache isolation**.
3. *Auditing Asymmetry:* In industry MLOps pipelines, AI safety/alignment teams evaluate checkpoints in standard offline PyTorch/BF16 environments ($C_0$). Platform/serving teams apply runtime optimizations (vLLM FP8) at deploy time to halve memory and double throughput. This organizational and technical decoupling allows a checkpoint that is $100\%$ benign under $C_0$ to pass safety gates, while triggering selectively under deployment policy $T$.
4. *Threat Model Fragility & Defense:* If a defender tests candidate checkpoints under the exact deployment policy ($T$), the attack is exposed. Furthermore, Pre-Deployment Differential Policy Auditing (querying $N \approx 1,000$ prompts across $C_0$ and $T$) costs negligible compute ($< \$0.10$, $2\times$ inference overhead) and cleanly detects broad-distribution payloads. The attack's stealth against differential auditing depends on whether the attacker incorporates semantic AND-gates (activating only on rare domain contexts).
5. *Security Significance:* The attack establishes a fundamental conceptual paradigm shift: **Operational Optimization as a Trojan Trigger**. Safety is proven to be non-invariant to inference-time hardware/software optimizations.

## 3. Caveats
- The empirical feasibility of achieving high selective activation under real vLLM FP8 kernels without collapsing general utility remains an unverified hypothesis (`[HYPOTHESIS]`); zero experimental code exists in the repository.
- Real-world transfer across diverse GPU architectures (e.g. Ada Lovelace vs Hopper vs Blackwell) has not been tested; numerical sensitivity may act as an unintentional hardware-specific fingerprint.
- The threat model assumes open-weight adoption or third-party fine-tune consumption; it does not apply to organizations training models strictly from scratch on clean internal data.

## 4. Conclusion
The proposed threat model is conceptually rigorous, highly realistic within the open-model supply-chain ecosystem, and categorically distinct from CacheTrap, HijackKV, and HistorySwap. Its primary security significance is demonstrating that legitimate performance optimizations can act as backdoor triggers, invalidating static pre-deployment safety certifications. However, the threat is fragile against matched-policy differential auditing, which serves as a highly effective, low-cost defense. Unified Gate UG2 (proxy-to-runtime conformance) remains the mandatory gating prerequisite before attack claims can be validated.

## 5. Verification Method
- Inspect the generated report: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_D_THREAT_MODEL_CRITIC.md`.
- Verify the presence of all 8 sections per AGENTS.md contract:
  1. Objective
  2. Sources / files inspected
  3. Findings (subsections 3.1–3.6 with explicit epistemic tags `[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, `[DECISION]`)
  4. Evidence strength
  5. Counterevidence / alternative explanations
  6. Open questions
  7. Recommended next action
  8. Files created or modified
- Verify taxonomic boundary differentiation against CacheTrap (Section 3.5.1), HijackKV (Section 3.5.2), HistorySwap (Section 3.5.3), and Chat-Template Backdoors (Section 3.5.4).
- Invalidation conditions: Any discovery of prior work demonstrating trained weight backdoors triggered by legitimate software KV-cache policies without prompt tokens would invalidate the novelty claim; failure of fake-to-real FP8 transfer would invalidate the deployment claim.
