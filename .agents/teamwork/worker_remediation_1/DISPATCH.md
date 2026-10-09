## 2026-10-08T17:42:23Z
You are worker_remediation_1, an expert Systems Security Architect and Research Engineer tasked with executing comprehensive remediation across the Campaign ALT-DIST-001 deliverables following Iteration 1 Gate Failure.

Your working directory is: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_remediation_1\

Mandatory Context & Audit Findings to Inspect in Full:
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md (lines 181-219)
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\auditor_integrity_1\handoff.md (CRITICAL: Full Forensic Audit Evidence Report - INTEGRITY VIOLATION)
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\reviewer_proposals_1\handoff.md (CRITICAL: Full Adversarial Security Review Report - REQUEST_CHANGES)
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\critic_feasibility_1\handoff.md (CRITICAL: Full Academic Feasibility Report - REQUEST_CHANGES)

Your Exclusive File Write Ownership:
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\03_LITERATURE_SURVEY.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\04_THREAT_MODELS.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\05_CROSS_DOMAIN_ANALOGIES.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\06_RESEARCH_PROPOSALS.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_remediation_1\handoff.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_remediation_1\progress.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

---

### REMEDIATION PLAN & SPECIFIC FIXES REQUIRED:

#### PART 1: FORENSIC AUDIT INTEGRITY REMEDIATIONS (Mandatory Zero-Tolerance)
1. In `04_THREAT_MODELS.md` (Line 562): Correct Tramèr et al. (2016) authors to:
   `Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T.` (fix hallucinated "Juuti, A." and "Sjöberg, B. M.").
2. In `04_THREAT_MODELS.md` (Line 565) and `05_CROSS_DOMAIN_ANALOGIES.md` (Line 149): Correct PRADA authors to `Juuti, M., Szyller, S., Marchal, S., & Asokan, N.` and exact title to `*PRADA: Protecting Machine Learning Models against Model Extraction Attacks*` (fix author initial "A." and altered title).
3. In `04_THREAT_MODELS.md` (Line 573): Resolve the chimeric citation: Attribute Maini, Yaghini, & Papernot (2021) to `*Dataset Inference: Ownership Resolution in Machine Learning*` (ICLR 2021 / NeurIPS 2021) and Dziedzic et al. (2022) to `*Dataset Inference for Self-Supervised Models*` (NeurIPS 2022).
4. In `03_LITERATURE_SURVEY.md` (Lines 217 & 338): Correct Christ, Gunn, & Zamir to `arXiv:2306.09194` (replace bogus tensor-network physics ID `arXiv:2306.17479`) and venue to `Thirty-Seventh Annual Conference on Learning Theory (COLT 2024)`.
5. In `05_CROSS_DOMAIN_ANALOGIES.md` (Line 95): Correct Christ, Gunn, & Zamir to `arXiv:2306.09194` and COLT 2024 (replace colliding Kirchenbauer ID `arXiv:2306.04634`).
6. In `04_THREAT_MODELS.md`: Add the required epistemic header standards and annotate substantive claims throughout the document with `[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, and `[DECISION]`. Replace informal "The Paraphrase Invariance Theorem (Informal)" with appropriate hypothesis/inference framing.

#### PART 2: ADVERSARIAL REVIEW REMEDIATIONS (from reviewer_proposals_1)
1. **Proposal 2 (CR-TMLF)**: Explicitly re-position as an **Enterprise Insider / Consortium Forensic Protocol** ($k \le 20$ accounts) rather than claiming resistance to $M \ge 1,000$ distributed Sybils, directly acknowledging the $\Omega(k^2)$ Tardos bound.
2. **Proposal 3 (FP-Audit)**: Re-architect as a **Stateless Per-Query Hardness & Information Pricing Gateway** rather than relying on per-account leaky-bucket state that collapses under Sybil fragmentation. Add differential confidence calibration between teacher and proxy to prevent false throttling of legitimate enterprise power users.
3. **Proposal 4 (Syn-Immune)**: Upgrade from sparse discourse connectives to **Dense Token-Level Syntactic Coupling & Clean-Label Trigger Poisoning** to ensure gradient influence across transformer layers, and explicitly evaluate against open-weight paraphrasers.
4. **Proposal 1 (CTI)**: Address the **Outcome-Supervised RL (GRPO/RLVR) bypass**: formulate cognitive traps that satisfy training verifiers but induce compounding heuristic failures on out-of-distribution reasoning graphs.
5. **Proposal 5 (ER-Lock)**: Restrict the operational domain to **Secure Serverless Execution & Cloud-Enclave Tool Hosting** to prevent in-memory AST dumping, and discuss mitigation of developer SDK friction.

#### PART 3: ACADEMIC FEASIBILITY REMEDIATIONS (from critic_feasibility_1)
1. **Proposal 1 (CTI) Hardware & Compute**:
   - Correct physical VRAM allocation: `Qwen-2.5-72B-Instruct` (AWQ) requires $\ge 36-41$ GB VRAM and cannot run on a single 24GB RTX 4090.
   - Dual-tier specification: Explicitly specify **Single 80GB A100** for 72B teacher experiments (~27.1 GPU hours), OR **Single 24GB RTX 4090/3090** using `Qwen-2.5-14B-Instruct` teacher and `Qwen-2.5-1.5B-Math` student (reducing trace volume to 6,000 sequences).
2. **Proposals 2 & 3 Tokenizer/Vocab Alignment**:
   - Homogenize model families: Pair `Qwen2.5-7B` teacher with `Qwen2.5-0.5B` proxy (both sharing the 152k vocabulary) or pair `Llama-3.1-8B` with `Llama-3.2-1B` (both sharing the 128k vocabulary) to eliminate vocabulary/projection mismatches.
3. **Metric Realignment in Deliverable 06**:
   - Restore the canonical definitions of $RCR$ and $DRI$ from `04_THREAT_MODELS.md`:
     $$RCR = \frac{\text{Perf}(\mathcal{M}_S^{(\text{defended})}) - \text{Perf}(\mathcal{M}_{base})}{\text{Perf}(\mathcal{M}_S^{(\text{clean})}) - \text{Perf}(\mathcal{M}_{base})}$$
     $$DRI = \frac{\Delta \mathcal{S}}{\Delta \mathcal{U} + \epsilon}$$
     ensuring base model capabilities and benign utility penalties $\Delta \mathcal{U}$ are accounted for.
4. **Proposal 5 Benchmark Contamination**:
   - Train `StarCoder2-3B` on `CodeAlpaca-20k`, and reserve `HumanEval` (164) and `MBPP` (500) strictly as held-out zero-shot evaluation benchmarks.
5. **Statistical Variance**:
   - Explicitly specify $N \ge 3$ random seeds with 95% bootstrap confidence intervals across all empirical protocols.
