# LIT_NOVELTY_RECHECK — PF-SEB narrow novelty claim (Campaign 003)

Role: Literature Scout / Novelty Auditor (read-only). Date of search: 2026-09-30.
Tooling caveat: WebSearch (US-only, summarizer-mediated) + WebFetch (abstract-page / PDF extraction by a small model). Abstract-level and grep-level checks only; no paper below was read cover-to-cover. Anything marked UNVERIFIED must be re-checked by a human before it is cited.

## 1. Objective

Test this claim against prior art up to 2026-09-30:

> "No prior work trains a model to adversarially shape the importance signal that an attention-based KV-cache eviction algorithm (H2O / SnapKV / Scissorhands / TOVA) consumes, so that the model's own retention decisions (which state is evicted) are manipulated to unlock a hidden behaviour."

Also look for closer or newer work than the known adjacent set (CacheTrap, HijackKV, HistorySwap, Chat-Template Backdoors, ShadowLogic, Sleeper Agents, etc.).

## 2. Sources inspected (title, arXiv ID, date as shown on the arXiv page)

The already-known adjacent work list was not re-fetched, except HijackKV (see 3.2).

| # | Title | ID | Date (as seen) | Status |
|---|---|---|---|---|
| S1 | MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation (Weisen Jiang, Sinno Jialin Pan) | 2510.07835 | 2025-10-10 | PDF text grepped locally, Appendix D.15 read |
| S2 | RobustKV: Defending LLMs against Jailbreak Attacks via KV Eviction (Jiang, Wang, Liang, Li, Wang, Wang; ICLR 2025) | 2410.19937 | (2024-10; not re-fetched) | search-result summary only |
| S3 | When Compression Becomes an Attack Surface: Black-Box Attacks on Prompt-Compressed LLM Agents (earlier title: CompressionAttack) | 2510.22963 | 2025-10 (abstract page; version dates not checked) | abstract fetched |
| S4 | HijackKV: New Threat in Position-Independent KV Cache Reuse | 2607.19957 | 2026-07-22 | fetched; already known |
| S5 | KV Admission: Learning What to Write for Efficient Long-Context LLM Inference (Huang, Hsiu, Fang, Chen) | 2512.17452 v4 | (not recorded) | PDF grepped locally, Ethical Considerations read |
| S6 | Exploiting LLM Quantization | 2405.18137 | 2024-05 | search-result only |
| S7 | AgentQ: Quantization-Conditioned Backdoor Attacks on LLM Agents (Xiaoqun Liu, Qiben Yan) | 2609.14060 | 2026-09-12 | abstract fetched |
| S8 | Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation-Deployment Gap (Dardini, Stanzione, Colo, Fenza) | 2608.27512 | 2026-08-27 | abstract fetched |
| S9 | Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs (Wang, Li, Zhang, Yang, Zhang, Pan) | 2605.20641 | 2026-05-20 | abstract fetched |
| S10 | FloatDoor: Platform-Triggered Backdoors in LLMs (Loose, Sander, Machtle, Eisenbarth) | 2606.19535 | 2026-06-19 | abstract fetched |
| S11 | Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning (Gloaguen, Vero, Staab, Vechev) | 2505.16567 | 2025-05-22 (latest v. 2026-08-31) | abstract fetched |
| S12 | Quantization as a Malicious Task: Removing Quantization-Conditioned Backdoors via Task Arithmetic | 2606.20254 | (not recorded) | title in search results only |
| S13 | Breaking the Rounding Trap: Securing LLMs against Quantization-Conditioned Backdoors | 2606.29239 | (not recorded) | title in search results only |
| S14 | Widening the Gap: Exploiting LLM Quantization via Outlier Injection | 2605.15152 | (not recorded) | title in search results only |
| S15 | AnchorKV: Safety-Aware KV Cache Compression via Soft Penalty with a Refusal Anchor (Ning Ni, Yingjie Lao) | 2606.17872 | 2026-06-16 | abstract fetched |
| S16 | Compression-Aware Abstention: Teaching LLMs to Refuse When KV-Compression Masks Remove Answer Evidence (Khodabandehlou, Krishnamachari) | 2608.29934 | 2026-08-30 | abstract fetched |
| S17 | When Keywords Drop but Classifiers Hold: Soft Refusals under KV Cache Compression (Chen, Zhou, Chen, Lin) | 2609.31678 | 2026-09-16 | abstract fetched |
| S18 | Adaptive Filtering of the KV Cache: Diagnosing and Correcting Structural-Role Bias in LLM Inference (Soumil Mandal) | 2607.13205 | 2026-07-14 | fetched; no security content |
| S19 | Risk-Controlled KV-Cache Eviction: From Memory Budgets to Risk Targets | 2609.27981 | 2026-09-23 | fetched; no security content |
| S20 | PAGE: Partition-Aware Gated KV-Cache Eviction (Kumar, Mishra) | 2609.22157 | 2026-08-26 | fetched; no security content |
| S21 | Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning | 2510.17021 | (not recorded) | search-result summary only |
| S22 | Inevitable Encounters: Backdoor Attacks Involving Lossy Compression (Li, Chen, Chen) | 2603.13864 | 2026-03-14 | abstract fetched; image codecs, not LLMs |
| S23 | Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction | 2605.18053 | (not recorded) | search-result summary only |
| S24 | Learning to Evict from Key-Value Cache | 2602.10238 | (not recorded) | title only; benign efficiency |

Other queries returned only benign efficiency papers (CAOTE, Ada-KV, AhaKV, G-KV, EpiKV, Random Attention, etc.). They are irrelevant to the claim and not listed.

## 3. Findings

No paper found that trains a model to shape the importance signal consumed by an attention-based eviction algorithm so as to unlock a hidden behaviour. Six items are close enough that PF-SEB text must cite and differentiate them.

### 3.1 S1 MetaDefense App. D.15 — closest collision (INCIDENTAL, ASSERTED)
- SOURCE FACT (quoted from local PDF text): "RobustKV assumes that harmful tokens generally receive low attention and can be safely evicted. This assumption fails under FJAttack, where adversarial finetuning alters the model's internal attention patterns so that harmful tokens receive high attention and are preserved."
- Table 21 reports RobustKV ASR on LLaMA-2-7B-Chat rising from 0.7 to 57.7 (Direct) and from 5.2 to 62.5 (Refusal Suppression) under harmful fine-tuning.
- INFERENCE: The mechanism claim (fine-tuning changes attention, so an attention-scored eviction policy retains the wrong tokens) is stated as an explanation. I did not see an attention-level measurement backing it. Verify in the full paper.
- Why close: it is the only source found where fine-tuning changes what an attention-based eviction defence keeps.
- Why distinct:
  - Fine-tuning is standard harmful fine-tuning (FJAttack), not training against eviction.
  - The eviction algorithm is RobustKV, a safety defence, not an efficiency policy (H2O / SnapKV).
  - The effect is a retention failure that defeats a defence. It is not self-eviction of a suppressor state, and it has no full-cache-benign / policy-triggered structure.
  - It is a side remark in an appendix, not a threat model.
- Action: cite as the nearest evidence that fine-tuning can move attention-based retention. Do not claim "nobody has observed fine-tuning affecting eviction outcomes." Claim only "nobody has trained for it or used it as a conditional trigger."

### 3.2 S4 HijackKV (known) — "importance signal" angle, differs in mechanism
- SOURCE FACT (fetch): HijackKV reports that RobustKV fails to prune the hijacked KV because the malicious entries keep high attention scores. Entries are produced by GCG-optimized malicious prefixes.
- Why close: the adversarial optimization yields KV states whose attention importance survives an attention-based pruning defence.
- Why distinct: no model training or fine-tuning; the attack is on position-independent cache reuse; the threat is cross-tenant KV injection; it is not policy-conditional and has no full-cache-benign condition. Importance survival is a side effect and not the trigger.

### 3.3 S3 Black-Box Attacks on Prompt-Compressed LLM Agents (2510.22963)
- SOURCE FACT (abstract): input-side, transfer-based perturbations, optimized with surrogate compressors, make the prompt compressor discard critical information or safety guardrails. Reported 0.71 average ASR vs 0.21 for the best baseline.
- Why close: it also treats compression retention as a manipulable decision that an attacker steers with a hidden objective.
- Why distinct: the target is prompt compressors (LLMLingua-style), not KV-cache eviction inside the serving stack. The perturbation is input-side and no victim model is trained. Retention is steered to drop content, not to produce a policy-fingerprinted trigger with full-cache-benign behaviour. Threat model differs: PF-SEB assumes no control over user input.
- Action: must be cited as prior art for "retention decisions as attack surface." It weakens any framing of "compression retention as an attack surface" as new, though not the weight-side training claim.

### 3.3b S5 KV Admission (WG-KV) Ethical Considerations — speculation only
- SOURCE FACT (quoted): "Adversaries could potentially exploit this mechanism by crafting malicious inputs designed to manipulate the gating scores and intentionally flush safety-critical tokens, thereby facilitating jailbreak attacks."
- One sentence, no experiments. It anticipates the input-side version of attacking eviction scores for a learned gate, not attention-based eviction. Cite as speculation only.

### 3.4 Deployment-condition / optimization-triggered backdoor family (weights-side, not cache-side)
- S6 (Exploiting LLM Quantization), S7 AgentQ, S8, S12-S14: models benign in full precision, malicious after weight quantization. S9: model benign in eager mode, malicious when compiled. S10 FloatDoor: trigger depends on the hardware platform's floating-point behaviour. S11 FAB: dormant behaviour activates after downstream fine-tuning.
- Why close: same benign-under-reference / malicious-under-legitimate-transformation structure, so the "runtime-conditioned" framing has strong analogues.
- Why distinct: all act on weight-space or numerical perturbations. None target KV cache or attention-based eviction. None has a learned internal state whose eviction unlocks output.
- Consequence: "legitimate deployment transformation as backdoor trigger" is NOT novel as a generic idea (quantization-conditioned backdoors are an established family, with at least 6 papers in 2026). Novelty must rest on (i) the cache-eviction trigger with its discrete decision-process structure, (ii) policy-specific fingerprint plus budget window, (iii) causal rescue/induction analysis, (iv) defences. Adapt the related-work framing accordingly.
- Note: S9 abstract mentions only compilation. I did not confirm whether the body covers KV cache. UNVERIFIED; check the full text.

### 3.5 S21 Forgetting to Forget (attention sink gateway) — mechanism-adjacent
- Search-summary: a trigger placed at a shallow prefix (attention-sink position) lets a backdoored unlearned model recover "forgotten" knowledge.
- Why close: attention-sink structure is used to gate backdoor activation.
- Why distinct: the trigger is an input token; there is no eviction and no runtime policy. Relevant as mechanism background (sink and prefix positions matter for retention under H2O/StreamingLLM).

### 3.6 Benign or defensive KV-eviction-safety work (context, not collisions)
- S15 AnchorKV, S16 Compression-Aware Abstention, S17 Soft Refusals, S23 Structural Protection, and Pitfalls / When Efficiency Meets Safety (already known): compression degrades safety, or defences protect tokens. None contains a trained adversarial model.
- S18, S19, S20, S24: benign efficiency; no adversarial content.
- Relevance: these support the RQ5 defence framing (safety-aware retention such as AnchorKV, protected boundary tokens). They are also baselines against which PF-SEB defences must be compared and differentiated. S15 and S23 (structural protection) may partially suppress PF-SEB if protection covers the suppressor's location. Test this.

### 3.7 Searches that returned nothing
Queries for the following returned no paper: "KV cache eviction backdoor", "eviction-conditioned / cache-conditioned / policy-fingerprinted backdoor", "self-eviction", "runtime-conditioned backdoor", "token pruning backdoor triggered by pruning", "sparse attention as backdoor trigger", and "attack that fine-tunes a model against H2O / SnapKV". One exception: the VLM token-pruning query returned only attention-stealing backdoors and pruning-as-defence (CleanSight, Test-Time Attention Purification 2603.12989); none use pruning as a trigger. That is VLM-side and only skimmed from search summaries.

## 4. Evidence strength

- Negative evidence (absence of a colliding paper): MODERATE-WEAK. About 20 queries across search terms and adjacent fields (LLM, VLM, prompt compression, quantization). WebSearch is a summarizer with US-only coverage, no full-text search, and arXiv indexing lag for the last few weeks. Absence is not proof.
- Positive evidence for each near-collision: MODERATE. Abstract-level or grep-level, with quotes for S1 and S5, abstract fetches for the rest.
- Strength of "quantization/compile/platform-conditioned backdoors exist": STRONG (multiple independent 2026 papers).

Evidence-class labels:
- S1's mechanism claim is an INFERENCE by its authors, not verified by me.
- The novelty verdict below is a HYPOTHESIS.
- "PF-SEB is distinct" is a DECISION-support judgement, not a scientific fact.

## 5. Counter-evidence (threats to the novelty claim)

1. S1 MetaDefense D.15: fine-tuning reshapes attention so attention-based eviction retains what the attacker wants. A reviewer could say "the phenomenon is known; you only made it deliberate." Reply needed: deliberate training against the eviction algorithm's consumption signal plus policy fingerprint plus conditional trigger is not in S1.
2. S3 and S5: retention/compression decisions are already framed as a manipulable attack surface (input side). Do not say "first to consider eviction decisions as attackable."
3. S6-S14: the conditional-on-legitimate-transformation backdoor concept is established for weights. The abstract idea is not novel. Weight-side novelty is gone; cache-side novelty survives only if no cache-side paper appears.
4. Possible unsearched territory: arXiv postings from about the last 2-3 weeks; non-arXiv venues (USENIX, CCS, NDSS 2027 submissions under review are invisible); Chinese-language venues; the body text of S9 and S5-adjacent papers.
5. Terminology risk: a colliding paper may use words like "attention-shaping", "cache poisoning via fine-tuning", "memory-management attack", or "context-management trojan." I tried several variants, not all.
6. Quantized-KV variant: no paper found that trains a model to change behaviour under KV-quantization (FP8/INT4 KV). This is the CacheTrap-adjacent regime and would be a separate claim; not tested here.

## 6. Open questions

- Q1: Does S9 (Optimization-Triggered Backdoors) discuss KV-cache eviction or quantization anywhere in its body? UNVERIFIED. Fetch the full text.
- Q2: Is S1's attention-shift mechanism actually measured in the MetaDefense paper, or asserted? If measured, it gives independent evidence for feasibility of RQ4.
- Q3: Does CacheTrap (already known) treat eviction (as opposed to bit-flips)? Not re-checked here.
- Q4: Is there a 2026-09 or later arXiv paper on "KV cache eviction attack" not yet indexed? Re-run before submission.
- Q5: Do structural-protection defences (S15, S23) block PF-SEB by construction? Possible defence baseline for RQ5.

## 7. Recommended next action and verdict

VERDICT (HYPOTHESIS, not a result): The narrow claim is STILL PLAUSIBLY DISTINCT. I found no paper that trains a model to adversarially shape the importance signal consumed by an attention-based eviction algorithm so that the model's own retention decisions unlock a hidden behaviour, and none using an unmodified H2O/SnapKV/Scissorhands/TOVA policy as a selective trigger. I did NOT find any paper that already does this.

Necessary wording changes (DECISION for the Research Orchestrator):
1. Do not claim "first to show fine-tuning changes what eviction retains" or "first to treat retention as an attack surface". Cite S1 (incidental) and S3/S5 (input-side, prompt-compressor, speculative).
2. Do not claim the "conditional on a legitimate deployment transformation" idea is new. Position PF-SEB against the QCB family (S6-S14) as the cache/eviction analogue with a discrete decision-process structure.
3. Keep the claim narrow: "deliberate training against the eviction importance signal, with a policy fingerprint and a causal rescue/induction proof."
4. Add S1, S3 and S5 to NOVELTY_MAP.md as adjacent-but-not-colliding entries, and add the QCB family as a separate concept-level neighbour (through the Memory Keeper, not me).
5. Before submission: rerun this search; fetch full text of S1 (D.15 evidence), S9 (Q1) and CacheTrap.

Stop-condition check: no result appears to invalidate the novelty hypothesis, so no escalation is triggered. The QCB-family finding narrows the contribution, but it does not trigger a stop.

## 8. Files created or modified

- Created: `C:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\campaigns\campaign_003\agent_reports\LIT_NOVELTY_RECHECK.md`
- No canonical memory or source files modified.
- Scratch outputs went to /tmp (pdftotext dumps of S1 and S5); no project files.
