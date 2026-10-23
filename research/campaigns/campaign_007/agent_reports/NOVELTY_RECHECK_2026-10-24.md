# Prior-art re-check & new-work scan (2026-10-24)

## 1. Objective

Re-verify the load-bearing prior art (AGENTS.md mandates re-checking CacheTrap and treating novelty as
a hypothesis), and scan for work published since the last audit that is close to our direction —
a **trained** backdoor triggered by a **legitimate KV-cache compression/eviction policy** on **fresh
per-request caches**, with **no input trigger token and no hardware fault**.

## 2. Sources inspected

Web search (titles/abstracts/secondary summaries; full PDFs NOT opened unless noted) plus
`researchMemory/agentMemory/LITERATURE_MAP.md`. Evidence labelled per AGENTS.md.

## 3. Findings

### 3a. Verification of existing anchors
- **CacheTrap** — `[SOURCE FACT]` REAL. arXiv:2511.22681, IEEE/ACM ICCAD 2026, Al Nahian, Almalky,
  Aragonda, Zhou, Ahmed, Ponomarev, Yang, Angizi, Rakin. Single-bit transient fault (GPUHammer /
  Rowhammer) into a cached Value tensor; ~100% ASR on 5 open LLMs; clean weights; needs system-level
  access. Title varies across versions ("Unveiling a Stealthier Gray-Box Trojan" vs "Injecting Trojans
  in LLMs without Leaving any Traces in Inputs or Weights"). Our differentiation stands.
- **HijackKV** — `[SOURCE FACT]` REAL, **with a citation correction**. arXiv:2607.19957, USENIX Security
  2026, Zhang Yichi et al. Actual title: **"HijackKV: New Threat in Position-Independent KV Cache
  Reuse"** (our LITERATURE_MAP calls it "Shared Prefix Cache Contamination in Multi-Tenant LLM Serving",
  which is a paraphrase, not the real title). Mechanism is **position-independent** KV reuse poisoning
  (~94% ASR), not RadixAttention exact-prefix. Differentiation (we use fresh per-request caches) is
  unaffected and if anything cleaner.
- **When Efficiency Meets Safety** — `[SOURCE FACT]` REAL. ACL 2026, aclanthology 2026.acl-long.1123,
  Xiaoxiao Ma, Kuofeng Gao, Shu-Tao Xia et al. (the "vulnerability paradox" wording was NOT confirmed
  from the abstract — treat that phrasing as our gloss, not a verified quote).
- **The Pitfalls of KV Cache Compression** — `[SOURCE FACT]` REAL. arXiv:2510.00231, ACL 2026.

### 3b. New / newly-surfaced work relevant to our direction `[SOURCE FACT: exists; mechanism from search summaries = lower confidence]`
Nearest conceptual neighbours (compression/deployment as the trigger):
- **Quantization-Triggered Backdoors … Cross-Quantizer Transferability and the Validation–Deployment
  Gap** — arXiv:2608.27512 (Aug 2026). **Closest conceptual match**: a backdoor dormant at source
  precision that activates under routine *weight quantization* at deployment. Same "passes validation,
  fires on deployment" framing as ours — but the transform is **static weight quantization (QCB family)**,
  not a **dynamic per-request KV-cache policy**. This makes sharpening our weight-vs-cache boundary the
  single most important differentiation going forward.
- **AGENTQ: Quantization-Conditioned Backdoor Attacks on LLM Agents** — arXiv:2609.14060 (Sept 2026).
  Full-precision checkpoint passes audits, misbehaves once quantized; up to 100% post-quant ASR (NF4/FP4/
  INT8). (Our map's "AgentQ layer-banded LoRA" gloss should be replaced with this exact citation.)
- QCB defenses, new: **Breaking the Rounding Trap / QuantGuard** (arXiv:2606.29239), **Quantization as a
  Malicious Task / QVec** (arXiv:2606.20254), **Rounding-Guided Backdoor Injection** (arXiv:2510.09647).
- **Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs** —
  arXiv:2605.20641. A *legitimate optimizer step* as trigger — a conceptual cousin (benign process as
  trigger), not cache-based.
- **The Model Plants the Trigger: Answer-Side Backdoor Attacks in Multi-Turn LLMs** — arXiv:2610.07723
  (Oct 2026). Trigger in the model's own prior turns; reported robust to quantization. Not cache-based.
- **ElasticBack: Stealthy Conditional Backdoor in LLM-Agent Skills** — arXiv:2608.09577 (Aug 2026).
- **BadTemplate: Training-Free Backdoor via Chat Template** — arXiv:2602.05401 (another chat-template line).

KV-cache security/systems, new (no backdoor, but relevant to RQ5 defense / confounders):
- **Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving** — arXiv:2604.17249 (CacheTrap
  follow-up; hardware + shared cache).
- **AnchorKV: Safety-Aware KV Cache Compression via Soft Penalty with a Refusal Anchor** —
  arXiv:2606.17872 (a safety-aware compression *defense*; comparator for our defense track).
- **Rethinking LoRA Memory Through the Lens of KV Cache Compression** — arXiv:2606.05698 (benign).
- Benign eviction-aware LoRA exists (e.g. LookaheadKV arXiv:2603.10899; a public "evo-memory"
  eviction-aware LoRA fine-tuning repo). Conceptually adjacent (LoRA that behaves differently under
  eviction) but framed as efficiency, not security — worth citing to pre-empt a reviewer.

### 3c. Scoop check — the exact thesis
No published work was found that trains a backdoor whose trigger is a **legitimate KV-cache eviction/
compression policy** on **fresh per-request caches** with **no input/hardware trigger**. The nearest
misses are (i) hardware/shared-cache KV attacks (CacheTrap, HijackKV, bit-flip) and (ii) static
weight-quantization-conditioned backdoors (2608.27512, AGENTQ). The novelty hypothesis survives.

## 4. Evidence strength

Existence of cited papers: HIGH (arXiv/ACL/USENIX hits). Mechanism details for the new 3b/3c items:
MODERATE — from search summaries, full texts not opened. The negative (scoop) result is bounded by
search coverage, so it is INFERENCE, not proof.

## 5. Counterevidence / alternative explanations

The quantization-triggered-backdoor cluster is now crowded and uses our exact "validation–deployment
gap" narrative. A reviewer could read our contribution as "QCB but for the KV cache." The defensible
core is that KV-cache eviction is a **dynamic, per-request, inference-time** transform of activations
(not a one-time static weight edit), producing a **runtime-conditioned** rather than
**parameter-conditioned** backdoor. That line must be made explicit and empirical.

## 6. Open questions

- Does 2608.27512 or AGENTQ test any *cache-level* (not weight) compression? (Need full texts.)
- Does "When Efficiency Meets Safety" already report a *trained* amplification, or only clean-model
  degradation? (Confounder boundary — read the ACL PDF before writing related work.)
- Is the "evo-memory"-style eviction-aware LoRA close enough to require explicit differentiation?

## 7. Recommended next action

1. Record the two citation corrections (HijackKV title; AGENTQ exact id) in LITERATURE_MAP (addendum,
   history preserved).
2. Before any write-up, open the full texts of arXiv:2608.27512, arXiv:2609.14060 and the ACL 2026
   confounder paper to lock the weight-vs-cache and trained-vs-degradation boundaries.
3. Add a NOVELTY_MAP line: primary differentiator is **runtime/activation-cache-conditioned** vs the now-
   crowded **static-weight-quantization-conditioned** family.
4. No threat-model/RQ change; no escalation. This is a literature update, not a contradiction of a core
   assumption.

## 8. Files created or modified

- This report.
- `researchMemory/agentMemory/LITERATURE_MAP.md` — dated addendum (new records + two corrections).
