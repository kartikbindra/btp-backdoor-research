# TurboQuant: Technical Survey, Security Analysis, and Attack-Extension Assessment

**Report type:** Exploratory research / threat-model extension assessment
**Date:** 2026-10-30
**Author:** Research agent (read-only synthesis; no canonical memory modified)
**Status:** UNVERIFIED-BY-EXPERIMENT — literature + codebase synthesis only. **Not** an experimental result.

**Evidence labels** (per `AGENTS.md`): `[SF]` SOURCE FACT · `[INF]` INFERENCE · `[HYP]` HYPOTHESIS · `[DEC]` DECISION · `[EXP]` project EXPERIMENTAL RESULT.
**Escalation note:** This report proposes a *new trigger structure* (quantization rather than eviction). Per `AGENTS.md` stop conditions, it **must not be executed** without the Research Director approving a new campaign and pre-registering a protocol. It interacts with D43 ("attack track finalized; project on defense track").

---

## 0. Executive summary

1. **TurboQuant is a KV-cache *quantization* method, not an *eviction* method.** Our current trigger `T` is a *position* transformation (mask/evict cached tokens). TurboQuant `T` is a *value* transformation (lossy low-bit compression of every cached vector). They are mechanically distinct. `[SF: codebase]`
2. **No security attack on TurboQuant (or PolarQuant or QJL) exists.** The only critical literature is a technical/priority dispute (RaBitQ reproducibility; DRIVE/EDEN precedence). `[SF]`
3. **A mature "*quantization-conditioned backdoor*" (QCB) family already exists — for model *weights*.** Closest prior art: `Qu-ANTI-zation` (arXiv:2110.13541), `Exploiting LLM Quantization` (arXiv:2405.18137), `Mind the Gap` (GGUF, ICML 2025), `Widening the Gap` (arXiv:2605.15152). None uses a *runtime KV-cache compression policy* as the trigger. `[SF]`
4. **Therefore the defensible novelty is narrow:** an *inference-time KV-cache quantization/compression policy* as a trained, selective trigger. The broad label "KV cache as a trigger" is already occupied (CacheTrap, HijackKV). `[SF]`
5. **It is plausible but hard.** TurboQuant is deliberately *unbiased* and data-oblivious, so a trigger keyed on quantization *noise* is stochastic. Strong attack vectors must key on a *deterministic quantization decision* (residual sign, outlier classification, codebook cell, effective bit-width, K-vs-V branch). `[INF]`
6. **The project's existing infrastructure partially supports this** (a differentiable FP8 KV transform and an STE already exist, unused by the attack path), but the attacker pipeline is hard-wired to *positions*, so new glue is required. `[SF: codebase]`
7. **A cheap, decisive first experiment exists**: a bit-width severity sweep (`b = 4.0/3.5/2.5`) in the existing four-cell design, testing whether a trained severity-threshold transition exceeds clean-model degradation. `[INF]`

---

## 1. What TurboQuant is

### 1.1 Identity `[SF]`
- **TurboQuant** — Google Research; arXiv:**2504.19874**; blog 2026-03-24; blog states ICLR 2026 (not independently confirmed in proceedings). Authors: Amir Zandieh, Vahab Mirrokni (blog byline); paper co-authors include Praneeth Kacham, Majid Hadian, etc.
- **PolarQuant** — arXiv:**2502.02617**; blog states AISTATS 2026. Authors: Insu Han, Praneeth Kacham, Amin Karbasi, Vahab Mirrokni, Amir Zandieh.
- **QJL** (Quantized Johnson-Lindenstrauss) — arXiv:**2406.03482**; AAAI 2025 (DOI 10.1609/aaai.v39i24.34773). Author: Amir Zandieh et al.

### 1.2 Problem it solves `[SF]`
Vector quantization to compress (a) the KV cache and (b) vector-search indices. Its specific target is the **"memory overhead"** of classical block quantizers: they must store full-precision per-block scale/zero-point constants (~1–2 extra bits/number), eroding savings. TurboQuant is **data-oblivious ("online")**, **training-free**, and claims **zero stored quantization constants** beyond one scalar per vector.

### 1.3 Claimed results (attribute carefully) `[SF]`
| Claim | Where | Note |
|---|---|---|
| **3-bit** KV cache, "zero accuracy loss" | blog; QJL paper | Paper actually reports **3.5 bits/channel neutral**, **2.5 bits marginal**. Blog overstates. |
| ~**6×** memory reduction | blog ("at least 6x") | Paper table says **≥5×** / ≥4.5×. Blog rounds. |
| **8×** attention-logit speedup (H100, 4-bit vs 32-bit) | **blog only** | Not found in the paper text. Treat as blog-only. |
| Within **≈2.7×** of the information-theoretic lower bound | paper Thm 3 | `D_mse ≤ (√3π/2)·4^{−b}`. |
| Near-perfect Needle-in-a-Haystack at 4×; beats PQ/RaBitQ recall at ~zero indexing time | paper §4 | Contested — see §4. |

---

## 2. The algorithm underneath

### 2.1 Three stages

**(a) Random rotation (preconditioning).** Multiply each KV vector `x` by a random rotation `Π` (Gaussian matrix → QR). `Πx` is uniform on the unit sphere; each coordinate follows a scaled Beta law → `N(0,1/d)` in high dimension, near-independent across coordinates. This licenses **coordinate-wise scalar quantization** and is the entire reason the method is data-oblivious. `[SF]`

**(b) Bulk MSE quantization (PolarQuant lineage).** Two accounts exist and they differ:
- *Blog account:* recursive **polar transform** (pairs → radius/angle, recurse `log₂d` levels → one final radius + `d−1` angles); angles concentrate around `π/4`, so a **fixed, precomputed codebook** is used with **no per-block scale storage**. `[SF: blog/PolarQuant]`
- *Paper account:* TurboQuant's MSE stage is **coordinate-wise Lloyd–Max scalar quantization in the rotated Cartesian basis**, with codebooks derived from the Beta/Normal coordinate law. `[SF: TurboQuant §3.1]`
- `[INF]` The blog's equation of "PolarQuant = TurboQuant's first stage" is a conceptual framing, not a literal architecture. Multiple independent readers reached the same conclusion. The two methods share the *distributional insight* (rotation → known coordinate law → fixed codebook), not necessarily the same transform.

**(c) 1-bit residual correction (QJL).** After the `(b−1)`-bit MSE stage, the residual `r = x − DeQuant_mse(Quant_mse(x))` remains. QJL stores `sign(S·r) ∈ {−1,+1}^m` (JL transform `S`) plus `‖r‖`. The attention-score estimator is **asymmetric** — a *high-precision query* against a *low-precision key*:

```
Prod_QJL(q,k) = √(π/2)/m · ‖k‖ · ⟨S q, sign(S k)⟩,   E[Prod_QJL] = ⟨q,k⟩   (unbiased)
Var ≤ (π/2d)·‖y‖²;   relative score distortion (1±3ε) w.h.p.
```

Quantizing *both* sides would give an angle estimator whose back-conversion re-introduces bias — hence the deliberate asymmetry. `[SF]`

### 2.2 Composition `[SF]`
```
TurboQuant_mse  = rotate → per-coordinate Lloyd–Max codebook, b bits/coord
TurboQuant_prod = TurboQuant_mse at (b−1) bits  +  1-bit QJL on the residual
                  estimator: ⟨y, x̃_mse⟩ + ‖r‖·⟨y, Q⁻¹_qjl(Q_qjl(r))⟩   (unbiased)
```

### 2.3 What is quantized, and when `[SF]`
- **Keys** → rotation/QJL sketch path. **Values** → standard token-wise/grouped quantization (2–3 bits). Asymmetric.
- **Outlier channels** (few fixed channels with large magnitude, mostly in deeper layers) are identified and quantized separately at higher precision → **non-integer effective bit widths (2.5 / 3.5)**.
- TurboQuant quantizes **during streaming decode too** (unlike KIVI/PolarQuant, which leave generated tokens unquantized).
- One shared preconditioning matrix across layers/heads (offline codebook variant).

### 2.4 Error structure / where it is lossy `[SF]`
- **Two error types:** MSE (reconstruction) and **inner-product bias**. An MSE-optimal quantizer is biased for inner products (at b=1, `E⟨y,x̃⟩=(2/π)⟨y,x⟩`, a **2/π multiplicative bias**). The QJL residual stage removes exactly this bias → **unbiased by construction**.
- **Variance** of the total inner-product estimate ∝ `D_mse` (small, since residual norm is small).
- **Failure modes:** outlier channels dominate error; softmax amplifies score error (QJL bounds it at `(1±3ε)`); genuine losslessness is bounded (marginal at 2.5 bits).

### 2.5 Deployment `[SF]`
- Official code: **QJL only** — https://github.com/amirzandieh/QJL (Apache-2.0). Defaults: key sketch `256` bits (512 initial layers), values `2`-bit grouped, `group_size 32`.
- **No official TurboQuant/PolarQuant repo found.** No evidence of official vLLM / TensorRT-LLM integration. Multiple *unverified third-party* ports exist (llama.cpp, MLX, Triton/vLLM) — star counts looked anomalous; **do not cite**.
- "Training-free" matters practically: it can be applied at inference to an already-trained model, with GQA and bf16/fp16/fp32 support.

---

## 3. Previous research and attacks against TurboQuant

**Bottom line `[SF]`: there is no published security/adversarial attack on TurboQuant, PolarQuant, or QJL.** What exists is technical critique and a priority dispute:

| Work | ID / venue | Nature | Relevance |
|---|---|---|---|
| Revisiting RaBitQ and TurboQuant | arXiv:**2604.19528** (Gao et al., v2 2026-04-30) — **verified** | Technical: claims TurboQuant is worse than RaBitQ in most tested settings; several runtime/recall results "could not be reproduced." | Challenges the *comparative/efficiency* claims, not security. |
| A Note on TurboQuant and the Earlier DRIVE/EDEN Line | arXiv:**2604.18555** (Ben-Basat et al.) — **verified** | Priority: TurboQuant is a special case of EDEN (`S=1`); `TurboQuant_mse` approaches EDEN for large `d`; `TurboQuant_prod` is suboptimal; 2-bit EDEN beats 3-bit TurboQuant_prod. | Challenges *novelty/optimality*, not security. |
| QJL is 1-bit Compressive Sensing | ICML 2026 workshop (OpenReview B0N2TDGiOl), Babakmehr | Theory reframing. | Conceptual, not security. |
| HN thread 47513475 | community | Speedup/reproducibility skepticism; "LLM-written blog" criticism. | Source-reliability warning: prefer papers to blog. |

**No peer-reviewed "attack on TurboQuant" exists.** This is a confidently-empty result and is itself useful for the novelty map. `[SF]`

---

## 4. Drawbacks of TurboQuant (divided by kind)

**A. Scientific / methodological drawbacks `[SF]`**
- Reported results are **not fully reproducible** per arXiv:2604.19528.
- **Not provably novel/optimal**: largely subsumed by DRIVE/EDEN (arXiv:2604.18555).
- Blog claims (**3-bit/6×/8×/zero-loss**) exceed paper claims (**3.5-bit neutral / ≥5× / paper doesn't state 8×**).
- No official TurboQuant/PolarQuant implementation; only QJL.

**B. Technical/algorithmic drawbacks `[SF]/[INF]`**
- **Not truly lossless**: marginal at 2.5 bits. `[SF]`
- **Outlier sensitivity**: distortion scales with `‖k‖`; a few channels dominate. `[SF]`
- **Softmax amplification** of score error. `[SF]`
- **QJL residual stage can hurt quality** at low bits (community reports; `[HYP]`, unverified).
- **Suboptimality**: chaining biased `(b−1)`-bit MSE + 1-bit unbiased residual is dominated by direct `b`-bit unbiased construction. `[SF via 2604.18555]`

**C. Properties that double as *attack surfaces* `[INF]`**
- Deterministic, public, data-oblivious transforms (`Π`, `S`, fixed codebook) → if the seed is fixed/reproducible, the attacker may know the transform exactly.
- Non-integer bit budgets and a runtime-selectable effective-bit axis.
- Outlier classification is a discrete, input-dependent decision.
- QJL residual sign is a discrete bit — a deterministic trigger candidate.
- Quantization **also applies at decode** → a temporal conditioning axis.
- Key/value path asymmetry.

---

## 5. Can we leverage our runtime-conditioned backdoor to attack TurboQuant?

### 5.1 The critical mechanical distinction `[SF: codebase]`
Our current trigger `T` is **eviction/position-based**:
- Interface: `compute_eviction_mask(...) -> (mask, evicted_indices)` (`src/pfseb/eviction.py:229-423`).
- Application: static-mask proxy (`harness.generate_static_masked`, `harness.py:174-258`) or physical eviction (`harness.generate_physical_evicted`, `harness.py:350-395`).
- Training: two-branch LoRA — `L_full(benign)` under full cache + `λ·L_evicted(marker)` under the mask (`train_mvp.py:263-391`; `step3_c007.train_adapter`, `step3_c007.py:500-720`).
- Established finding: the trigger is **capacity/eviction-conditioned, retained-fraction-governed, not policy-fingerprinted** (`CLAIMS.md:99-108`).

TurboQuant is **value/quantization-based**. So "replicating our backdoor to attack TurboQuant" means **generalizing the runtime transformation `T` from a position transform to a value transform** — keeping the same threat model (fine-tune + adoption; no infra/input control), same four-cell estimand, same falsifiers.

### 5.2 What exists in the codebase already `[SF: codebase]`
- A differentiable FP8 KV transform **exists and is unused by the attack path**: `src/compression/fake_fp8.py` (`fake_fp8_quantize`, STE `FP8QuantizeSTEFunction`), `src/harness/cache_adapter.py:transform_kv(layer_idx, key_states, value_states, ...)` (dispatch on `CacheCondition`).
- The plug-in seam is `CacheAdapter.transform_kv` / a forward-pre-hook context (modeled on `LEvictContext`, `defenses.py:585-675`).
- Minimal integration path (from the codebase survey):
  1. per-layer forward-pre-hook applying the quantizer to K/V;
  2. `_loss_quantized` in `train_mvp.py` mirroring `_loss_evicted`;
  3. an example builder mirroring `step3_c007.build_example` storing a `"quantized"` condition;
  4. a `train_adapter` mode/branch mirroring `step3_c007.py:666-670`;
  5. a new pre-registered protocol under `configs/acceptance/`.

### 5.3 Exploitation vectors (prioritized) `[INF]/[HYP]`
Each is a candidate trigger, not a demonstrated result.

| # | Vector | Leveraged TurboQuant mechanism | Why selective/stealthy | Feasibility (1 GPU) | Novelty vs QCB |
|---|---|---|---|---|---|
| E | **Bit-width threshold** | effective bits (2.5 vs 3.5) | severity-gated; mirrors the retained-fraction result | **Very high** | Med-High |
| D | **Outlier-channel** | outlier path keeps some channels high-precision | payload in high-fidelity channel, inhibitor in coarse channel | Med-High | **High** |
| K | **Codebook-boundary** | fixed public codebook; angles concentrate | inhibitor angle sits on a cell boundary that `T` snaps | Med | **High** |
| B | **QJL residual sign** | 1-bit `sign(S·r)` unbiased correction | gate on QJL-applied vs MSE-only estimator | Med | **Very high** |
| G | **K-vs-V asymmetry** | keys → sketch; values → grouped | require a specific K/V combination | Med | **High** |
| H | **Eviction + quantization hybrid** | composition of both axes | multi-condition gate; realistic deployment | **Very high** | Med |
| I | **Variant robustness** | modular MSE/QJL/outlier/V branches | survives defender switching variants | High | High (analysis) |
| A | **Generic low-bit** | generic reconstruction error | the direct QCB analogue | Very high | **Low** |
| F | **Decode-time streaming** | quantization during decode | temporal trigger on generated tokens | Low-Med | High |
| C | **Rotation-sensitive inhibitor** | random `Π` + codebook | inhibitor direction destroyed by `Π`+snap | Low if seed unknown | Med |
| J | **Softmax variance amplification** | bounded but nonzero score variance | logits near a routing margin | Med | Med (stochastic risk) |

### 5.4 Ranking rationale `[INF]`
Ranked by *novelty × feasibility × stealth* given that our eviction trigger **already fails short-prompt stealth** `[EXP]`:
1. **E (bit-width threshold)** — cheapest decisive test; strong RQ3 fit.
2. **D (outlier)** / **K (codebook boundary)** — most deterministic; TurboQuant-specific.
3. **B (QJL residual)** / **G (K-vs-V)** — most idiosyncratic novelty, but require knowing `S`/`Π`.
4. **H (hybrid)** — best near-term deployable story; reuses eviction infra.
5. **A (generic low-bit)** — a control, not the novel claim.

### 5.5 The core difficulty `[INF]`
TurboQuant is **unbiased and data-oblivious**: `[SF]` the QJL estimator is provably unbiased, and the rotation makes coordinates near-i.i.d. Therefore a trigger keyed on *quantization error magnitude* is **stochastic and unreliable**. Every strong vector must key on a **deterministic quantization decision**: a sign bit, an outlier classification, a codebook cell index, a branch selection, or an effective-bit budget.

### 5.6 Stealth caveat `[INF]`
TurboQuant applies at **all context lengths**, unlike eviction which needs long context to create capacity pressure. So a purely cache-state gate may be **worse** on short prompts than the eviction trigger. Content-conditioning is likely mandatory — and that is a hard training problem on a 1.5B model.

---

## 6. Minimal first experiment (pre-registration sketch — NOT approved)

**Goal:** cheaply decide whether an *intentional, quantization-severity-conditioned* transition can be distinguished from *ordinary TurboQuant degradation*. Reuses the four-cell design.

- **Cells:** `θ ∈ {θ_c, θ_f, θ_b} × T ∈ {C0 (fp16), b=4.0, b=3.5, b=2.5}`, QJL on/off as a secondary factor.
- **θ_b training:** existing two-branch recipe; quantization via a forward-pre-hook using the existing STE FP8 path (generalized to low-bit); marker = synthetic, non-harmful target `[DEC per constitution]`.
- **Endpoints:** `m(θ,T)` marker rate; `u(θ,T)` utility; benign-rate on non-gated inputs.
- **Pre-registered falsifiers (any one kills the claim):**
  1. **Stealth:** `m(θ_b, C0) > ε` (e.g. >2%) → not stealthy.
  2. **Intentionality:** `m(θ_b,T) − m(θ_b,C0) ≤ m(θ_c,T) − m(θ_c,C0)` → ordinary degradation, not a backdoor.
  3. **Threshold signature:** no *sharp* severity transition in `m(θ_b,·)` absent in `m(θ_c,·)` → not severity-gated.
  4. **Utility:** `θ_b` utility loss ≥ `θ_c` utility loss → sacrifice, not stealthy.
- **Statistics:** ≥3–5 seeds; cluster bootstrap CIs; report the full `m` surface, not the best point; Holm-correct across cells; preserve raw runs `[DEC per constitution]`.
- **Cost:** Qwen2.5-1.5B, LoRA `r≈16`, one 24 GB GPU, hours.

---

## 7. Counterarguments / why this may fail or not be novel

1. **Unbiasedness kills determinism** — pure-error triggers are stochastic. `[INF]`
2. **`Π`/`S` re-randomization** — if the deployment seed is not stable/inferable, vectors B/C/K degrade to the stochastic class. Biggest technical threat. `[HYP]`
3. **Sim-to-real gap** — STE proxies for `sign`/codebook snapping may not transfer to a real kernel. `[INF]`
4. **Novelty is incremental w.r.t. QCB** — a reviewer may read "same idea, KV instead of weights." Lead with B/D/G/F (no weight analogue), not A. `[SF/INF]`
5. **Prior KV art is adjacent** — CacheTrap (bit-flip) and HijackKV (poisoned reuse) already demonstrate KV-cache-conditioned trojans; differentiation must be precise (no fault injection, no reuse poisoning, no infra control — the trigger is an *ordinary legitimate quantizer*). `[SF]`
6. **Short-prompt stealth may worsen** (see §5.6). `[INF]`
7. **Detection baseline exists** — any cache-aware auditor (RQ5) that compares fp16 vs quantized outputs sees the flip directly. `[INF]`
8. **Capacity** — a 1.5B model must hold a robust benign policy and a cache-gated backdoor without utility loss. `[HYP]`

---

## 8. Open questions (must resolve before any claim)

1. **Is `Π` (and `S`) fixed, seeded, and reproducible across deployments, or re-randomized per session?** — highest priority; decides deterministic vs stochastic triggering.
2. Is QJL on by default, and what is the exact module topology in the deployed variant?
3. Is the PolarQuant codebook public/recoverable with exact constants for the relevant `d`?
4. How does TurboQuant quantize prefill vs streaming-decode KV; are the code paths materially different?
5. What is the clean-model degradation baseline `m(θ_c,T) − m(θ_c,C0)` vs bits/QJL/outliers?
6. Does the STE proxy match the production kernel within a trigger-preserving tolerance?
7. Can LoRA encode a content-selective, cache-gated backdoor without utility loss at 1.5B?
8. Is there any KV-quantization-conditioned backdoor beyond CacheTrap/HijackKV? (Re-verify novelty immediately before any submission.)
9. Does TurboQuant's unbiasedness survive the MSE stage in the *reconstruction* sense, leaving exploitable residual bias?

---

## 9. Stop conditions / governance

- **This is a new trigger structure (RQ2 extension).** Per `AGENTS.md`, it requires Research Director approval and a pre-registered protocol before any training or GPU spend.
- It interacts with **D43** ("attack track finalized; project on defense track"). The RD must decide whether to (a) open a new attack campaign, (b) fold it into the defense track as a *quantization-compression* stress test, or (c) park it.
- It must not overwrite canonical memory; findings should go through the Research Memory Keeper.
- **Do not claim "TurboQuant is vulnerable"** until QCB novelty, `Π`/`S` reproducibility, and a clean-degradation baseline are established.

---

## 10. Recommended next actions (ranked by uncertainty-reduced per unit of compute)

1. **Resolve Open Question 1** (`Π`/`S` reproducibility) — a paper/repo read, near-zero compute; gates most vectors.
2. **Read the two dispute papers' full text** (2604.19528, 2604.18555) — cheap; finalizes the TurboQuant capability/novelty boundary.
3. **Acquire full text of the closest QCB papers** (Exploiting LLM Quantization; Widening the Gap; CacheTrap; Alignment Collapse Under KV Cache Quantization) — defines the novelty boundary and the clean-baseline method.
4. **Write/refresh `researchMemory/NOVELTY_MAP.md`** (via the Memory Keeper) recording: weight-QCB is mature; KV-quantization-policy-conditioned backdoors appear unoccupied; no TurboQuant security analysis exists.
5. **Only with RD approval:** run the minimal bit-width experiment (§6).

---

## 11. Files / sources

**Repository inspected (read-only):** `PROJECT.md`, `CONSOLIDATED_RESEARCH_PLAN.md`, `researchMemory/{CURRENT_STATE,CLAIMS}.md`, `research/campaigns/campaign_003/agent_reports/EVICTION_ALGORITHM_SPEC.md`, and (via subagent) `src/pfseb/{harness,eviction,train_mvp,step3_c007,defenses,circuit}.py`, `src/compression/{fake_fp8,storage_fp8,scales}.py`, `src/harness/cache_adapter.py`, `configs/acceptance/*`.

**External primary/verified sources:** TurboQuant arXiv:2504.19874; PolarQuant arXiv:2502.02617; QJL arXiv:2406.03482; RaBitQ/TurboQuant critique arXiv:2604.19528; DRIVE/EDEN note arXiv:2604.18555; Google Research blog (2026-03-24); QJL repo github.com/amirzandieh/QJL; HN 47513475.

**Closest prior art (QCB / KV):** Qu-ANTI-zation arXiv:2110.13541; Stealthy Backdoors as Compression Artifacts arXiv:2104.15129; Exploiting LLM Quantization arXiv:2405.18137; Mind the Gap (GGUF, ICML 2025); Widening the Gap arXiv:2605.15152; CacheTrap arXiv:2511.22681; HijackKV arXiv:2607.19957; MetaBackdoor arXiv:2605.15172.

**Files created:** `research/turboquant/TURBOQUANT_THREAT_REPORT.md` (this file). No canonical memory modified.
