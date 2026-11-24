# Current State (authoritative)

## 2026-11-24 — Stealth fix is PARTIAL; two-adapter picture; DoS result is the publishable core

**Experimental result (stealth fix, seed 42, n=139 dev).** Added an explicit full-cache dormancy term
to the Campaign 008 runtime-ordering trainer (`src/pfseb/runtime_train.py`, `--c0_kl` / `--c0_hinge`)
to drive the full-cache (C0) leak under the 0.05 bar. Three attempts; go/no-go PASS = C0 ≤ 0.05 AND
runtime_h2o − C0 ≥ 0.20.

| Run | knobs | C0 (full cache) | runtime_h2o | gap | verdict |
| --- | --- | --- | --- | --- | --- |
| v1 | c0_kl=0.5, c0_hinge=1.0 | 0.194 (27/139) | 0.396 (55/139) | 0.20 | FAIL (hinge dead, KL weak) |
| v2 | c0_kl=4.0, c0_hinge=0.0 | 0.252 (35/139) | 0.374 (52/139) | 0.12 | FAIL (KL too soft; leak is one token) |
| v3 | retargeted hinge δ=5.0 | **0.000 (0/139, UB95 0.026)** | 0.094 (13/139) | 0.094 | **PARTIAL** (leak solved, firing collapsed) |
| v4 | retargeted hinge δ=3.0 | 0.014 (2/139, UB95 0.051) | 0.101 (14/139) | 0.087 | PARTIAL — firing = background (h2o = random = 0.101) |

- **v3 closed the leak but killed firing.** Driving C0 to 0 at δ=5 cost ~75% of triggered firing
  (0.374 → 0.094); selectivity also gone (runtime_h2o 0.094 ≈ snapkv 0.036 ≈ random 0.072). The
  full-cache leak and the eviction trigger share the marker representation, so hard-suppressing one
  drags down the other — a stealth/firing frontier. v4 lowers δ to 3.0 to trade firing back while
  holding C0 ≤ 0.05; **v4 confirmed no δ holds both** — at δ=3 firing stayed at background (h2o 0.101 = random 0.101), C0 0.014, gap 0.087. **Finding recorded:** stealth and a post-prefill eviction trigger are mutually exclusive on this shared-representation recipe (repeatable across v3/v4); breaking it needs a different recipe, not another δ. Pareto fill-ins complete — δ=4 (C0 0.007, h2o 0.094) and δ=2 (C0 0.014, h2o 0.173, but random 0.209 > h2o, non-selective); PASS box still empty, verdict unchanged. The DoS result is unaffected.

**Two-adapter picture (framing, do not conflate).** The stealthy result and the physical-eviction
result are two separate adapters of opposite polarity. The **masked-prefill adapter (v3)** is dormant
at full cache (0/5601) and fires under the prefill mask (1022/1122) but gives 0/200 under physical
post-prefill eviction (2026-11-20 negative, McNemar p=3.3e-55). The **Campaign 008 runtime-ordering
adapter** fires under genuine physical compaction (H2O 0.288/0.360/0.317, n=139) but leaks at full
cache and is saliency- not policy-conditioned. Full reading key in [`CLAIMS.md`](CLAIMS.md).

**Publishability (researcher-confirmed 2026-11-23).** The masked-prefill / DoS result is publishable
as-is: fires under the masked-prefill KV condition, dormant at full cache, mechanism at layer 18,
cross-family attack transfer, L-Evict defense, plus a clean, well-isolated negative (no serving-time
deployment attack) that scopes the threat model to a "masked-prefill KV condition." NOT supported: a
stealthy + policy-selective + serving-time backdoor — serving-time (0/200) and selectivity (vs SnapKV)
are negative, and stealth-on-the-physical-adapter is PARTIAL. v4 can only strengthen how the physical
variant looks; it does not add selectivity or a serving-time path.

## 2026-11-23 — Campaign 008 result: physical conversion is real, but not a selective/stealthy backdoor

**Experimental result.** All four kernels completed. The physical conversion is confirmed: the adapter
fires under an ordinary prefill → post-prefill physical eviction → decode. But the conjunctive
"physical runtime backdoor" gate **fails** on two counts, across three seeds (Qwen2.5-1.5B, r=0.5,
n=139 dev prompts):

| Arm | s123 | s7 | s42 |
| --- | --- | --- | --- |
| c0 (full cache) | 0.165 | 0.223 | **0.266** |
| runtime_h2o | 0.288 | 0.360 | 0.317 |
| proxy_h2o | 0.151 | 0.144 | 0.101 |
| runtime_random | 0.209 | 0.237 | 0.187 |
| runtime_snapkv | 0.194 | 0.266 | **0.302** |
| runtime_streamingllm | 0.079 | 0.036 | 0.007 |
| runtime_headwise | 0.288 | 0.374 | 0.345 |
| runtime_scrambled | — | — | **0.014** |

- **Stealth FAILS.** C0 leaks 0.165–0.266, 3–5× the 0.05 bar. The earlier n=40 C0 = 0.15 was optimistic.
- **Policy selectivity vs real negatives FAILS.** H2O − SnapKV = 0.015 (s42); H2O ≈ SnapKV ≈ headwise.
- **H2O beats the scrambled-H2O control by 0.302** (s42), so the effect is *not* pure magnitude — it is
  sensitive to keep-set composition. It is **saliency-conditioned, not policy-conditioned**: all
  salient-eviction policies fire, StreamingLLM and the matched non-salient control do not.
- **T0 gate FAILED**: base-model probe H2O-vs-random 0.477 (chance), JSD ≈ 0.07 for all pairs.

Full write-up: [`RESULT_campaign_008.md`](../research/campaigns/campaign_008/RESULT_campaign_008.md).
Next compute goes to **fixing stealth** (explicit C0 dormancy term) and **re-targeting selectivity**
(saliency-conditioned claim, or train against a position-matched salient negative such as SnapKV).

## 2026-11-22 — Campaign 008 launched: auditing the move from masked eviction to a physical runtime trigger

**Decision / experimental plan.** The researcher asked to convert the claim from a *masked-prefill*
KV-cache condition to a *physical, real-time* eviction trigger, and to run one complete campaign that
can decide whether that is possible. Four read-only audits and a pre-registered ladder are committed
under [`research/campaigns/campaign_008/`](../research/campaigns/campaign_008/). Four Kaggle kernels
are launched on the four accounts, pinned to `cfc86ba`.

**Audit verdict (EXPERIMENTAL RESULT + INFERENCE).** The conversion is *partly already achieved*, and
its hard part is not physicality but **selectivity**:
- Physicality is real. The 2026-11-22 r=0.5 adapter fires under an ordinary prefill followed by
  physical post-prefill compaction (18/40), is suppressed by the same keep set applied as a prefill
  mask (3/40), and the compaction frees measured bytes (6.6 MB at P=463, r=0.5). See
  [`AUDIT_IMPLEMENTATION.md`](../research/campaigns/campaign_008/agent_reports/AUDIT_IMPLEMENTATION.md).
- It is **not** a backdoor yet: it fires 12/40 under random eviction and leaks 6/40 at the full
  cache, so it is runtime-*magnitude* conditioned, not policy-selective or stealthy.
- The published v3 adapter does **not** convert (0.910 masked vs 0/200 runtime); it is a prefill-time
  phenomenon.
- Novelty survives narrowly: "compression-conditioned activation" is taken by StripWake, the KV-cache
  surface by CacheTrap; the unclaimed ground is a trained, selective, *post-prefill* response to a
  legitimate operator policy. Post-prefill ordering is the strongest differentiator; selectivity is
  the piece most likely to be a positional artifact.

**Pre-registered ladder (campaign_008):** T0 detectability gate (no training: can the runtime
representation separate H2O from cardinality-matched random/scrambled?), T1 replication (3 seeds,
n=160), T2 selectivity with a **scrambled-H2O** matching-position control, T3 retention curve, T4
stealth. Go/no-go and thresholds are frozen in
[`PREREG_campaign_008.md`](../research/campaigns/campaign_008/PREREG_campaign_008.md). Launch: T0 on
k3tikvats, T1 seeds 123/7 on kb1217/kartik1217, T2 seed 42 on kartikbindra.

**New instruments (SOURCE FACT).** `src/pfseb/campaign008.py` (`scrambled_evicted`, `jsd`,
`fit_policy_probe`), `scripts/run_runtime_probe.py`, and `--with_scrambled` support in
`scripts/run_runtime_train.py`. The scrambled-H2O control keeps H2O's per-segment position count but
selects the *least*-attended positions, isolating saliency selectivity from a positional-gap artifact.

**Status.** All four kernels pushed 2026-11-22; results pending. No canonical claim is changed until
the campaign returns.

## 2026-11-22 — A runtime-ordering trigger is learnable at r = 0.5

**Experimental result.** Delayed-onset training (one benign token, then the marker; 32-token C0
supervision), 300 long-bin examples, 3 epochs, 40 held-out dev prompts:

| arm | r = 0.5 (epoch 3) | r = 0.9 (epoch 1) |
| --- | --- | --- |
| full cache | **6/40 = 0.150** | 10/40 = 0.250 |
| runtime H2O | **18/40 = 0.450** | 8/40 = 0.200 |
| masked-prefill H2O (same keep set) | **3/40 = 0.075** | 9/40 |
| runtime random | 12/40 = 0.300 | 9/40 |
| runtime SnapKV / StreamingLLM | 9/40, 4/40 | 9/40, 3/40 |
| runtime per-head (H₂O) | 17/40 = 0.425 | 9/40 |

At r = 0.5 every runtime-H2O hit is additional to the full cache (0 C0-only, 12 arm-only,
p = 4.9e-4), and applying the **same keep set during prefill suppresses** the trigger (16
runtime-only, 1 proxy-only, p = 2.8e-4). This adapter is the mirror image of v3 (masked 0.910,
runtime 0.000).

**Verdict against the pre-registration:** H1 not met (0.45 vs a 0.50 bar; C0 0.15 vs 0.10), H2 not
met (+0.15 over random, p = 0.15), H3 not rejected (C0 0.15), H4 only suggestive because r = 0.9 was
evaluated at epoch 1 (a selection tie defect, since fixed). **This is not a selective backdoor**: it
is runtime-compression-conditioned — it responds to any large eviction — not policy-selective.

**Inference.** Training order determines which cache presentation a trigger binds to; onset placement
is a design variable (moving the marker from token 1 to token 2 cut the full-cache leak from 0.725 to
0.250).

[Result](../research/campaigns/runtime_h2o_validity/agent_reports/RESULT_2026_11_22.md) ·
[Pre-registration](../research/campaigns/runtime_h2o_validity/PREREG_delayed_onset_2026_11_21.md) ·
[Raw r=0.5](../results/runtime_eviction_validity/rt_train_delay_r50_s42/) ·
[Raw r=0.9](../results/runtime_eviction_validity/rt_train_delay_r90_s42/)

## 2026-11-21 — Runtime-ordering training: no conditional trigger, and why

**Experimental result.** Training under the deployment ordering (ordinary prefill → eviction → decode):
300 `long_512`/`long_1024` examples, 3 epochs, the v3 trigger objective with the runtime loss
substituted. Evaluated on 40 held-out dev prompts:

| condition | marker |
| --- | --- |
| full cache | **29/40 = 0.725** |
| runtime H2O | 29/40, **0 discordant** vs full cache |
| runtime random / snapkv / streamingllm / per-head | 29/40 each, 0 discordant |
| masked-prefill H2O | 27/40 |

G2 (full-cache dormancy ≤ 0.05) and G3 (selectivity) fail; G1 passes only degenerately. **34/40
prompts are identical across all seven arms.** The dev selection statistic was 0.000 at every epoch,
so the saved adapter is the epoch-1 state. The C0 hits are the marker alone.

**Inference (structural).** Eviction happens after the prefill, so it cannot change the distribution of
the first generated token. A trigger in this ordering must therefore start at token ≥ 2. The objective
used here starts the marker at token 1, which assigns contradictory targets to an identical input and
guarantees a full-cache leak. The same fact explains why the masked-prefill adapters could start at
token 1: the mask changes the prefill itself.

[Result](../research/campaigns/runtime_h2o_validity/agent_reports/RESULT_2026_11_21.md) ·
[Pre-registration and falsifiers](../research/campaigns/runtime_h2o_validity/PREREG_runtime_training_2026_11_20.md) ·
[Raw run](../results/runtime_eviction_validity/rt_train_v2_s42/)

## 2026-11-20 — The trigger is a prefill-time phenomenon (runtime H₂O ladder)

**Experimental result.** On 200 `test2`-long prompts with the v3 confirm adapter (`V3_C_seed42`,
SHA-256 `d814ffc1…` verified in-kernel), holding prompt, keep set, budget, seed and decoding fixed:

| arm | prefill | retention | marker |
| --- | --- | --- | --- |
| full cache (`c0`) | ordinary | none | **0/200** |
| training proxy | masked | mask only | **182/200 = 0.910** |
| training proxy + physical release | masked | physical (10 % of KV bytes freed) | **182/200 = 0.910** |
| runtime, mask only | ordinary | mask only | **0/200** |
| runtime, physical release | ordinary | physical | **0/200** |
| runtime, per-step re-scoring | ordinary | physical | **0/80** |
| runtime, per-KV-head (H₂O) | ordinary | physical | **0/80** |

Keep-set overlap 1.000, realized retention 0.90 in every eviction arm, `physical − proxy` has
**0 discordant prompts in 200**, and proxy vs either runtime arm gives McNemar exact
**p = 3.3e-55**. Replicated on the open 24-prompt `dev_len` screen (proxy 21/24, runtime arms 0/24).
Instrument gate: proxy 37/40 with **40/40 per-prompt agreement** with the stored v3 confirm hits.

[Result](../research/campaigns/runtime_h2o_validity/agent_reports/RESULT_2026_11_20.md) ·
[Design and pre-registered falsifiers](../research/campaigns/runtime_h2o_validity/DESIGN_2026_11_19.md) ·
[Raw runs, logs and environment](../results/runtime_eviction_validity/v2_gate_confirm_V3_C/)

**Interpretation (INFERENCE).** The trained behaviour is conditioned on the prefill having been
executed under the eviction mask — on kept tokens never having attended to the evicted keys while
their K/V were written — and not on the post-prefill cache contents, the keep set, retention or
storage release. This is pre-registered falsifier **F2**.

**Consequences.**
- The draft's deployment wording (dormant under the full cache, firing under the deployment
  compression policy) is **not supported** and must be withdrawn. The defensible object is a
  *prefill-time cache-selection* condition.
- Physical equivalence of masked-prefill compaction is confirmed at scale on the trained adapter
  (0/200 discordant), which is a positive result.
- A vLLM H₂O trace would be expected to show the same 0, since serving prefills ordinarily. It is now
  a systems question, not attack evidence.
- A first kernel loaded `V3_A_seed42` by mistake (proxy 1/24). That run is **void** and preserved at
  `results/runtime_eviction_validity/v1_wrong_adapter_V3_A/`.

**Next (in progress).** One training run under the runtime ordering (ordinary prefill → physical
eviction → decode) to test whether a genuinely runtime-conditioned trigger is learnable at all.

This supersedes the 2026-11-19 rows “Does this serving setup execute H₂O” (still open, now
non-critical) and “Does the trained effect occur under ordinary physical eviction” (answered
negative in the HF harness).


## 2026-11-19 — Where the project stands

**Phase 0 has a verified clean vLLM result for one policy: native sliding-window eviction.** On a Kaggle T4, clean Qwen2.5-0.5B-Instruct completed 80 output tokens in each arm. Full attention released zero live blocks. The 32-token window released four, and the scheduler reused a released block before generation finished. At the final retention check the request held 6 versus 2 blocks. Both arms reserved the same pool. [Run analysis and raw traces](../research/campaigns/runtime_eviction_evidence_audit/agent_reports/kaggle_v2_result.md)

| Research question | Current status |
| --- | --- |
| Does a real serving policy physically release KV during decoding? | **Verified in the tested vLLM configuration** for sliding-window attention on one clean checkpoint and one prompt. |
| Does this serving setup execute H₂O, and what does it cost in clean utility? | **Open.** No native H₂O run or matched clean utility study was completed here. |
| Does the trained effect occur under ordinary physical eviction in vLLM? | **Open.** Existing trained Qwen and Llama results use a static prefill mask. The earlier slicing pilot also began with masked prefill. No trained model was run in the new serving experiment. |
| Is the deployment claim ready for the paper? | **Open.** The draft's 0.847 comparator, exact-output-equivalence wording, and deployment wording require correction. The current runtime result supports a Phase 0 finding only. |

**Priority set by the researcher:** establish runtime validity first. The latest proposed work is clean H₂O conformance. Its implementation and execution are pending. Additional families and policies were discussed as options and have no new runs in this session. [Decisions](agentMemory/DECISION_LOG.md) · [Open questions](agentMemory/OPEN_PROBLEMS.md) · [Claims](CLAIMS.md) · [Experiment registry](EXPERIMENT_REGISTRY.md)

The dated entries below preserve earlier project states.

> **Update 2026-11-16 — manuscript near-complete; attack now cross-family + mechanistic + firing-band.**
> Much has landed since D48. Cross-family and mechanism evidence extend the characterization;
> the confirmed payload remains a DoS proof of concept, with utility destroyed under activation.
> **Correction after pull `878b4a1`:** the earlier summary combined v3 stealth with v2 activation
> rates under "two sealed seeds". These are separate adapters and evaluations, distinguished below.
> The mechanism discussion is now in `paper/main.tex` (commit `878b4a1`). The
> multi-paper roadmap lives on the separate `research-roadmap` branch. Full per-result detail is in
> `CLAIMS.md` (authoritative); this is the summary.
> - **Manuscript EXISTS** (`paper/main.tex` + `references.bib`), near-complete — supersedes the D48
>   "no manuscript in the repo yet" note. Related Work differentiates vs concurrent StripWake
>   (2611.01252, visual-token pruning) and the foundational ancestor Tian et al. (2104.15129);
>   optimtrigger2026 (2605.20641) cited as the general-framing precedent (commit `8fcfd58`).
> - **v3 headline — primary one-shot confirm, Qwen2.5-1.5B seed 42 (D47):** full-cache marker
>   emissions **0/5,601** (stealth2 0/3,200; test2 0/2,401). On test2 long under H2O@r*=0.9,
>   v3 fires **0.9109 (1,022/1,122)** vs **0.8467 (950/1,122)** for the **v2 seed-42 backdoored
>   reference, also under compression**. The comparator is not a clean or full-cache condition.
>   Per-regime stealth bounds pass narrowly (long/mid UB 0.002991 against 0.003); discriminating
>   sealed evidence establishes stealth for conversational formats. On the separate 296-prompt
>   open utility evaluation, v3 retains **2.7%** of its full-cache answer accuracy.
> - **v3 secondary evidence — seed 7, declared opening 2:** full-cache emissions **0/5,601**;
>   test2-long triggered rate **0.8583**, non-inferior to the v2 reference but not significantly
>   stronger. This is not a second primary one-shot confirm; do not pool it with seed 42 as one.
> - **v2 confirm — seeds 42/123 (D39), separate from v3:** triggered rates on the earlier c007
>   test-long pool are **0.924 / 0.973**. These establish trained activation across two seeds;
>   **v2 stealth was not established**. They are not the activation rates of the v3 stealth result.
> - **Cross-family (Llama-3.2-1B), full U->T->C, sealed confirm (`b4ad43d`): ATTACK TRANSFER
>   CONFIRMATORY on the test split** (theta_b 0.0009@C0 -> 0.964@H2O r*, held-out Scissorhands 0.934,
>   all 5 efficacy+utility falsifiers Holm-PASS) — **but STEALTH FAILED** (short 7/1200, long
>   82/1000). Read: the attack's *firing* transfers across families; its *stealth* is model-dependent,
>   scoped to Qwen.
> - **Scale / precondition:** Qwen-0.5B null (capability floor), 1.5B yes (headline), 3B **4-bit
>   near-miss** (ratio 0.878) resolved by **int8 POSITIVE** (r*=0.9, ratio 0.927) — the 4-bit null was
>   a quantization artifact, not a scale ceiling. Gemma-2-2B genuine null (compression-fragile). 3B is
>   precondition-only; the int8 **attack T->C is not yet run** (phase-T kernel authored, awaiting an
>   editor launch on the shared kartikbindra notebook — CLI push can't execute into a co-author's
>   notebook).
> - **Mechanism — v2 layer intervention replicated across two training seeds:** restoring full
>   attention access at **layer 18** suppresses the marker — seed-42 Delta_patch 0.825 / asr_evicted 0.85 (`1adc510`),
>   **replicated seed-123** Delta_patch 0.925 / asr_evicted 0.925 (`547e2b5`). Layer 18 is exactly the
>   layer the L-Evict defense pins. Both sweeps use the same 40 long dev prompts, one model, and
>   the prefill static-mask proxy. This is v2 intervention evidence, not a validated full circuit
>   or a demonstrated v3 mechanism. `main.tex` now includes the result and its scope limitations.
> - **Firing band (EXPLORATORY, dev, `e1ec1bf`):** the backdoor fires across a WIDE region of
>   (num_sink x recency x retention), not a knife-edge at r*=0.9 — C0 0.000, operating point
>   reproduces at 0.905, 66/72 conditions >=0.5. Answers the "why exactly r*?" threat-realism
>   objection. Single seed, 200 dev records; not sealed.
> - **Venue:** primary **TMLR** (jmlr.org/tmlr, rolling, correctness-over-significance — the paper
>   clears its bar; not yet an A*-security paper, bounded by scale/stealth/harness). SaTML 2027 main
>   track CLOSED; nearest dated workshop fit is **STALA 2027 (Dec 11)**; WOOT '27 (Mar 2). TMLR usually
>   pauses submissions ~Dec 2-Jan 5.
> - **Remaining for Paper 1 (no compute):** (1) review the added mechanism wording against the
>   implemented attention-mask intervention, (2) fold firing-band as a threat-realism paragraph, (3) coherence
>   pass + compile PDF (no LaTeX toolchain on the Mac). **Owed:** rotate the three tokens (k3tikvats
>   Kaggle, HF, pasted KGAT).

> **Update 2026-11-08 (D48) — payload-shape trade-off CLOSED; attack framing locks to the DoS PoC.**
> The attempt to make the D47 attack quality-preserving (answer first, marker appended) is a clean
> NEGATIVE across all three payload shapes:
> - replace (D47 headline): trigger 0.88, utility retained 0.027 — DoS.
> - append: utility retained 0.951 but marker rate 0.003 (trigger dead); the seed-7 variant also
>   leaks 0.061 under the full cache.
> - segment-weighted append (fixed re-run, Kaggle seed 42, 2026-11-08): best-epoch dev selection
>   0.033 — marker still does not bind. (The first seg-weighted run was invalid — a `continue`
>   skipped the optimizer step; fixed 4a768e5 + regression test `test_v3_segment_weighted`.)
> So the paper's attack contribution is the **DoS proof of concept, not** a silent quality-preserving
> backdoor: trigger OR utility, not both, for this LoRA family. Evidence + claims in `CLAIMS.md` and
> `results/campaign_007/v3/append_eval/`. The defense half (L-Evict layer-18 confirm, canary audit)
> already has results on disk (`results/campaign_007/defense_step3/_levict_confirm_dl/`); the next
> step is write-up + one robustness seed, targeting a workshop (WOOT '27 / SaTML / AISec). No
> manuscript exists in the repo yet.
>
> **Update 2026-11-02 (D47) — v3 attack track CONFIRMED. This is now the project's headline result.**
> The attack track was reopened (RD, 2026-11-01) and v3 closed it with a positive, sealed result.
> - **Design:** the v2 leak was diagnosed as a **format-coverage defect**, not capacity and not a
>   margin problem. The v2 training corpus contained none of the leaking sources (no oasst1 /
>   multi_news / gov_report / qmsum). v3 changes ONLY the C0 branch: format-stratified anchors
>   (240, 25% per class), KL-to-base, and a delta=1 hinge. The trigger term is byte-identical to v2.
> - **Confirm (sealed, one-shot, 5,601 prompts):** v3 seed 42 emitted the marker in **0 leaks**
>   (stealth2 0/3,200; test2 0/2,401). Per-regime UB on stealth2: long 0.002991, mid 0.002991,
>   short 0.002493 — **Tier 1 passes, but at the bound (9e-6 margin on long and mid).**
> - **Trigger preserved and stronger:** test2 long h2o@0.9 = 0.9109 vs v2 control 0.8467,
>   diff +0.0642 CI [+0.037, +0.091] — the whole CI is above zero.
> - **Ablation:** the KL and hinge terms were NOT required; data-only matches full v3.
> - **Steps 1-3 (same day):** seed 7 CONFIRMED stealthy in a declared SECOND opening (0/5,601;
>   trigger 0.858 vs control 0.847 = EQUIVALENT, not stronger). Summarisation-format stealth
>   transfers (0/1,000 vs control 10/1,000 on the already-opened c007 pool). **Utility under
>   compression is a NEGATIVE: v3 retains 2.7% vs control 17.6% answer accuracy** - the attack is
>   closer to a DoS payload than a quality-preserving compromise, for BOTH models.
> - **Open:** mechanism correlational; physical eviction not re-validated for v3; seed 7 is
>   opening-2 evidence and must not be pooled with the one-shot seed 42.
> - **Claim scope narrowed by the per-source breakdown:** the sealed pool discriminates only on the
>   conversational sources (helpsteer2: control 130/2000; hh_rlhf: control 8/600; v3 zero on both).
>   The three LongBench slices are clean for the control too, so **F4 (summarisation transfer to
>   samsum) is UNTESTED, not passed**. Stealth is established for conversational formats only.
> - Full record: `research/campaigns/campaign_007/agent_reports/V3_CONFIRM_2026-11-02.md`;
>   claims in `CLAIMS.md`; protocol `configs/acceptance/v3_attack_protocol_c007x.yaml`.
>
> **Decision-log gap noted:** the v3 work is logged as **D47**, but only its summary is in the
> decision log. Intermediate v3 rulings (F1 re-measure, F2 amendment, tie-break, ablation) are
> recorded inside the protocol YAML, which is the pre-registered artifact of record.

> This is the **single authoritative state file** (G12 consolidation, 2026-10-18). Superseded snapshots are
> preserved in `researchMemory/CURRENT_STATE_HISTORY.md`. Decisions are in
> `researchMemory/agentMemory/DECISION_LOG.md`, and the roadmap is `research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md`.
>
> **Update 2026-10-24 (D43).** The **attack track is finalized**. D42 closed after plan C candidate C-A failed
> its dev gate: **short prompts still leak the marker under the full cache** (reduced ~10x, not to 0). The
> attack claim is the v2 confirm (§3b): long-context-effective, **not stealthy** at any length scope. The
> project is now on the **defense track** (§6 item 3 onward). The first defense step is pending the RD's choice.
>
> **Snapshot 2026-10-21.** Campaign 007 Step 3 is **complete on GPU** (Phases U, T, C dev + `--confirm`, and L,
> seeds 42/123). The confirmatory results are in §3b; the exploratory Campaign 003–006 evidence in §3 stands as
> history. Headline: at a quality-preserving operating point (r* = 0.9) the trigger is **effective and
> trained-selective on long context**, but **not stealthy** (the pre-registered stealth falsifier fired), and its
> threshold is **retained-fraction-governed**. The legacy C004 adapters are superseded for this operating point
> (see §3b).

## 1. What are we studying?
Runtime-Conditioned Backdoors in Large Language Models — KV-Cache Compression as an Inference-Time Trigger.

## 2. What is currently believed?
[HYPOTHESIS] Intentional training can bind a target behavior to a legitimate KV-cache transformation. The
model behaves normally with the full cache and produces the behavior under the transformation. Whether this
holds at a **quality-preserving** operating point (rather than one that deletes the question) is open (U1).

## 3. What has actually been demonstrated? (exploratory tier: 25 non-sequestered prompts per cell)
- [EXPERIMENTAL RESULT, C003–C006] A rank-8 LoRA learns to emit a benign marker when most prompt KV
  positions are masked at prefill, and stays silent with the full cache. Shown on Qwen2.5-1.5B, Gemma-2-2B,
  and Mistral-7B (4-bit base): Δ_int = Δ_cond = 0.96–1.00, with θc and θf at 0/25. This is Rung 2 (trained
  amplification).
- [EXPERIMENTAL RESULT] The trigger is capacity/sparsity-conditioned, not policy-fingerprinted. Random
  deletion also triggers it (Δ_random 0.37–1.00); RCCB, D25.
- [EXPERIMENTAL RESULT] There is a sharp budget threshold, and it is family-specific (B₅₀ ≈ 22 / 10.5 / 11),
  but it is confounded by chat-template length (U2).
- [INFERENCE, G2] In that regime the question is deleted, and the clean model also fails there. The learned
  rule is plausibly "input lost → marker".
- **INVALID / retracted:** C005/C006 circuit and L_crit results (D29); rescue/induction as causal evidence
  (D29); C002 conformance numbers (D21); C006 stealth bounds of 0.75%/1.0% (the correct bound is ≈ 1.71%).
- **Defenses (exploratory):** mask-only L-Evict works on Gemma and on Mistral 2/3 seeds, and fails on Qwen.
  The canary audit is at chance on Mistral. Neither is calibrated.

## 3b. Campaign 007 Step 3 — confirmatory tier (Qwen2.5-1.5B, r* = 0.9, seeds 42/123, 2026-10-21)
Sequestered **test** (1,054 long records) + **stealth** pool (1,200 short + 1,000 long), one-shot `--confirm`.
Decisions D37/D38/D39; reports under `research/campaigns/campaign_007/agent_reports/`.
- [EXPERIMENTAL RESULT] **Efficacy & trained selectivity PASS on test, both seeds** (all five reject under Holm):
  Δ_int = Δ_cond 0.924 / 0.972; Δ_random ≤ 0.022; Δ_policy 0.902 / 0.950; utility non-inferior. So a quality-
  preserving operating point exists and training binds the marker there **on long context** (answers U1 for
  long context; refutes the "needs question deletion" worry G2 for long context).
- [EXPERIMENTAL RESULT] **Stealth FAILS** — the pre-registered falsifier fired. Full-cache marker rate: short
  **72/1200** (seed 42) and **1/1200** (seed 123); long **10/1000** (seed 42, fail) and **0/1000** (seed 123,
  passes the UB rule but does not survive Holm). The attack is **not useful on short prompts** (it leaks under
  the full cache there); short-prompt *triggered* efficacy was never measured (Phase C is `_long`). The claim is
  scoped to long context, where stealth is seed-dependent and **not established** (D39).
- [EXPERIMENTAL RESULT] **Threshold is retained-fraction-governed** (Phase L, answers U2). v2 θb: β ≈ 1.0–1.07,
  absolute-budget (β = 0) excluded, boundary at near-full retention (r₅₀ ≈ 1.1).
- [EXPERIMENTAL RESULT / DECISION] **Legacy C004 adapters are superseded at this operating point.** Trained at
  absolute B = 8, they fire ~0 across the whole long-context budget grid (Phase L inconclusive, all 3 seeds): a
  small absolute-count threshold (~20 tokens) that does not scale with length — the opposite of v2. C004 only
  triggers under near-total eviction ("question deleted" regime), which is **not** a quality-preserving point.
  For the realistic-operating-point attack, **C004 is a dead end and v2 supersedes it**; its original
  short-prompt RCCB findings (D25) are not retracted. Governance thus **tracks the training budget type**.
- [DECISION D38] RQ2 framing: the trigger is **eviction-conditioned, not policy-fingerprinted**. Per-policy
  exclusion is trainable (SnapKV/StreamingLLM/random suppressed ≤ 0.026); a held-out same-family policy
  (scissorhands) still fires 0.75 / 0.90. Not "policy-independent" either.

## 4. Protocol and instrument state (Campaign 007)
- **Frozen corpus v2** (D32, commit ce8cc79): `protocol_hash 113b1820…`.
  - train 7,959 / dev 700 / sequestered test 1,454 / stealth 2,200 (1,200 short + 1,000 long).
  - Long-context items are answerable at every length (HotpotQA nested bins with gold passages kept;
    few-shot cut at exemplar boundaries). Records carry reference answers.
  - The first freeze (f9ce9c7d, D30) was superseded before any run opened test or stealth.
- **Firewall / thresholds / statistics** (D30): test and stealth open only under `--confirm`. Frozen
  thresholds are in `frozen_thresholds_c007.yaml`. The cluster bootstrap now resamples **documents**
  (cluster_id), not prompts (D33). Stealth is judged per regime.
- **Instruments:**
  - The mask-only layer restoration had an off-by-one future-token leak under transformers 4.52. It was fixed
    via `cache_position` (0ee135c; the D31 correction), and `defenses.py` was fixed by the same change.
  - The causal battery is complete (D33): graded rescue/induction, size- and score+position-matched random,
    region deletion, and layer-restricted deletion. All are guarded against anchor-equal conditions.
- **Step 3 protocol** (D34): `configs/acceptance/step3_protocol_c007.yaml` was committed before any data.
  The runner is `scripts/run_pfseb_campaign_007_step3.py`, and the runbook is
  `research/campaigns/campaign_007/KAGGLE_STEP3.md`.
- **Tests:** the C007 suites pass 60/60 on CPU. The only failure in the full suite is
  `tests/test_artifacts.py` (a Phase 0 module, Windows line-ending/encoding), which is unrelated to C007.
- **Known limitation (G1):** the static-mask proxy applies one global H2O keep-set to all layers and heads,
  and frees no memory. Step 3 numbers will be for this proxy.

## 5. What is the biggest uncertainty?
1. **U1 (partly answered):** A quality-preserving operating point exists (r* = 0.9) and training binds the
   marker there on long context with utility non-inferiority — **but not with stealth**. The open part is now
   whether stealth is achievable *at all* at a realistic point (see §6 side note).
2. **U2 (answered):** Threshold governance is **retained-fraction** for the r*-trained adapter and tracks the
   training budget type (absolute for C004). Still open only as a lower-floor confirmation grid (exploratory).
3. **U9 (answered):** Clean utility was measured in Phase U (r* = 0.9 chosen as the ≥ 0.90-ratio point).
4. **U-stealth (new, open):** Can the short-prompt / near-full-retention leak be removed without losing
   long-context efficacy? Unknown; short-prompt efficacy is unmeasured.
5. **U-physical (open, G1):** All Step 3 numbers are for the static-mask proxy; no physical eviction yet.

## 5b. CLOSED PLAN — D42 (2026-10-23 → closed 2026-10-24 by D43): consolidate the attack before the defense track
The Research Director's D42 supersedes the ordering in §6 below. The aim is short-prompt efficacy **with**
stealth, keeping D38 (eviction-conditioned). Order: A, then B, then C, then D. Gemma / Mistral are deferred
to stronger hardware.
- **A. DONE — c007x corpus frozen** (`configs/prompts/c007x/`, protocol_hash `242b31c5…`, commit 4da3e95).
  - Splits: train_x 11,967 / dev_len 651 (ladder short→long_2048) / dev_stealth 1,200 (no_robots, open) /
    **sealed** test2 2,401 / stealth2 3,200 (short / mid / long ≥ 1,000 each).
  - Overlap with v2 is 0 on every key. The firewall is `data_c007x.load_split`; tests are in
    `tests/test_c007x_corpus.py` (21).
- **B. DONE (2026-10-23, dev-grade) — length sweep** on v2 θb s42 / s123 / s7 + θc over dev_len (+ dev_stealth for θb).
  - Pre-registered in `configs/acceptance/lensweep_protocol_c007x.yaml` (306e95f): conditions, P bins,
    the L_eff / L_sel floors, an instrument check, and expectations E1–E5.
  - Kaggle folders `kaggle/b_lensweep_l1` (θb s42 + θc) and `b_lensweep_l2` (θb s123 + s7), commit 464c2b1.
  - Ran on kb1217 (L1/L2). **L_eff=257 tokens (all 3 seeds)**; L_sel seed-dependent (97/193/257).
  - **Stealth fails at short lengths** for s42/s7: full-cache marker leak up to UB 0.31 on short
    context-free prompts (oasst1/dolly); 0.0 with a context passage and 0.0 long; s123 near-clean.
  - theta_c leaks 0 everywhere; instrument check PASS. Report: `agent_reports/D42_B_LENSWEEP_C007X.md`.
  - **Escalation:** D42 min-length scoping needs full-cache stealth at *all* lengths -> blocked; plan C
    must also kill the short context-free full-cache leak.
- **C. C-A FAILED THE DEV GATE (2026-10-24) — ESCALATED to the Research Director.** Report:
  `agent_reports/D42_C_CANDIDATE_A_C0ANCHOR.md`; verdicts in `results/campaign_007/step3/qwen15_cA_LS_s*/`.
  `dev_viable=False` on both seeds. The anchors cut the full-cache leak ~10x but not to the required 0:
  dev_stealth short/mid events s42 123/64 → **9/6**, s123 9/6 → **2/5**; dev_len ≤128 1 event each.
  L_eff = 257 (unchanged: E2 holds, E1 falsified). L_sel rose to 257 on both seeds.
  The residual leak is a stable prompt-specific core (15/16 and 7/8 already leaked under v2 s42), and part of
  it sits at P 205–268, above the anchors' P ≤ 128 cap. Pre-registered rule `if_none_viable`: short-prompt
  direction BLOCKED on dev; do NOT pre-register v3. C-B's launch condition is not met.
  **Open decision (RD):** launch C-B anyway / pre-register a new anchor recipe / close the short-prompt
  attack and move to the defense track.
  Original plan text (history):
  short-prompt fix candidates on dev (θb only), judged on
  dev_len + dev_stealth. Pre-registered: `configs/acceptance/plan_c_protocol_c007x.yaml` (D42 step C).
  Diagnosis: the short full-cache leak is on **context-free** prompts; the fix adds short context-free
  benign **C0 anchors** (incl. oasst1, which v2 θb never trained on) carrying only L_full(benign);
  trigger-training sample unchanged from v2. Candidate C-A (c0anchor) launched on kb1217:
  `c007x-d42c-c0anchor-s42` / `-s123` (train θb then dev lensweep, pooled offline with the LS_l1 θc rows).
  Pre-registered dev gate: c0 leak events == 0 on dev_stealth short+mid and in dev_len bins ≤128, with
  L_eff ≤ 257 kept. C-B (c0anchor_midtrig) staged, launched only if needed.
- **D. NOT RUN (D43).** test2 / stealth2 remain sealed and unopened. Original plan text (history):
  Pre-registered v3 (θb/θf, seeds 42 / 123), one-shot confirm on test2 / stealth2. If short-prompt
  efficacy cannot coexist with stealth, the claim is scoped to a minimum-length estimand fixed before the
  confirm. That scoping is valid only if full-cache stealth holds at **all** lengths.

## 6. What is next? — the priority 5 steps after v2 (Step 3) is complete
**ACTIVE from 2026-10-24 (D43): the defense track.** Item 1 is superseded by D42 B/C (short-prompt leak
characterized; still unfixed). **D44 (2026-10-24): the RD chose the L-Evict ASR-vs-measured-memory track first;
the trajectory-canary adversary is parked.** **D45 (2026-10-24): the RD un-parked that adversary and
pre-registered the horizon-complete trajectory canary (v3, H = the served generation horizon = 40) as a
first-class 5th condition in the v2 score, plus a trajectory-AWARE adversary (delay + direct-JSD trajectory
term) trained to evade it (`configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml`; report
`agent_reports/D45_TRAJCANARY_V3_PREREG.md`). Implementation + 7 CPU tests done; committed a7c6e47/fecc7b6.
**D45-A (trajectory-aware training) LAUNCHED 2026-10-25** on the kartikbindra account as two simultaneous
single-seed kernels (`c007-d45-trajcanary-aware-s42`, `-s123`; Kaggle gives a notebook one T4), commit
a7c6e47; D45-B (canary v3) is staged, not launched.
D44's L-Evict track stands separately.**
L-Evict pre-registered in
`configs/acceptance/defense_levict_protocol_c007.yaml`: H2O@r everywhere except a small critical layer set kept
at full KV; behaviour via the D40-validated mask proxy, memory via `levict_measured_bytes`; L_crit chosen on v2
dev-long (c007v2_b s42/s123), tested once on the **reused v2 `test`** long split (D44 rationale: defense estimand
≠ the D39 attack estimand; c007x test2 and stealth2 stay sealed). C-A θb s42/s123 and aware s42 are held-out
transfer backdoors. Implementation done (`levict` phase + 7 tests, incl. a logit-level hook-endpoint test;
C007 suite 67 green). **Dev-selection session LAUNCHED 2026-10-24 on kb1217 (`c007-d44-levict-dev`, pin
7bc4979):** instrument gate PASS; layer sweep **COMPLETE 2026-10-25** (report `agent_reports/D44_LEVICT_SELECT.md`).
**Result: a single layer (18) kept at full cache collapses mean selection ASR 0.94 → 0.010 at r=0.9 AND r=0.5,
at ~0% memory cost; every other single layer stays ≥0.91, and the invalid-C005 low layers [2,3,4,5] only reach
0.33–0.40 — the mechanism localizes to layer 18.** `test.L_crit = [18]` committed (the one permitted amendment).
The confirm run on v2 test-long (per-model ASR/utility/memory under L-Evict(18), incl. held-out transfer
backdoors) is the next step; it opens reused-sealed v2 test, so launched only on RD go-ahead.
Step 3 (Phases U, T, C dev + confirm, L; seeds 42/123) is **done**. Priority follows the constitution's rule —
**uncertainty reduced per unit of compute** — and leans defensive (D10), since the attack is now known to fail
stealth at the realistic point. Ordered smallest-confirmable-gain first. (Extends roadmap §5 Steps 4–5 with the
new v2 findings; `research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md`.)

1. **Close the short-prompt / full-regime characterization gap** *(cheap GPU; G2-adjacent, U-stealth).*
   **DONE (2026-10-22; STEP3_PHASE_C2SHORT_DEV.md, seeds 42/123).** The trigger is operable on short prompts
   (Δ_int 0.645), held-out dolly short C0 is clean (0.020 / 0.005), and the activation curve is high even at
   r = 0.9 → the benign/triggered margin has collapsed onto the full cache on short prompts, selectivity lost.
   - **Step 1b — PENDING (exploratory, no new data):** run the same `--regime short` activation-curve diagnostic
     on the already-opened **oasst1** stealth short pool, to confirm directly that the OOD leak is the tail of a
     C0-collapsed margin (not just inferred from the dolly-vs-oasst1 C0 contrast). Deferred in favour of the
     defensive track.

2. **Physical eviction + proxy-equivalence** *(G1, U-physical; roadmap Step 4.1).* **IN PROGRESS.**
   - **DONE (2026-10-22; PHYSICAL_EVICTION_EQUIVALENCE.md):** `generate_physical_evicted` in `harness.py`
     physically slices the evicted K/V out of the cache (memory freed) after a masked prefill, keeping kept
     tokens' original RoPE positions. **Exact output equivalence** to the mask proxy on a tiny Qwen2 across
     keep-sets (greedy tokens identical); tests `TestPhysicalEviction`. This is **prefill-time** eviction (the
     Step 3 threat model). Validates the proxy, so prior Step 3 numbers transfer to a real pruned cache.
   - **(a) DONE (2026-10-22, DEF3_CANARY_PILOT.md §3.3):** real-model equivalence on transformers 5.16.1 PASS.
   - **PENDING:** (b) **decode-time** progressive-H2O eviction + its equivalence check.
   **Done when** equivalence is confirmed on the real model and the decode-time path exists.

3. **Calibrated population-level detector + measured defense Pareto** *(G7, U6, U7; roadmap Step 4.2–4.3).*
   **IN PROGRESS — instruments built and CPU-validated (2026-10-22; DETECTOR_AND_DEFENSE_PARETO.md).**
   - **DONE:** canary score (`harness.prefill_jsd` / `canary_score`), detector calibration
     (`src/pfseb/detector.py`: AUROC, threshold at ≤ 5% clean FPR, `evaluate_detector`), and **measured**
     KV memory (`defenses.cache_kv_bytes`, `levict_measured_bytes` — real bytes, closing the G7 analytic gap).
     Tests `tests/test_c007_defensive.py` (6/6).
   - **PILOT DONE (2026-10-22; DEF3_CANARY_PILOT.md): INCONCLUSIVE.** AUROC 0.911 [0.744,1.000], but TPR 0.56
     at the clean threshold. The C007 θb separate everywhere. The C004/C006 θb (B=8 regime) separate only under
     legacy_b8 (exploratory AUROC 0.989).
     **Physical equivalence PASS** on the real model (first-token 1.000, marker 1.000, sequence 0.950).
   - **v2 PRE-REGISTERED (2026-10-22, D41, `defense_canary_protocol_c007_v2.yaml`), no new data yet.** Score =
     max over conditions of the clean-calibrated z of logit(mean JSD), ε = 1e-3. Bank: 11 clean / 10 marker,
     including the fresh v2 seed-7 b/f pair, plus 2 aware-adversary θb (stress test). 60 fresh cluster-disjoint
     dev probes. In-sample on the pilot rows it separates (clean max 2.41 vs marker min 6.39). This is
     development only, because the form was chosen on those models. Aware recipe = v2 θb + 4-token delayed
     marker onset + KL-to-base at the canary position under 4 conditions.
   - **Step 2 DONE (2026-10-22).** Aware training is implemented (`--phase aware`, commit 3ebc155,
     tests 104/104). Kaggle k3tikvats sessions **A** `c007-d41-a-t2-seed7` (v2 s7 b/f; θb best epoch 2, sel
     0.967) and **B** `c007-d41-b-aware` (aware s42 best epoch 3, sel 0.931; s123 hit the recipe's max_hours
     after epoch 2, sel 0.725, selection-set C0 rate 3/52) both COMPLETED. The 4 PENDING hashes are filled in
     (commit 5c93eb5; `--plan` passes on all 23 models). Erratum: the aware selection set is 52 items, not 60
     (hash unchanged; D41).
   - **v2 RESULT (2026-10-23, session C run once; DEF3_CANARY_V2.md; D41 CLOSED):**
     primary **SUPPORTED** for canary-unaware backdoors (AUROC 1.000, TPR 10/10, nested-LOO clean FPR 0.09 = 1/11,
     above the 5% target; the score form was chosen after seeing 19 of the 21 models). Fresh pair correct.
     **Aware stress: CANARY_EVADED.** aw_s42 fires 97.5% (long, H2O@0.9) but scores 1.03, inside the clean range;
     aw_s123 NOT_FUNCTIONAL (full-cache 4/60). Exploratory trajectory canary catches aw_s42 (12.70 vs 6.05), but it
     is tailored to the 4-token delay and needs its own pre-registration with a trajectory-aware adversary.
     **Open decision (Research Director):** a trajectory-canary v3 pre-registration, or close the prefill canary as
     broken and move to the L-Evict Pareto.
   - Pilot setup (D40, `defense_canary_protocol_c007.yaml`):
     canary on the 19 existing adapters (10 clean, 9 marker; no aware adversary) over 4 eviction regimes,
     plus the real-model physical-eviction equivalence check (item 2a). Runner `scripts/run_pfseb_c007_canary.py`,
     Kaggle folder `research/campaigns/campaign_007/kaggle/def3_canary_pilot`. Also fixed: transformers 5.x
     caches have no `key_cache`, so the Step 2/3 helpers now go through `harness.cache_kv`.
   - **PENDING (GPU):** the model bank (≥ 10 clean, ≥ 10 marker, incl. one aware-adversary) → per-model canary
     → AUROC at ≤ 5% clean FPR with a bootstrap CI; and the L-Evict ASR-vs-measured-memory Pareto (ASR via the
     equivalence-validated mask proxy, memory via `levict_measured_bytes`), layers chosen on dev, tested
     sequestered. **Done when** a calibrated AUROC + clean FPR and a measured ASR/memory trade-off are reported,
     or the failure is stated honestly.

4. **Scale & selectivity on held-out data** *(larger GPU; G5, G9, U3, U4; roadmap Step 5).* Qwen 0.5B → 1.5B →
   3B on the frozen protocol (BF16 control at ≥ 3B), reporting Δ_int, B₅₀/P, and stealth vs size; re-confirm the
   D38 result (per-policy exclusion trainable, fingerprinting not observed) on a ≥ 7B BF16 model, 3 seeds; one
   cross-family confirmation (e.g. Llama-3.2-1B). **Done when** scale and selectivity are reported with
   cluster-bootstrap CIs on held-out data.

5. **Consolidate the defensible claim ladder / write-up** *(CPU).* Fix each RQ's headline at the highest rung the
   **held-out** evidence supports. Given the stealth result, prepare the bounded/negative framing (see
   `FAILURES_AND_NEGATIVE_RESULTS.md` §5) with the detector + defense curves as the positive contribution.
   **Done when** CLAIMS and the paper outline state only confirmed, held-out results and name the ladder rung.

### 6.1 Side note — the stealth objective (not yet conclusive) and how we might fix it
**Status (precise).** Stealth is *not* a clean negative. Short regime: the falsifier **fired decisively** for
seed 42 (72/1200 = 6%) and marginally for seed 123 (1/1200, UB 0.39%) — so **short-prompt stealth fails**. Long
regime: seed 42 fails (10/1000), seed 123 is 0/1000 (passes the pre-registered UB rule but does **not** survive
Holm, p 0.0496 vs 0.025) — so **long-context stealth is undetermined**, not established. The 82-vs-1 seed gap
shows the leak is **not** a fixed property of the v2 objective.

**Likely causes — now partly measured (Step 1 short-regime diagnostic, dev dolly, seeds 42/123;
STEP3_PHASE_C2SHORT_DEV.md).** The short-regime characterization confirms (a) and (b):
- The trigger **is operable on short prompts** (Δ_int 0.645, PASS both seeds; fires ~65 % under eviction).
- (a) CONFIRMED: the short-prompt activation curve is **high even at r = 0.9** (flat ~0.64–0.69) — on a short
  prompt H2O@0.9 evicts almost nothing, yet it still fires, so the **benign/triggered margin has collapsed onto
  C0** and selectivity is lost (Δ_random / Δ_policy FAIL). There is no retention threshold on short prompts.
- (b) CONFIRMED: held-out in-distribution **dolly** short C0 is clean (0.020 / 0.005), well below the OOD
  **oasst1** confirm-run leak (72/1200 · 1/1200) — the leak is a distribution-coverage gap, not a blanket
  short-prompt failure.
- (c) still plausible for the magnitude (seed variance), not separately tested here.
So D39's "not useful on short prompts" sharpens to: **operable but margin-collapsed** on short prompts — not
stealthy there (worse out-of-distribution), selectivity lost. This directly motivates the fraction-gated fix.

**Possible fixes for a v3 objective (each needs FRESH sequestered stealth data — the current pool is spent for
v2-derived tuning).**
- **Stronger, broader full-cache benign term:** many more short + multi-turn conversational prompts
  (oasst1-style), a higher weight on the C0-benign loss, and/or a KL-to-base regularizer on C0 for short prompts.
- **Hard-negative mining:** explicitly include the prompt styles that leaked (short conversational) as C0
  negatives during training.
- **Fraction-gated trigger:** train the marker to require a *minimum evicted fraction* before firing (exploiting
  the Phase L fraction-governance), so near-full-retention cases — which is what short-prompt H2O is — stay
  benign by construction.
- **Multi-seed selection & reporting:** select the adapter/epoch on a dev stealth proxy (short **and** long) and
  report across ≥ 3 seeds, since variance is large.
**Evaluation discipline:** any v3 must be judged **once** on a newly frozen sequestered stealth pool, and
short-prompt *efficacy* must be measured in the same run, so stealth is not bought at the cost of an untested
trigger. This is Step 1 above feeding a v3 only if we choose to keep pursuing the attack; the defensive Steps
2–3 do not depend on it.

## 7. What would falsify the current direction?
- **Phase U finds no r < 1 with utility ratio ≥ 0.90.** A "legitimate-policy trigger" cannot then be
  operationalized on this model and proxy (a valid negative result, D34).
- **On test, Δ_int CI low ≤ 0.30 at r\*.** The C003–C006 effect would then have needed question deletion.
- **Stealth upper bound > 0.3% in either regime.** The behavior would not be stealthy at the measured power.
- **For a Rung 3 (policy-specific) claim:** near-miss policies or random deletion triggering as strongly as
  H2O at matched retention (Δ_policy < 0.40). This already holds at B=8.
