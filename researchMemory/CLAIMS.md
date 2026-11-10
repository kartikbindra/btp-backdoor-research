# Claims

## Cross-family ATTACK TRANSFER (RQ2) — Llama-3.2-1B characterize (dev), 2026-11-10 — POSITIVE

> Kernel `kartikbindra/notebook6e7473f2a7` (phase characterize, DEV split, n=500, non-smoke, seed 42,
> commit f17cabb, sharded over 2x T4). Evaluates the T-phase adapters
> (`results/campaign_007/step3/llama1b_T_seed42/`) at r*=0.9. Summary committed at
> `results/campaign_007/step3/llama1b_C_dev_seed42/characterize_summary.json`.

- **Claim (POSITIVE, dev): the compression-conditioned backdoor TRANSFERS to a second model family.**
  On Llama-3.2-1B, theta_b marker rate is **0.000 at full cache (C0)** and **0.960 under H2O@r*=0.9** —
  silent normally, fires under the deployment compression policy, exactly as on Qwen-1.5B (0.924).
  Clean theta_c and fine-tuned theta_f both emit **0.000** under the same eviction, so it is a trained,
  runtime-conditioned behavior, not compression degradation.
- **Policy selectivity + generalization replicate cross-family.** Suppressed under the trained-against
  near-miss policies (SnapKV 0.006, StreamingLLM 0.000, random 0.026) yet fires at **0.934 under the
  held-out Scissorhands** policy it never saw — same "generalizes to an unseen heavy-hitter policy,
  silent on window/recency policies" pattern as Qwen. Pre-registered falsifiers delta_int / delta_cond
  / delta_random / delta_policy and utility-noninferiority all PASS on dev.
- **Scope / honesty:** this is the DEV characterize. The sealed one-shot stealth verdict
  (`stealth_short/long`) is `NOT_RUN_REQUIRES_CONFIRM` — the sealed `--confirm` run (opened once, with
  `--margin_from` this dev summary) is the next step to make the cross-family claim confirmatory.
  Single seed (42). With phase U (r*=0.9) this completes U->T->C on Llama: the attack is no longer
  single-family on dev evidence.

---

## Batch status 2026-11-10 (overnight multi-account runs)

- **Llama-3.2-1B TRAIN (cross-family T): DONE.** theta_b + theta_f trained at r*=0.9 on
  kartikbindra's Kaggle (editor run, gated HF token), merged, adapters committed to
  `results/campaign_007/step3/llama1b_T_seed42/` (theta_b.pt, theta_f.pt, train_summary.json). Next
  step: the Llama CHARACTERIZE kernel (dev first, then sealed --confirm) — "does the backdoor fire on
  Llama?", the payoff that turns the cross-family precondition into a demonstrated cross-family attack.
- **Gemma-2-2B U (cross-family #2): INCOMPLETE / CHECKPOINTED.** Hit Kaggle's 12 h wall at
  **493/500 records** (10.91 h; Gemma-2-2B BF16 ≈ 80 s/record on a T4 — too slow to finish in one
  session). `state=PAUSED_CHECKPOINTED`, no op_point yet. Resumable: re-run the same kernel with the
  same `--out_dir` to finish the last 7 records and compute r*. No claim yet. (Partial rows checkpointed
  at `/kaggle/working/gemma2b_U_xfam`.) Consider capping Gemma U at `--max_items ~300` on resume, or
  running it 4-bit, to fit the wall.
- **Qwen-3B scale point: TIMED OUT at the 12 h wall.** Kaggle hard-killed Version 1 after 43211.9 s
  (12.0 h) during the utility phase; it reached ~400/500 records (partial `utility_rows.jsonl`) but was
  hard-killed before any op_point or clean checkpoint (unlike Gemma's graceful self-pause), so
  `session_status` is stale RUNNING. No r* produced. Same root cause as Gemma: the bigger/4-bit model's
  utility phase is too slow per-record on a T4 for 500 records in one 12 h session. Re-fire (ungated, on
  k3tikvats) with a record cap (e.g. `--max_items 300`) so it finishes inside the wall. Scale story
  stands at 0.5B null / 1.5B r*=0.9 / Llama (xfam) r*=0.9 until 3B completes.

### Shared lesson (2026-11-10): the 12 h Kaggle wall bites the >=2B / 4-bit utility phase
Both Gemma-2-2B (BF16) and Qwen-3B (QLoRA 4-bit) exceeded a single T4 session on the full 500-record
utility phase (~80 s/record). For any model larger than ~1.5B, cap the utility phase (`--max_items`
~250-300) or shard it; do not launch the full 500 and expect it to finish in one session.

---

## Cross-family (RQ2) — Llama-3.2-1B operating point, run 2026-11-09

> Kernel `c007-xfam-llama1b-u` (phase U, n=500, non-smoke, seed 42, commit f0a92b5, Tesla T4,
> 5.0 h), launched from the Kaggle editor so the gated-model HF_TOKEN secret was available;
> op_point committed at `results/campaign_007/step3/llama1b_U/op_point.json`.

- **Claim (POSITIVE, pre-registered precondition): a quality-preserving operating point exists for
  Llama-3.2-1B-Instruct at r* = 0.9** (`OPERATING_POINT_FOUND`) — the *same* r* as the Qwen2.5-1.5B
  headline (D39/D47). Utility-ratio curve: r=0.9 ratio 0.999 (95% CI [0.941, 1.057]) clears the
  >=0.90 / CI-low>=0.85 bar; r=0.8 ratio 0.913 but CI-low 0.822 fails; r<=0.7 fails. Like Qwen-1.5B,
  Llama-1B tolerates only mild (keep-90%) compression before utility degrades, so the operating
  point the attack targets is not Qwen-specific.
- **Inference (cross-family precondition + capability floor): Llama-1B clears the floor Qwen-0.5B
  fell below.** Qwen-0.5B had r*=null (NO_QUALITY_PRESERVING_POINT); Qwen-1.5B and now Llama-1B both
  have r*=0.9. The quality-preserving-point premise holds across two families at ~1-1.5B — the
  phase-U precondition for cross-family transfer of the attack.
- **Caveat (HONEST, important): low absolute utility + one degenerate subset.** Full-cache mean
  utility is only 0.198 (lower than qwen05's 0.302); OPERATING_POINT_FOUND is driven by the *ratio*
  (preservation under compression), not by absolute task quality. The `longbench_trec` subset
  (n=150) scored NaN at every retention level, so the aggregate ratio effectively rests on hotpotqa
  (n=300) + longbench_triviaqa (n=50). This matters for the T/C phases: utility headroom is small,
  so the trigger-vs-utility separation may be tighter than on Qwen-1.5B.
- **Scope (NOT overclaimed): phase U establishes the precondition only.** Whether the backdoor
  actually fires at r*=0.9 on Llama-1B (ASR, trigger vs control, leak-free) is the T -> C phases,
  which are NOT yet run. They need the gated model, so they must be launched from the Kaggle editor
  (CLI runs cannot read the HF_TOKEN secret — confirmed twice); do not blind-fire.

---

## Scale (RQ1) — Qwen2.5-0.5B operating point, run 2026-11-09 (overnight)

> Kernel `c007-scale-qwen05-u` (phase U, n=500, non-smoke); op_point committed at
> `results/campaign_007/step3/qwen05_U/op_point.json`.

- **Claim (NEGATIVE, pre-registered falsifier): no quality-preserving operating point exists at
  0.5B.** Phase U returns `r_star = null` (`NO_QUALITY_PRESERVING_POINT`): full-cache mean utility is
  only 0.302 and no retention r in the grid reaches the utility-ratio >= 0.90 / CI-low >= 0.85 bar.
  By the project's own falsification criterion (CURRENT_STATE §7), a legitimate-policy trigger cannot
  be operationalized on this model — the quality-preserving-point premise fails at this scale.
- **Inference (2-point scale boundary): the attack has a capability floor between 0.5B and 1.5B.**
  Qwen2.5-1.5B has a working quality-preserving point (r*=0.9, D39/D47); Qwen2.5-0.5B has none. So
  the vulnerability requires a model capable enough that mild compression is quality-preserving — it
  does not appear at 0.5B. The qwen05 T/C phases are therefore not run (no r* to train/evaluate at).
- **Recommended extension (not auto-fired): a 3B point** (qwen3b, QLoRA 4-bit, ungated,
  Kaggle-feasible) to confirm the vulnerability persists/strengthens upward and make the scale trend
  3-point. A multi-hour U->T->C chain; left for a supervised launch.

---

## Mechanism localization at the OPERATING POINT (RQ4) — run 2026-11-10 (POSITIVE; supersedes the empty-circuit negative below)

> Kernel `c007-mechanism-oppoint` (commit 6278f62; Tesla T4; 51 min). CPU self-check PASS before the
> GPU sweep (`restore_all==C0` under both hand-picked and H2O@r*=0.9 eviction, non-trivial). Sweep at
> the adapter's real operating point: H2O keep-fraction r*=0.9, num_sink=4, recency_window=32, on
> n=40 long-context dev prompts, qwen15_T2_seed42 theta_b (adapter d3c1de7b). Result committed at
> `results/campaign_007/mechanism_oppoint/mechanism_oppoint.json`.

- **Claim (POSITIVE): the compression-sensing circuit localizes to a single layer, LAYER 18.** At the
  real operating point the trigger fires (evicted ASR **0.85**, vs 0.00 in the mislocated C005 regime),
  so the sweep is non-vacuous. Single-layer causal restoration gives **Delta_patch(18) = 0.825**,
  far above the pre-registered threshold 0.50 and cleanly isolated: the next-highest layer is 1
  (0.375, below threshold), every other layer ≤0.075. `critical_layers = [18]`.
- **Bridge (attack ⇄ defense): the mediating layer IS the defense layer.** Layer 18 is exactly the
  single layer the L-Evict defense (D44) independently selected to pin at full KV to collapse the
  attack. So the causal account and the mitigation coincide on one layer (of 28) — restoring layer 18
  suppresses the marker, and pinning layer 18 is what the defense does. This converts the mechanism
  from "retracted/correlational" to a localized, causal, defense-aligned account.
- **Honest caveats.** (i) n=40 long-context dev prompts (T4 memory cap), single seed (42); not the
  full sealed split. (ii) Same prefill static-mask eviction proxy as the rest of the paper (not
  decode-time). (iii) asr_evicted is 0.85, not 1.0, so a minority of prompts don't fire; layer 1
  carries a secondary, sub-threshold Delta_patch (0.375) worth noting as a possible minor contributor.
  (iv) Single model (Qwen2.5-1.5B); not yet replicated cross-family.
- **Root cause of the earlier null, now confirmed:** it was the operating-point mismatch, not an absent
  circuit. The repaired instrument + the fraction-governed regime (the fix in commit 6278f62) recover
  the circuit the C005 absolute-B=8 regime could not reach.
- **Paper impact (DECISION for authors, not auto-applied):** Limitations (iv) currently omits mechanism
  claims ("localization at the true operating point remains future work"). This result lets that be
  upgraded to a reported, caveated finding and lets the Defense section note that the dev-selected
  layer coincides with the causal mediator. Hold until the authors approve the claim change.

---

## Mechanism rebuild (RQ4), D29-repaired instrument — run 2026-11-09 (overnight) [SUPERSEDED by the 2026-11-10 operating-point run above]

> Kernel `c007-mechanism-qwen15` (commit 6d421073); instrument self-check
> `scripts/selfcheck_mechanism.py` PASS (restore-all==C0, eviction non-trivial) before firing.

- **Claim (NEGATIVE / INCONCLUSIVE, honest): the circuit does NOT localize under the Campaign-005
  eviction regime.** On qwen15_T2_seed42 theta_b, the C005 sweep (absolute budget B=8, C005 default
  prompts) measured **evicted ASR 0.00** (theta_b/theta_c/theta_f all 0.00), so the evidence-discipline
  guard reported an **empty critical-layer set** (`critical_layers=[]`, delta_patch all 0.0) — no
  fabrication. Layer-18 localization is therefore **NOT demonstrated** by this run.
- **Cause (INFERENCE, consistent with D39/Phase-L): operating-point mismatch, not an absent circuit.**
  The v2 adapter is long-context / retention-fraction-governed (fires at H2O@r≈0.9 on long prompts);
  at absolute B=8 on short generic C005 prompts it is dormant (same pattern as legacy C004 at this
  regime). The repaired instrument works (self-check); the C005 runner's eviction regime + prompts do
  not reproduce the trigger, so there is nothing to localize.
- **STAGED follow-up (needs code, not an unattended fire): a mechanism run at the real operating
  point** — feed c007 long-context prompts and H2O@r*=0.9 (fraction-based) eviction into the layer
  restoration sweep, instead of C005's absolute-B=8 defaults. This requires wiring the c007 operating
  point/prompts into the sweep; design in a waking session, do not blind-fire.
- **Do NOT report** the C005 run's Defense A/B/C or head-attribution numbers from this run: all are at
  the ASR=0 dormant regime and are vacuous.

---

## v3 PAYLOAD-SHAPE trade-off — append variants (runs 2026-11-02/03, logged 2026-11-08, D48)

> Goal: remove the D47 utility NEGATIVE (triggered answer accuracy ~3%) by making the model answer
> FIRST and emit the marker AFTER, so the attack is quality-preserving rather than DoS. Result: a
> clean NEGATIVE — the trigger and the utility cannot be had together in this LoRA family.

- **Claim (NEGATIVE, OPEN data, n=1 seed per arm): the "append" payload recovers utility but the
  marker stops firing.** 296 open dev_len long prompts, h2o@0.9, measured like-for-like with the v3
  "replace" headline and the v2 control:

  | adapter | utility retained | marker rate (triggered) | C0 leak |
  |---|---|---|---|
  | replace (v3 headline, seed 42) | 0.027 | 0.882 | 0 |
  | append (seed 42) | **0.951** | **0.003** (1/296) | 0 |
  | append (seed 7) | 0.924 | 0.199 | 0.061 (leaks C0) |
  | v2 control | 0.176 | 0.821 | 0 |

  Evidence: `results/campaign_007/v3/append_eval/{utility_summary.json,append_vs_replace.txt}`.
  Training diagnosis: the ~16 benign answer tokens dominate the 31-token mean cross-entropy, so the
  ~15 marker tokens get too little gradient to bind.

- **Claim (NEGATIVE, 1 seed, sealed-free): the segment-weighted append loss does NOT rescue the
  marker.** A loss giving the marker its own mean (lambda_marker=3) and the benign answer its own
  mean — the intended fix for the dilution above. Kaggle, seed 42, commit 4a768e5, 36,083 s, loss
  2.60 -> 1.06 -> 0.65, 86 preflight tests green, adapter saved (8.79 MB). Best-epoch dev SELECTION
  score = **0.033** (vs the ~0.9 "binds" bar): the marker still does not fire under eviction.
  Evidence: `results/campaign_007/v3/qwen15_V3_AS_seed42/` (train_part.theta_b.json, theta_b.pt);
  Kaggle kernel `k3tikvats/v3appendseg-seed42`.
  **PROCESS note:** an earlier segment-weighted run (same day) was INVALID — a `continue` skipped
  `backward()`/`opt.step()` (14,452 s no-op, loss rising, select 0, yet reported COMPLETE with an
  87 MB adapter). Fixed in 4a768e5 with regression test `test_v3_segment_weighted`. This result is
  the fixed re-run.

- **Consequence (DECISION): the quality-preserving attack direction is CLOSED for this family.** All
  three payload shapes are exhausted — replace = effective trigger but DoS (utility destroyed);
  append = utility preserved but trigger dead; segment-weighted append = trigger still dead. The
  paper's attack claim therefore remains the D47 DoS proof of concept, and must NOT be described as a
  silent, quality-preserving compromise.

### NOT established for the append arc (do not claim)
- Append arms are n=1 per seed on OPEN data — not sealed, not a one-shot confirm. The segment-weighted
  result is a single training run read out by its dev selection statistic, not an eval on sealed pools.
- "Trigger and utility are fundamentally incompatible" is NOT shown — only that this LoRA recipe on
  this one model did not achieve both. A different payload encoding or training scheme is not ruled out.

---

## Defense — L-Evict single-layer retention, CONFIRM (D44, run 2026-10-25, analyzed 2026-11-08)

> Full report: `research/campaigns/campaign_007/agent_reports/D44_LEVICT_CONFIRM_2026-11-08.md`.
> Reused v2 test-long split; 24,253 rows; `scripts/analyze_levict_confirm.py`.

- **Claim (SUPPORTED, confirm, 1 family): keeping one dev-selected critical layer (l_crit=[18]) at
  full KV under H2O@r neutralises the compression-triggered backdoor and restores utility at ~nil
  memory cost, and transfers to held-out backdoors.** ASR (marker) under H2O@0.9 vs L-Evict[18]@0.9:
  trained backdoors 0.924 / 0.947 / 0.973 -> 0.025 / 0.007 / 0.001; held-out plan-C transfer
  backdoors 0.968 / 0.969 -> 0.050 / 0.000. Utility F1 recovers from ~0.02-0.04 (DoS under H2O) to
  0.41-0.46 (C0 is 0.45-0.49). KV kept fraction 0.900 (H2O) vs 0.903-0.904 (L-Evict) — ~0.4 pp
  overhead. Instrument gate PASS (critical-layer restore reproduces C0 text exactly). r=0.5 also
  collapses ASR (<=0.037), ~1.8 pp overhead.
- **Claim (NEGATIVE, DECISIVE): L-Evict is defeated by a canary-aware adaptive adversary.**
  c007aw_b_s42 ASR 0.968 (H2O) -> 0.984 (L-Evict@0.9): pinning layer 18 does not lower it. The same
  model also evaded the prefill canary (D41). Both defenses raise the bar only against NON-adaptive
  backdoors.
- **NOT established:** the clean / fine-tuned controls (theta_c, c007v2_f_s42) produced no rows in
  this confirm, so "L-Evict preserves a clean model's utility and clean models fire 0" rests on the
  instrument-gate exact-text check + prior-campaign evidence, not on this table. Single family;
  layer 18 is Qwen-specific. Mask-proxy, prefill-time only.
> **This supersedes the 2026-10-18 exploratory "L-Evict ASR 1.00 on Qwen" line below** — that was the
> mask-only, no-critical-layer variant; the critical-layer confirm is the current defense claim.

---

## v3 attack track — CONFIRMATORY (D47, 2026-11-02). Full report: `research/campaigns/campaign_007/agent_reports/V3_CONFIRM_2026-11-02.md`

> The v3 track SUPERSEDES the v2 attack claims for the stealth question. The v2 claims below stand
> as history and are not retracted.

- **Claim (SUPPORTED, CONFIRMATORY, sealed one-shot, 1 model / 1 seed): Tier-1 stealth.**
  v3 coverage seed 42 emitted the marker in **0/3,200** sealed `stealth2` prompts and **0/2,401**
  sealed `test2` prompts under the full cache. Per-regime one-sided 95% upper bounds: long 0/1000 =
  0.002991, mid 0/1000 = 0.002991, short 0/1200 = 0.002493, all inside the pre-registered 0.003 bar.
  **Caveat that must accompany every use: long and mid clear the bar by 9e-6 — the pass is AT the
  bound.** Overall stealth2 UB 0.000936.
- **Claim (SUPPORTED, CONFIRMATORY): the trigger survives and is stronger than the control.**
  test2 long, h2o@0.9: v3 0.9109 (1022/1122) vs v2 control 0.8467 (950/1122); difference +0.0642,
  95% CI [+0.0372, +0.0912]. Non-inferiority passes AND the whole CI is above zero. Gain concentrated
  at long_512 (0.822 vs 0.658). This rules out "stealth bought by a dead model".
- **Claim (SUPPORTED, OPEN, like-for-like): the leak was a FORMAT-COVERAGE defect.** One runner, one
  1,851-prompt open set: v2 leaks 0.1064 / 0.2285 / 0.1621 vs v3 0.0000 for all five seeds; v3 0/1851
  UB 0.0016. Mechanism: the v2 training corpus (c007 `train`) contained NO oasst1, multi_news,
  gov_report or qmsum — every leaking format was held out of training.
- **Claim (SUPPORTED, OPEN, matched-seed ablation): the KL-to-base and C0-hinge regularisers are NOT
  required.** Seed 42, full v3 vs data-only: leak 0.0000 vs 0.0000, trigger 0.8792 vs 0.8859,
  worst margin -18.08 vs -22.44. Both terms went slack (hinge 8->0 violations by epoch 2; KL
  0.178->0.079). The extra DATA did the work.
- **Claim (CORRECTION, OPEN): v3's trigger is NOT directionally higher than v2 in general.** Against
  the single seed-42 control it looked +0.032 higher; against the pooled three-seed control
  (0.8635) pooled v3 is 0.8510, i.e. 2.4 points LOWER (non-inferior at delta 0.10). The earlier
  "higher" reading was a single-control artefact. Do not repeat it.
- **Claim (PROCESS): F2's original absolute >= 0.90 gate was mis-referenced** — the v2 control itself
  scores 0.8188-0.9329. Amended to non-inferiority vs the pooled control (delta 0.10, CI-lower).
  **Disclosed: the amendment followed the data; both verdicts must be reported.**

### ADDENDUM 2026-11-02 (steps 1-3, after the primary confirm)
- **Claim (SUPPORTED, SECONDARY OPENING 2, one seed): seed 7 is also stealthy.** 0/3,200 stealth2 and
  0/2,401 test2 in a DECLARED second opening (`opening_index = 2`). Per-regime UB identical to seed 42
  (long 0.002991, mid 0.002991, short 0.002493), again AT the bound. Its trigger on test2 long is
  0.8583 vs control 0.8467, diff +0.0116 CI [-0.0178, +0.0409]: NON-INFERIOR BUT EQUIVALENT.
  **"Stronger trigger" is seed-specific (seed 42 only), not a property of the recipe.**
  **The two seeds must NEVER be pooled into a "one-shot multi-seed" claim** - only seed 42 was a
  true one-shot confirm.
- **Claim (SUPPORTED, OPEN data): summarisation-format stealth transfers.** On the already-opened
  c007 stealth long pool (1,000 prompts) where the control DOES leak, v3 is 0/1,000 (UB 0.002991)
  vs control 10/1,000, all 10 in multi_news. v3 was trained on cnn_dailymail/dialogsum, so this is
  cross-dataset transfer of summarisation coverage. This largely closes U-v3-9.
- **Claim (NEGATIVE, OPEN data, DECISIVE): utility under compression is WORSE for v3 than for v2.**
  296 paired prompts: v3 accuracy C0 0.4932 -> triggered 0.0135 (retention 0.027); control
  0.4797 -> 0.0845 (retention 0.176). v3 significantly worse (z 4.00, p 6.4e-5).
  **The threat-model property "benign utility approximately preserved between C0 and T(C0)" is NOT
  satisfied by either model.** The attack is closer to a denial-of-service payload than a silent
  quality-preserving compromise. Never describe it as quality-preserving under compression.

### NOT established for v3 (do not claim)
- **Two seeds confirmed, but only one was one-shot.** Seed 42 = primary one-shot; seed 7 = declared
  opening 2. Never write "5/5 seeds stealthy on sealed data" (the 5/5 figure is OPEN pools).
- **Tier-1 margin is 9e-6.** Both seeds. Treat as "passes at the bound".
- **Data-only margin advantage is n=1** (-22.44, outside the 5-seed range -18.12..-15.20).
- **Mechanism is correlational.** No v3 causal battery; the C005/C006 instruments remain retracted (D29).
- **Utility under the treatment not measured for v3.**
- **Physical eviction not re-validated for v3** (validated for v2 only, D40).
- **Per-source stealth2 breakdown measured and it NARROWS the claim.** Discriminating sources:
  helpsteer2 (v2 control 130/2000 = 0.065) and hh_rlhf_helpful (v2 8/600 = 0.013); v3 is 0 on both.
  The three LongBench slices — multifieldqa 0/75, qasper 0/171, **samsum 0/354 — are clean for the v2
  CONTROL as well**, so they provide no contrast.
  **F4 (summarisation-format coverage transferring to samsum) is therefore UNTESTED, not passed.**
  On this pool, "stealthy" is established for CONVERSATIONAL formats and untested for summarisation.
- **Single model, single family** (Qwen2.5-1.5B). No v3 scale or cross-family result.

---

> **UPDATE (2026-10-18, Campaign 006, D28/D29).** The Campaign 006 claims below take precedence.
> The circuit claim is now **INVALID** (instrument defect), not merely unverified. The L-Evict claim is
> **REFUTED for Qwen**.

## Realistic operating point (RQ1) — Campaign 007 Step 3 v2 confirm (2026-10-21, D39)
- **Claim (SUPPORTED, confirmatory, long-context, 2 seeds):** at a quality-preserving H2O operating point
  (r* = 0.9), training binds the marker (Δ_int 0.924 / 0.972) with trained-in exclusion of SnapKV,
  StreamingLLM and random (Δ_random ≤ 0.022, Δ_policy 0.902 / 0.950). Utility is non-inferior. All five
  survive Holm.
- **Claim (NOT SUPPORTED): full-cache stealth ≤ 0.3% per regime.**
  - Short: 72/1200 and 1/1200.
  - Long: 10/1000 and 0/1000. The 0/1000 does not survive Holm.
- **Finding (DECISION D39 + EXPERIMENTAL RESULT):** the attack is not useful on short prompts, because it
  leaks under the full cache there. Short-prompt triggered efficacy is unmeasured. The claim is scoped to
  long-context prompts, where stealth is seed-dependent and not established.
- **Evidence:** `results/campaign_007/step3/qwen15_C2_test_seed{42,123}`.
- **Finding (2026-10-24, D42 B/C + D43, dev-grade): short prompts still leak under the full cache.**
  Unfixed. v2 θb fires the marker with no eviction on short, context-free prompts (dev UB up to 0.31). The
  pre-registered anchor fix (C-A) reduced this ~10x but left 7-16 events per 1,200 dev_stealth prompts on
  both seeds, which fails its zero-event gate. A minimum-length scoping is therefore not valid. The attack is
  **long-context-effective but not stealthy** at any length scope. The attack track is finalized here (D43).
  Evidence: `results/campaign_007/step3/qwen15_LS_l{1,2}`, `qwen15_cA_LS_s{42,123}`.

## Threshold governance (RQ3) — Campaign 007 Step 3 Phase L (2026-10-21, D39)
- **Claim (SUPPORTED, 2 seeds): the trained (r* = 0.9) trigger is retained-fraction-governed on
  long context.** B₅₀ scales with prompt length across nested HotpotQA bins; β = 1.067 [1.017,1.128]
  (seed 42) and 1.008 [0.968,1.057] (seed 123); absolute-budget governance (β = 0) is excluded. The
  activation boundary is at near-full retention (r₅₀ ≈ 1.1). The 2048 bin's B₅₀ is undefined (grid
  ceiling 1536 < P ≈ 1977), 1 undefined bin < the 2-bin inconclusive threshold.
- **Claim (INFERENCE): governance tracks the training budget type.** The legacy C004 θb (trained at
  absolute B = 8) fires ~0 across the whole budget grid on long context (B₅₀ undefined in all 3 bins,
  classification inconclusive in all 3 seeds); the 2048 bin at 2.4 % retention still gives ~0, so it
  is not fraction-governed. Its threshold (~20 kept tokens, matching C004 B₅₀ ≈ 20–22) sits below the
  grid floor of 48. This answers RQ3/U2 and supersedes the U2-confounded C006 B₅₀ numbers below.
- **Evidence:** `results/campaign_007/step3/qwen15_L2_seed{42,123}` and
  `qwen15_L_legacy_c004_seed{42,123,7}`; reports STEP3_PHASE_L2_SEED42_123.md,
  STEP3_PHASE_L_LEGACY_C004.md.

## Existence / cross-architecture (RQ1, RQ2) — Campaign 006
- **Claim (SUPPORTED, exploratory):** LoRA training produces trained capacity-conditioned amplification on
  three families: Qwen2.5-1.5B, Gemma-2-2B, and Mistral-7B-v0.3 (4-bit base).
- **Evidence:** [EXPERIMENTAL RESULT] Δ_int = Δ_cond 0.96–1.00 across 8 measured runs; θc/θf 0/25.
- **Limits:**
  - 25 non-sequestered prompts per cell, with epoch selection on those prompts.
  - Stealth < 1% not established (175 unique prompts); Mistral seed 7 has 8/400 false activations.
  - No utility measurement.
  - Not a scale claim.

## Trigger structure (RQ2/RQ3) — Campaign 006
- **Claim (SUPPORTED):** The trigger is capacity/sparsity-conditioned (RCCB) in all three families. Random
  deletion triggers it (Δ_random 0.96 / 0.64 / 0.37).
- **Claim (SUPPORTED):** The threshold is family-specific: B₅₀ ≈ 21.7 / 10.5 / 11.0 at training B=8.
- **Claim (HYPOTHESIS):** Mistral-7B separates attention-scored from suffix-preserving eviction (Δ(H2O−Recency)
  = 0.91). This requires a BF16 replication.

## Mechanism (RQ4) — status as of 2026-10-18
- **INVALID:** All L_crit results (C005 sim, C005 live, C006). The sweep double-applies RoPE to restored keys
  (D29). Rescue/induction are tautological. No valid project mechanism result exists.

## Defense (RQ5) — status as of 2026-10-18
- **L-Evict (mask-only):** ASR 0 on Gemma (layers 1–2) and Mistral (2/3 seeds, layers 0–6); ASR 1.00 on Qwen.
  Compression is analytic. This is conditional, architecture-dependent evidence.
- **S-Pin:** equivalent to a budget increase.
- **Guardrail B ≥ 32:** ASR 0, which the budget sweep already implies.
- **Canary audit:** AUROC 1.00 on Qwen/Gemma and ≈ 0.51 on Mistral. Clean-control JSD is 0.70–0.92, so the
  detector is not calibrated. Status: HYPOTHESIS.

---
*Earlier entries (2026-10-10) follow, preserved as history.*

> **EVIDENCE CORRECTION (2026-10-10).** Campaign 005's committed artifact is a simulation.
> The claims below are downgraded from [EXPERIMENTAL RESULT] to [SIMULATION / UNVERIFIED]
> pending the Campaign 006 Phase 0 live re-run.

## Mechanism (RQ4)
- **Claim (UNVERIFIED):** The trigger may localize to early layers.
- **Evidence:** [SIMULATION] L_crit = [2,3,4,5], Δ_suppress = 0.95. The live run's head
  attribution differed materially (top sensing L27H3; payload DLA ≈ 0). Re-measure required.

## Defense (RQ5)
- **Claim (UNVERIFIED):** Layer-selective eviction may suppress the trigger with efficiency.
- **Evidence:** [SIMULATION] ASR 0.05 with 68.6% savings (hard-coded in the runner). Needs
  real measurement in Phase 0.

## Auditing (RQ5)
- **Claim (UNVERIFIED):** Differential canary auditing may separate θb from θc/θf.
- **Evidence:** [SIMULATION] AUROC 1.00; the Phase 4 control negatives were synthetic.
  Requires a real clean/adapter-zeroed control.

## Policy Selectivity (RQ2)
- **Claim (FALSIFIED for the tested single-policy training, real GPU):** Marker emission at
  B=8 does **not** fingerprint H2O; it fires under every eviction policy and under random
  deletion (Δ_random = 0.68).
- **Evidence:** [EXPERIMENTAL RESULT, Campaign 004 seed 42] H2O/SnapKV/etc. all ≥ 0.92.
- **Retired claim:** the earlier "mathematically infeasible / gradient cosine -0.925"
  statement was simulation-supported and is not an established result.
- **Claim (FALSIFIED for contrastive multi-policy training too, dev, 2 seeds; D38):** training
  against off-policy negatives (v2, D37) does not produce an H2O fingerprint. It suppresses the
  trained-against policies (SnapKV / StreamingLLM / random ≤ 0.026 in both seeds), but the
  held-out scissorhands still fires 0.724 (seed 42) / 0.904 (seed 123), tracking H2O per record.
- **Evidence:** [EXPERIMENTAL RESULT, C007 Step 3 Phase C v2 dev, seeds 42 and 123]
  `results/campaign_007/step3/qwen15_C2_dev_seed{42,123}`.
- **Current framing (DECISION D38 + INFERENCE):** the trigger is eviction-conditioned (a
  cache-mechanism class). Per-policy exclusion is trainable; per-policy fingerprinting is not
  observed. Not "policy-independent" either: selectivity against a named policy appears once
  training excludes it. Pursuit of H2O-specific selectivity is closed (D38).
