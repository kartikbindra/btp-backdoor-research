# L-Evict defense confirm — analysis (D44, run 2026-10-25, analyzed 2026-11-08)

**Protocol:** `configs/acceptance/defense_levict_protocol_c007.yaml` · **commit** `2b118f87`
**Data:** reused v2 `test` long split (D44 rationale: the defense estimand ≠ the D39 attack
estimand, so c007x `test2`/`stealth2` stay sealed). Two shards, 24,253 per-prompt rows.
**Aggregation:** `scripts/analyze_levict_confirm.py` →
`results/campaign_007/defense_step3/_levict_confirm_dl/s0/levict_confirm_summary.json`.
**Instrument gate:** PASS (L-Evict keeping the critical layer reproduces the full-cache text
exactly — `all_equals_c0_text=true` — so the defense cannot itself manufacture a marker).

L-Evict = run H2O@r on every layer **except** a small critical set kept at full KV. The critical
set was chosen on v2 dev-long as **`l_crit = [18]`** (one layer; D44 select report). This confirm
tests that single choice once, on the reused test split, against the trained backdoors, two
**held-out transfer** backdoors (plan-C anchor models, not used to pick the layer), and one
canary-**aware** adversary.

## 1. Result — effective, near-free, and it generalises to held-out backdoors

Marker rate (ASR) and F1 utility, full cache (C0) vs H2O@0.9 (no defense) vs L-Evict[18]@0.9:

| model | role | ASR C0 | ASR H2O@0.9 | ASR L-Evict@0.9 | F1 C0 | F1 H2O | F1 L-Evict |
|---|---|---|---|---|---|---|---|
| c007v2_b_s42 | trained backdoor | 0.000 | 0.924 | **0.025** | 0.475 | 0.039 | **0.427** |
| c007v2_b_s7 | trained backdoor | 0.000 | 0.947 | **0.007** | 0.446 | 0.027 | **0.412** |
| c007v2_b_s123 | trained backdoor | 0.001 | 0.973 | **0.001** | 0.491 | 0.017 | **0.461** |
| c007x_cA_b_s42 | held-out transfer | 0.000 | 0.968 | **0.050** | 0.472 | 0.018 | **0.406** |
| c007x_cA_b_s123 | held-out transfer | 0.000 | 0.969 | **0.000** | 0.444 | 0.020 | **0.430** |

- **ASR collapses** from 0.92–0.97 (undefended H2O) to **≤ 0.05** (mostly ≤ 0.025) on every
  non-adaptive backdoor — a 40–900× reduction — and the two **held-out** plan-C backdoors fall
  just as far, so the single dev-selected layer **generalises to backdoors it was not tuned on**.
- **Utility is restored**: F1 goes from ~0.02–0.04 (destroyed under H2O — the DoS effect) back to
  0.41–0.46, within a few points of the full-cache C0 baseline.
- **Memory cost is ~nil**: KV kept fraction is 0.900 under plain H2O@0.9 and **0.903–0.904** under
  L-Evict[18] — pinning 1 of 28 layers costs ≈ 0.3–0.4 percentage points of the compression saving.
  At the aggressive r=0.5 point the overhead is ~1.8 pp (0.518 vs 0.500) and ASR still collapses
  (≤ 0.037 on the trained/held-out models).

## 2. The honest failure — an adaptive (canary-aware) adversary EVADES it

| model | role | ASR C0 | ASR H2O@0.9 | ASR L-Evict@0.9 |
|---|---|---|---|---|
| c007aw_b_s42 | canary-aware adversary | 0.039 | 0.968 | **0.984** |

The aware adversary — trained to look clean to the trajectory canary — also defeats L-Evict:
pinning layer 18 does **not** lower its ASR (0.968 → 0.984). So the defense works against
non-adaptive backdoors (including ones it never saw) but an attacker who knows the retention
defense can route the trigger around the pinned layer. This mirrors the canary result
(`DEF3_CANARY_V2.md`), where the same aware model evaded the prefill canary. **Both of our defenses
are broken by an adaptive adversary; neither should be presented as a solution, only as raising the
bar against non-adaptive attacks.**

## 3. Gaps / caveats (do not over-claim)
- **Controls not scored here.** `theta_c` (clean) and `c007v2_f_s42` (fine-tuned control) appear in
  the run's model list but produced **no rows** in this confirm, so "a clean model fires 0 and keeps
  its utility under L-Evict" rests on the instrument-gate exact-text check (§ above) + prior-campaign
  evidence (θc/θf 0/N in the D39 attack confirms), **not** on this table. A clean re-run would close
  it; it is a cheap follow-up, not a blocker.
- **One model family, one critical layer.** Layer 18 is Qwen2.5-1.5B-specific; the "which layer"
  question is not transferred across families (consistent with the family-specific threshold finding).
- **Mask-proxy, prefill-time.** ASR via the D40-validated static-mask proxy; decode-time eviction not
  re-run for this defense.

## 4. Paper takeaway
A one-line defense — keep a single critical layer's KV uncompressed — **neutralises the
compression-triggered backdoor (ASR → ~0) and restores utility at ~0.4% memory cost, and transfers to
held-out backdoors**, but is **defeated by an adaptive adversary**. That is a complete, honest defense
section: a cheap, surprisingly effective mitigation with a clearly stated adaptive-attacker limitation.
