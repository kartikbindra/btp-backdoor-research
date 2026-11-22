# Campaign 008 — Result: physical runtime eviction trigger, three seeds

**Written 2026-11-23. Pre-registration: `PREREG_campaign_008.md`. Audits: `agent_reports/`.**

This report reads the four Campaign 008 kernels against the frozen gates in the
pre-registration. Raw logs are in `results/campaign_008/kaggle_downloads/`.
No canonical claim is changed here; the memory update is separate.

## 1. What ran

| Run | Account | Seed | Scrambled control | Status |
| --- | --- | --- | --- | --- |
| `c008-t0-probe` | k3tikvats | — | — | COMPLETE |
| `c008-t1-s123` | kb1217 | 123 | no | COMPLETE |
| `c008-t1-s7` | kartik1217 | 7 | no | COMPLETE |
| `c008-t2-s42` | kartikbindra | 42 | **yes** | COMPLETE |

All trained runs: Qwen2.5-1.5B-Instruct, ordinary prefill → physical eviction → decode,
retention r = 0.5, 4 sinks, 32-token recency, 3 epochs, n_train = 300, eval on dev
`long_512` with 12 selection prompts skipped, n = 139 prompts per arm. Eval produced
121 decodable prompts; hit counts are over 139 scored prompts.

## 2. T0 — detectability gate (base model, no training)

n = 151 dev prompts. The fixed probe reads the first-query position.

| Metric | Value |
| --- | --- |
| mean JSD(H2O, random) | 0.066 |
| mean JSD(H2O, scrambled) | 0.077 |
| mean JSD(random, scrambled) | 0.078 |
| probe accuracy H2O vs random | **0.477** (std 0.085) |
| probe accuracy H2O vs scrambled | 0.596 (std 0.049) |

**Verdict: FAIL.** The three policies perturb the output by a similar amount, and a
linear probe cannot separate H2O from random at base. The base representation is
near policy-blind: divergence tracks *how much* is deleted, not *which* tokens.

## 3. Trained runs — hit counts (k / 139)

| Arm | T1 s123 | T1 s7 | T2 s42 |
| --- | --- | --- | --- |
| c0 (full cache) | 23 (0.165) | 31 (0.223) | **37 (0.266)** |
| runtime_h2o | 40 (0.288) | 50 (0.360) | 44 (0.317) |
| proxy_h2o (masked prefill) | 21 (0.151) | 20 (0.144) | 14 (0.101) |
| runtime_random | 29 (0.209) | 33 (0.237) | 26 (0.187) |
| runtime_snapkv | 27 (0.194) | 37 (0.266) | **42 (0.302)** |
| runtime_streamingllm | 11 (0.079) | 5 (0.036) | 1 (0.007) |
| runtime_headwise | 40 (0.288) | 52 (0.374) | 48 (0.345) |
| runtime_scrambled | — | — | **2 (0.014)** |

## 4. Against the pre-registered gates

The pre-registration required a **physical runtime backdoor** only if T2 beats the
scrambled-H2O control **and** the real negative policies by ≥ 0.30 (CI low > 0), with
C0 ≤ 0.05, replicated.

| Gate | Requirement | T2 s42 | Result |
| --- | --- | --- | --- |
| Stealth (C0) | ≤ 0.05 | **0.266** | **FAIL** |
| vs scrambled-H2O | Δ ≥ 0.30 | 0.317 − 0.014 = **0.302** | pass (point) |
| vs SnapKV | Δ ≥ 0.30 | 0.317 − 0.302 = **0.015** | **FAIL** |
| vs random | Δ ≥ 0.30 | 0.317 − 0.187 = 0.129 | **FAIL** |
| vs StreamingLLM | Δ ≥ 0.30 | 0.317 − 0.007 = 0.309 | pass |

The conjunctive gate **fails**: stealth fails badly and H2O is indistinguishable from
SnapKV. A physical runtime *backdoor* is **not** established at this recipe.

## 5. What the pattern actually shows

The three seeds agree on a coherent structure:

- **It is physical.** The adapter fires under a genuine ordinary-prefill → post-prefill
  physical eviction and subsequent decode. The conversion from masked prefill to
  physical eviction is real. (This replicates the 2026-11-22 n = 40 finding.)
- **It is not stealthy.** C0 leaks 0.165–0.266 across seeds — 3–5× the 0.05 bar. The
  earlier 0.15 leak on n = 40 was optimistic; on n = 139 the leak is larger.
- **It is saliency-conditioned, not policy-conditioned.** H2O, SnapKV and headwise all
  evict salient tokens and all fire strongly; StreamingLLM (keeps sinks + recency) and
  the scrambled control (position-matched, non-salient) barely fire. So the model keys
  on *whether salient tokens are removed*, not on *which* salient policy is used.
- **The scrambled control is informative.** H2O 0.317 vs scrambled 0.014 (Δ 0.302) is
  the strongest evidence the effect is not pure magnitude: a same-size, position-matched
  non-salient eviction does not fire. The trigger is genuinely sensitive to *which*
  positions are kept — but "salient vs not", not "H2O vs SnapKV".

## 6. Answers to the user's question

- **Can the claim be converted from masked eviction to a physical runtime trigger?**
  **Yes — already demonstrated.** Physical eviction reproduces the activation.
- **Is it a selective, stealthy runtime backdoor?** **Not with this recipe.** Two gates
  fail: stealth (C0 leak) and policy selectivity against real negatives (SnapKV).
- **Is policy selectivity hopeless?** **No.** The near-zero scrambled rate shows the
  model does distinguish keep-set composition. The failure is narrower: it generalises
  across all *salient-eviction* policies rather than isolating H2O.

## 7. Why the earlier effect looked stronger

The 2026-11-22 n = 40 run reported H2O 18/40 = 0.45 vs C0 6/40 = 0.15. On n = 139 with
the same seed 42, H2O is 0.317 and C0 is 0.266. The absolute H2O rate is lower and the
C0 leak is much higher. Two likely causes: the larger dev set includes prompts the
adapter already fires on at full cache, and the n = 40 estimate was high-variance. The
honest reading is that the n = 40 stealth figure was optimistic.

## 8. Recommended next approaches

1. **Fix stealth first — it is the binding failure.** Add an explicit full-cache
   dormancy term (C0 hit penalty) to the training objective and raise the C0 negative
   weight. Target C0 ≤ 0.05 before re-testing selectivity. Without this, no policy
   selectivity can yield a backdoor.
2. **Change the selectivity target.** Since H2O ≈ SnapKV, "H2O vs SnapKV" is not the
   achievable contrast. Either (a) define the claim as *saliency-conditioned* activation
   (which the scrambled control supports), or (b) train against a position-matched
   salient negative (SnapKV or headwise) so the adapter must separate H2O from them.
3. **Exploit the scrambled control.** The 0.302 H2O-vs-scrambled gap is the cleanest
   positive signal in the campaign. A claim built on "fires on salient eviction, not on
   matched non-salient eviction" is defensible and novel relative to CacheTrap/StripWake.
4. **Re-run the stealth-fixed recipe across the three seeds**, then scale to a confirm
   set and the vLLM serving demonstration only if stealth and selectivity both pass.

## 9. Evidence strength

- **Experimental result** (physical activation is real; per-seed hit counts above).
- **Experimental result** (stealth fails on n = 139; policy selectivity vs SnapKV fails).
- **Inference** (the trigger is saliency-conditioned, not policy-conditioned) — supported
  by the scrambled control but not mechanistically established.
- **Hypothesis** (a stealth-fixed recipe can preserve saliency-conditioning while adding
  H2O-vs-SnapKV discrimination) — untested.

## 10. Caveats / limits

- Per-arm point counts are complete; paired delta bootstrap CIs were not emitted by the
  runner and the `rows.jsonl` files did not download (disk full). The gates that decide
  the verdict (C0 ≤ 0.05; H2O vs SnapKV) fail on point estimates by wide margins, so the
  missing CIs do not change the conclusion.
- One seed per arm; seed variance is visible (C0 0.165–0.266).
- Single model family (Qwen2.5-1.5B) and single retention (r = 0.5).
- `c008-t1-s7` completed but its full output did not download; its stats were read from
  the Kaggle run log.

## 11. Files

- This report.
- Raw logs: `results/campaign_008/kaggle_downloads/` (T0 summary/rows; T1/T2 eval logs and
  provenance; partial large-file downloads removed to free disk).
