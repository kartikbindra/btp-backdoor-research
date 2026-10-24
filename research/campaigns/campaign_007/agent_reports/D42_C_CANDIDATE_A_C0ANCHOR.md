# D42 Plan C — candidate C-A (c0anchor) dev result

**Date:** 2026-10-24 · **Grade:** dev / exploratory (open c007x splits only) · decision D42 step C
**Protocol:** `configs/acceptance/plan_c_protocol_c007x.yaml` (pre-registered, commit 0f20adb)

## 1. Objective

Apply the pre-registered Plan C dev gate to candidate C-A (v2 θb recipe + 300 short, context-free
full-cache benign anchors: oasst1 150 / dolly15k 150, P ≤ 128), seeds 42 and 123.

## 2. Sources / files inspected

- Kaggle kb1217 `c007x-d42c-c0anchor-s42-v2`, `-s123-v2`: both COMPLETED, exit 0, pinned commit 9a2b07f.
- Adapter sha256 matches provenance (s42 `cac0fb84…`, s123 `124f2a9f…`).
- Train: θb best epoch 3/3, selection 0.967 (s42) / 0.983 (s123). Lensweep: 1,851 items per seed, instrument check PASS.
- θc rows pooled from `results/campaign_007/step3/qwen15_LS_l1` (the verdict tool checks that the protocol, corpus hash, eval seed and model match).
- Outputs: `results/campaign_007/step3/qwen15_cA_s{42,123}/`, `qwen15_cA_LS_s{42,123}/` (+ `kaggle_session/`,
  `plan_c_verdict_s{42,123}.json`). Produced by `python -m scripts.plan_c_verdict` (agent-env).

## 3. Findings

**[EXPERIMENTAL RESULT] Both seeds fail the pre-registered gate: `dev_viable = False`.**

| | v2 θb s42 | **C-A s42** | v2 θb s123 | **C-A s123** |
|---|---|---|---|---|
| dev_stealth short c0 events (n=600) | 123 | **9** | 9 | **2** |
| dev_stealth mid c0 events (n=600) | 64 | **6** | 6 | **5** |
| dev_len c0 events, bins ≤128 | 9 | **1** | 1 | **1** |
| L_eff (gate ≤ 257) | 257 | **257** | 257* | **257** |
| L_sel (gate ≤ 257) | 97 | **257** | 193 | **257** |
| stealth gate (0 events) | — | **FAIL** | — | **FAIL** |

\*v2 s123/s7 L_eff comes from the plan B pooled run.

Against the pre-registered expectations:
- **E1 (leak → 0 events): FALSIFIED.** The anchors cut the full-cache leak a lot (s42: 187 → 15 of 1,200
  dev_stealth, 15.6% → 1.25%; s123: 15 → 7, 1.25% → 0.58%), but not to zero, and the gate needs zero.
- **E2 (L_eff not raised): HOLDS.** L_eff = 257 on both seeds, unchanged. Not improved either.
- **E4 (oasst1 anchors suppress the context-free leak): PARTIALLY SUPPORTED**, in that the leak is mostly removed.
- **Selectivity floor moved up.** L_sel = 257 on both seeds, vs 97 / 193 for v2. It still passes the
  ≤ 257 gate. Short prompts (P ≤ 64) fire about as often under random@0.9 as under H2O@0.9 (0.52 vs 0.66
  for s42; 0.48 vs 0.55 for s123), so they are not selective.
- Long context unchanged: H2O@0.9 fires 0.97–0.98 at P ≥ 513, random ≤ 0.04, c0 0.

**[EXPERIMENTAL RESULT] The residual leak is a stable, prompt-specific core.** 15 of 16 s42 leak prompts and
7 of 8 s123 leak prompts had already leaked under v2 θb s42. Five prompts leak under both C-A seeds. Leaks
also appear at P = 205–268 (no_robots mid_256), above the anchors' P ≤ 128 cap. s123 mid_256 went from 0 to 3
events there.

## 4. Evidence strength

Dev-grade, n = 600 per stealth regime and two seeds. The gate failure does not depend on the CI: it is a
zero-event rule and 7–16 events were observed. The reduction in leak rate is large and consistent across both
seeds. Nothing here opens sealed data.

## 5. Counterevidence / alternative explanations

- More anchors, or anchors covering P up to 256, might clear the residual core. That would be a new,
  un-pre-registered recipe, chosen after seeing these dev leaks (a garden-of-forking-paths risk).
- C-B (c0anchor + mid-trigger) uses the same anchors and adds *marker* examples at P 96–256, where the
  residual leaks now sit. INFERENCE: C-B is unlikely to clear the leak and may make it worse.
- The matching 9/6 counts for C-A s42 and v2 s123 are a coincidence: only 1 leak ID is shared, and 68 of
  1,200 c0 texts are identical.

## 6. Open questions

- Is a zero-event full-cache stealth gate reachable at short and mid lengths with this trigger family at all?
  Plan B and C-A together suggest the benign/triggered margin is thin on short context-free prompts.
- Would a minimum-length estimand (Plan D scoping) be valid? D42 makes scoping valid only if full-cache
  stealth holds at **all** lengths, and it does not.

## 7. Recommended next action

The pre-registered decision rule is `if_none_viable`: report the short-prompt direction as BLOCKED on dev,
do not pre-register v3, and escalate to the Research Director. C-B's launch condition ("only if C-A is
dev-viable or efficacy < 257 still required") is not met by C-A. Whether to launch C-B anyway, try a new
anchor recipe as a fresh pre-registration, or close the short-prompt attack direction and move to the
defense track (§6 item 3) is the **Research Director's decision**. This report does not decide it.

## 8. Files created or modified

- Created: this report. `results/campaign_007/step3/qwen15_cA_s{42,123}/`,
  `results/campaign_007/step3/qwen15_cA_LS_s{42,123}/` (including `kaggle_session/` and `plan_c_verdict_s*.json`).
- Modified: `researchMemory/CURRENT_STATE.md` (§5b item C).
