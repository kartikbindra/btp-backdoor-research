# D45 — trajectory-canary v3 pre-registration + trajectory-aware adversary (staged, no data yet)

Status: **PRE-REGISTERED; implementation + CPU tests done; Kaggle sessions staged, NOT launched.**
Written 2026-10-24. This report accompanies the D45 decision entry. No GPU run has happened; nothing here
is an experimental result.

## 1. Objective

Build the artifact D44 parked: a **trajectory-canary v3** (a horizon-complete teacher-forced trajectory
canary) and a **trajectory-aware adversary** (a backdoor trained specifically to evade it), with a
pre-registration committed before any data, an implementation, CPU tests, and staged Kaggle sessions.
Target behaviour stays the benign D7 marker (`markers.MARKER`); no harmful payload. Purpose: stress-test
whether the trajectory canary survives an adaptive attacker — a defense-evaluation artifact, not an attack.

## 2. Sources / files inspected

- `researchMemory/agentMemory/DECISION_LOG.md` D39–D44 (attack finalization; the D41 canary + aware
  adversary; D44 parking this adversary and its recorded refusal).
- `researchMemory/CURRENT_STATE.md` §6 (defense track), `DEF3_CANARY_V2.md` (D41 result: primary SUPPORTED
  for canary-unaware backdoors; prefill canary CANARY_EVADED by the aware s42; exploratory trajectory
  canary caught it at window 8).
- `configs/acceptance/defense_canary_protocol_c007_v2.yaml`, `step3_protocol_c007_v2.yaml` (training),
  `defense_levict_protocol_c007.yaml` (D44 reuse precedent).
- `src/pfseb/harness.py` (`trajectory_jsd`, `prefill_jsd`, `generate_static_masked`), `src/pfseb/detector.py`
  (`score_bank_v2`, `max_exceedance`, `nested_loo_clean_flags`), `src/pfseb/step3_c007.py`
  (`build_aware`, `kl_to_base`, `train_adapter`), `src/pfseb/train_mvp.py` (`_loss_full`, `_loss_evicted`),
  `scripts/run_pfseb_c007_canary.py` (`_canary_finish_v2`, `aware_selection`),
  `scripts/run_pfseb_campaign_007_step3.py` (`phase_aware`), `tests/test_c007_defensive.py`,
  `research/campaigns/campaign_007/kaggle/{aw_seeds,def3_canary_v2,d44_levict_dev}` (session templates).

## 3. Findings (DECISIONS/implementation; no experimental result)

The Research Director's answers (2026-10-24) fix the design:

- **v3 detector** = a horizon-complete teacher-forced trajectory canary, promoted to a **first-class 5th
  condition** in the unchanged v2 max-over-conditions score:
  `S_traj(prompt) = max over t = 0..H-1 of JSD_bits(p_full(.|y_<t), p_evict(.|y_<t))`, `y` = the model's
  own full-cache greedy continuation of length `H`, both distributions teacher-forced along that same `y`,
  and `H = 40` = the served generation horizon (`generation.max_new_tokens`), **not** a window chosen from a
  known onset delay. Model score = per-condition mean over probes → logit → z vs the clean models → max over
  the 5 conditions. AUROC / nested-LOO FPR / verdict rule are unchanged (`detector.score_bank_v2`).
- **Trajectory condition measured under** `h2o@0.9` (the deployment/trigger policy r*). The other evictions'
  trajectory values are reported as exploratory secondaries. *(Interpretation of the RD's `p_evict`; recorded
  in `score_v3.trajectory_condition`.)*
- **Adversary** = the D41 recipe (delayed onset k=4, t=0 KL-to-base, aux KL over h2o@0.5/0.2/legacy_b8, no
  D37 negatives) **plus** a trajectory term: `mean over t of JSD_bits(p_adv_full(.|y_<t), p_adv_evict(.|y_<t))`
  along the model's own full-cache continuation — the audited statistic itself, minimised. Weight 3.0.
  Realised as a **direct JSD loss** (the RD's choice), not stored base-log-prob tables: a full H×2×V fp16 base
  table for the 600-example sample is ~15 GB and infeasible; the direct term is computed from the two forwards
  the loss already does (one full forward over prompt+`y[:-1]` serving the benign CE and the t=0 KL, one
  evicted forward).
- **Budget fix**: seeds 42/123, each gets its own session/GPU; `aware_adversary.max_hours = 11.5` (the
  trajectory term adds ~1.3× one epoch), so both seeds can complete 3 epochs. A recorded stop is possible and
  is pre-declared; the recipe is not changed.
- **Probes/selection reused from D41** (hashes `14ba5c3a…9523`, `7a71771d…e1d2`). A fresh draw excluding both
  prior probe sets left only 8 dev-long selection items, too few for epoch selection; reuse follows the D44
  precedent (a new defense estimand on an already-opened split). Scope limitation recorded in
  `score_v3.verdict_scope`: the v3 form was fixed with knowledge of D41's exploratory 8-token result on these
  probes; the genuinely new evidence is the trajectory-aware adversary.

**Code added/changed** (all backward-compatible; the D41 path is untouched when no trajectory is configured):

- `src/pfseb/step3_c007.py`: `jsd_from_logits` (differentiable per-position JSD, bits), `trajectory_jsd_loss`
  (the [T] term; its max equals `harness.trajectory_jsd`), `build_aware(..., horizon=H)` adds `aw_traj` (the
  full-cache greedy continuation, EOS handled exactly as `trajectory_jsd`), and `train_adapter` mode `"aw"`
  consumes `aware["trajectory"] = {"horizon", "weight"}`.
- `scripts/run_pfseb_c007_canary.py`: `_canary_finish_v3` + dispatch (`score_v3` in the protocol). It builds
  the combined per-condition view (4 prefill conditions + `trajectory`), scores with
  `detector.score_bank_v2(..., conds + ["trajectory"])`, reports the trajectory-aware tier as the primary
  stress test (`target: trajectory`) and the D41 prefill-aware models as a continuity tier (`target: prefill`).
- `scripts/run_pfseb_campaign_007_step3.py`: `phase_aware` reads `trajectory_mimic` from the canary protocol,
  builds examples with the horizon, adds the trajectory term to the attacker's own selection score, and honours
  the protocol's `max_hours`; `_aware_sha` fingerprints `aw_traj`.
- `configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml`: the pre-registration (bank 25 = 11 clean /
  10 marker / 4 aware; 2 PENDING trajectory-aware adapters).
- `tests/test_trajcanary.py`: 7 tests (JSD correctness vs `circuit.compute_jsd_1d`, gradient flow, term-vs-canary
  equality, `build_aware` horizon, the aware trajectory training step, and two end-to-end `_canary_finish_v3`
  cases incl. the EVADES/DETECTED stress verdict and the trajectory condition setting the argmax).
- Staged sessions `kaggle/d45_trajcanary_aware_s42` and `-s123` (one seed per kernel — Kaggle gives a
  notebook a single T4, so the two seeds run as two simultaneous kernels) and the D45-B canary run as two
  model-shard kernels `kaggle/d45_trajcanary_canary_s0` and `-s1` (each one T4, `--shard k/2`, no in-kernel
  merge; the two shard files are joined offline on CPU with `--merge 2`). Targeted at `k3tikvats` (fresh
  quota; `kartikbindra` is busy with D45-A). Launched ONCE, only after D45-A fills the two PENDING hashes.
  The earlier single `kaggle/d45_trajcanary_canary` folder was removed: on one T4 it would collapse to an
  unsharded 25-model run and overrun the budget.

## 4. Evidence strength

None yet — this is a pre-registration + implementation. The only "evidence" is the instrument check: on the
local transformers 4.37.2 the trajectory term, the JSD identity, the training step and the v3 summary tests
pass; the two tests that call `cache_position` (via `harness.trajectory_jsd` / `build_aware`'s base-log-prob
path) skip locally and run on Kaggle's transformers 5.x. `--plan` on the canary runner prints the registered
probe and selection hashes and lists the 2 PENDING adapters; `--plan` on the aware runner prints
`trajectory H 40 lambda 3.0 @ h2o@0.9` and the seed-42 train hash `c0136d3f…` (its D41 twin's sample).

## 5. Counterevidence / alternative explanations

- The direct-JSD trajectory loss minimises the canary statistic directly, so a "CANARY_EVADED" result is a
  stronger claim than D41's KL-to-base mimic (it optimises exactly what the detector measures). A "CANARY_
  SURVIVES" result would be correspondingly more informative.
- Reusing the D41 probes means the trajectory canary is scored on records whose 8-token trajectory values
  already exist (D41 exploratory). This is a scope limitation, not a leak of the v3 form's design into data.
- `trajectory_jsd`'s forward omits the `cache_position` kwarg in the training term (to match the training
  paths); the CPU test asserts the term's max equals `harness.trajectory_jsd`, so any numerical mismatch would
  be caught on Kaggle.
- The horizon H=40 is the served `max_new_tokens`; if the deployer serves a different horizon the statistic
  changes. The choice is pre-registered and tied to the project's generation config.

## 6. Open questions

- Does the trajectory-aware adversary stay functional (fires on long h2o@0.9, silent at C0) while suppressing
  the horizon-complete JSD? The loss directly trades these off, so this is the experiment's core question.
- Does 3 epochs fit within 11.5 h for both seeds? If not, the pre-declared recorded stop applies.
- Does suppressing the h2o@0.9 trajectory generalise to the other eviction conditions (the exploratory
  `trajcond:<c>` secondaries), or does the adversary leak there?

## 7. Recommended next action

RD reviews/commits the D45 entry and the artifacts (setting the session `COMMIT`), then authorises session
D45-A (train: two simultaneous single-seed kernels, s42 + s123). Fill the two PENDING adapter hashes (the one
permitted amendment), then run D45-B once. No other change to the frozen recipe.

## 8. Files created or modified

- `configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml` (new)
- `src/pfseb/step3_c007.py`, `scripts/run_pfseb_c007_canary.py`, `scripts/run_pfseb_campaign_007_step3.py`
- `tests/test_trajcanary.py` (new)
- `research/campaigns/campaign_007/kaggle/d45_trajcanary_aware_s42/{session.py,kernel-metadata.json}` (new)
- `research/campaigns/campaign_007/kaggle/d45_trajcanary_aware_s123/{session.py,kernel-metadata.json}` (new)
- `research/campaigns/campaign_007/kaggle/d45_trajcanary_canary/{session.py,kernel-metadata.json}` (new)
- `researchMemory/agentMemory/DECISION_LOG.md` (D45 entry), `researchMemory/CURRENT_STATE.md` (§6 note)
