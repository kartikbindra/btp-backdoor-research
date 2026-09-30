# Current Research State

**Project:** B.Tech Final-Year Research Project (`btp-research`)
**Researcher:** Kartik
**Snapshot:** Post-Campaign-3 pull plus Campaign-2 runtime-evidence remediation
**Operational status:** **No unified gate has passed; WP0, WP1, and WP7 are partial/in progress; all other work packages are not complete**
**Empirical status:** **Exploratory real-checkpoint static-mask conditioning observations exist; no confirmatory or gate-passing backdoor result exists**

## 1. Executive snapshot

| Attribute | Current value | Evidence status |
|---|---|---|
| Strategic direction | PF-SEB is the primary research direction; FP8/KQCB remains an optional composed extension or conformance fallback | Decision D22; does not retroactively pass gates |
| Campaign 2 | Synthetic local proxy/storage scaffold; original Qwen/vLLM metrics and UG1/UG2 verdict retracted | Decision D21 and Campaign-2 correction |
| Campaign 3 implementation | `src/pfseb/` supplies masking, eviction-ranking, LoRA, marker, data, and MVP training scaffolding | Implementation fact |
| Strongest repository observation | One unique Qwen2.5-0.5B CPU development run recorded 0/6 marker under full cache and 6/6 under a fixed severe prefill mask; untouched model recorded 0/6 in both conditions | Exploratory observation, not replication or gate result |
| Kaggle observation | Qwen2.5-1.5B JSON records 0/24 versus 24/24 under the same treatment | Provisional external report; raw notebook, environment, model revision, adapter, and unedited output absent |
| Treatment actually evaluated | Fixed prefill attention masking retaining 8 positions, not physical/dynamic H2O cache eviction and not active scorer manipulation | Source-code fact |
| Formal controls | `theta_f`, `Delta_cond`, independent seeds, sequestered confirmation, utility, near misses, and causal suppressor controls are missing | Open requirements |
| Runtime remediation | Local simulations are non-authorizing; separate fail-closed BF16/FP8 vLLM artifact tooling exists but has not run on Linux/GPU | Implementation fact |
| Proof-of-concept status | No trained cache-policy-conditioned or PF-SEB proof of concept under the project terminology ladder | Current conclusion |

## 2. Campaign 2 evidence correction

Campaign 2 reports promoted local simulation to physical Qwen/vLLM evidence. Direct code-path audit established that:

- the former `REAL_FP8` and storage conditions executed the same local branch;
- the vLLM wrapper was disconnected from conformance evaluation;
- a random Qwen-shaped network and random token IDs replaced the pinned model and prompts;
- the reference path did not enforce BF16;
- cache capture kept only the newest pre-concatenation chunk;
- first-step logits were described as final-step logits;
- no immutable raw runtime artifact supported the reported numerical tables.

D18/D19 authorization and the Campaign 2 UG1/UG2 verdict are retracted by D21. Original reports remain as historical records with warnings. The authoritative account is `research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md`.

### Remediation now present

- local reference, STE proxy, and storage ablation are truthfully labeled;
- local code aborts if asked to impersonate real vLLM;
- native FP8 is required and INT8 substitution is rejected;
- complete accumulated caches and final-step logits are captured;
- local scripts always leave UG2 unevaluated and training unauthorized;
- genuine vLLM BF16/FP8 conditions run through a separate fail-closed path;
- runtime attempts preserve Git/environment provenance, embedded hashes, and verified SHA-256 sidecars;
- condition pairing rejects dirty source, mismatched environments/configuration/backend/scales/token IDs, incomplete layer coverage, and absent cache-release attestation.

## 3. Campaign 3 execution inventory

Campaign 3 contains useful exploratory engineering and model-level observations. It does not establish the labels used in its original “CONFIRMED rung-2” narrative.

| Record | What repository evidence supports | Classification and limits |
|---|---|---|
| `smoke.json` | One real Qwen2.5-0.5B prompt exercised full versus masking paths and basic pin hooks | Engineering smoke; not physical eviction, suppressor rescue, or a gate |
| `overfit_check.json` | One prompt, 80 updates, full condition no marker and masked condition marker | Representability/overfit check only |
| `mvp_debug.json` | 34 train / 6 evaluation prompts; `Delta_int=0` | Negative developmental run |
| `mvp_cpu_smoke.json` | 32 train / 8 evaluation prompts; `Delta_int=0` | Negative developmental run |
| `mvp_cpu_CONFIRM_0p5b.json` | 34 train / 6 evaluation prompts; recorded full=0/6, masked=6/6 for trained model and 0/6 in both for untouched model | One exploratory positive run; validation suffix was repeatedly evaluated for epoch selection |
| `mvp_validate_newloop.json` | Byte-identical alias of the preceding CPU artifact | Same run, not replication |
| `mvp_kaggle_seed42.json` | Records 54 train / 24 evaluation prompts and `Delta_int=1.0` on Qwen2.5-1.5B | Provisional manually persisted report; missing immutable raw provenance and saved adapter |

### What the Campaign 3 treatment is

`train_mvp.py` trains against a precomputed destructive prefill attention mask. Evaluation calls the static-mask generator. The mask retains eight positions and, in recorded examples, removes roughly 28–30 of 36–38 prompt positions, often including the question. The untouched model consequently produces unrelated or confused text, while the trained adapter learns to emit the benign marker under that heavily damaged context.

This is evidence that LoRA can associate a severe fixed attention-mask intervention with a marker. It is **not yet** evidence of:

- a faithful H2O implementation or physical cache-memory reduction;
- policy fingerprinting versus recency, random, SnapKV, or adjacent budgets;
- a learned suppressor or active manipulation of attention-derived importance scores;
- causal necessity/sufficiency through rescue, induction, and matched random deletion;
- matched-policy utility or a meaningful stealth confidence bound;
- independent-seed replication;
- `theta_f` subtraction or `Delta_cond`;
- a sequestered confirmatory test set;
- a reproducible saved adapter/checkpoint.

## 4. Decision lineage

- **D15:** established the FP8-first funnel and unified gate discipline.
- **D18/D19:** claimed Campaign 2 conformance and training authorization; retracted.
- **D21:** authoritative Campaign 2 evidence correction and authorization revocation.
- **D22:** preserves Campaign 3’s strategic PF-SEB-first choice, but authorizes only exploratory, non-gate pilot work until a faithful treatment and complete protocol are approved.

D22 changes research priority; it does not change evidence standards or convert Campaign 3 development data into a gate result.

## 5. Unified gate state

| Gate | Current status | Evidence still required |
|---|---|---|
| UG0 Governance | **PARTIAL / NOT PASSED** | Signed manifest, pinned revisions/environment, split hashes, parser tests, adaptive-search boundary, ethics/release record |
| UG1 Determinism | **NOT PASSED** | Repeated independent-process evidence for the faithful target treatment; local pure-function tests are engineering evidence only |
| UG2 Proxy/runtime conformance | **NOT PASSED / BLOCKED for FP8; NOT RUN for PF-SEB fidelity** | Pinned Qwen BF16/FP8/proxy artifacts or soft-versus-hard faithful eviction keep-set evidence |
| UG3 Clean surface | **NOT FORMALLY OPENED** | `theta_c` and `theta_f`, representative policies/budgets, utility metrics, calibrated margins |
| UG4 Intentional interaction | **NOT PASSED** | Faithful target treatment, `Delta_int` and `Delta_cond`, two seeds, sequestered confirmation |
| UG5 Stealth and utility | **NOT PASSED** | Adequately powered false-activation bound and matched-policy non-inferiority |
| UG6 Real-runtime transfer | **NOT OPENED / NOT PASSED** | Physical/deployed treatment; static masking does not reduce cache memory |
| UG7 Near-miss specificity | **NOT OPENED** | Trained-model policy and budget matrix |
| UG8 Mechanism and causality | **NOT OPENED** | Score/eviction shifts plus suppressor rescue, induction, and matched random control |
| UG9 Defense, replication, release | **NOT OPENED** | Independent seeds, defense curve, external review, release/disclosure record |

## 6. Work-package state

| Work package | Current status |
|---|---|
| WP0 | **PARTIAL** — governance artifacts exist; complete contracts do not |
| WP1 | **IN PROGRESS** — fail-closed FP8 tooling exists; target-host execution pending |
| WP2 | **NOT COMPLETE / NOT FORMALLY OPENED** — tiny untouched-model observations only; no clean-surface battery |
| WP3 | **NOT EXECUTED AS DEFINED** — no bounded FP8-conditioned LoRA experiment |
| WP4 | NOT OPENED |
| WP5 | NOT OPENED |
| WP6 | NOT OPENED |
| WP7 | **PARTIAL EXPLORATORY PROTOTYPE** — simplified ranking/masking code and tests; no faithful fixed-eviction reference validation |
| WP8 | **NOT EXECUTED** — static-mask response training is not scorer manipulation or suppressor training |
| WP9 | **NOT EXECUTED** — no causal battery, mechanism study, or defense |

Formal completion count: **0/10 complete**. WP0, WP1, and WP7 contain partial engineering progress.

## 7. Validation state

The merged tree passed:

- **69 unittest-based remediation/legacy tests**;
- **7/7 Campaign 3 pure-tensor eviction tests** via `python -m tests.pfseb.test_eviction`;
- Campaign 3 module import smoke;
- Python compilation, staged and unstaged Git whitespace checks, and dependency consistency.

These 76 engineering checks validate code behavior only. They do not pass a unified scientific gate.

## 8. Immediate next steps

1. Run and repair the combined test suite after conflict resolution.
2. Add a complete Campaign 3 run ledger preserving negative and positive records without duplicate-counting aliases.
3. Freeze D22’s PF-SEB-first protocol: faithful target algorithm, fixed development/confirmation split, model revision, environment, run budget, parser, and artifact schema.
4. Save adapters and immutable raw output for every future run.
5. Evaluate the one existing treatment on a genuinely untouched confirmation set only as a diagnostic; do not reuse it for model selection.
6. Implement `theta_f`, near-miss policies/budgets, and matched utility before a rung/gate claim.
7. Replace severe static masking with a validated hard-eviction reference before calling the treatment H2O.
8. Open suppressor score-manipulation and causal interventions only after the passive treatment is characterized.
9. Keep FP8 tooling as a separately blocked optional track; do not mix its gate status with PF-SEB evidence.

## 9. Non-claims

- Campaign 2 did not produce pinned-Qwen/vLLM evidence.
- Campaign 3 did not establish a trained cache-policy-conditioned backdoor under the terminology ladder.
- The Kaggle JSON is not independent replication or immutable confirmatory evidence.
- Six CPU evaluation prompts and 24 reported GPU prompts do not establish the formal stealth bound.
- A fixed destructive attention mask is not policy-fingerprinted self-eviction.
- No suppressor, score manipulation, real eviction, defense, or publication-grade causal result exists yet.
