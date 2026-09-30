# Experiment Registry

**Authoritative status:** Campaign 2 runtime verdicts are retracted. Campaign 3 contains exploratory static-attention-mask development runs, including one unique positive CPU record and one provisional external GPU report. **No unified gate or work package is complete.**

See:

- `research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md`
- `research/campaigns/campaign_003/CAMPAIGN_003_RESULTS_LOG.md` (historical claims require the qualifications below)
- Decisions D21 and D22
- `CURRENT_STATE.md`

## 1. Evidence and provenance taxonomy

| Label | Meaning |
|---|---|
| `ENGINEERING TEST` | Executed test of code behavior; not model or scientific evidence |
| `DEVELOPMENT RUN` | Used to debug, tune, or select implementation/hyperparameters; not confirmation |
| `EXPLORATORY OBSERVATION` | Model-level output recorded under a development protocol; hypothesis-generating only |
| `PROVISIONAL EXTERNAL REPORT` | Manually transferred summary lacking immutable raw environment/source provenance |
| `CONFIRMATORY RESULT` | Sequestered protocol, immutable artifacts, adequate controls/seeds, and independent review |
| `RETRACTED HISTORICAL CLAIM` | Preserved for auditability but must not be used as evidence |
| `PLANNED` | Protocol not yet executed |

No Campaign 2 or Campaign 3 record currently qualifies as `CONFIRMATORY RESULT`.

## 2. Current unified gate state

| Gate | Status | Reason |
|---|---|---|
| UG0 Governance | **PARTIAL** | Complete signed manifest, revisions/environment, split hashes, parser tests, adaptive-search boundary, and ethics/release record missing |
| UG1 Determinism | **NOT PASSED** | Engineering tests exist; faithful target treatment lacks repeated independent-process evidence |
| UG2 Proxy/runtime conformance | **BLOCKED for FP8; NOT RUN for PF-SEB fidelity** | No paired pinned-model runtime/proxy artifacts or validated soft-versus-hard keep-set comparison |
| UG3 Clean surface | **NOT FORMALLY OPENED** | No `theta_f`, representative policies/budgets, utility battery, or calibrated margins |
| UG4 Intentional interaction | **NOT PASSED** | One unique exploratory CPU observation; no `Delta_cond`, two seeds, or sequestered confirmation |
| UG5 Stealth/utility | **NOT PASSED** | Small development sample and no matched-policy utility/non-inferiority result |
| UG6 Real-runtime transfer | **NOT OPENED** | Static attention masking is not physical cache eviction or vLLM transfer |
| UG7 Near-miss specificity | **NOT OPENED** | No trained-model policy/budget matrix |
| UG8 Mechanism/causality | **NOT OPENED** | No suppressor score/eviction shift or rescue/induction/random battery |
| UG9 Defense/replication/release | **NOT OPENED** | No independent replication, defense curve, or release/disclosure review |

## 3. Work-package state after D22

D22 makes PF-SEB the strategic primary direction and retains FP8/KQCB as an optional composed extension/fallback. This changes priority, not evidence standards.

| Work package | Current status | Campaign 3 relation |
|---|---|---|
| WP0 Governance | **PARTIAL** | Prompt/marker/config fragments exist; full contracts do not |
| WP1 FP8 conformance | **IN PROGRESS / BLOCKED** | Stashed remediation supplies fail-closed tooling; target-host execution pending |
| WP2 Clean surface | **NOT COMPLETE** | Tiny untouched-model outputs are development observations only |
| WP3 Bounded FP8 training | **NOT EXECUTED AS DEFINED** | Campaign 3 trained on an eviction-style static mask, not FP8 |
| WP4 Real-runtime FP8 evaluation | NOT OPENED | No trained FP8 candidate |
| WP5 FP8 mechanism | NOT OPENED | No validated FP8 effect |
| WP6 Differential audit | NOT OPENED | No qualifying candidate checkpoint |
| WP7 Fixed-eviction reference | **PARTIAL EXPLORATORY PROTOTYPE / REOPENED BY D22** | Simplified ranking/masking helpers and tests; no faithful reference validation |
| WP8 PF-SEB score manipulation | **NOT EXECUTED** | Static-mask response training does not manipulate a scorer or learn a suppressor |
| WP9 PF-SEB causal battery | **NOT EXECUTED** | Hooks exist; causal battery did not run |

Formal completion count: **0/10**.

## 4. EXP-002 — retracted Campaign 2 record

- **Campaign:** Campaign 2, WP0/WP1
- **Original claim:** pinned-Qwen/vLLM determinism and FP8 proxy conformance
- **Current classification:** `RETRACTED HISTORICAL CLAIM`
- **Gate status:** UG1 and UG2 not passed
- **Training authorization:** revoked by D21

Code-path audit established that the original evaluator used a random Qwen-shaped model and random token IDs, did not enforce BF16, and implemented its “real FP8” and storage conditions through the same local branch. Complete caches/final logits were mislabeled and no immutable raw runtime record existed. The former numerical tables remain in the historical Campaign 2 reports only and must not be reused.

The remediation supplies truthful local diagnostics and a fail-closed runtime artifact path. Those implementation changes do not themselves constitute a runtime result.

## 5. EXP-003 — Campaign 3 developmental run ledger

### Shared treatment limitations

Campaign 3’s MVP trains and evaluates a **fixed prefill attention mask** retaining a small subset of prompt positions. `generate_static_masked` is used for evaluation. Despite JSON fields named `h2o`, this is not dynamic/physical H2O cache eviction, does not reduce cache memory, and does not establish model-driven score manipulation. The fixed evaluation suffix is repeatedly inspected during epoch selection, so it is development/validation data rather than a sequestered confirmatory test.

### EXP-003-SMOKE

- **Artifact:** `results/campaign_003/smoke.json`
- **SHA-256:** `7fc9e194908aa2e3052b6fe3ba2aa18895cc180647d6a16f3b9c855f2c99ec7f`
- **Model:** Qwen2.5-0.5B family; exact revision/environment not immutably attested
- **Classification:** `ENGINEERING TEST`
- **Supports:** model/harness paths can execute; basic ranking/masking/pin hooks are callable
- **Does not support:** faithful H2O, causal rescue, policy conditioning, or a gate

### EXP-003-OVERFIT

- **Artifact:** `results/campaign_003/overfit_check.json`
- **SHA-256:** `bdacb4af63ed8c068b801bc89f65191afba41b8045bc62b05d59ff0894470685`
- **Model/device:** `Qwen/Qwen2.5-0.5B-Instruct`, CPU
- **Design:** one prompt, 80 updates, 1,081,344 trainable parameters; mask removes 28/36 positions
- **Recorded outcome:** full condition returns a normal answer without marker; masked condition emits marker
- **Classification:** `DEVELOPMENT RUN`
- **Interpretation:** LoRA can overfit an association between one severe mask and the marker
- **Does not support:** generalization, selectivity, suppressor mechanism, or a formal causal claim

### EXP-003-DEBUG-A

- **Artifact:** `results/campaign_003/mvp_debug.json`
- **SHA-256:** `22e7857adfd2289515fe426a950e1a8adddefee350d002e2ac5e609facfdabe6`
- **Model/device:** Qwen2.5-0.5B, CPU
- **Design:** 34 train / 6 development-evaluation prompts, 2 epochs, seed 42
- **Recorded outcome:** trained full=0/6, trained masked=0/6; untouched model 0/6 in both; `Delta_int=0`
- **Classification:** `DEVELOPMENT RUN — NEGATIVE`

### EXP-003-DEBUG-B

- **Artifact:** `results/campaign_003/mvp_cpu_smoke.json`
- **SHA-256:** `2a920e1ae551b77731f35bb1d21b317e189e3d231c28239b1c85333be1a4f9c1`
- **Model/device:** Qwen2.5-0.5B, CPU
- **Design:** 32 train / 8 development-evaluation prompts, 16 epochs, seed 42
- **Recorded outcome:** trained full=0/8, trained masked=0/8; untouched model 0/8 in both; `Delta_int=0`
- **Classification:** `DEVELOPMENT RUN — NEGATIVE`

### EXP-003-CPU-POS

- **Primary artifact:** `results/campaign_003/mvp_cpu_CONFIRM_0p5b.json`
- **Duplicate alias:** `results/campaign_003/mvp_validate_newloop.json`
- **Shared SHA-256:** `756a7f7b75e6a68e30520b087e0b60e1141799956a46b5d1be8e1f5ee24db7f0`
- **Duplicate policy:** the two files count as **one run**, not replication
- **Model/device:** `Qwen/Qwen2.5-0.5B-Instruct`, CPU; exact model revision and package environment absent
- **Design:** 34 train / 6 repeatedly evaluated validation prompts, 4 epochs, seed 42; best recorded evaluation at epoch 2; no adapter saved
- **Recorded outcome:** untouched full=0/6 and masked=0/6; trained full=0/6 and masked=6/6; development `Delta_int=1.0`
- **Classification:** `EXPLORATORY OBSERVATION`
- **Interpretation:** evidence that this LoRA recipe can associate the severe fixed mask with the benign marker on the selected validation suffix
- **Does not support:** `Delta_cond`, replication, formal stealth, utility, policy specificity, physical eviction, or PF-SEB

### EXP-003-GPU-REPORT

- **Artifact:** `results/campaign_003/mvp_kaggle_seed42.json`
- **SHA-256:** `f406d1ef45a4d8b33a938d2663823dc3196e85927b6fd0bade83d990d0e50380`
- **Model/device claimed:** `Qwen/Qwen2.5-1.5B-Instruct`, Kaggle T4/CUDA
- **Design recorded:** 54 train / 24 evaluation prompts, 20 epochs, seed 42; best epoch 4
- **Recorded outcome:** untouched full/masked=0/24; trained full=0/24 and masked=24/24; `Delta_int=1.0`
- **Classification:** `PROVISIONAL EXTERNAL REPORT`
- **Provenance limits:** manually persisted JSON; one generated sample summarized/edited; no raw notebook/stdout, source/environment hash, exact model revision, loss/evaluation trajectory, or saved adapter
- **Interpretation:** potentially useful signal that must be independently reproduced from a frozen clean protocol
- **Does not support:** independent replication, stability through epoch 20, formal stealth bound, policy fingerprinting, utility, or PF-SEB

## 6. Aggregate Campaign 3 interpretation

### Supported

- Actual Hugging Face Qwen-family checkpoints were exercised in Campaign 3 development code.
- The LoRA path is capable of learning a marker under a severe fixed prefill attention mask.
- Negative runs preceded the positive configuration and are preserved.
- The CPU positive artifact is internally explicit about the six recorded indicator outcomes.

### Not supported

- “MVP confirmed,” “decisive,” “rung-2 established,” “stealthy,” “selective,” “generalizing,” or “stable epoch 4–20” as publication-grade conclusions;
- real H2O or any physical cache eviction;
- policy fingerprinting or budget threshold behavior;
- active suppressor score manipulation;
- rescue/induction/random-deletion causal proof;
- matched-policy utility;
- adequate false-activation confidence bound;
- independent seed/model replication;
- complete six-cell `theta_c`/`theta_f`/`theta_b` causal design.

The strongest permissible statement is:

> **Campaign 3 provides exploratory evidence that a LoRA can associate a severe fixed prefill attention-mask intervention with a benign marker. Formal rung assignment and all unified-gate claims remain open.**

## 7. Next registered protocols

### EXP-004 — faithful passive eviction characterization (`PLANNED`)

1. Pin exact model revision, environment, scorer, per-layer/per-head aggregation, sink, recency, budget, and eviction timing.
2. Separate development from immutable confirmation prompts.
3. Save every adapter and raw run artifact.
4. Evaluate untouched and ordinary-fine-tuned controls under full, target, random, recency, SnapKV-like, and budget near misses.
5. Report utility and marker outcomes without selecting the confirmation set or epoch on final metrics.

### EXP-005 — PF-SEB score-manipulation pilot (`PLANNED`, blocked on EXP-004)

Require a learned score shift and eviction-probability shift relative to `theta_f`; static mask response is insufficient.

### EXP-006 — suppressor causal battery (`PLANNED`, blocked on EXP-005)

Require preregistered suppressor positions, rescue as necessity, induction as sufficiency, and position/layer/norm-matched random deletion.

## 8. Artifact rules for future records

Every run must record source commit and dirty state, complete environment/model/tokenizer revisions, split hashes and role, predeclared configuration, all epoch/evaluation history, raw per-prompt outcomes, adapter/checkpoint hash, and exclusion reason if invalid. Duplicate file hashes must never be counted as replication. Development-selected prompts cannot become confirmatory data.

The complete causal definitions, terminology ladder, run budgets, ethics policy, and release constraints remain governed by `CONSOLIDATED_RESEARCH_PLAN.md` unless superseded through an explicit reviewed decision.
