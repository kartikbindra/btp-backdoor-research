# Campaign 008 — Design and statistical audit

**2026-11-22. Read-only audit. No source modified.**

## 1. Objective

Give a blunt verdict on what the 2026-11-22 r=0.5 result does and does not establish, size the
statistical power needed for a decisive claim, and specify the pre-registered test ladder for
Campaign 008.

## 2. Sources

- `researchMemory/CURRENT_STATE.md`, `CLAIMS.md`, `agentMemory/DECISION_LOG.md`, `agentMemory/OPEN_PROBLEMS.md`
- `research/campaigns/runtime_h2o_validity/` pre-registrations and the three RESULT reports
- `results/runtime_eviction_validity/rt_train_delay_r50_s42/{summary.json,rows.jsonl}`
- `src/pfseb/stats_c007.py` (clopper_pearson, cluster_bootstrap, p_one_sided)

## 3. Findings — what the r=0.5 result is

Decompose the phenomenon into four nested claims:

| Claim | Evidence on 2026-11-22 | Verdict |
| --- | --- | --- |
| (i) generic compression/perturbation response | fires under **any** large eviction; runtime random 12/40, snapkv 9/40, streamingllm 4/40 | **ESTABLISHED (EXPERIMENTAL RESULT)** |
| (ii) eviction-magnitude conditioning | r=0.5 binds (18/40), r=0.9 did not (8/40) | **SUGGESTIVE only** (r=0.9 evaluated at epoch 1) |
| (iii) runtime-ordering conditioning | runtime 18/40 vs masked-prefill 3/40, same keep set, p=2.8e-4 | **ESTABLISHED (EXPERIMENTAL RESULT)** |
| (iv) policy selectivity (H2O ≠ random) | 18 vs 12, exact p=0.15 | **NOT ESTABLISHED** |
| (v) stealth at C0 | leak 6/40 = 0.15, above the 0.10 target | **NOT ESTABLISHED** |

**INFERENCE.** The defensible new object is (iii): a behaviour conditioned on the prefill *not*
having seen the evicted keys — i.e. a genuine post-prefill, physical, runtime-ordering trigger. This
is exactly the "conversion" the project wants. (iv) and (v) remain open and are the gates to a
paper-worthy backdoor.

## 4. Power analysis

Unit = prompt. Primary statistic = paired McNemar exact on discordant prompts (C0 vs arm) and on
(H2O vs random). Let δ = true difference, ψ = discordant-pair rate.

Paired-sample size (McNemar): `n ≈ ( z_{1-α/2}√ψ + z_{1-β}√(ψ-δ²) )² / δ²`, α=0.05, power 0.80.

- δ=0.30, ψ≈0.30 (as observed): n ≈ ( 1.96·0.548 + 0.84·√(0.30-0.09) )²/0.09
  = (1.074 + 0.385)²/0.09 = 2.13/0.09 ≈ **24 prompts**. The 2026-11-22 n=40 is therefore *adequate*
  for the C0-vs-runtime contrast (and indeed p=4.9e-4).
- δ=0.10, ψ≈0.20: n ≈ ( 1.96·0.447 + 0.84·√(0.20-0.01) )²/0.01 = (0.876+0.366)²/0.01 = 1.544/0.01
  ≈ **155 prompts**.
- δ=0.15, ψ≈0.25: n ≈ ( 0.98+0.84·√0.2275 )²/0.0225 = (0.98+0.401)²/0.0225 = 1.905/0.0225 ≈ **85 prompts**.

**DECISION.** Set the Campaign 008 eval pool to **n = 160 held-out dev prompts** per arm. This gives
≥80% power to detect a 0.15 difference (and >95% for 0.20) between H2O and the strongest negative, at
2,000-bootstrap cluster CIs. Three seeds give a variance estimate, not a rate.

## 5. Threats to validity (must be pre-empted)

1. **Position-pattern confound** (see `AUDIT_IMPLEMENTATION.md` finding 5): add a **scrambled-H2O**
   control — the same number of kept positions drawn to match H2O's per-segment density but with
   saliency destroyed. If H2O does not beat scrambled-H2O, the effect is positional, not policy.
2. **Selection statistic is uninformative** (constant 0). Fix by selecting on a *paired* dev statistic
   (H2O − max(C0, random, scrambled)) with the `>=` tie-break already added, and report the chosen
   epoch.
3. **Full-cache leak** is the stealth gate; any rate quoted must carry its C0 leak beside it.
4. **Researcher degrees of freedom**: the retention grid and thresholds must be frozen in the
   pre-registration before any arm is read.

## 6. Recommended next action — the pre-registered ladder

See `../PREREG_campaign_008.md`. In brief:

- **T0 (gate, no training):** hidden-state / logit separability probe of H2O vs random vs scrambled
  at matched cardinality. If a linear probe cannot separate them above chance, (iv) is near-impossible
  and we stop before spending GPU.
- **T1 (replication):** r=0.5, 3 seeds, n=160 eval, fixed selection. Is the runtime-ordering effect
  real and size-stable?
- **T2 (selectivity training):** explicit policy-contrast objective + per-epoch all-negative cycling +
  scrambled-H2O negative; target H2O − strongest negative ≥ 0.30.
- **T3 (threshold):** single adapter evaluated at r ∈ {1.0, 0.9, 0.75, 0.5, 0.25, 0.1}; is firing
  monotone in evicted fraction (runtime-magnitude conditioning) or a step?
- **T4 (stealth):** v3-style C0 hinge/KL on the runtime objective; target C0 ≤ 0.05.

## 7. Evidence strength

The decomposition (section 3) follows directly from the saved `summary.json`. The power numbers are
standard McNemar algebra (INFERENCE), not measured. The confound list is grounded in
`AUDIT_IMPLEMENTATION.md`.

## 8. Counterevidence / open questions

- A negative T2 does not disprove selectivity in general — only at this capacity, corpus and budget
  family; state the scope.
- A positive T1 with a negative T2 is still a publishable *mechanism* result (runtime-ordering
  conditioning exists) even without selectivity.

## 9. Files created

- `research/campaigns/campaign_008/agent_reports/AUDIT_DESIGN.md` (this file)
