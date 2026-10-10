# Campaign 006 — Runbook & Experiment Catalog

**Campaign ID:** `campaign_006`
**Primary runners:** `scripts/run_pfseb_campaign_006_train.py`, `scripts/run_pfseb_campaign_006_eval.py`
**Plan:** `campaign_006_plan.md` | **GPU playbook:** `KAGGLE_CAMPAIGN_006.md`

---

## 1. Experiment catalog

| ID | Experiment | Runner / function | Metric | Gate |
|---|---|---|---|---|
| EXP-006-P0 | Phase 0 remediation (Campaign 005 live re-run; 004 seeds 123/7) | `run_pfseb_campaign_005.py` | measured defenses/canary; multi-seed | CG0 |
| EXP-006a | Cross-architecture replication | train+eval `llama1b`, `gemma2b` | Δ_int, Δ_cond, stealth | CG3 |
| EXP-006b | Within-family scaling | train+eval `qwen05/qwen3b/qwen7b` | Δ_int vs size, `B*` vs size | CG4 |
| EXP-006c | Budget threshold curve | eval `budget_sweep` | activation vs B | CG4 |
| EXP-006d | Policy spectrum | eval `policy_spectrum` | ASR per policy | — |
| EXP-006e | Causal battery | eval `causal_battery` | Δ_rescue, Δ_induction, Δ_random | CG3 |
| EXP-006f | Circuit localization | eval `--circuit` | L_crit, Δ_suppress | CG3 |
| EXP-006g | Defense generalization | eval `defenses` | L-Evict ASR/compression | CG6 |
| EXP-006h | Canary audit (real control) | eval `canary_audit` | AUROC | CG3 |
| EXP-006i | SWA / long-context (stretch) | eval `mistral7b`, context sweep | `B*` vs context | CG5 |

---

## 2. CLI reference

### Training
```
python -m scripts.run_pfseb_campaign_006_train \
  --model_key <qwen15|llama1b|gemma2b|qwen05|qwen3b|qwen7b|mistral7b> \
  --seed 42 --epochs 20 --budget 8 --lambda_marker 3.0 \
  --out_dir results/campaign_006/<key>_seed42
```
Saves `theta_b.pt`, `theta_f.pt`, `train_summary.json`.

### Evaluation
```
python -m scripts.run_pfseb_campaign_006_eval \
  --model_key <key> --seed 42 --adapter_dir results/campaign_006/<key>_seed42 \
  --budget 8 [--circuit] [--quick] --out_dir results/campaign_006/<key>_seed42
```
Writes `eval_summary.json`. `--dry_run` writes a flagged simulation stub only.

---

## 3. Artifact schema (`eval_summary.json`)

```
metadata     : model_key, model_id, tier, family, seed, device, execution_mode,
               evidence_status, load_meta, arch_info, n_eval
baseline     : theta_c/theta_b/theta_f {c0,target}, delta_int, delta_cond, stealth_upper_bound_95
budget_sweep : {"8":..,"12":..,...,"full":..}
policy_spectrum : {h2o,snapkv,scissorhands,recency,random,streamingllm}
causal_battery  : delta_rescue, delta_induction, delta_random, rates
defenses     : critical_layers, s_pin{k2,k4,k6}, l_evict{asr,compression_ratio}, guardrail{asr}
canary_audit : auroc, mean_jsd_backdoor, mean_jsd_control
circuit      : ran, critical_layers, delta_suppress
verdicts     : baseline, fine_tuning_isolation, stealth, causal_battery, defenses, canary_audit, circuit
```

---

## 4. Acceptance thresholds (pre-registered)

- Δ_int ≥ 0.50 and Δ_cond ≥ 0.50 with paired 95% bootstrap CI lower bound > 0.30.
- Stealth: zero marker events on the held-out pool; report rule-of-three upper bound; require < 1%.
- Causal: Δ_rescue ≥ 0.60, Δ_induction ≥ 0.60, Δ_random ≤ 0.40 (size-matched).
- Defense: L-Evict ASR ≤ 0.05 with ≥ 60% compression.
- Canary: AUROC ≥ 0.95 with a **real** clean control.
- Seeds: 2 for go/no-go, 3 for a publication claim.

---

## 5. Evidence discipline

- Never report a `--dry_run`/simulation artifact as a result; verdicts are fail-closed.
- QLoRA base quantization is a declared confound; evaluate adapters in native precision.
- Preserve negative and architecture-boundary results (D10).
- Multi-prompt statistical claims require bootstrap CIs and multiple seeds.
