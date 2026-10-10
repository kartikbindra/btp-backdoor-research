# Kaggle GPU Execution Playbook — Campaign 006

**Environment:** Kaggle Notebook, NVIDIA Tesla T4 (16 GB VRAM)
**Target:** cross-architecture & scale generalization of the RCCB (see `campaign_006_plan.md`)
**Default compute:** T4-only. Tier B models ≥3B use 4-bit QLoRA (declared confound).

---

## 0. Phase 0 prerequisite (blocking)

Do **not** start Campaign 006 training until Phase 0 is closed (`campaign_006_plan.md §2`):

1. Re-run Campaign 005 **live** with the θb adapter using the remediation-fixed runner
   (defenses and canary controls now measured, verdicts fail-closed) and commit the artifact.
2. Run Campaign 004 seeds 123 and 7 and commit artifacts.
3. Save/hash the Campaign 003 1.5B adapter (or disarm the claim).
4. Fix canonical memory to label the committed Campaign 005 artifact as simulation.

---

## 1. Setup (Cell 1)

```bash
!git clone https://github.com/kartikbindra/btp-backdoor-research /kaggle/working/btp-research
%cd /kaggle/working/btp-research

# Base deps (Tier A)
!pip install -q torch==2.4.0 transformers==4.45.1 accelerate==0.34.2 scipy numpy

# For Tier B (3B-7B QLoRA), additionally:
# !pip install -q bitsandbytes peft
```

Upload the Campaign 004 θb adapter (`theta_b_seed42.pt`) to
`results/campaign_004/checkpoints/` (used by the Campaign 005 Phase 0 re-run).

---

## 2. Phase 0 — Campaign 005 live re-run

> **ADAPTER PROVENANCE WARNING (2026-10-10).** The previously committed
> `theta_b_seed42.pt` was a **0.5B adapter** (hidden 896) mistakenly used against the 1.5B
> model (hidden 1536). It has been renamed to `theta_b_seed42_0p5b_WRONGARCH.pt`. Always
> confirm the adapter matches the base model: a 1.5B LoRA has **224** keys and `.A` rows of
> shape `[8, 1536]`; a 0.5B LoRA has **192** keys and `[8, 896]`. The runner now aborts on
> any architecture mismatch (no silent base-model fallback).

Regenerate the seed-42 1.5B adapter (its checkpoint was the bad one), then run:
```bash
# Regenerate the correct 1.5B seed-42 theta_b (and theta_f)
!python -m scripts.run_pfseb_campaign_004 \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" --budget 8 --epochs 20 --eval_every 4 \
  --lr 5e-4 --lambda_marker 3.0 --train_frac 0.69 --benign_len 16 \
  --max_new_tokens_eval 40 --grad_clip 1.0 --warmup_frac 0.05 --n_bootstrap 2000 \
  --device cuda --seed 42 --out results/campaign_004/kaggle_decisive_seed42.json --save_checkpoints

# Live Campaign 005 re-run with the fixed runner
!python -m scripts.run_pfseb_campaign_005 \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --device cuda --seed 42 --budget 8 --safe_budget 32 \
  --n_bootstrap 2000 \
  --adapter_b_path results/campaign_004/checkpoints/theta_b_seed42.pt \
  --out_dir results/campaign_005_live_seed42
```

*Quick alternative if you don't want to re-train seed 42:* use the already-valid seed-7
1.5B adapter (`--adapter_b_path results/campaign_004/checkpoints/theta_b_seed7.pt
--seed 7`) and label the run as seed 7.

Verify in `results/campaign_005_live_seed42/run_pfseb_campaign_005.json`:
- `metadata.execution_mode == "live"`
- `metadata.defenses_measured == true` and `metadata.canary_measured == true`
- `metadata.baseline_verification.theta_b_evicted_asr` is high (≈1.0) — if it is 0.0, the
  adapter did not apply and the run is invalid.
- `verdicts` contains no `NOT_MEASURED` / `SIMULATION` value.
- `circuit_localization.critical_layers` is EMPTY when the eviction ASR is 0 (no fabrication).

---

## 3. Campaign 006 — Tier A (cross-architecture, 1B class)

For each model key in `{qwen15, llama1b, gemma2b}` and seed in `{42, 123, 7}`:

```bash
# Seed 42, Qwen anchor
!python -m scripts.run_pfseb_campaign_006_train \
  --model_key qwen15 --seed 42 --epochs 20 --budget 8 --lambda_marker 3.0 \
  --out_dir results/campaign_006/qwen15_seed42

!python -m scripts.run_pfseb_campaign_006_eval \
  --model_key qwen15 --seed 42 --adapter_dir results/campaign_006/qwen15_seed42 \
  --budget 8 --circuit --out_dir results/campaign_006/qwen15_seed42

# Cross-family Llama-3.2-1B
!python -m scripts.run_pfseb_campaign_006_train --model_key llama1b --seed 42 \
  --out_dir results/campaign_006/llama1b_seed42
!python -m scripts.run_pfseb_campaign_006_eval --model_key llama1b --seed 42 \
  --adapter_dir results/campaign_006/llama1b_seed42 --circuit \
  --out_dir results/campaign_006/llama1b_seed42

# Cross-family Gemma-2-2B
!python -m scripts.run_pfseb_campaign_006_train --model_key gemma2b --seed 42 \
  --out_dir results/campaign_006/gemma2b_seed42
!python -m scripts.run_pfseb_campaign_006_eval --model_key gemma2b --seed 42 \
  --adapter_dir results/campaign_006/gemma2b_seed42 --circuit \
  --out_dir results/campaign_006/gemma2b_seed42
```

**Tier A gate (CG3):** at least one non-Qwen family reproduces Δ_int ≥ 0.50
(CI low > 0.30), Δ_cond ≥ 0.50, stealth 95% upper bound < 1%.

---

## 4. Campaign 006 — Tier B (scale, QLoRA)

Within-family Qwen scale points (0.5B / 3B / 7B) and Mistral-7B (SWA):

```bash
# Qwen 0.5B (BF16 LoRA)
!python -m scripts.run_pfseb_campaign_006_train --model_key qwen05 --seed 42 \
  --out_dir results/campaign_006/qwen05_seed42
!python -m scripts.run_pfseb_campaign_006_eval --model_key qwen05 --seed 42 \
  --adapter_dir results/campaign_006/qwen05_seed42 --out_dir results/campaign_006/qwen05_seed42

# Qwen 3B / 7B and Mistral-7B use 4-bit QLoRA automatically from the registry.
# Review VRAM; reduce --epochs or use --quick for a smoke pass first.
!python -m scripts.run_pfseb_campaign_006_train --model_key qwen3b --seed 42 \
  --out_dir results/campaign_006/qwen3b_seed42
```

**QLoRA confound:** evaluation loads the base in native precision and merges the adapter;
report the base-quantization delta explicitly. A BF16-BF16 control should be run for at
least one ≥3B model if compute allows.

---

## 5. Download & verify

Download per model/seed:
- `results/campaign_006/<model>_seed<k>/train_summary.json`
- `results/campaign_006/<model>_seed<k>/eval_summary.json`
- `results/campaign_006/<model>_seed<k>/{theta_b.pt,theta_f.pt}` (optional; large)

Place them in the local repo under `results/campaign_006/`. Confirm
`metadata.execution_mode == "live"` in every `eval_summary.json`.

---

## 6. Evidence rules

- A `--dry_run` artifact is a **simulation** and must never be reported as a result.
- Every claim requires paired 95% bootstrap CIs and, for a publication claim, 3 seeds.
- Report `B*` (budget threshold) vs model size and vs context length.
- Negative/architecture-boundary results are valid and publishable (D10).
