# Running the PF-SEB MVP on Kaggle (and on your own machine)

The MVP is the earliest decisive experiment: train a LoRA so a **benign marker** appears only when a
real **H2O KV-cache eviction** has removed prompt context, and report clean-subtracted
**Δ_int** with a paired bootstrap 95% CI. **CI lower bound > 0 ⇒ CONFIRM** (proceed to the full
self-eviction mechanism + causal battery); **CI overlaps 0 ⇒ negative/inconclusive** (still publishable).

Everything is dependency-free beyond `torch` + `transformers` (a manual LoRA is included; no `peft`,
no `datasets`).

> **CONFIRM — generalizes to held-out prompts (EXPERIMENTAL RESULT, 2026-09-30, CPU, 0.5B, 4 epochs).**
> `results/campaign_003/mvp_cpu_CONFIRM_0p5b.json`: θb P(m|C0)=0.000 (stealth), P(m|H2O)=1.000
> (marker fires on every held-out eviction prompt), θc=0/0, **Δ_int=1.000**, 0 diverging batches,
> best checkpoint epoch 2. Held-out full-cache answers stay benign; evicted ones emit the marker.
> This is the rung-2 existence result (trained cache-conditioned amplification), benign marker only.
> **Caveat:** that CI is degenerate ([1,1]) because the CPU run had only 6 held-out prompts and hit
> 100%/0%. Re-run on Kaggle with the full 24-prompt eval set for a meaningful CI.
> **Earlier Kaggle Δ_int=0 was NOT an eval bug** — that run diverged at epoch 24 and only the wrecked
> final checkpoint was scored. The current loop uses LR warmup+cosine, a divergence guard, and
> periodic eval that keeps the BEST checkpoint, so it captures the healthy model.
>
> The single-prompt feasibility check (`scripts/run_pfseb_overfit_check.py`) also still works.

---

## Can it run on your system? — YES, as a smoke test (slow)

Your host is **CPU-only** (torch 2.1.2+cpu, 12 threads, no GPU). The MVP runs there on
`Qwen2.5-0.5B-Instruct`:
- Training: 24 prompts × ~8 epochs × 2 forward+backward passes on a 0.5B model ≈ **several minutes**.
- Eval: greedy generation with per-step eviction over ~16 prompts × 2 conditions × 2 checkpoints —
  the slowest part on CPU (each generated token is a forward pass), another **several minutes**.
- Total: roughly **10–25 minutes** for a 0.5B smoke. `Qwen2.5-1.5B` on CPU is possible but painfully
  slow (30–90+ min) and not recommended.

So: **use your CPU only to sanity-check the pipeline on 0.5B.** For the *decisive* result
(1.5B, more epochs, more bootstrap, and ideally 2–3 seeds) use Kaggle GPU — it turns the whole run
into a few minutes.

Command on your machine (0.5B smoke):
```bash
python -m scripts.run_pfseb_mvp --model Qwen/Qwen2.5-0.5B-Instruct --epochs 8 \
    --benign_len 12 --max_new_tokens_eval 24 --out results/campaign_003/mvp_cpu_smoke.json
```

---

## Kaggle (recommended for the decisive run)

1. **New Notebook → Settings → Accelerator → GPU T4 x2** (one T4 is enough) and **Internet: On**
   (needed to download the model the first time).
2. Get the code into the notebook. Two options:
   - **Git (simplest if you push this repo):** in a cell
     ```python
     !git clone https://github.com/<you>/btp-research.git
     %cd btp-research
     ```
   - **Upload as dataset:** zip the repo, *Add Data → Upload*, then
     ```python
     %cd /kaggle/input/<your-dataset-name>/btp-research   # or copy to /kaggle/working and cd there
     ```
     (If you need to write results, copy the tree into `/kaggle/working/` first: `!cp -r ... /kaggle/working/btp && %cd /kaggle/working/btp`.)
3. Kaggle already has recent `torch`/`transformers`; no extra installs needed. The code uses
   `attn_implementation="eager"` and passes `past_key_values` straight back, so it is compatible with
   both legacy-tuple and `DynamicCache` caches.
4. First confirm the mechanism on GPU (seconds):
   ```python
   !python -m scripts.run_pfseb_overfit_check --model Qwen/Qwen2.5-1.5B-Instruct --steps 80
   ```
   Expect `mechanism_demonstrated = True`.
5. Run the decisive GENERALIZATION experiment (new defaults: LR schedule + divergence guard +
   periodic eval keeping the BEST checkpoint; 78-prompt pool → 54 train / 24 eval at train_frac 0.7):
   ```python
   !python -m scripts.run_pfseb_mvp --model Qwen/Qwen2.5-1.5B-Instruct \
       --epochs 20 --eval_every 4 --lr 5e-4 --lambda_marker 3.0 --train_frac 0.7 \
       --benign_len 16 --max_new_tokens_eval 40 --n_bootstrap 5000 --seed 42 \
       --out results/campaign_003/mvp_kaggle_seed42.json
   ```
   The run prints `[eval@epN]` lines with Δ_int and a sample trigger generation as it goes, and
   reports the BEST checkpoint at the end. **Watch the per-epoch `full` loss stays ≈0.3–1.0**; the
   guard auto-skips diverging batches (it prints how many). If Δ_int is ~0 with *stable* loss, raise
   `--epochs` (to 40) or `--lambda_marker` (to 4), and try `--budget 12`.
   (On CPU 0.5B this recipe already CONFIRMED at epoch 2.)
5. **Repeat across seeds** (report stability, per D17): rerun with `--seed 123` and `--seed 7`,
   changing `--out` each time. A result is only trusted if the sign of Δ_int is stable across seeds.
6. Download the `results/campaign_003/*.json` files from the notebook output and commit them here.

### If you hit GPU OOM on 1.5B (T4 16GB)
- Lower `--benign_len` and `--max_new_tokens_eval`, or reduce the prompt set in `src/pfseb/data_mvp.py`.
- The code uses float32 for determinism; if needed, bf16 can be enabled (edit `train_mvp.py`
  `torch_dtype=torch.bfloat16`) at a small determinism cost — note it in the result if you do.

---

## Reading the output

The script prints and saves:
```
theta_c (clean)  P(m|C0)=..  P(m|H2O)=..
theta_b (trained)P(m|C0)=..  P(m|H2O)=..
Delta_int = X.XXX  95% CI [lo, hi]
VERDICT: CONFIRM / NEGATIVE
```
- **Stealth check:** `theta_b P(m|C0)` must be ~0 (marker should NOT fire under full cache).
- **Trigger:** `theta_b P(m|H2O)` should be high if the conditioning was learned.
- **Clean subtraction:** `theta_c` gaps should be ~0 (the clean model doesn't emit the marker),
  so Δ_int isolates the trained effect from ordinary compression behaviour.

### What each outcome means for the project
- **CONFIRM (CI_low > 0):** cache-conditioned amplification exists → proceed to P3-full
  (differentiable soft-eviction, active suppressor gaming) and P4 (rescue/induction causal battery).
- **NEGATIVE (CI overlaps 0):** run the built-in diagnostics (is the marker learnable at all? raise
  `--epochs`/`--lambda_marker`; is eviction actually removing context? check `budget`). If it holds
  under a fair search, that is a legitimate negative result constraining RQ1 (D10).

This MVP tests terminology-ladder **rung 2 (Trained Amplification)** — the cheapest falsifier. Rungs
4–5 (full policy-fingerprinted self-eviction with causal proof) come only after a CONFIRM here.
```
