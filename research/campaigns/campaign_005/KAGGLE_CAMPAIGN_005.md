# Kaggle GPU Execution Playbook: Campaign 005
**Environment:** Kaggle Notebook with NVIDIA Tesla T4 (16GB VRAM)  
**Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Revision: `560647970498b8c199e8471c6155fe7f1c1f5138`)  
**Expected Runtime:** ~25–35 minutes per seed  

---

## 1. Setup in Kaggle Notebook Cell 1

```bash
# Cell 1: Environment Setup with Pinned Dependencies
!git clone https://github.com/your-username/btp-research.git /kaggle/working/btp-research
%cd /kaggle/working/btp-research

# Install pinned dependencies
!pip install -q torch==2.4.0 transformers==4.45.1 accelerate==0.34.2 peft==0.12.0 datasets scipy numpy
```

---

## 2. Multi-Seed Execution Commands

### Seed 42 (Primary Production Run):
```bash
!python -m scripts.run_pfseb_campaign_005 \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --device "cuda" \
  --seed 42 \
  --budget 8 \
  --safe_budget 32 \
  --ceiling_gb 7.0 \
  --n_bootstrap 2000 \
  --out_dir "results/campaign_005"
```

*(Note: If evaluating with trained Campaign 004 LoRA weights, add `--adapter_b_path "models/checkpoints/seed42_theta_b"`)*

### Seed 123 (Replication Seed 1):
```bash
!python -m scripts.run_pfseb_campaign_005 \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --device "cuda" \
  --seed 123 \
  --budget 8 \
  --safe_budget 32 \
  --ceiling_gb 7.0 \
  --n_bootstrap 2000 \
  --out_dir "results/campaign_005"
```

### Seed 7 (Replication Seed 2):
```bash
!python -m scripts.run_pfseb_campaign_005 \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --device "cuda" \
  --seed 7 \
  --budget 8 \
  --safe_budget 32 \
  --ceiling_gb 7.0 \
  --n_bootstrap 2000 \
  --out_dir "results/campaign_005"
```

---

## 3. Post-Run Download & Verification

Download the generated files from Kaggle:
1. `results/campaign_005/run_pfseb_campaign_005.json`
2. `results/campaign_005/circuit_attribution_heatmap.json`

Place them into your local project repository at `results/campaign_005/`.
Confirm all 5 verdicts in the JSON evaluate to `"PASS"`.
