# Campaign 004 Codebase Survey & Technical Audit Report

**Date:** 2026-10-06  
**Investigator:** Explorer 1 (Generation 2)  
**Investigation Scope:** Checkpoint persistence, cache policy implementations, inference cache hooking, and execution harness audit for Campaign 004.  
**Integrity Standard:** AGENTS.md Research Integrity Rules (SOURCE FACT, INFERENCE, HYPOTHESIS, EXPERIMENTAL RESULT, DECISION).

---

## 1. Executive Summary

A comprehensive, read-only architectural investigation was performed across the codebase to determine the current state of trained checkpoints, cache policy implementations, KV-cache hooking mechanics, and the Campaign 004 execution scripts.

Key findings:
1. **Checkpoint Storage & Persistence (SOURCE FACT):** There are **zero saved model weights or LoRA adapter files** anywhere in the repository (no `.pt`, `.safetensors`, `.bin`, or `.pth` files; no `models/` or `checkpoints/` directories exist). All prior campaigns (including Campaign 003's successful Kaggle and CPU runs) trained lightweight LoRA adapters **strictly on-the-fly in memory**, evaluated them, dumped metrics and completion samples to JSON artifacts (`results/campaign_003/*.json`), and terminated without persisting model parameters.
2. **Cache Policy Implementations (SOURCE FACT):**
   - **H2O** is fully implemented using token-level attention mass aggregation across all heads and layers, preserving attention sinks and a recency window while keeping top-$k$ heavy hitters.
   - **Recency-Only** and **Random Eviction** near-miss baselines are implemented.
   - **SnapKV and Scissorhands are currently stubs with zero specialization:** In `src/pfseb/eviction.py`, `snapkv_keep_mask` and `scissorhands_keep_mask` are identical pass-through wrappers to `topk_keep_mask`. In `src/pfseb/harness.py` (`_eviction_decision`) and `scripts/run_pfseb_campaign_004.py`, policies `h2o`, `snapkv`, and `scissorhands` fall into the exact same branch and use the exact same prefill attention scores. Consequently, **SnapKV and Scissorhands produce byte-identical eviction masks to H2O in the current codebase**.
3. **Inference & KV-Cache Hooking (SOURCE FACT):** Hooking does **not** use PyTorch module hooks (`register_forward_hook`), custom attention layers, Hugging Face `generate()` hooks, custom cache classes, or vLLM. Instead, eviction is simulated behaviorally in a manual greedy decode loop via **2-D attention masking** (`attention_mask=pmask`), passing 0 at evicted prompt indices and explicitly managing `position_ids`. This was explicitly chosen to avoid breaking Hugging Face's RoPE rotary embedding tables.
4. **Campaign 004 Runner Script Audit (`scripts/run_pfseb_campaign_004.py`) (SOURCE FACT & INFERENCE):** The runner contains **8 major structural gaps and bugs**, including:
   - An **18 GB VRAM memory explosion** caused by concurrently initializing three 1.5B float32 models (`base_model`, `b_model`, `f_model`) on Kaggle's 16 GB T4 GPU;
   - A **mathematically broken candidate selection** in the Random Deletion control that evicts only ~6 tokens instead of ~30 tokens when $B=8$;
   - Omission of the evicted branch loss in fine-tuned control $\theta_f$ training;
   - Omission of the divergence guard and best-checkpoint selection proven necessary in Campaign 003.

---

## 2. Trained Model Checkpoints & Storage Mechanics

### 2.1 Directory & File Audit
A systematic filesystem search across `results/campaign_003/`, `models/`, `checkpoints/`, `src/pfseb/`, and `research/campaigns/` revealed:
- `models/`: **Does not exist.**
- `checkpoints/`: **Does not exist.**
- Binary model files: Searching for `*.pt`, `*.bin`, `*.safetensors`, and `*.pth` across the workspace yielded **0 results**.
- `results/campaign_003/` contains 7 JSON result files:
  1. `mvp_kaggle_seed42.json` (3,503 bytes): Authoritative Campaign 003 decisive run on Kaggle GPU (T4, Qwen2.5-1.5B-Instruct, 20 epochs, seed 42, 54 train / 24 held-out eval prompts, $\Delta_{int} = 1.0$).
  2. `mvp_cpu_CONFIRM_0p5b.json` (3,690 bytes): Campaign 003 confirmation run on local CPU (Qwen2.5-0.5B-Instruct, 4 epochs, seed 42, $\Delta_{int} = 1.0$).
  3. `mvp_cpu_smoke.json` (1,318 bytes): Initial 0.5B CPU sanity check.
  4. `mvp_debug.json` (1,219 bytes): Diagnostic run during loop debugging.
  5. `mvp_validate_newloop.json` (3,690 bytes): Validation run verifying the scheduler and divergence guard.
  6. `overfit_check.json` (450 bytes): Single-prompt mechanism feasibility check.
  7. `smoke.json` (2,277 bytes): Clean-model A/B/C generation test.

### 2.2 Weight Storage vs. On-the-Fly Training
- **Implementation in `src/pfseb/lora.py`:**
  - Defines `LoRALinear(nn.Module)` (lines 13–27), which wraps standard `nn.Linear` layers with trainable low-rank matrices $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$.
  - `add_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=8, alpha=16)` (lines 29–38) modifies the model in-place.
- **Training & Discard Cycle in `src/pfseb/train_mvp.py` & `scripts/run_pfseb_mvp.py`:**
  - `train_and_eval` (lines 188–293) downloads the base model from Hugging Face (`AutoModelForCausalLM.from_pretrained`), freezes all base parameters, inserts `LoRALinear`, trains on-the-fly using AdamW, evaluates the held-out set, packages metrics into a dict, and returns.
  - Neither `train_and_eval` nor `scripts/run_pfseb_mvp.py` ever invokes `torch.save()`, `state_dict()`, or `save_pretrained()`.
  - The model weights reside purely in volatile RAM/VRAM during execution and are discarded when the Python process exits.
- **Handling in `scripts/run_pfseb_campaign_004.py`:**
  - Similarly, `run_campaign_004` initializes `b_model` and `f_model`, trains them on-the-fly, evaluates them, and exits.
  - There are **no CLI arguments** (`--load_adapter`, `--adapter_path`, etc.) and no logic to load pre-trained checkpoints or export trained adapters.

---

## 3. Cache Policy Implementations & Near-Miss Definitions

### 3.1 Policy Locations
Cache policy definitions reside in three primary locations:
1. `src/pfseb/eviction.py`: Tensor-level keep-mask functions and `EvictionConfig`.
2. `src/pfseb/harness.py`: Generation loops and online ranking function `_eviction_decision`.
3. `scripts/run_pfseb_campaign_004.py`: Inline policy routing in `evaluate_policy_single`.

### 3.2 H2O Implementation Details
H2O (Heavy-Hitter Oracle) is implemented as a heavy-hitter retention policy over accumulated attention mass:
- **Configuration (`EvictionConfig`, `src/pfseb/eviction.py:33–49`):**
  - `policy="h2o"`
  - `budget: int` (positions retained)
  - `recency_window: int` (default 8 in config, 2 in MVP)
  - `num_sink: int` (default 4 in config, 2 in MVP)
- **Scoring (`src/pfseb/harness.py:46–51`):**
  ```python
  def _aggregate_step_scores(attentions, num_positions: int, device) -> torch.Tensor:
      acc = torch.zeros(num_positions, device=device)
      for a in attentions:
          acc += a[0].sum(dim=0).sum(dim=0)  # [n_q_heads, q_len, kv_len] -> [kv_len]
      return acc
  ```
  Attention mass is summed across all layers and all query heads into a 1D sequence-level score tensor.
- **Eviction Decision (`src/pfseb/eviction.py:62–87` and `src/pfseb/harness.py:54–85`):**
  - Positions `[0, num_sink)` (attention sinks) and `[seq_len - recency_window, seq_len)` (recency window) are unconditionally protected.
  - Non-protected positions are ranked descending by score.
  - The top `remaining = budget - len(protected)` positions are kept; all other positions are marked for eviction.
  - Monotonicity is preserved in `harness.py:187`: once evicted, a position is never restored.

### 3.3 Near-Miss Policies & The Critical Implementation Gap
The project requirements (ORIGINAL_REQUEST.md §R1 and CAMPAIGN_004_PLAN.md §WP4.1) require evaluating H2O against four near-miss policies: `snapkv`, `scissorhands`, `recency`, and `random`.

#### 1. Recency-Only Policy (Implemented)
- **Code:** `src/pfseb/eviction.py:108–112`, `src/pfseb/harness.py:70–71`, and `scripts/run_pfseb_campaign_004.py:51–55`.
- **Definition:** Ranks positions purely by temporal order (`rank = torch.arange(seq)`). Sinks `[0, num_sink)` and the most recent `budget - num_sink` positions are retained; all intermediate positions are evicted regardless of attention weight.

#### 2. Random Eviction Policy (Implemented)
- **Code:** `src/pfseb/eviction.py:115–120`, `src/pfseb/harness.py:68–69`, and `scripts/run_pfseb_campaign_004.py:56–63`.
- **Definition:** Ranks positions by uniform random noise (`torch.rand()`), protecting sinks and the recency window, and randomly evicting `P - budget` candidates.

#### 3. SnapKV & Scissorhands (CRITICAL GAP: Identical to H2O)
- **In `src/pfseb/eviction.py` (lines 95–106):**
  ```python
  def snapkv_keep_mask(windowed_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
      return topk_keep_mask(windowed_scores, cfg.budget, cfg.num_sink, cfg.recency_window)

  def scissorhands_keep_mask(persistence_counts: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
      return topk_keep_mask(persistence_counts, cfg.budget, cfg.num_sink, cfg.recency_window)
  ```
  Both functions are simple aliases for `topk_keep_mask`. The docstrings state that `windowed_scores` and `persistence_counts` should be calculated upstream in `attention_cache.py`.
- **The Missing File:** `attention_cache.py` **does not exist** anywhere in the repository (referenced only in comments).
- **In `src/pfseb/harness.py` (lines 72–74):**
  ```python
  else:  # h2o / snapkv / scissorhands all rank by their accumulated `scores` here
      rank = scores
  ```
  In `_eviction_decision`, `snapkv` and `scissorhands` fall through to the exact same branch as `h2o`, using unwindowed cumulative attention scores.
- **In `scripts/run_pfseb_campaign_004.py` (lines 65–68):**
  ```python
  else:
      # Score-based: H2O, SnapKV, Scissorhands
      cfg = EvictionConfig(policy=policy, budget=budget, recency_window=2, num_sink=2)
      evicted = _prompt_evicted_positions(model, ids, cfg)
  ```
  `_prompt_evicted_positions` performs a standard forward prefill on the prompt, extracts `out.attentions`, computes cumulative attention mass via `_aggregate_step_scores`, and calls `_eviction_decision`.
- **Impact (INFERENCE):** In `scripts/run_pfseb_campaign_004.py`, evaluating `snapkv` and `scissorhands` executes the **exact same code on the exact same tensor** as `h2o`. The evicted token sets are 100% byte-identical. As written, Campaign 004 does NOT test true SnapKV (observation window pooling) or Scissorhands (persistence count across steps); it evaluates H2O three times under different labels.

---

## 4. Inference & KV-Cache Hooking Mechanics

### 4.1 Absence of Invasive Hooks
Inspection of `src/pfseb/harness.py` demonstrates:
- **No PyTorch Hooks:** Neither `register_forward_hook` nor `register_full_backward_hook` is utilized.
- **No Custom Attention Layers:** The model uses standard Hugging Face attention (`attn_implementation="eager"`).
- **No vLLM or PagedAttention:** Everything runs within native PyTorch.
- **No Custom Cache Class:** The cache is passed back and forth as Hugging Face's standard `past_key_values` structure (`DynamicCache` or tuple-of-tuples).

### 4.2 The 2-D Attention Masking Paradigm
Instead of physical memory pruning, the codebase simulates eviction via **2-D attention masking**:
- **Why Masking over Pruning (`src/pfseb/harness.py:8–15`):**
  In modern Hugging Face causal models (such as Qwen2.5), Rotary Position Embeddings (RoPE) are applied to keys prior to cache entry. Physical pruning of `past_key_values` shortens the cached tensor length. When subsequent tokens are passed with their true absolute position IDs, standard Hugging Face rotary embedding tables mismatch or overflow.
  Masking the evicted positions (assigning `attention_mask = 0`) zeroes out their attention weights in softmax:
  $$\text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right), \quad M_{i, j} = -\infty \text{ if } j \in \text{evicted}$$
  This is **behaviorally and numerically identical** to physical eviction regarding the model's logits and generated tokens, while preserving exact RoPE positional coordinates.

### 4.3 Static vs. Dynamic Masking
The harness implements two distinct generation functions:
1. `generate_static_masked(model, tokenizer, input_ids, evicted_positions, max_new_tokens)` (`harness.py:87–129`):
   - Masks a fixed set of prompt positions from token 0 onward.
   - At prefill: passes `attention_mask=pmask(P)`.
   - At decoding step `step`: passes single token with `position_ids=[[P + step]]` and `attention_mask=pmask(P + step + 1)`.
   - **Why this is used for training and eval:** As documented in Campaign 003 Results Log (§4.2), the model's decision to emit the marker occurs at token 0 of generation. In dynamic decoding, eviction only triggers after several tokens have been generated; thus the model never received the trigger signal at marker onset. Prefill-time static masking aligns train (`_loss_evicted`) and eval.
2. `generate_with_eviction(model, tokenizer, input_ids, cfg, max_new_tokens, ...)` (`harness.py:131–202`):
   - Full autoregressive dynamic eviction loop.
   - Computes attention weights at each decode step (`output_attentions=True`), accumulates running scores, and evicts positions when active length exceeds `cfg.budget`.

---

## 5. Audit of Existing Scripts & Campaign 004 Runner

A thorough line-by-line inspection of `scripts/run_pfseb_campaign_004.py` in comparison with `scripts/run_pfseb_mvp.py` and `src/pfseb/train_mvp.py` uncovered eight critical defects and design flaws:

```
┌────────────────────────────────────────────────────────────────────────┐
│               SUMMARY OF CAMPAIGN 004 CODE DEFECTS & GAPS             │
├────┬─────────────────────────────┬─────────────────────────────────────┤
│ #  │ Defect / Gap                │ Severity & Consequence              │
├────┼─────────────────────────────┼─────────────────────────────────────┤
│ G1 │ 3 Concurrent 1.5B Models    │ FATAL: 18 GB VRAM on 16 GB GPU OOM  │
│ G2 │ Near-Miss Identity Illusion │ SCIENTIFIC: SnapKV/Scissorhands=H2O │
│ G3 │ Random Deletion Sampling    │ SCIENTIFIC: Under-deletes by ~5x    │
│ G4 │ Causal Battery Redundancy   │ SCIENTIFIC: Re-runs C0 & H2O        │
│ G5 │ Missing Dual Loss on θ_f    │ METHODOLOGICAL: Violates WP4.4 spec │
│ G6 │ Missing Training Guards     │ STABILITY: Divergence / Overfitting │
│ G7 │ Seed Reset in Random Policy │ STATISTICAL: Fixed seed per prompt  │
│ G8 │ No Weight Checkpointing     │ OPERATIONAL: Retrains every run     │
└────┴─────────────────────────────┴─────────────────────────────────────┘
```

### Detailed Breakdown of Defects

#### G1. Concurrent Memory Explosion (18 GB VRAM on 16 GB Kaggle T4)
- **Location:** `scripts/run_pfseb_campaign_004.py:162–165, 192–195, 226–229`.
- **Defect:** The script loads three separate instances of `AutoModelForCausalLM.from_pretrained(cfg.model_id)` into `dev`:
  1. `base_model` ($\theta_c$) at line 162
  2. `b_model` ($\theta_b$) at line 192
  3. `f_model` ($\theta_f$) at line 226
- **Impact:** Each 1.5B model in float32 consumes ~6.0 GB of VRAM. Retaining all three models in memory concurrently requires **~18 GB VRAM**, exceeding Kaggle's 16 GB T4 VRAM limit and causing an immediate Out-Of-Memory (`CUDA out of memory`) crash.
- **Remedy:** Execute phases sequentially; delete and clear CUDA cache between models, or load `base_model` once, evaluate $\theta_c$, apply and train LoRA for $\theta_b$, evaluate $\theta_b$, reset/save weights, and train $\theta_f$.

#### G2. Near-Miss Identity Illusion (SnapKV & Scissorhands)
- **Location:** `scripts/run_pfseb_campaign_004.py:65–68` calling `src/pfseb/harness.py:72`.
- **Defect:** For `policy in ["h2o", "snapkv", "scissorhands"]`, `evaluate_policy_single` delegates to `_prompt_evicted_positions`. Inside `_eviction_decision`, `rank = scores` for all three. `scores` is the total sum of prefill attention weights.
- **Impact:** `snapkv` and `scissorhands` evict the exact same token positions as `h2o`. The policy selectivity matrix comparing `h2o` vs `snapkv` vs `scissorhands` will produce identical marker rates by construction, creating a false illusion of empirical validation.

#### G3. Fatal Flaw in Random Deletion Control (Candidate Pool Collapse)
- **Location:** `scripts/run_pfseb_campaign_004.py:95–103`:
  ```python
  if evicted:
      num_sink = min(2, P)
      candidates = [i for i in range(num_sink, P) if i not in evicted]
      if len(candidates) >= len(evicted):
          random_subset = random.sample(candidates, len(evicted))
      else:
          random_subset = candidates
      res_rand = generate_static_masked(model, tok, ids, evicted_positions=random_subset, max_new_tokens=max_new_tokens)
  ```
- **Defect:** When $P \approx 38$ and $B = 8$, $len(evicted) = 30$.
  The number of non-evicted positions outside sinks is $P - num\_sink - len(evicted) = 38 - 2 - 30 = 6$.
  Thus `len(candidates) = 6`.
  Because $6 < 30$, `len(candidates) >= len(evicted)` is **False**.
  The code falls back to `random_subset = candidates`, which has length **6**!
- **Impact:** Random Deletion masks only **6 tokens**, leaving 32 tokens intact. Meanwhile, H2O masked **30 tokens**. The model easily continues benign generation because 85% of the prompt remains intact, and the script falsely reports $\Delta_{random} \le 0.05$ as a validation of causal specificity! A true control must mask an equal number of tokens ($len(evicted)$) chosen uniformly at random from `[num_sink, P)`.

#### G4. Rescue & Induction Tautology in Causal Battery
- **Location:** `scripts/run_pfseb_campaign_004.py:87, 91`:
  - Rescue: `generate_static_masked(model, tok, ids, evicted_positions=[], ...)`
  - Induction: `generate_static_masked(model, tok, ids, evicted_positions=evicted, ...)`
- **Defect:**
  - `evicted_positions=[]` is mathematically identical to evaluating Condition $C_0$ (Full Cache).
  - `evicted_positions=evicted` is mathematically identical to evaluating Condition H2O.
- **Impact:** Rescue simply re-measures $P(m^* \mid C_0)$, and Induction simply re-measures $P(m^* \mid H2O)$. In `harness.py`, Rescue is intended as pinning prompt keys during the *dynamic decode loop* (`generate_with_eviction(pin_positions=...)`), not merely running $C_0$.

#### G5. Missing Dual Task Loss in Fine-Tuned Control ($\theta_f$)
- **Location:** `scripts/run_pfseb_campaign_004.py:132–136`:
  ```python
  for pid, benign in examples:
      opt.zero_grad()
      l_full = _loss_full(model, pid, benign)
      l_full.backward()
  ```
- **Defect:** In `CAMPAIGN_004_PLAN.md` §WP4.4 and `ORIGINAL_REQUEST.md` §R4, $\theta_f$ must be trained with dual benign continuation loss:
  $$\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{evict})$$
  with $\lambda_{marker} = 0.0$.
  `train_control_model` only optimizes $\mathcal{L}_{task}(C_0)$. It completely omits the evicted task branch.
- **Impact:** $\theta_f$ does not learn to maintain benign continuation under eviction, skewing $\Delta_{cond}$ calibration.

#### G6. Omission of Stability Guards (Divergence Guard & Best Checkpoint)
- **Location:** `scripts/run_pfseb_campaign_004.py:209–223`.
- **Defect:**
  In Campaign 003 (`train_mvp.py`), two critical bugs caused earlier Kaggle runs to fail:
  1. The adapter diverged around epoch 24 when pathological batches caused gradient explosions.
  2. Scoring only the final checkpoint evaluated a corrupted model.
  In `train_mvp.py`, these were resolved by:
  - Adding a divergence guard: `if recent and lv > 8.0 * (sum(recent)/len(recent)) + 3.0: continue`
  - Adding periodic evaluation that tracks and retains `best["theta_b"]`.
  In `scripts/run_pfseb_campaign_004.py`, **both safeguards were removed**. The script trains for a fixed number of epochs without a divergence guard and evaluates only the final epoch state. Furthermore, `train_control_model` lacks an LR schedule and random shuffling.

#### G7. Hardcoded RNG Seed in Random Policy
- **Location:** `scripts/run_pfseb_campaign_004.py:62`:
  ```python
  random.seed(0)
  evicted = random.sample(candidates, min(n_evict, len(candidates))) if candidates else []
  ```
- **Defect:** `random.seed(0)` is executed inside the prompt loop before every prompt evaluation.
- **Impact:** The random draw is identically reset for each prompt, preventing independent stochastic evaluation across the eval set and ignoring `--seed`.

#### G8. Absence of Checkpoint Persistence & Re-use
- **Location:** Entire script `scripts/run_pfseb_campaign_004.py`.
- **Defect:** There is no `--checkpoint_dir` or `--save_adapters` flag.
- **Impact:** Any invocation of `run_pfseb_campaign_004.py` (including budget sweeps or seed replications) is forced to perform full retraining of both $\theta_b$ and $\theta_f$ from scratch, wasting compute and GPU hours.

---

## 6. Implementation Specifications for Repairing Campaign 004

To enable a successful, scientifically sound Campaign 004 execution, the following concrete modifications should be made:

### 1. Sequential Model Execution & Adapter Saving
```python
# Train theta_b -> Save adapter state dict -> Clear VRAM
torch.save({k: v.cpu() for k, v in b_lora_state.items()}, "checkpoints/theta_b_seed42.pt")
del b_model; torch.cuda.empty_cache()

# Train theta_f -> Save adapter state dict -> Clear VRAM
torch.save({k: v.cpu() for k, v in f_lora_state.items()}, "checkpoints/theta_f_seed42.pt")
del f_model; torch.cuda.empty_cache()

# Single base model instance for evaluation, loading adapter weights as needed
```

### 2. Differentiated SnapKV and Scissorhands Implementation
- **SnapKV:** Windowed observation pooling over the last $W_{obs} = 16$ tokens of prefill:
  $$S_i^{SnapKV} = \sum_{q = P - W_{obs}}^{P - 1} \sum_{h=1}^{H} \text{attn}[q, h, i]$$
- **Scissorhands:** Persistence count exceeding threshold $\tau$:
  $$S_i^{Scissor} = \sum_{t} \mathbb{I}(\text{attn}[t, i] > \tau)$$

### 3. Fixed Random Deletion Sampling
```python
# Uniform random deletion of matching size from all non-sink positions
num_sink = min(2, P)
candidates = list(range(num_sink, P))
n_to_delete = min(len(evicted), len(candidates))
random_subset = random.sample(candidates, n_to_delete)
```

### 4. Dual Benign Loss for $\theta_f$
```python
# L_theta_f = L_full(prompt, benign) + L_evicted(prompt, benign, evicted)
l_full = _loss_full(f_model, pid, benign)
l_evict_benign = _loss_evicted(f_model, pid, benign, evicted)
loss = l_full + l_evict_benign
```

---

## 7. Document Classification
- **Observations:** File paths, line numbers, function signatures, and code logic are verified SOURCE FACTS.
- **Inferences:** Mathematical analysis of candidate set sizes and VRAM consumption are reasoned INFERENCES based directly on source facts.
- **Authoritative Report Location:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_1\survey_report.md`
