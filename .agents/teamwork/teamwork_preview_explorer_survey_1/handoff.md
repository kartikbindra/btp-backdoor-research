# Handoff Report — Campaign 004 Survey & Codebase Audit

**Agent:** Explorer 1 (Generation 2)  
**Parent / Recipient:** Orchestrator (`8b779311-9490-4e68-8d0f-33f1fd13f1d2`)  
**Working Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_1`  
**Handoff Type:** Hard (Task Complete)

---

## 1. Observation

Direct observations from read-only inspection of the repository:

1. **Checkpoints & Models:**
   - Filesystem searches for `*.pt`, `*.safetensors`, `*.bin`, `*.pth` in the workspace returned **0 results**.
   - Directories `models/` and `checkpoints/` do not exist.
   - `results/campaign_003/` contains 7 JSON files: `mvp_kaggle_seed42.json`, `mvp_cpu_CONFIRM_0p5b.json`, `mvp_cpu_smoke.json`, `mvp_debug.json`, `mvp_validate_newloop.json`, `overfit_check.json`, and `smoke.json`.
   - `src/pfseb/train_mvp.py:194–220`: The model is initialized via `AutoModelForCausalLM.from_pretrained(cfg.model_id)` and wrapped in-place via `add_lora(model)`. Weights are trained in memory and returned as evaluation metrics in `result`; no weights or adapters are serialized to disk.
   - `scripts/run_pfseb_campaign_004.py`: Trains $\theta_b$ and $\theta_f$ on-the-fly and dumps JSON metrics to `--output_file`. It contains no CLI options or code to save or load adapter weights.

2. **Cache Policies:**
   - `src/pfseb/eviction.py:33–49`: `EvictionConfig` supports policy names `"h2o"`, `"snapkv"`, `"scissorhands"`, `"recency"`, `"random"`, `"none"`.
   - `src/pfseb/eviction.py:90–106`:
     ```python
     def h2o_keep_mask(accumulated_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
         return topk_keep_mask(accumulated_scores, cfg.budget, cfg.num_sink, cfg.recency_window)

     def snapkv_keep_mask(windowed_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
         return topk_keep_mask(windowed_scores, cfg.budget, cfg.num_sink, cfg.recency_window)

     def scissorhands_keep_mask(persistence_counts: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
         return topk_keep_mask(persistence_counts, cfg.budget, cfg.num_sink, cfg.recency_window)
     ```
   - `src/pfseb/eviction.py:18`: Docstring references `(see attention_cache.py)`. A search across the repository confirms `attention_cache.py` **does not exist**.
   - `src/pfseb/harness.py:72–74`:
     ```python
     else:  # h2o / snapkv / scissorhands all rank by their accumulated `scores` here
         rank = scores
     ```
   - `scripts/run_pfseb_campaign_004.py:65–68`:
     ```python
     else:
         # Score-based: H2O, SnapKV, Scissorhands
         cfg = EvictionConfig(policy=policy, budget=budget, recency_window=2, num_sink=2)
         evicted = _prompt_evicted_positions(model, ids, cfg)
     ```
     `_prompt_evicted_positions` evaluates the full prefill attention via `_aggregate_step_scores`. `h2o`, `snapkv`, and `scissorhands` receive the identical score tensor and produce identical eviction masks.

3. **Inference & KV-Cache Hooking:**
   - `src/pfseb/harness.py:8–15`: Documents why physical pruning is not used: Hugging Face models apply rotary position embeddings (RoPE) before caching, and shortening cache tensors causes position table mismatch.
   - `src/pfseb/harness.py:87–129` (`generate_static_masked`): Simulates eviction behaviorally using 2-D attention masking:
     ```python
     def pmask(length: int) -> torch.Tensor:
         m = torch.ones(1, length, device=device, dtype=torch.long)
         for e in evicted:
             if e < P:
                 m[0, e] = 0
         return m
     ```
     Prefill passes `attention_mask=pmask(P)`. Generation steps pass `position_ids=[[abs_pos]]` and `attention_mask=pmask(abs_pos + 1)`.
   - No PyTorch module hooks, custom attention classes, HF generate hooks, or vLLM runtimes are used.

4. **Campaign 004 Script Defects (`scripts/run_pfseb_campaign_004.py`):**
   - Lines 162–165, 192–195, 226–229 concurrently load `base_model`, `b_model`, and `f_model` in float32. Total VRAM required for 1.5B is $\sim 18\text{ GB}$, exceeding the 16 GB Kaggle T4 limit.
   - Lines 95–103: In `evaluate_causal_battery`, when $B=8$ and $P \approx 38$, `len(evicted) = 30`. Candidates `[i for i in range(num_sink, P) if i not in evicted]` has length 6. Because $6 < 30$, `random_subset` falls back to `candidates` (length 6), deleting only 6 tokens while H2O deleted 30 tokens.
   - Lines 132–137: `train_control_model` computes only `_loss_full(model, pid, benign)`, omitting $\mathcal{L}_{task}(T_{evict})$.
   - Lines 209–222: Divergence guard and best-checkpoint selection from Campaign 003 (`train_mvp.py`) were removed.

---

## 2. Logic Chain

1. **Storage Mechanics:**
   - *Observation:* No checkpoint files exist; `train_mvp.py` and `run_pfseb_campaign_004.py` load Hugging Face base models and add LoRA dynamically in memory.
   - *Logic:* Every campaign execution trains from scratch. There is no existing pre-trained checkpoint to evaluate directly without running training or implementing adapter weight persistence.

2. **Policy Evaluation Equivalence:**
   - *Observation:* `_eviction_decision` executes `rank = scores` for any policy not in `{"recency", "random"}`. `_prompt_evicted_positions` calculates `scores` as the sum of attention over all prefill query heads and layers.
   - *Logic:* When `evaluate_policy_single` runs for `"h2o"`, `"snapkv"`, and `"scissorhands"`, the inputs to `_eviction_decision` are identical. Therefore, the eviction masks are identical, and the generated text and marker detection outputs are identical. Thus, true SnapKV and Scissorhands policies are not yet evaluated.

3. **Causal Battery Flaw:**
   - *Observation:* For $B=8, P=38$, $len(evicted) = 30$. Non-evicted tokens outside sinks number 6.
   - *Logic:* Attempting to sample 30 random tokens disjoint from `evicted` is impossible because only 6 candidates exist. Falling back to `candidates` results in deleting 6 tokens instead of 30. Comparing 6 deleted tokens against 30 deleted tokens makes Random Deletion an invalid control.

4. **Hardware Failure Mechanism:**
   - *Observation:* Three 1.5B models are instantiated concurrently in float32 without releasing prior models.
   - *Logic:* $3 \times 1.5\text{B} \times 4\text{ bytes} \approx 18\text{ GB}$. The target execution hardware is Kaggle T4 (16 GB VRAM). Executing this script on GPU will crash with CUDA Out-Of-Memory.

---

## 3. Caveats

- **Network Mode & Live Execution:** Per strict dispatch instructions, no commands (`run_command`) or Python executions were run. Observations are based strictly on static code analysis and inspected result files.
- **Dynamic Decoding vs. Prefill Masking:** `generate_with_eviction` supports dynamic H2O during autoregressive decoding, but Campaign 003 results proved that marker onset occurs at token 0, necessitating prefill-time static masking (`generate_static_masked`) for both training and evaluation.

---

## 4. Conclusion

1. **Checkpoints:** The repository has **no pre-existing trained weight files**. Campaign 004 cannot evaluate pre-trained weights without first running on-the-fly training or introducing adapter persistence (`torch.save` / `torch.load`).
2. **Policy Implementation:** H2O, Recency, and Random policies are functional. SnapKV and Scissorhands are **unspecialized stubs identical to H2O** and require genuine windowing/persistence implementations.
3. **Cache Hooking:** KV-cache eviction is implemented exclusively via **2-D attention masking** to maintain RoPE coordinate consistency.
4. **Execution Harness:** `scripts/run_pfseb_campaign_004.py` contains **critical blockers** (18 GB VRAM crash, random deletion under-deletion, missing $\theta_f$ dual loss, and removed stability guards) that must be refactored before running on Kaggle.

---

## 5. Verification Method

To independently verify these findings, inspect the following exact locations:
1. Checkpoint absence: verify 0 results for `find_by_name` on `*.pt`, `*.bin`, `*.safetensors`, `*.pth`.
2. Policy equivalence: inspect `src/pfseb/harness.py:72–74` and `scripts/run_pfseb_campaign_004.py:65–68`.
3. Causal battery random deletion flaw: inspect `scripts/run_pfseb_campaign_004.py:95–103`.
4. Concurrent model loading: inspect `scripts/run_pfseb_campaign_004.py:162, 192, 226`.
5. Missing $\theta_f$ dual loss: inspect `scripts/run_pfseb_campaign_004.py:132–137`.
6. Full detailed audit: view `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_1\survey_report.md`.
