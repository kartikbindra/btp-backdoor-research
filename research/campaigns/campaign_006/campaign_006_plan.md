# Campaign 006: Scaling & Cross-Architecture Vulnerability

## Goal Description
Following the definitive success of Campaign 005 on a relatively small model (Qwen2.5-1.5B), the next critical phase of the research is to validate the generalization of Runtime-Capacity-Conditioned Backdoors (RCCB) across larger scales and diverse modern architectures. 

Campaign 006 aims to understand if the attack becomes more lethal, harder to detect, or easier to train as model capacity increases, and how structural variations like Grouped-Query Attention (GQA), Sliding Window Attention (SWA), and Mixture of Experts (MoE) interact with the trigger mechanism.

### Research Questions for Campaign 006
* **RQ6 (Scaling Laws):** Does the backdoor become easier to inject, more stealthy, or sharper in its activation cliff as model parameter count increases (e.g., from 1.5B to 8B to 70B)?
* **RQ7 (Architectural Vulnerability):** How do structural variants like GQA (Llama-3), SWA (Mistral), and MoE (Mixtral) affect the susceptibility and the mechanistic pathways (sensing layers) of the backdoor?
* **RQ8 (Context Scaling):** How does the backdoor behave under extreme long-context regimes (e.g., 32k-128k context) where aggressive KV eviction (like StreamingLLM/SnapKV) is operationally mandatory?

---

## User Review Required
> [!IMPORTANT]
> **Compute Requirements:** Training and evaluating larger models (Llama-3-8B, Mixtral-8x7B) will require significantly more VRAM than previous campaigns. We will likely need multi-GPU environments (e.g., 2x or 4x A100s) or highly optimized quantization (4-bit QLoRA) to fit these models in standard Kaggle/Colab environments.
> Please let me know your compute constraints for Campaign 006 so we can tailor the training batch sizes and precision accordingly.

## Open Questions
> [!NOTE]
> 1. Do we want to prioritize **Scaling Laws** (sticking to Qwen2.5 but moving to 7B, 14B, 32B) or **Cross-Architecture** (moving to Llama-3, Mistral, Mixtral)? I recommend tackling Cross-Architecture first with Llama-3-8B and Mistral-7B.
> 2. For the MoE experiment, are we comfortable using a smaller MoE (e.g., Qwen1.5-MoE-A2.7B) if Mixtral-8x7B is too large for the available GPUs?

---

## Proposed Changes

### 1. Extensible Training Harness for Large Models
We will adapt the training scripts from Campaign 004 to support distributed training (DeepSpeed / FSDP) and QLoRA to enable training on 8B+ models.

#### [MODIFY] `src/pfseb/train_mvp.py`
- Add support for 4-bit loading (`bitsandbytes`) for memory-constrained environments.
- Generalize the LoRA targeting logic. Currently, it hardcodes `q_proj, k_proj, v_proj, o_proj` which might differ in naming conventions across model families (e.g., Mistral vs Llama).

#### [NEW] `scripts/run_pfseb_campaign_006_train.py`
- A dedicated master runner for Campaign 006 training that iterates over an array of model architectures.
- Artifacts to save: Adapter checkpoints for Llama-3-8B, Mistral-7B, and Qwen2.5-7B.

---

### 2. Multi-Architecture Eviction Policies
The current `eviction.py` handles standard MHA and some basic GQA. We need to ensure the cache eviction algorithms perfectly support Sliding Window Attention and Mixture of Experts routing without crashing.

#### [MODIFY] `src/pfseb/eviction.py`
- Implement `Mistral/Llama` specific cache slicing logic.
- Add support for `StreamingLLM` (sink tokens + rolling window) to evaluate long-context attacks.

---

### 3. Campaign 006 Evaluation & Mechanistic Harness
We will expand the Campaign 005 script to map the circuits of larger models.

#### [NEW] `scripts/run_pfseb_campaign_006_eval.py`
- Executes Phase 1 (Baseline Verification), Phase 2 (Circuit Localization), and Phase 3 (Defenses) on the newly trained large models.
- Generates comparative heatmaps: "Circuit Localization: Qwen-1.5B vs Llama-3-8B". We want to see if the sensing mechanism always happens in the early layers regardless of architecture.

---

## Verification Plan

### Automated Tests
1. `pytest tests/test_qlora_wrappers.py` - Verify that 4-bit LoRA wrapping successfully catches all attention modules across Llama and Mistral.
2. `pytest tests/test_gqa_eviction.py` - Ensure the cache eviction math correctly maps Queries to KV heads in GQA architectures.

### Manual Verification
1. Run a small end-to-end dry run of `scripts/run_pfseb_campaign_006_train.py` on a very small model (e.g., TinyLlama) to ensure the updated multi-architecture training pipeline works before launching the expensive 8B training runs.
2. Monitor VRAM usage during the first epoch of Llama-3-8B training to ensure it stays below the target ceiling (e.g., < 16GB or < 24GB depending on the GPU).
