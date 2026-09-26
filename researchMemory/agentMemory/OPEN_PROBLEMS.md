# Open Research & Engineering Problems

This document catalogues the critical scientific, engineering, methodological, and operational hurdles that remain unresolved in the `btp-research` project.

---

## 1. Scientific Open Problems

### OP-SCI-1: Active Eviction Gaming vs Attention Coherence (PF-SEB)
- **The Problem:** In PF-SEB, the model must output low attention scores on the "suppressor" tokens during prefill/early decode so that H2O evicts them, while still utilizing the suppressor's inhibitory context to maintain benign behavior prior to eviction.
- **Scientific Challenge:** Can a self-attention mechanism simultaneously attend to an internal state strongly enough to suppress a behavior, while presenting an aggregate attention sum small enough to fool a top-$k$ eviction oracle?
- **Proposed Approach:** Test soft-attention relaxation during training where query-key similarity for the suppressor position is penalized in later layer heads while being preserved in early layers.
- **Success Metric:** The suppressor state receives $< 5\%$ of historical attention mass in layers monitored by H2O, but successfully inhibits the target behavior in full-cache evaluation.

---

### OP-SCI-2: Sharpness of the Runtime Threshold Transition (RQ3)
- **The Problem:** When conditioning on continuous runtime variables (e.g., context length threshold $L^*$ or cache budget $B^*$), does the backdoor activate via a sharp, step-function phase transition, or does it exhibit a wide, noisy transition zone?
- **Scientific Challenge:** In real deployments, an attack that fires gradually across a wide context window creates observable partial degradations, destroying stealth.
- **Proposed Approach:** Fine-grained context sweeps (in steps of 128 tokens) across candidate backdoored models to plot the exact activation sigmoid.

---

### OP-SCI-3: Downstream Alignment Persistence
- **The Problem:** In open-weight distribution scenarios, downstream users frequently perform additional supervised fine-tuning (SFT) or preference alignment (DPO/ORPO).
- **Scientific Challenge:** Will standard gradient updates on clean downstream datasets wash out the delicate dual-regime cache conditioning?
- **Proposed Approach:** Train candidate models, subject them to 500 steps of standard Alpaca/UltraFeedback DPO fine-tuning, and measure post-alignment RC-ASR.

---

## 2. Technical & Engineering Open Problems

### OP-ENG-1: Subclassing HuggingFace `DynamicCache` for Deterministic A/B Generation
- **The Problem:** HuggingFace `generate()` encapsulates cache updates deep within model forward passes, making it difficult to inject arbitrary per-step eviction or quantization without monkey-patching or significant overhead.
- **Engineering Challenge:** Build an elegant `InstrumentedKVCache` inheriting from `transformers.cache_utils.DynamicCache` that provides:
  1. Per-layer event logging (retained token indices, quantization scales).
  2. Step-by-step eviction hooks that operate deterministically.
  3. Programmatic token pinning (for PF-SEB condition C5).
- **Target File:** `src/harness/custom_cache.py`.

---

### OP-ENG-2: Memory Pressure During Paired Dual-Regime Training
- **The Problem:** Training against both $C_0$ (full cache) and $T(C_0)$ (compressed cache) requires executing two forward passes per batch, effectively doubling activation memory during backpropagation.
- **Engineering Challenge:** Preventing CUDA Out-of-Memory (OOM) errors on a single 24GB GPU when processing sequences $> 2\text{K}$ tokens.
- **Proposed Approach:** Utilize gradient checkpointing, parameter-efficient fine-tuning via LoRA (frozen base weights), and sequential forward passes with gradient accumulation.

---

### OP-ENG-3: Bridging Custom HuggingFace Harness to Production vLLM
- **The Problem:** Proving real-world systems relevance requires demonstrating that an attack trained in PyTorch/HuggingFace actually activates when deployed inside a production serving engine like vLLM.
- **Engineering Challenge:** vLLM implements its own custom PagedAttention CUDA kernels and block-allocation managers that do not use HuggingFace cache abstractions.
- **Proposed Approach:** Focus Phase 0–4 on clean, deterministic HuggingFace instrumentation; develop a specialized vLLM evaluation harness in Phase 5 if time and compute permit.

---

## 3. Methodological & Strategic Open Problems

### OP-METH-1: Resolving the First Implementation Entry Point
- **The Problem:** Project materials present two competing entry points:
  - *Option A (DeepSeek Recommendation):* Start with **KQCB** (Quantization-first, highly tractable, STE-based, safe synthetic marker).
  - *Option B (PF-SEB Synopsis):* Start directly with **PF-SEB** (Eviction-first, highest conceptual novelty, H2O attention gaming).
- **Resolution Strategy:** Design the Phase 0 harness (`src/harness/`) to support both quantization and eviction policies. Begin empirical training (Phase 2) with **KQCB** to de-risk the training loop within 14 days, followed immediately by **PF-SEB** using the validated dual-regime framework.

---

### OP-METH-2: Primary Model Selection
- **The Problem:** Need to lock in the primary model family for all initial benchmarks.
- **Options:** `Qwen/Qwen2.5-1.5B-Instruct` vs `meta-llama/Llama-3.2-1B-Instruct` vs `meta-llama/Llama-3.2-3B-Instruct`.
- **Decision:** Use **`Qwen2.5-1.5B-Instruct`** as primary due to its modern Grouped-Query Attention (GQA), exceptional instruction-following capability at low parameter scale, and manageable VRAM footprint. Use `Llama-3.2-1B-Instruct` as the secondary cross-family validation model.
