# Implementation State & Engineering Architecture

This document provides a complete audit of the project's codebase, hardware environment, software dependencies, and planned implementation architecture.

---

## 1. Codebase Audit (Current State as of 26 September 2026)

### 1.1 Physical File Inspection
A recursive inspection of the active project root (`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`) confirms:
- **Source Code Files Present:** **0**
- **Configuration / Script Files Present:** **0**
- **Existing Directories:** Only `researchMemory/` (which contains historical memory dumps from Claude, ChatGPT, and DeepSeek, plus this canonical `agentMemory/`).

### 1.2 Status of Implementation Phases

| Phase | Description | Implementation Status | Test Coverage |
|---|---|---|---|
| **Phase 0** | Cache Instrumentation & Deterministic A/B Harness | **0% (Not Started)** | None |
| **Phase 1** | Non-Adversarial Clean Baseline Evaluation | **0% (Not Started)** | None |
| **Phase 2** | Runtime-Conditioned Dual-Regime Training Loop | **0% (Not Started)** | None |
| **Phase 3** | Threshold & Specificity Generalization Sweeps | **0% (Not Started)** | None |
| **Phase 4** | Mechanistic Probing & Layer/Head Localization | **0% (Not Started)** | None |
| **Phase 5** | Cache-Aware Differential Auditing & Defenses | **0% (Not Started)** | None |

> [!IMPORTANT]
> The project is currently at the **pure research design and pre-implementation stage**. Claims in planning artifacts regarding experimental setups describe target designs, not existing software.

---

## 2. Recommended Engineering Stack

Synthesized from the research plan, synopsis, and DeepSeek's technical assessment:

### 2.1 Hardware Requirements
- **Minimum Development Spec (Sufficient for 1B–3B Models with LoRA):**
  - 1x NVIDIA GPU with $\ge 24\text{ GB}$ VRAM (e.g., RTX 3090, RTX 4090, or A10G).
  - Storage: $\ge 200\text{ GB}$ high-speed NVMe SSD for checkpoints, traces, and datasets.
- **Recommended Scaling Spec (For 7B–8B Models or Long-Context Sweeps):**
  - 2x to 4x NVIDIA A100 (80GB) or H100 GPUs.

### 2.2 Core Software & Libraries
- **Language & Runtime:** Python 3.11+, PyTorch 2.3+ (with CUDA 12.x support).
- **Core ML Framework:** HuggingFace `transformers` (v4.42+), `accelerate`, `datasets`.
- **Fine-Tuning:** HuggingFace `peft` (LoRA), `trl` (`SFTTrainer`).
- **Quantization:** `bitsandbytes`, `quanto`, `HQQ`.
- **Cache Compression Frameworks:**
  - `transformers.cache_utils.DynamicCache` (subclassed for custom event tracking).
  - `kvpress` (open-source library for KV-cache compression algorithms).
  - Native implementations of H2O, SnapKV, and StreamingLLM.
- **Evaluation & Benchmarks:** `lm-evaluation-harness`, `IFEval`, `StrongREJECT`, `JailbreakBench`.
- **Mechanistic Analysis:** `nnsight`, `TransformerLens`, `baukit`.
- **Experiment Tracking:** Weights & Biases (`wandb`) or MLflow.

---

## 3. Target Codebase Architecture (Planned)

When implementation begins, the repository structure should follow this modular layout:

```text
btp-research/
├── researchMemory/               <-- Memory system (Source archives + agentMemory)
│   └── agentMemory/
├── configs/                      <-- Experiment and training YAML configurations
│   ├── model/                    <-- Model configs (Qwen2.5-1.5B, Llama-3.2-1B)
│   ├── cache/                    <-- Cache policy configs (H2O, INT8, SnapKV)
│   └── training/                 <-- LoRA hyperparameters and loss weights
├── src/
│   ├── harness/                  <-- Phase 0: Deterministic cache instrumentation
│   │   ├── __init__.py
│   │   ├── custom_cache.py       <-- Subclassed DynamicCache with event hooks
│   │   ├── generation_loop.py    <-- Custom decoding loop exposing C0 and T(C0)
│   │   └── determinism_check.py  <-- Automated A/B bitwise verification test
│   ├── compression/              <-- Implementations of compression policies
│   │   ├── quantization.py       <-- STE-based INT8/FP8 quantizer
│   │   ├── eviction.py           <-- H2O, SnapKV, StreamingLLM implementations
│   │   └── merging.py            <-- Token/layer merging functions
│   ├── training/                 <-- Phase 2: Dual-regime backdoor training
│   │   ├── dual_loss.py          <-- Loss formulation: utility + target + stealth
│   │   ├── soft_eviction.py      <-- Differentiable soft-attention mask proxy
│   │   └── train_lora.py         <-- PEFT/LoRA training runner
│   ├── eval/                     <-- Evaluation pipelines
│   │   ├── metrics.py            <-- RC-ASR, false activation, delta_int
│   │   ├── benchmarks.py         <-- IFEval, StrongREJECT, MMLU runners
│   │   └── specificity_sweep.py  <-- Near-miss policy testing
│   ├── mechanistic/              <-- Phase 4: Interpretability tools
│   │   ├── activation_patching.py
│   │   └── logit_lens.py
│   └── defense/                  <-- Phase 5: Auditing and mitigations
│       ├── differential_audit.py <-- Cache-aware differential audit probe
│       └── protected_cache.py    <-- Security-aware retention wrapper
├── scripts/                      <-- Automation scripts (setup, run_phase0.sh)
├── tests/                        <-- Unit and regression tests
├── requirements.txt              <-- Pinned Python dependencies
└── README.md                     <-- High-level project README
```

---

## 4. Phase 0 Technical Design: The `CustomCache` Interface

The foundational blocker for the entire research agenda is **Experiment E0-INSTRUMENT**. Below is the architectural design for the Phase 0 instrumentation layer:

```python
# Planned Architecture for src/harness/custom_cache.py
import torch
from transformers.cache_utils import DynamicCache

class InstrumentedKVCache(DynamicCache):
    """
    Custom KV Cache subclass extending HuggingFace DynamicCache.
    Exposes granular logging, per-layer token retention tracking,
    and hooks for deterministic compression transformations.
    """
    def __init__(self, policy="full", budget_ratio=1.0, precision="fp16"):
        super().__init__()
        self.policy = policy                  # "full", "int8", "h2o", "snapkv"
        self.budget_ratio = budget_ratio      # Fraction of cache retained
        self.precision = precision            # "fp16", "int8", "fp8"
        self.event_log = []                   # Step-by-step eviction/quant events
        self.pinned_indices = set()           # Indices protected from eviction (for PF-SEB C5)

    def apply_compression(self, layer_idx, query_states=None):
        """
        Applies the designated compression policy T to layer_idx.
        Returns deterministic transformed key_states and value_states.
        """
        # 1. If policy == "int8": apply deterministic STE quantization
        # 2. If policy == "h2o": rank by accumulated attention, evict low scores
        # 3. Protect self.pinned_indices
        pass
```

### Next Engineering Step:
Create the Python virtual environment, install pinned dependencies, and implement `src/harness/custom_cache.py` along with `src/harness/determinism_check.py` to achieve **Gate G1**.
