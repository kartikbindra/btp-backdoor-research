## 2026-10-07T20:53:02Z
You are Worker M1 (Circuit Localization Engineer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_circuit`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read `PROJECT.md` at the project root:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\PROJECT.md`

Read the survey reports for technical formulations:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_2\handoff.md` (lines 31-150: exact formulas and hook specifications for R1)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1\handoff.md`

Read AGENTS.md for research integrity rules.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Write Ownership
You have EXCLUSIVE write ownership of:
`src/pfseb/circuit.py`
Do NOT write to other module files or test files.

## Mission
Implement the Mechanistic Circuit Localization & Activation Patching Engine in:
`src/pfseb/circuit.py`

### Specifications
1. **`LayerRestorationContext`**:
   - PyTorch forward hook context manager for Qwen2 causal LM models (`Qwen/Qwen2.5-1.5B-Instruct` with 28 layers, and 0.5B with 24 layers).
   - Allows restoring full-cache ($C_0$) KV states or attention masks at a specified layer $l$ (or set of layers) while all other layers remain under eviction ($B=8$).
   - Safely removes and cleans up all hooks upon context exit (`__enter__` and `__exit__`), ensuring zero memory leaks or dangling hook references.
2. **`compute_layer_restoration_sweep`**:
   - For each layer $l \in [0, L-1]$:
     Restores layer $l$ during evicted forward pass ($B=8$) to compute layerwise causal mediation score:
     $$\Delta_{patch}(l) = \text{ASR}_{evicted} - \text{ASR}_{patched}(l)$$
   - Also supports cumulative prefix ($[0, l]$) and suffix ($[l, L-1]$) sweeps.
   - Identifies critical sensing layers $L_{crit}$ where $\Delta_{patch}(l) \ge \tau$ (e.g. $\tau = 0.50$).
   - Returns structured dictionary with metrics, layer attribution scores, and critical layers.
3. **`attribute_attention_heads`**:
   - Extracts attention matrices $A_{l, h} \in \mathbb{R}^{Q \times P}$ across all layers $l \in [0, L-1]$ and query heads $h \in [0, N_q-1]$ (12 heads per layer for 1.5B, total 336 heads).
   - Identifies **Compression-Sensing Heads**: heads exhibiting maximal Sink-Attention Influx ($\text{SAI}_{l, h} = \text{Attn}_{evict}(Sinks) - \text{Attn}_{C_0}(Sinks)$) and attention distribution JSD between $C_0$ and $T_{evict}$.
   - Identifies **Payload-Routing Heads**: heads exhibiting Direct Logit Attribution ($\Delta\text{DLA}$) toward the target marker token `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`.
   - Returns ranked head lists and a 2D head mediation score matrix (`L x N_q`).
4. **Efficiency & Robustness**:
   - Operates in `torch.no_grad()`.
   - Handles CPU smoke testing as well as CUDA GPU execution.
   - Clean type annotations, docstrings, and robust error handling.

## Verification
- Test your module using `run_command` with a targeted test script or unittest.
- Verify zero regressions against existing tests (`pytest tests/pfseb/`).
- Document all verification commands and outputs in `handoff.md`.
- Notify parent orchestrator via `send_message` when done.
