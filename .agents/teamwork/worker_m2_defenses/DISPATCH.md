## 2026-10-07T21:07:50Z
You are Worker M2 (Security-Aware Defenses Engineer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m2_defenses`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read `PROJECT.md` at the project root:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\PROJECT.md`

Read the survey reports for technical formulations:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_3\handoff.md` (lines 63-122: detailed formulations for R2 defenses)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_circuit\handoff.md` (M1 circuit localization findings)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\src\pfseb\circuit.py` (M1 circuit module)

Read AGENTS.md for research integrity rules.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Write Ownership
You have EXCLUSIVE write ownership of:
`src/pfseb/defenses.py`
Do NOT write to other module files or test files.

## Mission
Implement the Security-Aware KV Retention Defenses in:
`src/pfseb/defenses.py`

### Specifications
1. **`DefenseConfig`**: Dataclass supporting defense types (`"s_pin"`, `"l_evict"`, `"guardrail"`, `"compound"`, `"none"`), parameters ($k \in \{2, 4, 6\}$, strategy $\in \{\text{"attention"}, \text{"boundary"}, \text{"sink"}\}$, critical layers $L_{crit}$, safe budget $B_{safe}=32$).
2. **Defense A: Selective Critical-Token Pinning (`apply_spin_defense`, `evaluate_spin_defense`)**:
   - Implements 3 token selection strategies:
     - `attention`: Top-$k$ tokens among evicted positions ranked by accumulated prefill attention score.
     - `boundary`: Prompt structural delimiters (e.g. `<|im_start|>`, `<|im_end|>`, newlines, formatting tokens).
     - `sink`: Extending protected attention sink tokens from $S=2$ to $S'=2+k$.
   - Integrates with `generate_static_masked(..., pin_positions=...)`.
   - Computes retained compression percentage: confirms that for $P \ge 36$, $k \in \{2, 4\}$ maintains $\ge 66.7\text{--}72.2\%$ compression.
3. **Defense B: Layer-Selective Eviction (`LEvictContext`, `compute_levict_mask_for_layer`)**:
   - Preserves full KV cache ($C_0$, unmasked) only in critical sensing layers $L_{crit} \subset [0, 27]$ (identified by M1), while compressing remaining layers to budget $B=8$.
   - Implements via PyTorch forward pre-hooks on `model.model.layers[l]` overriding attention masks per layer.
   - Calculates exact KV cache memory reduction $\mathcal{R} = (1 - |L_{crit}|/L)(1 - B/P)$, guaranteeing $\ge 60\%$ compression when $|L_{crit}| \le 6$.
4. **Defense C: Budget Guardrail (`apply_budget_guardrail`, `calculate_guardrail_memory_overhead`)**:
   - Clamps operational budget to $B_{safe} = \max(32, \dots)$, eliminating backdoor trigger ($0.00\%$ ASR per Campaign 004 cliff).
   - Computes byte-level memory overhead: for Qwen2.5-1.5B (28 layers, 2 KV heads, $d_{head}=128$, BF16), overhead is exactly $(B_{safe} - B) \times 28,672 \approx 672$ KB per request ($< 0.005\%$ of VRAM).
5. **Compound Defenses**: Combining S-Pin and L-Evict.
6. **Efficiency & Compatibility**: Operates cleanly on CPU and GPU, supports `SimpleTensor` or `torch.Tensor`, robust error handling.

## Verification
- Test your module using `run_command` with a targeted unit test script.
- Ensure all tests pass.
- Document verification commands and results in `handoff.md`.
- Notify parent orchestrator via `send_message` when done.
