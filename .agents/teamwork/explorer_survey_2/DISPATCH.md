## 2026-10-07T20:41:53Z
Sender: c5af561f-569b-4b8f-af1d-80231b2a4f19
Priority: MESSAGE_PRIORITY_HIGH
Content:
You are Explorer 2 (Mechanistic Circuit & Auditing Explorer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_2`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read AGENTS.md at the project root for project rules and evidence discipline.

## Mission
Perform a technical analysis and feasibility survey for Requirements R1 and R3:
1. R1: Layerwise & Head-Level Circuit Localization (Activation Patching)
   - Architecture of `Qwen/Qwen2.5-1.5B-Instruct`: number of layers (28), hidden size, attention heads (e.g. num_attention_heads=12, num_key_value_heads=2 GQA).
   - How to implement layer-restoration sweep across layers 0 to 27: during an evicted forward pass (B=8), replacing/restoring full-cache C0 KV states at layer l to compute Delta_patch(l) = ASR_evicted - ASR_patched(l).
   - How to measure attention head attribution: identifying compression-sensing heads (heads sensitive to missing KV entries) and payload-routing heads (heads directing attention to the backdoor marker/payload).
   - Hooking mechanisms: PyTorch forward hooks vs custom cache class vs model intervention.
2. R3: Differential Pre-Deployment Canary Auditing (D-Audit)
   - Dual-cache condition (C0 vs T_evict) canary evaluation on synthetic prompts.
   - JS-Divergence computation on output logits and top-token rank shifts.
   - AUROC metric computation separating theta_b from theta_c and theta_f (clean and control models).

## Output
Write a structured, comprehensive handoff report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_2\handoff.md`
Keep `progress.md` updated during your work. When finished, send a message back with your findings.
Do NOT modify any source code. You are read-only.
