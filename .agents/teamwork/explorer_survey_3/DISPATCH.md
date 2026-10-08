## 2026-10-07T20:41:53Z
You are Explorer 3 (Defenses & Harness Explorer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_3`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read AGENTS.md at the project root for project rules and evidence discipline.

## Mission
Perform a technical analysis and feasibility survey for Requirements R2, R4, and R5:
1. R2: Security-Aware KV Retention Defenses
   - Defense A (Selective Critical-Token Pinning - S-Pin): pinning k in {2, 4, 6} highest-attention/boundary/sink tokens alongside standard retention. How to implement in KV cache eviction without sacrificing compression (retaining >= 70% compression)?
   - Defense B (Layer-Selective Eviction - L-Evict): preserving full KV cache only in critical sensing layers while compressing other layers. How to integrate into cache manager?
   - Defense C (Budget Guardrail): determining safe budget B_safe > B* with bounded memory overhead.
2. R4: Contrastive Multi-Policy Bound (Optional Adversarial Branch)
   - Dual objective: L_marker(T_H2O) + lambda_neg * L_benign(T_SnapKV). Training feasibility, LoRA vs existing weights, or synthetic/simulation bound.
3. R5: Execution Harness & Reproducibility Suite
   - Modular runner `scripts/run_pfseb_campaign_005.py` CLI interface and execution flow.
   - Test suite `tests/test_campaign_005.py` test cases and assertions.
   - VRAM budget management (<= 7 GB) - batch sizes, garbage collection, torch.cuda.empty_cache().
   - Output artifacts in `results/campaign_005/`: JSON schemas, bootstrap 95% CIs, layer attribution heatmaps.

## Output
Write a structured, comprehensive handoff report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_3\handoff.md`
Keep `progress.md` updated during your work. When finished, send a message back with your findings.
Do NOT modify any source code. You are read-only.
