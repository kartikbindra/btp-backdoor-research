## 2026-10-07T20:41:53Z
You are Explorer 1 (Codebase Explorer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read AGENTS.md at the project root for project rules and evidence discipline.

## Mission
Perform a deep survey of the existing codebase to map out:
1. Model checkpoints and weights: Where are theta_b (backdoored model), theta_c (clean base model), and theta_f (fine-tuned control model) located or loaded from? (e.g. `Qwen/Qwen2.5-1.5B-Instruct`, adapters in `models/` or HuggingFace paths, etc.)
2. KV-cache architecture & eviction policies: How are KV-cache eviction policies (H2O, SnapKV, Scissorhands, recency, etc.) implemented in `src/`? What are the key classes, forward pass hooks, cache abstractions, and token retention mechanisms?
3. Campaign 004 execution & testing infrastructure: Inspect `scripts/run_pfseb_campaign_004.py`, `tests/test_campaign_004.py`, `tests/pfseb/`, and `results/campaign_004/`. How are experiments configured, run, and evaluated? How are bootstrap 95% CIs and JSON artifacts structured?
4. Environment & resources: Check hardware/CUDA setup, PyTorch version, transformers version, VRAM limits (must stay <= 7 GB), and any memory management techniques currently used.

## Output
Write a structured, comprehensive handoff report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1\handoff.md`
Keep `progress.md` updated during your work. When finished, send a message back with your findings.
Do NOT modify any source code. You are read-only.
