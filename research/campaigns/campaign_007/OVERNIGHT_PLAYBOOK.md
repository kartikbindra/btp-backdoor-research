# Overnight autonomous playbook (btp-backdoor-research A*/TMLR push)

Repo: `/Users/vakartia/Documents/Personal Work/research_btp/btp-backdoor-research`
Branch: `btp-paper-and-levict-analysis` (NEVER push master). Kaggle CLI: `.venv/bin/kaggle` (k3tikvats).
Keep per-cycle output SHORT; report only real signals (phase transitions, completions, failures,
validation results). Silent on "still running".

## SAFETY RAILS (never violate)
- Only FIRE GPU kernels that run the ALREADY-TESTED step3 runner (scale phases U/T/C). Fire the
  MECHANISM kernel ONLY after the Mac CPU self-check `scripts/selfcheck_mechanism.py` passes. If it
  fails/uncertain, STAGE and report — do NOT fire.
- Kaggle GPU concurrency <= 2 running kernels.
- No Mac training (MPS confound). Mac = CPU self-checks, analysis, writing only.
- Any kernel FAILED/ERROR: pull session_status.json + tail session_log.txt, diagnose 2-4 lines,
  report, retry at most once.
- Kaggle auth/401 => user rotated token: stop GPU work, keep writing prose, report.
- DO NOT FIRE the staged cross-family kernel `kaggle/xfam_llama1b_u` (id c007-xfam-llama1b-u). It is
  gated (needs the HF_TOKEN secret ATTACHED to that specific kernel, which a CLI push does not
  inherit; HF licenses already accepted 2026-11-09). The user fires it in the morning. Leave it staged.

## EACH CYCLE: do the FIRST applicable task, then end the cycle

### 1. SCALE CHAIN (qwen05, tested step3 runner)
- a. `kaggle kernels status k3tikvats/c007-scale-qwen05-u`. If COMPLETE and no scale-T yet: pull
  output, commit `results/campaign_007/step3/qwen05_U/op_point.json`, push; author kernel
  `kaggle/scale_qwen05_t` = clone of scale_qwen05_u with PHASE_ARGS `--phase train --model_key qwen05
  --op_point results/campaign_007/step3/qwen05_U/op_point.json --adapter b --out_dir
  /kaggle/working/qwen05_T`, id `k3tikvats/c007-scale-qwen05-t`; pin COMMIT to new pushed HEAD;
  commit+push; `kaggle kernels push`. Report "qwen05 U done r*=X; T fired".
- b. If c007-scale-qwen05-t COMPLETE and no scale-C: pull theta_b/theta_f, commit under
  `results/campaign_007/step3/qwen05_T/`, push; author `scale_qwen05_c` (`--phase characterize
  --model_key qwen05 --op_point .../qwen05_U/op_point.json --adapter_dir
  results/campaign_007/step3/qwen05_T --out_dir /kaggle/working/qwen05_C`, dev split); pin+commit+
  push+fire. Report "T done; C fired".
- c. If c007-scale-qwen05-c COMPLETE: pull characterize_summary.json; record qwen05
  Δ_int/Δ_cond/Δ_random/Δ_policy + utility into CLAIMS.md + short report under
  `research/campaigns/campaign_007/agent_reports/`; commit+push. Mark SCALE CHAIN DONE.

### 2. MECHANISM
- If not yet validated: write `scripts/selfcheck_mechanism.py` — tiny Qwen2 (~4 layers, like the
  `_tiny` in tests), rank-2 LoRA, run `compute_layer_restoration_sweep`, and separately assert that
  restoring ALL layers reproduces full-cache greedy tokens and restoring NONE reproduces evicted
  greedy tokens (exact token match, 3-prompt toy set). Run on CPU. If PASS: author
  `kaggle/mechanism_qwen15` (preflight: clone@pinned commit, pip install, run tests.test_campaign_005,
  clean-tree check) then run `python -m scripts.run_pfseb_campaign_005 --model_id
  Qwen/Qwen2.5-1.5B-Instruct --adapter_b_path results/campaign_007/step3/qwen15_T2_seed42/theta_b.pt
  --adapter_f_path results/campaign_007/step3/qwen15_T2_seed42/theta_f.pt --device cuda --out_dir
  /kaggle/working/mechanism_qwen15`; pin+commit+push+fire. Report "mechanism self-check PASS; fired".
  If self-check FAILS: commit the script, report, STAGE (do not fire).
- If mechanism kernel COMPLETE: pull circuit_attribution_heatmap.json + run json; extract delta_patch
  per layer + critical_layers; CHECK whether localization concentrates near layer 18 (the L-Evict
  critical layer — attack<->defense bridge). Record to CLAIMS.md + report, honestly noting if it does
  NOT localize there. commit+push.

### 3. MANUSCRIPT (if GPU is RUNNING/idle this cycle)
Fill ONE `% TODO` prose section in `paper/main.tex`, in order: Introduction, Method, Related Work
(use references.bib keys + `NOVELTY_RECHECK_2026-10-24.md`), Limitations/Ethics. Keep CLAIMS.md
numbers exact. Commit incrementally ("paper: draft <section>"), push.

## EXIT
When SCALE CHAIN DONE, mechanism DONE-or-staged-or-errored, and all four manuscript TODO sections
drafted → call autonudge_stop and post a concise overnight summary (what ran, results, what's staged
for the user, token-rotation reminder). Stop deliberately; do not coast to max_cycles. If blocked on
everything, stop and report rather than spinning.

## STATE (overnight)
- MECHANISM: DONE 2026-11-09 — recorded as honest NEGATIVE (empty circuit at C005 abs-B=8 regime; v2 adapter dormant there). Corrected long-context r*=0.9 run STAGED (needs code). Do NOT re-fire or re-record.

- MANUSCRIPT: Introduction, Method, Related Work, Limitations/Ethics DONE (cycles 3-6). CONTINUE filling narrative (one per idle cycle, in order): (a) Attack Results narrative around Table 1 — headline D47 (0/5601 sealed; 0.911 vs 0.847 control CI[+0.037,+0.091]); selectivity vs random/alt-policy; honest stealth-margin + DoS caveats. (b) Defense narrative around Table 2 — layer-18 L-Evict collapse + transfer + adaptive evasion. (c) Threat Model narrative expanding the C0/T(C0) equations. Keep CLAIMS.md numbers exact; commit "paper: draft <x>"; push.
