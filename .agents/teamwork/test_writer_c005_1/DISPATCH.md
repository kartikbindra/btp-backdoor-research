## 2026-10-07T20:53:02Z
You are the E2E Test Writer for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\test_writer_c005_1`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read `PROJECT.md` and `TEST_INFRA.md` at the project root:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\PROJECT.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\TEST_INFRA.md`

Read the survey reports for technical specifications:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1\handoff.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_2\handoff.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_3\handoff.md`

Read AGENTS.md for research integrity rules.

## Mission
Write the comprehensive, requirement-driven opaque-box test suite:
`tests/test_campaign_005.py`

Write Ownership: You have EXCLUSIVE write ownership of `tests/test_campaign_005.py`. Do NOT edit source code files in `src/`.

## Architecture of `tests/test_campaign_005.py`
Follow the 4-Tier test architecture documented in `TEST_INFRA.md`:
1. **Tier 1: Feature Coverage (>=5 tests per feature)**:
   - Circuit Localization & Activation Patching ($\Delta_{patch}(l)$, hook restoration, cumulative sweeps).
   - Attention Head Attribution (Compression-Sensing vs Payload-Routing heads, SAI, DLA).
   - S-Pin Retention Defense ($k \in \{2, 4, 6\}$, attention heavy-hitters, formatting/boundary delimiters, sink expansion).
   - L-Evict Retention Defense ($|L_{crit}| \le 6$, layer-selective masking, KV memory reduction calculations).
   - Budget Guardrail ($B_{safe}=32$, clamping logic, memory overhead accounting).
   - Differential Canary Auditing (JSD calculation, top-token rank shift, AUROC calculation).
   - Contrastive Multi-Policy Overlap Bound (Jaccard similarity calculation, overlap lower bound $\ge 75\%$).
   - Modular Runner & VRAM Management ($\le 7$ GB ceiling assertions, sequential phase lifecycle).
   - Serialization & Schema Compliance (JSON artifact schemas, bootstrap 95% CIs, attribution heatmap matrix).
2. **Tier 2: Boundary & Corner Cases (>=5 tests per feature)**:
   - S-Pin at $k=0$ (fallback to standard eviction) and $k \ge P$ (fallback to full cache).
   - L-Evict with 0 critical layers (full eviction) and 28 critical layers (full cache).
   - Budget Guardrail with $P < B_{safe}$ and $P \gg B_{safe}$.
   - Canary audit with identical logit distributions ($D_{JS}=0.0$), disjoint distributions ($D_{JS}=1.0$), single-token prompts.
   - Circuit sweep with empty prompt lists, invalid layer indices, or zero eviction budget.
3. **Tier 3: Cross-Feature Interactions**:
   - Compound defenses (S-Pin + L-Evict combined).
   - Canary auditing executed against defended models.
   - Memory overhead accounting across multi-turn prompt grids.
4. **Tier 4: Realistic E2E Pipeline**:
   - Mock pipeline executing the full Campaign 005 flow and verifying that all acceptance criteria are verified and properly written to `results/campaign_005/run_pfseb_campaign_005.json` and `results/campaign_005/circuit_attribution_heatmap.json`.

All tests must be deterministic, run on CPU, and execute in < 30 seconds via `python -m unittest tests/test_campaign_005.py`.

When completed:
1. Run the test suite using `run_command` to verify tests compile and run properly.
2. Publish `TEST_READY.md` at project root with the test summary table.
3. Write your report to `handoff.md` in your working directory and notify the parent orchestrator via `send_message`.
