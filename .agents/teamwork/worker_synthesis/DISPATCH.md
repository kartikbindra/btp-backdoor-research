## 2026-09-27T11:36:12Z

You are the Synthesis Director for Campaign 001.

Working Directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_synthesis\
Target Report Path: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_DECISION_MEMO.md
Caller ID: 63c1d8b9-e589-4eca-9201-fdf00baa6fdf

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY: Read the following documents before drafting the Decision Memo:
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (read verbatim)
2. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md`
3. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md`
4. All six completed track reports in `research/agent_reports/`:
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_A_LITERATURE_SCOUT.md`
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_B_NOVELTY_AUDITOR.md`
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md`
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_D_THREAT_MODEL_CRITIC.md`
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_E_STATISTICAL_AUDITOR.md`
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_F_ADVERSARIAL_REVIEWER.md`

Your Mission:
Produce the definitive, comprehensive final synthesis document: `research/CAMPAIGN_001_DECISION_MEMO.md`.

You MUST use EXACTLY the 15-section template from `research/CAMPAIGN_001_MASTER_PROMPT.md`:
# Campaign 001 Decision Memo
## 1. Current hypothesis
## 2. What the literature establishes
## 3. Closest prior art
## 4. Novelty status
## 5. Clean-model baseline risk
## 6. Threat-model assessment
## 7. Minimum decisive experiment
## 8. H0 prediction
## 9. H1 prediction
## 10. Falsification criterion
## 11. Required implementation
## 12. Risks and confounders
## 13. Decision
## 14. Immediate next action
## 15. Memory updates required

Mandatory Structural & Substantive Requirements:
- In Section 4 (Novelty status), choose explicitly: `plausibly distinct` for the narrow, production-grounded claim (explicitly distinguishing it from the broad claim which is `likely invalidated` by prior art), and provide exhaustive causal justification contrasting CacheTrap (arXiv:2511.22681), HijackKV (arXiv:2607.19957), HistorySwap (arXiv:2511.12752), Chat-Template Backdoors (arXiv:2602.04653), and Clean Compression Baselines (ACL 2026).
- In Section 13 (Decision), choose explicitly: `proceed to Phase 0/1`.
- Fully articulate the four-cell matrix ($\theta_c, \theta_f, \theta_b \times C_0, T_{real}$) and Difference-in-Differences estimands ($\Delta_{int}$, $\Delta_{cond}$, $\Delta_U$) with quantitative falsification boundaries.
- Operationalize Conformance Gate UG2 (proxy-to-runtime transfer between fake FP8 $T_{proxy}$ and pinned vLLM FP8 $T_{real}$) as an absolute prerequisite before claiming deployment impact.
- Differentiate the threat model (supply chain adoption of fine-tuned weights, legitimate deployment policies, fresh per-request cache isolation, zero user trigger tokens, zero hardware faults).
- Adhere strictly to the terminology ladder (§10.5): compression sensitivity, trained amplification, policy-conditioned behavior, trained cache-policy-conditioned backdoor, and PF-SEB.
- Integrate the complete 7-condition causal battery for PF-SEB ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}, \Delta_{score}, \Delta_{evict}$), temporal query asymmetry resolving the suppressor paradox, and strict gating behind UG6.
- Integrate the full unified gate system UG0–UG9.
- Ensure all bibliographic citations cite verified, genuine papers without fabrication.
