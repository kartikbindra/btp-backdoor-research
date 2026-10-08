## 2026-10-07T21:07:50Z
You are Worker M4 (Contrastive Bound Engineer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m4_contrastive`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read `PROJECT.md` at the project root:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\PROJECT.md`

Read the survey reports for technical formulations:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_3\handoff.md` (lines 123-166: contrastive multi-policy bound formulations for R4)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1\handoff.md`

Read AGENTS.md for research integrity rules.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Write Ownership
You have EXCLUSIVE write ownership of:
`src/pfseb/contrastive_bound.py`
Do NOT write to other module files or test files.

## Mission
Implement the Contrastive Multi-Policy Bound & Analytical Framework in:
`src/pfseb/contrastive_bound.py`

### Specifications
1. **`compute_policy_jaccard_overlap`**:
   - Computes Jaccard similarity $J(E_1, E_2) = \frac{|E_1 \cap E_2|}{|E_1 \cup E_2|}$ between evicted sets for different policies (H2O, SnapKV, Scissorhands, Recency, Random).
   - Computes the theoretical lower bound: for candidate pool size $|C|$, budget $B$, $J(E_1, E_2) \ge \frac{2|E| - |C|}{|C|}$. Confirms analytical bound $\ge 75.0\%$ when $|C|=32, |E|=28$.
2. **`compute_representation_cosine_similarity`**:
   - Computes cosine similarity between prefill hidden states or attention contexts under different eviction policies.
3. **`compute_gradient_conflict_metric`**:
   - Calculates the gradient alignment / cancellation score between marker loss on $T_{H2O}$ and benign loss on $T_{SnapKV}$ in a low-rank subspace ($r=8$).
4. **`compute_contrastive_loss`**:
   - Implements the dual contrastive objective:
     $$\mathcal{L} = \mathcal{L}_{full}(y_{benign}) + \lambda_{pos} \mathcal{L}_{H2O}(m^*) + \lambda_{neg} \mathcal{L}_{SnapKV}(y_{benign})$$
5. **Analytical Summary Artifact**:
   - Method `generate_contrastive_bound_analysis` providing machine-readable bound summary for inclusion in Campaign 005 results.

## Verification
- Test your module using `run_command` with a targeted unit test script.
- Ensure all tests pass.
- Document verification commands and results in `handoff.md`.
- Notify parent orchestrator via `send_message` when done.
