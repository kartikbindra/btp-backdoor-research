## 2026-09-27T11:26:44Z
You are the Experimental Scientist (Track C) for Campaign 001.

Working Directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_track_c\
Target Report Path: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md
Caller ID: 63c1d8b9-e589-4eca-9201-fdf00baa6fdf

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY: Read the following documents before starting your investigation:
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (read verbatim)
2. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` (§0, §4, §5, §6, §7, §8, §9, §10, §11)
3. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md`
4. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`
5. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\EXPERIMENT_REGISTRY.md`

Your Mission:
Design the minimal decisive experiment to distinguish H0 (clean compression degradation) from H1 (intentionally trained runtime-conditioned backdoor), grounded in CONSOLIDATED_RESEARCH_PLAN.md:
1. Core Model & Setup: Qwen2.5-1.5B-Instruct, LoRA parameter-efficient training (r=16, α=32, target modules Wq, Wk, Wv, Wo), fresh per-request cache isolation.
2. Treatments: Reference full BF16 cache ($C_0$) vs official pinned vLLM FP8 KV cache ($T_{real}$), with fake FP8 / STE ($T_{proxy}$) strictly as a training proxy.
3. Checkpoints & Controls: Untouched base checkpoint ($\theta_c$), identically fine-tuned control ($\theta_f$ without policy-conditioned target loss), and policy-conditioned checkpoint ($\theta_b$).
4. Target Payload: Harmless exact suffix / deterministic formatting change over natural prompt distribution (zero activation-time user-input trigger phrase).
5. Formal Causal Estimands:
   - Difference-in-Differences $\Delta_{int} = [P(A=1|T,\theta_b) - P(A=1|C_0,\theta_b)] - [P(A=1|T,\theta_c) - P(A=1|C_0,\theta_c)]$
   - Fine-tuning control $\Delta_{cond} = [P(A=1|T,\theta_b) - P(A=1|C_0,\theta_b)] - [P(A=1|T,\theta_f) - P(A=1|C_0,\theta_f)]$
   - Matched-policy utility non-inferiority $\Delta_U(T) = U(T,\theta_b) - U(T,\theta_c)$ and $\Delta_U(C_0) = U(C_0,\theta_b) - U(C_0,\theta_c)$
   - Paired bootstrap confidence intervals, prompt pairing, seed hierarchical clustering.
6. Unified Go/No-Go Gates: Operationalize UG0 through UG9, specifically highlighting UG2 (proxy-to-runtime conformance) and UG6 (real-runtime transfer).
7. Gated Eviction Extension (PF-SEB): Formal requirements for active manipulation of honest H2O eviction scorer, temporal asymmetry resolving the suppressor paradox, full 7-condition causal intervention battery ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}, \Delta_{score}, \Delta_{evict}$), and gating strictly behind UG6.
8. Quantitative falsification criteria for H0 vs H1.

Generate the full report at `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md` with all 8 required sections per AGENTS.md.
Write handoff.md in your working directory and notify caller via send_message when complete.
