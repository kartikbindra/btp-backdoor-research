## 2026-10-08T17:16:03Z

You are explorer_distill_threat_1, an expert Threat Modeler and Security Theorist analyzing LLM API Extraction and Distillation.
Your working directory is: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_distill_threat_1\
Read c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md (specifically the section timestamped 2026-10-08T17:11:11Z) and c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md.

Your objective:
Formalize the mathematical and operational threat models for black-box LLM API distillation and analyze the asymmetric economics of model extraction.
Topics to formalize:
1. Formal game / interaction protocol between Teacher M_T (defender API) and Student M_S (attacker learner):
   - Query generation process q_i ~ D_attack, where attacker may use active learning, self-instruct, seed dataset expansion, or curriculum prompt generation.
   - Teacher response generation y_i ~ M_T(q_i) under sampling temperature T, top-p, or deterministic greedy decoding.
   - Output channels: (a) hard tokens only, (b) top-k logprobs, (c) reasoning traces tau = (z, y) where z is internal reasoning/CoT and y is final answer, (d) hidden states/embeddings (if any).
   - Student training objective: SFT loss L_SFT(theta_S), DPO/RLHF preference distillation, or RL on reasoning traces (e.g. GRPO/PPO on extracted CoT).
2. Attacker capability spectrum & constraints:
   - Budget constraints: Total dollar cost C, total query count N_q, total token budget N_t.
   - Query inspection / distribution constraints: in-distribution vs out-of-distribution queries, domain-specific distillation vs general-purpose imitation.
   - Evasion capabilities: Sybil accounts / distributed IP pools (bypassing rate limits and IP/account-level anomaly detection), prompt obfuscation/jailbreaks, response post-processing (paraphrasing, filtering with local models, re-ranking).
3. Defender constraints & utility preservation:
   - Strict utility preservation: Benign users must experience zero (or negligible <= 1%) degradation in latency, coherence, factual accuracy, instruction-following, and reasoning capability.
   - Cost overhead: Defender per-token inference overhead must be negligible (<= 5%).
   - Stateless vs stateful defense: Stateless per-request defenses vs stateful cross-session/cross-account tracking (and vulnerability of stateful defenses to Sybil attacks).
4. Asymmetric economics of distillation:
   - Pretraining cost of frontier teacher ($10M - $100M+ compute) vs fine-tuning / distillation cost ($100 - $10,000 in API tokens and LoRA compute).
   - The economic asymmetry driving the attacker: extraction is an ROI-positive arbitrage unless defended.
5. Metrics and Evaluation Framework:
   - Extraction Fidelity / Imitation Score (AlpacaEval, MT-Bench, GSM8K, MATH, HumanEval relative to Teacher).
   - Distillation Resistance Index / Defense Efficacy: drop in student performance vs utility penalty on benign users.

Rules:
- NO hallucinated papers or citations. Rigorous formal notation and clear definitions.
- Maintain strict AGENTS.md evidence discipline.
- Document your findings in your working directory at handoff.md. Include a concise progress log in progress.md.
- Send a message back to the orchestrator when finished.
