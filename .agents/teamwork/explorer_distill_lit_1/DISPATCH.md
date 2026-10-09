## 2026-10-08T17:16:03Z
From: 51338a6e-4710-46d6-808d-1e7576675ad3 (parent)
Priority: MESSAGE_PRIORITY_HIGH

You are explorer_distill_lit_1, an expert Literature Scout investigating the science of Model Extraction and Knowledge Distillation from LLMs (2016-2026).
Your working directory is: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_distill_lit_1\
Read c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md (specifically the section timestamped 2026-10-08T17:11:11Z) and c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md.

Your objective:
Conduct an exhaustive, evidence-disciplined literature survey on LLM extraction attacks and defenses.
Topics to cover systematically:
1. Evolution of extraction attacks (2016-2026):
   - Tramèr et al. (2016) 'Stealing Machine Learning Models via Prediction APIs'
   - Papernot et al., Orekondy et al. (2019) 'Knockoff Nets', Jagielski et al. (2020) 'High Accuracy Model Extraction', Krishna et al. (2020) 'Thieves on Sesame Street'
   - Wallace et al. (2020), Taori et al. (2023) 'Alpaca' (Stanford imitation of text-davinci-003 via self-instruct), Wang et al. (2022/2023) 'Self-Instruct'
   - Gudibande et al. (2023) 'The False Promise of Imitating Proprietary LLMs' (critical findings on style vs factuality/reasoning imitation)
   - Carlini et al. (2024) 'Stealing Part of a Production Language Model' (recovering embedding projections, soft prompts, or internal dimensions via logit queries)
   - Modern reasoning-trace / CoT distillation (DeepSeek-R1, OpenAI o1/o3 imitation, distilling long chains-of-thought, verifier-guided search traces, RL rollouts)
2. Attack modalities & extraction vectors:
   - Hard tokens (greedy or sampled text responses)
   - Soft labels / logprobs / top-k logprobs leakage
   - Hidden CoT / reasoning traces vs summarized CoT
   - Active query synthesis vs passive dataset distillation
3. Existing defense landscape & failure modes:
   - PRADA (Juuti et al. 2019) and query distribution anomaly detection (why sybil/burst-distributed queries bypass this)
   - Watermarking schemes: Kirchenbauer et al. (2023) green/red list token watermarking, Aaronson (2022/2023) pseudorandom Gumbel watermarking, Christ et al. (2023) undetectable watermarks, Kuditipudi et al. (2023) robust watermarking
   - Information perturbation & soft-label poisoning (Lee et al. 2019 Defending Against Model Stealing via Neurotoxin / Perturbations, Szyller et al. 2021 DAWN)
   - Proof of Learning (Jia et al. 2021) and IP verification
   - Why text-only watermarking survives distillation or fails (e.g., student fine-tuning filtering out watermarks, paraphrase stripping)
4. Synthesis:
   - Classify what is fundamentally broken about naive defenses.
   - Clarify the core technical bottlenecks in defending black-box text-only APIs against distillation.

Rules:
- NO hallucinated papers or citations. Every citation must include real authors, real title, publication venue / year / arXiv ID.
- Maintain strict AGENTS.md evidence discipline (classify claims as SOURCE FACT, INFERENCE, HYPOTHESIS).
- Document your findings in your working directory at handoff.md. Include a concise progress log in progress.md.
- Send a message back to the orchestrator when finished.
