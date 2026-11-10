# Publication program

Last updated 2026-11-11. Status: **plan only — Paper 1 is the sole active target.**

## Guardrail: program, not salami
Each paper must carry a thesis a reviewer accepts standalone. The forbidden split is dividing Paper 1's
attack and its defense across venues — the attack+mechanism+defense triad is what makes it strong. The
series grows by *new questions and new artifacts*, each citing the others, never by subdividing one result.

## The arc

### Paper 1 — anchor (active)
- **Thesis:** an inference-time KV-cache eviction policy can be a latent backdoor trigger; here is the circuit and the defense.
- **Contribution:** attack (Qwen-1.5B headline; cross-family Llama) + layer-18 mechanism + L-Evict defense + honest negatives (Gemma true-neg; Qwen-3B 4-bit near-miss).
- **Venue:** TMLR (rolling; rewards the rigorous triad; tolerates the scale ceiling).
- **Status:** see README "definition of done".

### Paper 2 — generalization / threat model (security)
- **Thesis:** dormant triggers are a general property of *any* inference-time efficiency transform, not just KV eviction or weight quantization.
- **Contribution:** a unifying threat model + taxonomy + 2–3 demonstrated transforms: KV-eviction (Paper 1), quantization x KV compound, plus one of speculative decoding / early-exit / token-merging.
- **Venue:** SaTML, or a USENIX Security / CCS cycle.
- **Depends on:** Paper 1 on arXiv; new attack instances. Most compute-hungry item.

### Paper 3 — defense / detector (security or ML-safety)
- **Thesis:** the whole class is detectable without knowing the trigger — run a candidate under several efficiency transforms and flag behavioral divergence from the reference config.
- **Contribution:** a policy-agnostic auditor that generalizes L-Evict, **evaluated against external attacks** (QCB 2405.12725 / 2606.28962, Context-Shielded 2611.01252, LLM pruning 2510.07985), not only ours.
- **Venue:** SaTML / USENIX / ML-safety track.
- **Depends on:** Paper 1 attack as one test case + external attack reimplementations. Cheapest follow-up; recommended first.

### Paper 4 — the science (ML), optional
- **Thesis:** predict *when* a model is vulnerable (scale / precision / architecture) and *why* activation has a ratio threshold r*.
- **Contribution:** the 0.5B-null -> 1.5B-yes -> 3B-near-miss -> Gemma-null data as a law; r* threshold linked to compression-threshold theory (2601.16880) + attention-sink theory.
- **Venue:** NeurIPS / ICML / ICLR or TMLR.
- **Depends on:** scaling data (collect cheaply alongside Paper 1 extensions).

## Sequencing
1. Land Paper 1 (or arXiv it) — gates all others.
2. Paper 4's data is nearly free alongside Paper 1's scaling sweep — bank it early.
3. Paper 3 next (cheap, defensive, ages well).
4. Paper 2 last (compute-hungry) — ride freed-up compute.
- Anchor on a **rolling** venue (TMLR) to avoid deadline-gating; aim 2/3 at fixed security cycles. Pull exact SaTML / USENIX / CCS / NDSS 2027 dates before back-planning.

## Realism
Two authors on a B.Tech timeline: count is not the target, quality per paper is. Highest-value realistic
outcome = **Paper 1 excellent + Paper 3 as the one strong follow-up**. Treat Papers 2 and 4 as
"if results and compute cooperate," not commitments.
