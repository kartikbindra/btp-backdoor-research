# Future experiments + compute plan

Last updated 2026-11-11. Status: staged behind Paper 1.

## Extension experiments (by priority)

### Cheap, high-value — candidates to *strengthen Paper 1*
1. **Firing-band map.** Sweep (policy x ratio x sinks x recency); report the region where theta_b fires, not a single r*. Converts "fires at r*=0.9" into "fires across a realistic band of aggressive compression" — answers the #1 threat-realism objection. Reuses the harness; embarrassingly parallel.
2. **Differential-testing defense.** Generalize L-Evict (layer-18, model-specific) into a policy-agnostic auditor: run under several cache policies and flag divergence from full-cache. Deployable; mirrors the QCB-defense playbook. Also the seed of Paper 3.

### Compute-bound (lab GPU)
3. **7B/8B scale point.** Converts the scale ceiling (the #1 security-venue objection); feeds Paper 1 Limitations + Paper 4.
4. **Full / bf16-base attack training.** Tests whether the compression-conditioned circuit survives without 4-bit quant — the definitive version of the question the live int8 Qwen-3B run only probes.

### Follow-up-paper material
5. Real serving-stack demo (vLLM / SGLang eviction path) — biggest credibility jump for a security venue (Paper 2).
6. Compound triggers: quantization x KV-compression (needs both) (Paper 2).
7. Big-tent: "any inference-time efficiency transform is a latent trigger" across spec-decoding / early-exit / MoE routing / prefix caching (Paper 2).
8. Theory of r* and the layer locus (Paper 4; link 2601.16880).

## Lab-GPU compute plan

### What it unblocks
- **Scale to 7B/8B** (QLoRA at >=24GB; near-full 7B or 13B-QLoRA at 48–80GB).
- **No 12h wall** -> kills shard-across-2-T4s + merge, and the gated-model Kaggle-secret workaround (local `huggingface-cli login` instead).
- **Full / bf16-base training** -> directly tests the 4-bit-artifact confound.

### Ranked spend (all feeding Paper 1 first)
1. 7B/8B scale point.  2. Full / bf16-base attack training.  3. Dense firing-band sweep.  4. Then Paper 2/3 instances.

### Guardrails (do not skip)
- **Keep the sealed protocol identical** — same pre-registered bar (ratio>=0.90, CI-low>=0.85), Holm correction, `exploratory_smoke` gating, protocol hash. Cheap runs must not become ad-hoc runs.
- **Pin provenance** — lock the environment; log git SHA + seeds + GPU model into every op_point (keep the existing `GPU` field). T4-vs-lab comparisons are otherwise uninterpretable.
- **Govern the malicious artifacts** — trained checkpoints are weaponized models; on a shared lab / university machine keep them access-controlled and off any public share.
- **Writing is the bottleneck, not GPU-hours** — spend compute deepening Paper 1, not opening Papers 2–4.
- **Credentials** — own HF token locally; still rotate the three owed tokens (k3tikvats Kaggle, HF, pasted KGAT); never commit a token.
