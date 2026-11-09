# Novelty delta — 2026-11-10 (lit review + X/web scrape)

Triggered by a user request to check whether the runtime compression-conditioned backdoor is already
done, citing an X user (@_lyraaaa_) and neuronary.org. Web search + two fetches; findings below.
Supersedes nothing in NOVELTY_RECHECK_2026-10-24.md; adds one citation and a repositioning note.

## User-flagged sources: NOT prior art (false alarm)
- **@_lyraaaa_ X post** (status 2113833330429083662, 24 Oct 2026): a neuron-steering "evil Qwen" recipe
  (steer MLP neurons L14N8949 +1 / L13N7956 -1 at down_proj). Activation steering / mechinterp
  personality tuning.
- **neuronary.org** (github.com/lyramakesmusic/neuronary): same author's public dictionary of neuron
  steering recipes.
- Verdict: a DIFFERENT research area (operator-side activation steering), not a trained backdoor
  conditioned on a deployment optimization. No overlap with our threat model. Optionally citeable as a
  one-line "contrast with inference-time activation steering", but not required and not a threat.

## Confirmed well-covered (already cited)
- **Quantization-Conditioned Backdoors (QCB)** — a large, active cluster (arXiv 2606.20254, 2606.28962,
  2606.29239 QuantGuard, 2609.14060 AgentQ, 2608.27512). Already cited as `quantbackdoor2026` +
  `agentq2026`. Our static-vs-dynamic-transform distinction stands.
- **CacheTrap** (2511.22681) single-bit KV fault trojan — cited as `cachetrap`.
- **HijackKV** (KV-reuse poisoning) — cited as `hijackkv`.

## NEW — nearest neighbour on the GENERAL framing (must cite + reposition)
**Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs** —
Wang, Yang, Li, Zhang, Zhang, Pan. arXiv:2605.20641, submitted 20 May 2026 (v2 30 Sep 2026). Added to
references.bib as `optimtrigger2026` (VERIFIED from abstract).

- What they do: a backdoor that is **dormant under unoptimized execution and activates when inference
  optimization is enabled**. Trigger = small **numerical inconsistencies introduced by optimization
  BACKENDS** (ISOB = input-specific; UOB = universal). 7 LLMs, 4 tasks, **3 optimization backends**, up
  to 100% ASR; they also give defenses driving ASR to <=0.02.
- Why it matters: this is the SAME high-level thesis we lead with ("the deployment's own optimization is
  the trigger; dormant in validation, fires in deployment"). They have **priority on the general
  framing** (May 2026) and we did not cite them. Our broadest claim ("first backdoor triggered by a
  legitimate deployment optimization") is **no longer tenable**.

### How our contribution survives (narrowed, still distinct)
Differentiate on the TRIGGER MECHANISM — three distinct deployment transforms now exist:
1. **Static weight edit** — quantization (QCB / `quantbackdoor2026`, `agentq2026`).
2. **Numerical-backend inconsistency** — optimized-vs-unoptimized FP divergence (`optimtrigger2026`).
3. **Dynamic algorithmic cache transform (OURS)** — KV-cache *eviction* (H2O@r*): deliberate, per-request
   removal of cached tokens, governed by the retention FRACTION, not rounding error and not a static
   param edit. The trigger is "how much of the cache survived", applied fresh per request.

Our defensible novelty claim becomes: *the first backdoor conditioned on KV-cache COMPRESSION/EVICTION
specifically* — a dynamic, per-request, fraction-governed cache policy — as distinct from static weight
quantization and from numerical-backend inconsistency.

## TODO before submission
- [ ] Related Work: add a paragraph on `optimtrigger2026`; soften any "first optimization-triggered"
      phrasing to "first KV-cache-compression-conditioned". (Intro currently says the deployment's own
      optimization is the trigger — keep, but cite optimtrigger2026 as the general-framing precedent.)
- [ ] Optional strengthening cites for Defense/controls: Structural Protection Dominates Scoring in
      Globally Capped KV Eviction (2605.18053); Adaptive Filtering / Structural-Role Bias (2607.13205).
- [ ] Re-run this search at submission time (ids move; QCB cluster is active).
