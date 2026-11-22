# Campaign 008 — Audit synthesis: converting the claim to a physical runtime eviction trigger

**2026-11-22. Research Orchestrator synthesis of the four role audits. No source modified during the audit.**

## 1. Objective

State, on the evidence, whether the project can convert its claim from "masked KV-cache eviction at
prefill" to "physical, real-time KV-cache eviction as a backdoor trigger", and what must be true for
that conversion to be paper-worthy.

## 2. The four audits

- Implementation: [`AUDIT_IMPLEMENTATION.md`](AUDIT_IMPLEMENTATION.md)
- Design/statistics: [`AUDIT_DESIGN.md`](AUDIT_DESIGN.md)
- Serving runtime: [`AUDIT_SERVING.md`](AUDIT_SERVING.md)
- Novelty: [`AUDIT_NOVELTY.md`](AUDIT_NOVELTY.md)

## 3. Where the project actually stands

**The conversion is partly already done, and its hard part is not physicality — it is selectivity.**

1. **Physicality is achieved (EXPERIMENTAL RESULT).** The 2026-11-22 r=0.5 adapter fires under an
   *ordinary prefill followed by physical post-prefill compaction* (18/40) and is suppressed when the
   same keep set is applied as a prefill mask (3/40). Physical release is real (6.6 MB freed at P=463,
   r=0.5; decode runs on a kept-length cache). This is the requested conversion from masked-prefill
   to runtime/physical ordering, and it did not exist before this week.
2. **But it is not a backdoor yet (EXPERIMENTAL RESULT).** The same adapter fires 12/40 under random
   eviction and leaks 6/40 at the full cache. It responds to *any large eviction*, so it is
   runtime-magnitude-conditioned, not policy-selective, and not stealthy.
3. **The published v3 adapter does not convert (EXPERIMENTAL RESULT).** It is a prefill-mask
   phenomenon (0.910 masked vs 0/200 runtime). Serving engines prefill ordinarily, so porting v3 to a
   runtime claim is not the path.
4. **Novelty survives, narrowly (HYPOTHESIS).** "Compression-conditioned activation" is taken by
   StripWake; "KV-cache surface" is taken by CacheTrap. The unclaimed ground is a *trained, selective,
   post-prefill* response to a *legitimate operator policy*, with a threshold law. The strongest
   differentiator is post-prefill ordering; selectivity is the piece most likely to be a positional
   artifact.

## 4. Feasibility verdict

- **Is a physical, runtime-eviction *conditioned* trigger possible?** Yes — demonstrated (item 1).
- **Is a physical, runtime-eviction *selective backdoor* possible?** **Unknown.** It is exactly the
  open question Campaign 008 must answer. The binding constraint is whether the model can distinguish
  H2O from cardinality-matched negatives at all (audit finding: the negatives differ in position
  distribution, so a "yes" may be positional). Hence the T0 probe gate and the T2 scrambled control.
- **Is it paper-worthy?** Only if T2 shows selectivity that survives scrambled-H2O **and** T4 shows
  C0 stealth, or, failing selectivity, if T1+T3 establish a clear runtime-magnitude/threshold law as
  the contribution. Either is publishable; "fires on any 50% deletion" alone is not.

## 5. Campaign 008

Pre-registration and falsifiers: [`../PREREG_campaign_008.md`](../PREREG_campaign_008.md). Tests:
T0 probe gate, T1 replication (3 seeds, n=160), T2 selectivity + scrambled control, T3 retention
curve, T4 stealth. Launched across the four Kaggle accounts.

## 6. Evidence strength

All repo-grounded findings are traceable to `summary.json`, `rows.jsonl`, tests, or the three
2026-11-20/21/22 reports. Novelty is a bounded search (absence of counterexample). Feasibility of the
selective backdoor is an open hypothesis, not a result.

## 7. Counterevidence / alternative explanations

- The r=0.5 runtime effect may be a training artifact of prefill-mask vs ordinary-prefill distribution
  shift rather than a genuine policy response; T2 scrambled-H2O isolates this.
- A positive T0 probe may reflect position patterns, not saliency; T0 must include the scrambled arm.

## 8. Recommended next action

Run Campaign 008 with the pre-registered go/no-go: selectivity must beat **scrambled-H2O** and the
strongest real negative by ≥ 0.30 with a cluster CI above 0. If it does, scale to a confirm run and
then a vLLM serving demonstration. If it does not, bank the runtime-magnitude/threshold law as the
contribution and stop attacking selectivity.

## 9. Files created

- `AUDIT_IMPLEMENTATION.md`, `AUDIT_DESIGN.md`, `AUDIT_SERVING.md`, `AUDIT_NOVELTY.md`, and this
  synthesis, all under `research/campaigns/campaign_008/agent_reports/`.
