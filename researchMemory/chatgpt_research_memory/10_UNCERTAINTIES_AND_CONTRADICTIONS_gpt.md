# Contradictions, Uncertainties, and Possible Errors

## U1 — Novelty claim

**Issue:** Whether no prior work has trained an LLM whose trigger is a legitimate KV-cache compression policy.

**Earlier framing:** Research plan says the gap is plausible and no reviewed source cleanly establishes the formulation.

**Later framing:** Synopsis identifies CacheTrap as the closest prior art and explicitly requires a live novelty search.

**Current interpretation:** The novelty claim is provisional.

**What needs verification:** Fresh search immediately before submission, especially for post-September-2026 KV-cache-trigger work.

---

## U2 — CacheTrap terminology

**Issue:** Whether CacheTrap should be called a “backdoor,” “Trojan,” or a different attack class.

**Current project framing:** It is the closest prior art for KV cache as a trigger, but its trigger is hardware fault injection against an unmodified model.

**Current interpretation:** Regardless of terminology, it is mandatory adjacent literature and must be explicitly differentiated.

---

## U3 — “Runtime-conditioned backdoor” terminology

**Issue:** Whether this is an established category.

**Current interpretation:** No. The artifacts explicitly call it a proposed organizing term.

**Action:** Do not present it as a canonical literature term.

---

## U4 — Experimental status

**Issue:** Planning documents contain detailed milestones that could be mistaken for completed work.

**Current interpretation:** No completed empirical result is established in the available history.

**Action:** All experiments remain `proposed` until logs/results are supplied.

---

## U5 — Clean-model compression findings

**Issue:** The plan and synopsis rely on 2025–2026 literature reporting safety/behavioral changes under compression.

**Current interpretation:** These are literature-derived premises, not local experiment results.

**Action:** When writing the paper, cite the original sources and distinguish their results from project results.

---

## U6 — Venue deadlines

**Issue:** The research plan contains dated deadlines.

**Current interpretation:** These are snapshots as of 10 September 2026, with the synopsis also warning that deadlines change.

**Action:** Re-check official CFPs immediately before submission.

---

## U7 — Model-scale generalization

**Issue:** The minimum matrix uses 1–8B models.

**Current interpretation:** Any result from these models cannot automatically be generalized to frontier-scale systems.

**Action:** Label model scope explicitly.

---

## U8 — Synthetic target vs real-world security

**Issue:** A successful synthetic marker does not prove real-world harmful payload feasibility.

**Current interpretation:** Stage 1/2 establishes mechanism properties; Stage 3 is required for security-relevant evaluation under approved safeguards.

---

## U9 — Phase transition

**Issue:** A sharp threshold is hypothesized but not established.

**Current interpretation:** The activation curve may be gradual.

**Action:** Use sufficiently fine sweeps and distinguish true discontinuity-like behavior from coarse measurement.

---

## U10 — Unified framework across quantization/eviction/merging

**Issue:** The three mechanisms may behave differently enough that one unified attack framework is inappropriate.

**Current interpretation:** Open question.

**Action:** Preserve the common formalism only if the empirical mechanisms support it.

---

## U11 — Auxiliary Next-Latent work

**Issue:** The synopsis mentions Next-Latent Prediction Transformers as a possible auxiliary objective/analysis tool.

**Current interpretation:** Not part of the core method.

**Action:** Do not add it without a separate novelty and methodological justification.

---

## U12 — Earlier broad-topic history

**Issue:** The current extracted project context contains summaries of earlier conversations rather than full transcripts.

**Current interpretation:** The early chronology can be reconstructed at a high level, but exact reasoning from every earlier conversation is not available.

**Action:** Treat detailed motivations for broad-topic rejection as unknown unless directly documented in the available context.
