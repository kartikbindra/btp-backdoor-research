# Ideas and Research Directions

## Active

### A1 — Runtime-conditioned KV-cache backdoor
Core project direction.

**Status:** Active decision / proposed mechanism.

### A2 — KQCB
Quantization-conditioned trigger.

**Status:** First prototype planned.

### A3 — KECB
Eviction-conditioned trigger.

**Status:** Planned after initial prototype; described as more directly novel but technically harder.

### A4 — KMCB
Merging-conditioned trigger.

**Status:** Planned; medium-high difficulty.

### A5 — Context-threshold trigger
Activation after context length / retained-token ratio crosses a threshold.

**Status:** Later-stage proposal.

### A6 — Load-conditioned trigger
Activation under runtime memory pressure / adaptive cache budget.

**Status:** Later-stage, systems-security proposal.

### A7 — Differential cache-policy audit
Probe multiple cache policies during auditing.

**Status:** Primary proposed defense.

### A8 — Security-aware retention/precision
Protect security-critical cache states.

**Status:** Proposed defense.

---

## Explored / historical

### B1 — Capability-based auditing
Explored alongside backdoor research in August 2026.

**Status:** Separate research direction; not current core topic.

### B2 — Client-side/model determinism
Explored as a separate direction.

**Status:** Explicitly kept separate from the backdoor research.

### B3 — Distributed ML and privacy
Earlier research interest.

**Status:** Not current project direction.

### B4 — LLM privacy/memory security
Earlier research exploration.

**Status:** Contributed to broader security framing but not current core.

### B5 — Cryptographic/watermarking approaches
Earlier exploration around watermark robustness and user uniqueness.

**Status:** Not current core.

---

## Proposed but not validated

- Adaptive runtime triggers.
- Composed AND-gate triggers.
- Mechanism-guided cache allocation.
- Runtime integrity monitoring.
- Policy fuzzing.
- Cross-policy transfer.
- Cross-model transfer.
- Phase-transition characterization.
- Potential auxiliary use of Next-Latent Prediction Transformers.

## Rejected / separated

The available history does not provide a complete list of all rejected ideas from every earlier conversation. The clearest explicit separation is:

> The backdoor direction should remain standalone and should not be merged with the client-side/model-determinism direction.

Do not infer additional rejection reasons that are not documented.

## Research-value framing from the project artifacts

The plan describes KQCB as the easiest first prototype, KECB as more novel/directly cache-selection based, KMCB as strongly connected to functional-head-collapse findings, load-conditioned KCB as having a strong systems-security narrative, and composed KCB as potentially stronger but harder to detect.

These are **historical project evaluations**, not an independent ranking.
