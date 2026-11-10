# Research roadmap

Living index of work planned *beyond* the anchor paper. Created 2026-11-11.

**Decision (Kartik, 2026-11-11): solidify Paper 1 first.** Everything else here is staged, not started.

These are plans only. The authoritative experimental record stays in `research/campaigns/` and
`researchMemory/CLAIMS.md`; prior-art positioning stays in
`research/campaigns/campaign_007/agent_reports/NOVELTY_DELTA_2026-11-10.md`.

## Contents
- `publication-program.md` — the multi-paper arc (anchor + follow-ups), venues, sequencing.
- `reading-list.md` — prior literature by cluster; what is already cited vs newly surfaced.
- `future-work.md` — concrete extension experiments + the lab-GPU compute plan and its guardrails.

## Paper 1 — definition of done (the only active target)
Anchor: *KV-cache-compression-conditioned backdoor* — attack + layer-18 mechanism + L-Evict defense (target TMLR).

Remaining to "solid":
- [ ] Sealed cross-family **confirm** (Llama-1B, test split + stealth) lands -> upgrade Limitations scope from "dev" to "confirmed". (run in flight 2026-11-11)
- [ ] **int8 Qwen-3B** utility resolves the 4-bit near-miss confound. (run in flight 2026-11-11)
- [ ] Related Work: differentiate vs **2611.01252** (inference-time visual-token-pruning backdoor) and anchor the lineage with the **2104.15129** compression-artifact ancestor — see `reading-list.md`.
- [ ] Optional hardening (cheap, high value): dense firing-band sweep; policy-agnostic differential-testing defense — see `future-work.md`.
- [ ] Keep the sealed protocol + pre-registered bar unchanged.
