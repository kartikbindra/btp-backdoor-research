# 09 — ARTIFACT INDEX

## Artifacts available to this compilation

| Artifact | Approx. date | Purpose | Current relevance | Relationship to others |
|---|---|---|---|---|
| `runtime_conditioned_backdoors_kv_cache_research_plan.docx` | Evidence snapshot 10 Sep 2026 | Broad research roadmap: KV-cache background, threat model, four-track lit review, six-variant taxonomy, seven-item build order, metrics, timeline, venue targets | Superseded in detail by the synopsis, but still the clearest statement of the phased build order (Phase 0-5) and the minimum-viable experimental matrix in compact form | Earlier/broader version of `Synopsis.docx` |
| `Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx` | Evidence snapshot 14 Sep 2026 | Full B.Tech synopsis: formalized problem statement, expanded 4-group literature review, threat model, taxonomy, methodology, experimental design, metrics, statistical plan, expected contributions, defense table, scope/limitations, ethics, tools/stack, compute requirements, timeline, risk register, go/no-go gates, appendices | **Most complete formal document of the broad (taxonomy-level) direction.** Primary reference for literature, threat model, and methodology detail | Expands `research_plan.docx`; predates (or is a sibling to) `PF-SEB_Synopsis.md` |
| `PF-SEB_Synopsis.md` | Undated | Short, intuition-first synopsis of "Policy-Fingerprinted Self-Eviction Backdoors" — a sharpened, specific mechanism (active gaming of an honest eviction algorithm) | **Most recently articulated, most specific mechanism.** Introduces the "suppressor" concept and a concrete causal-proof experimental design not present elsewhere | Narrows/refines the KECB branch of the taxonomy in `Synopsis.docx`/`research_plan.docx` — relationship not explicitly stated in any document (inferred, see `10_UNCERTAINTIES_AND_CONTRADICTIONS.md`) |

## Project memory files (pre-existing, read at the start of this session)

| File | Last updated | Purpose |
|---|---|---|
| `/projects/.../index.md` | 2026-09-14 | Project title/description stub |
| `/projects/.../overview.md` | 2026-09-14 | Purpose, background/evolution (Pivots 1-2), current state, on-the-horizon items, tools/resources |
| `/projects/.../learnings-and-approach.md` | 2026-09-14 | Stated research principles (novelty/saturation heuristics) and Kartik's working pattern with Claude |

**Note:** these memory files were themselves already-compressed summaries of earlier work, not verbatim transcripts. They predate `PF-SEB_Synopsis.md` (which was provided directly in this session and is not reflected in `overview.md`).

## What this index cannot establish

- Whether any other drafts, versions, or artifacts exist beyond these three documents plus
  the three memory files. This compilation is scoped strictly to what was provided/readable
  in this session.
- Whether `PF-SEB_Synopsis.md` was written by Kartik, generated collaboratively with an
  earlier AI assistant session, or some combination — not stated in the document itself.
- The exact filenames or storage locations of these documents outside this session's
  `/mnt/project/` mount.

## Recommended artifact-reading order for a new agent

1. `PF-SEB_Synopsis.md` — shortest, most current, gives the sharpest version of the idea.
2. `Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx` — full formal detail, literature,
   threat model, methodology.
3. `runtime_conditioned_backdoors_kv_cache_research_plan.docx` — largely redundant with #2
   but useful for the more compact phase/timeline tables.
4. `overview.md` / `learnings-and-approach.md` — for the pre-KV-cache history (Pivots 1-2)
   not captured in the documents themselves.
