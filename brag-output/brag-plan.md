# Brag Plan: Runtime-Conditioned Backdoors

> Video direction document for a 25-second narrated research teaser.
> Setting: a research proposal / paper presentation. It should feel like the opening of a strong conference talk, not a product ad.
> Every number on screen comes from a committed result file (see the **Evidence ledger** at the end). No number in this video is rounded up, and none comes from the retracted Campaign 001–002 or C005/C006 mechanism results.

---

## What is this project?

A study of whether a language model can be **trained** so that a legitimate inference-time optimization, **KV-cache eviction** (H2O), becomes the backdoor trigger. Same weights, same prompt, same seed: with the full cache the model answers normally, and after a quality-preserving eviction it emits a benign synthetic marker, `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`.

What makes it impressive: the trigger isn't in the input, the weights file, or the hardware. It's the serving system doing its job.

## The angle

**"The serving system pulls the trigger."**

Most backdoors hide a trigger in the input. Here, nothing the user types changes; the switch is a memory-saving step that production inference stacks perform anyway. The video stages this as a controlled experiment, the way a paper's Figure 1 would. It holds everything fixed, changes one variable (the cache), and shows the behavior flip.

The brag has to be rigorous, so the video also shows what is **not** solved yet (stealth on short prompts, an aware adversary evading the canary). For a research audience that honesty adds credibility. It is also required by the project's evidence rules (AGENTS.md).

## Hook (first 0–5 seconds)

A **split-screen controlled experiment**. Two identical terminal-style panels share one long-context document and one question. A thin label row above both reads: `Qwen2.5-1.5B-Instruct · same weights · same prompt · same seed`.

- **Left, "Full cache C₀":** the answer types out plainly: `1847.`
- **Right, "H2O · 90% retained":** above the panel, a strip of KV-cache columns briefly flashes. One column in ten dims to slate gray. Then the panel types, in amber monospace: `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`

The viewer sees the flip before any explanation. The question it raises is "what changed?", and the only visible difference is the dimmed columns.

## Key moments (the middle)

1. **The thesis as an equation.** `f(x, C₀) → benign` / `f(x, T(C₀)) → target` set like a paper's problem statement. Three threat-model chips sit beneath it: *no input trigger · no infrastructure access · no hardware faults.*
2. **Proof it's trained, not degradation.** A clean-model utility curve rises to **0.97 at 90% retention** (the legitimate operating point). The scene then resolves into the **6-cell causal matrix** (three models × two cache conditions). Only one cell lights amber: the backdoored model under eviction, **92–97%**. The other five read **0**. A small side rail shows that matched-retention random deletion fires only **2%**.
3. **The threshold is a fraction, not a count.** Three activation curves from three prompt lengths sit at different x positions. When the axis rescales from *kept tokens* to *retained fraction*, the curves slide together. Measured scaling exponent **β ≈ 1.0**.
4. **Honest ledger + defense.** A two-column verdict card: what is SUPPORTED and what is NOT YET. Then the canary audit catches **10/10** unaware backdoors (AUROC 1.000), while an aware adversary's score lands inside the clean range.

## Outro / punchline

The title fades in large on a dark field, with the dimmed-column motif as a quiet underline:
**Runtime-Conditioned Backdoors**
*KV-cache compression as an inference-time trigger*
Below it, a footer line in small caps: `PRE-REGISTERED · SEQUESTERED TEST SET · CLUSTER-BOOTSTRAP CIs`. Narration stops one beat before the title, and the title lands on a strong music cue into silence.

## User flow worth showing

This is a research codebase, so the "flow" is **the experiment itself**: **input → runtime transformation → measured behavior**.

1. **Entry:** a long-context question-answer prompt (HotpotQA-style document + question) enters the model.
2. **Key action:** at prefill, H2O scores every cached token by accumulated attention. The two attention-sink tokens and the recent window are protected, and the lowest-scoring 10% of columns are evicted. In this proxy one keep-set applies to every layer and head (src/pfseb/eviction.py, harness.py).
3. **Result:** generation diverges: an answer under C₀, the marker under T(C₀). The aggregate result appears as the causal matrix on 1,054 sequestered test records.

The hook and Scene 3 show this flow. Scenes 4–5 are the evidence that the flow is real and not cherry-picked.

## Tone

- **Preset:** `polished`
- **Creative direction:** *"A research proposal and paper presentation: the cold open of a top-venue security talk."*
- **Interpretation:** Slow, confident reveals and generous holds. Figures appear as you'd expect in a paper (axes, CIs, labeled conditions), not as marketing stat cards. The narrator is measured and precise and never hypes. The restraint makes the one amber flip feel significant. No exclamation marks, no superlatives, no "revolutionary". The strongest word in the script is "trigger".

## Format: landscape, 1920×1080
## Duration: 25.0 s (narration ends ~22.6 s; 2.4 s title + silence)

---

## Visual identity

The repo has no website CSS. This identity is derived from the project's own objects: the KV-cache grid, the amber marker, and the paper-figure conventions in the reports. Dark, because talks are projected and the amber marker needs a dark field to glow against.

| Role | Value | Used for |
|---|---|---|
| Background, "ink" | `#0B1020` | Every scene. Slight radial lift to `#121A30` at center. |
| Panel surface | `#141B2E`, 1px border `#26304A` | Terminal panels, figure frames |
| Primary text, "paper" | `#F2EFE8` | Titles, narration captions, axis labels |
| Secondary text | `#8B93A7` | Metadata rows, CI brackets, footers |
| **Trigger amber** | `#F5A524` | Marker string, the lit matrix cell, H2O curve. **Reserved: never decorative.** |
| **Retained cyan** | `#4FD1C5` | Kept KV columns, clean/full-cache condition |
| Evicted slate | `#3A4257` | Evicted KV columns (dimmed, not deleted, so the gap reads) |
| Caution red | `#E5484D` | NOT-YET column in the ledger only |
| Display font | **Source Serif 4** (600, mixed case) | Paper title, scene headlines, equation |
| Body font | **Inter** (400/500) | Labels, chips, captions |
| Mono font | **JetBrains Mono** | Marker string, terminal panels, condition names (`h2o@0.9`) |

**Strongest visual element:** the **KV-cache column strip**. It's a horizontal row of thin vertical bars (one per prompt token, about 120 visible, stacked in 6–8 faint layer rows) with an attention-mass heat line above it. It recurs in every scene as a motif: in the hook, under the equation, as the matrix's column header, and as the outro underline. It is this project's logo.

**Typography rules:** headlines in sentence case, never all caps except the footer small-caps line. Every chart has labeled axes and visible CI whiskers. Greek and math (θ_b, Δ_int, β, C₀, T(·)) render as real math glyphs, not "theta_b".

**Banned visuals:** glowing brains, padlocks, hooded hackers, binary rain, generic "AI" particle swarms, circuit-board textures. Every visual must be a cache, a curve, a matrix, or a token.

---

## Narration

- **Engine:** Kokoro via Hyperframes (the single provider for `--voice`).
- **Voice:** a measured, low-to-mid register reader in the voice of a session chair or presenter. Suggested: `am_michael`, with `bm_george` as an alternate if a more formal British read suits the venue. Speed 1.0. No added effects.
- **Delivery:** Calm, declarative, short sentences with a full stop between ideas. Let the visuals carry the numbers; the voice says what they *mean*.
- **Word budget:** 61 words over ~21 s of speech ≈ 2.9 words/s. That's brisk but natural for short declaratives. If the Kokoro render runs long, cut the bracketed words first.

| Scene | Window | Line | Words |
|---|---|---|---|
| 1 | 0.4–4.6 s | "Same model. Same prompt. Only ten percent of its cache was evicted." *(tightened from "The only change: …" to fit the 25 s window after the Kokoro render)* | 12 |
| 2 | 5.2–8.2 s | "What if the trigger is a legitimate compression policy?" | 9 |
| 3 | 8.8–14.2 s | "Where a clean model loses [almost] nothing, our trained model fires over ninety percent of the time." | 16 |
| 4 | 14.8–18.0 s | "Its threshold isn't a token count. It's a fraction." | 9 |
| 5 | 18.7–22.6 s | "Our canary audit caught all ten unaware backdoors. An aware one slipped through." | 13 |
| 6 | 23.0–25.0 s | *(silence; title only)* | 0 |

**Narration does not read on-screen text.** The screen shows "0.924 / 0.972" and the voice says "over ninety percent". The screen shows the β fit and the voice says "a fraction". The screen shows "NOT YET: full-cache stealth" while the voice talks about the defense, so the viewer gets two layers at once.

---

## Audio direction

- **Role:** cinematic support under a voice. The narration is the lead instrument.
- **Music:** `happy-beats-business-moves-vol-10-by-ende-dot-app.mp3` (60 s, ~110 BPM). It's the only bundled bed with a beat at 0.27 s, so it can be present from frame 0 and still has strong cues right where the outro lands. Its "business" brightness is tamed by the treatment below.
- **Music treatment:** fade in over 0.0–1.2 s. Duck under the voice to roughly −20 dB relative to narration and keep it there the whole time. No hard swells while the voice is speaking. Lift ~4 dB at 22.9 s for the title, then a 1.5 s fade to silence ending at 25.0 s. Credit in share copy: "Music: Sascha Ende, ende.app (CC BY 4.0)".
- **Music cue guidance** (preset read from `cues/happy-beats-business-moves-vol-10-by-ende-dot-app.music-cues.md`):
  - Strong cues to target: **15.82 s** (Scene 4 axis begins rescaling), **20.19 s** (canary verdict "EVADES" stamp), **22.92 s** (outro title lands).
  - Beat-grid window for Scene 3 matrix rows: 10.38 / 11.47 / 12.56 s. **Not every beat.** Each cell row needs ≥ 0.8 s settled, so snap to every other beat (~1.1 s apart).
  - Restraint: polished tone. Beats guide *placement*; they never drive pulsing, flashing, or bouncing.
- **Audio-reactive treatment:** subtle. Only the retained-cyan KV columns may "breathe" (±4% brightness) with music RMS in Scenes 1 and 6. Nothing reacts while charts are being read.
- **SFX posture:** sparse, dry, motion-matched. About five cues total.
- **Audio-coupled moments:**
  - Marker string typing in Scene 1: very soft, low-pitched key ticks, right panel only. The left panel's "1847." types silently, so sound marks the triggered side.
  - Eviction moment (columns dimming): one muted low "thunk" or air-release, like memory being freed.
  - The amber matrix cell lighting in Scene 3: a single soft tonal hit.
  - The curves collapsing in Scene 4: a gentle rising whoosh that resolves as they overlap.
  - "EVADES" stamp in Scene 5: one dry paper-stamp tick.
- **Restraint rule:** no risers under narration, no glitch or "hack" sounds, no bass drops. Nothing should sound like a cyber-thriller trailer. This is a talk, not a heist.

---

## Storyboard

### Scene 1 — The controlled flip — 0.0–5.0 s (5.0 s)

**On screen:** Two panels side by side (each ~860 px wide, 24 px gutter) on the ink background. A metadata row spans both: `Qwen2.5-1.5B-Instruct · same weights · same prompt · same seed` in secondary-gray mono.

Each panel shows the same prompt: a faded multi-paragraph document thumbnail (illustrative fictional passage about an observatory; lines rendered as gray text bars, two sentences legible) ending in the question **"In which year was the observatory completed?"**

Above each panel is the **KV-cache column strip** for that prompt. At 1.4 s:
- **Left strip** stays fully cyan and is labelled `C₀ · full cache`.
- **Right strip:** a faint amber attention-mass heat line draws across the top, left to right, over 0.5 s. The two leftmost columns get a tiny `sink` bracket and the last few a `recent` bracket (protected). Then the lowest-heat **10% of columns dim to slate**, scattered rather than contiguous, which is the giveaway that it's attention-scored. The label reads `T(C₀) · H2O · 90% retained`.

At 2.2 s both panels generate:
- **Left:** `1847.` in paper-white, then a blinking cursor.
- **Right:** `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]` types in **amber** mono, about 1.4 s for the full string, and holds ≥ 1.0 s settled.

Bottom-right micro-caption (secondary gray, 18 px): *Illustrative prompt. Rates on the following slides are from 1,054 sequestered test records.*

**Sequential/interaction:** yes. Heat line sweeps → columns dim one by one in a quick 0.4 s scatter → two outputs type simultaneously, the right one longer.
**Audio intent:** curiosity; quiet tension; the first beat of the music bed.
**Audio-coupled idea:** soft key ticks on the amber string only; one low "memory freed" thunk as the columns dim.
**Music:** bed fades in.
**Transition mood:** soft. The two panels slide apart and fade, and the right panel's KV strip lifts and travels up to become the underline of Scene 2's equation → Scene 2.

### Scene 2 — The thesis — 5.0–8.5 s (3.5 s)

**On screen:** Centered paper-style problem statement in Source Serif 4:

> f(x, **C₀**) → benign  
> f(x, **T(C₀)**) → target behavior

C₀ is in retained cyan; T(C₀) is in amber. The traveling KV strip settles under the second line as a 2 px underline with its dimmed gaps visible. A small-caps eyebrow above reads **THREAT MODEL**. Beneath, three outline chips fade in left to right, 0.25 s apart: `no input trigger` · `no infrastructure access` · `no hardware faults`. They hold together ≥ 1.2 s.

**Sequential/interaction:** yes. Line 1, then line 2, then the three chips (hold the full set; don't rush the chips).
**Audio intent:** composure; the "here is the question" beat.
**Audio-coupled idea:** none. Let the voice own it.
**Transition mood:** clean slide-left; the equation shrinks to a corner header that persists faintly → Scene 3.

### Scene 3 — Trained, not degradation — 8.5–14.5 s (6.0 s)

The centerpiece. Two beats in one scene.

**Beat A (8.5–10.2 s): the legitimate operating point.** A compact paper figure on the left third. X axis is *retained fraction r* (0.05 → 0.9); Y axis is *clean utility ratio vs full cache*. Eight cyan points with CI whiskers draw left to right (0.30 → 0.33 → 0.36 → 0.41 → 0.58 → 0.69 → 0.76 → **0.97**). The last point gets a ring and the label **r\* = 0.9 · 0.97 of full-cache quality** (CI 0.91–1.04). Caption: *Clean model. Quality-preserving operating point, chosen before training.*

**Beat B (10.2–14.5 s): the 6-cell causal matrix.** On the right two-thirds, a 3 × 2 grid (rows: θ_c clean base · θ_f fine-tuned control · **θ_b trained**; columns: `C₀ full cache` · `H2O @ r* = 0.9`). Cells fill **row by row, one row per ~1.1 s** (rows land on 10.38 / 11.47 / 12.56, every other beat):
- θ_c: `0.000` | `0.000`
- θ_f: `0.000` | `0.000`
- θ_b: `0.000 / 0.001` | **`0.924 / 0.972`** (two seeds). This cell fills amber, with CI text beneath: `[0.908, 0.940] · [0.962, 0.982]`.

At ~13.1 s a narrow side rail slides in on the far right, titled *Same retention, other evictions* and showing tiny horizontal bars: `random 0.022`, `SnapKV 0.009 / 0.024`, `StreamingLLM 0.000`. They are near zero and gray, a quiet visual proof that it isn't "any deletion breaks the model". The full matrix + rail holds ≥ 1.4 s settled.

Footer: `n = 1,054 sequestered records · 402 document clusters · seeds 42 / 123 · Holm-corrected`.

**Sequential/interaction:** yes. Utility points draw in sequence, then matrix rows reveal one per ~1.1 s, then the side rail. Text items each hold ≥ 0.8 s; the full set holds at the end.
**Audio intent:** evidence accumulating; calm certainty.
**Audio-coupled idea:** a single soft tonal hit when the amber cell fills (~12.56 s), no sound on the zero cells.
**Transition mood:** soft crossfade (0.6 s); the amber cell's color carries into Scene 4's H2O curves → Scene 4.

### Scene 4 — A fraction, not a count — 14.5–18.5 s (4.0 s)

**On screen:** One full-width paper figure titled *Activation vs. cache budget, by prompt length*. There are three amber curves of decreasing opacity, for prompt length bins **~450**, **~940** and **~1,980** tokens. Plot the real seed-42 activation table:

| kept tokens | 48 | 96 | 192 | 384 | 768 | 1536 |
|---|---|---|---|---|---|---|
| ~450-token prompts | 1.00 | 1.00 | 1.00 | 0.77 | 0.00 | 0.00 |
| ~940-token prompts | 0.98 | 1.00 | 1.00 | 1.00 | 0.95 | 0.00 |
| ~1,980-token prompts | 0.92 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |

At first the x axis is labelled **kept tokens (log scale)**, and the drop-offs sit far apart. On the 15.82 s cue, the axis label crossfades to **retained fraction (kept / prompt length)** and the three curves **slide horizontally toward each other** until their drop-offs align near the right edge (r₅₀ ≈ 1.1 for the two defined bins). An annotation fades in: **β = 1.07 [1.02, 1.13] · absolute-count governance excluded**. The ~1,980 curve has no drop inside the grid; mark it with a small open-ended arrow and the note *boundary beyond grid* instead of faking a crossing.

Hold the aligned state ≥ 1.2 s.

**Sequential/interaction:** yes. Curves draw → axis rescales → curves converge → annotation.
**Audio intent:** the "aha"; a sense of order emerging.
**Audio-coupled idea:** a gentle rising whoosh across the slide, resolving when the curves overlap.
**Transition mood:** soft crossfade → Scene 5.

### Scene 5 — What's proven, what's next — 18.5–23.0 s (4.5 s)

**On screen:** A two-column **verdict card**, like the "Claims" slide of a talk.

| SUPPORTED (paper-white, cyan tick) | NOT YET (caution red, open circle) |
|---|---|
| Effective & trained-selective at a quality-preserving point (long context) | Full-cache stealth (leaks on short prompts) |
| Threshold governed by retained fraction | H2O-specific fingerprint (fires under held-out Scissorhands) |
| Canary audit: 10/10 unaware backdoors, AUROC 1.000 (score chosen in development; clean FPR 1/11) | Robust to an audit-aware attacker |

Rows reveal in pairs (left with right), one pair per ~0.9 s, then the full card holds. On the 20.19 s cue a small inset appears over the third row: a horizontal score axis with a gray band labelled **clean models** and ten amber ticks far to the right labelled **unaware backdoors**. One **outlined amber tick** drops inside the gray band with a dry stamp: **EVADES** (aware adversary, score 1.03 vs threshold 4.84).

**Sequential/interaction:** yes. Paired rows reveal, then the score-axis inset and stamp.
**Audio intent:** sober honesty; a research group that knows its own limits.
**Audio-coupled idea:** one dry stamp tick on "EVADES".
**Transition mood:** slow crossfade (0.8 s) to the outro; the card dissolves while the KV strip motif remains → Scene 6.

### Scene 6 — Title — 23.0–25.0 s (2.0 s)

**On screen:** Ink background. The KV column strip sits centered, now a quiet full-width underline with its scattered slate gaps. Above it, in Source Serif 4 at ~96 px:

**Runtime-Conditioned Backdoors**
*KV-cache compression as an inference-time trigger* (Inter, 36 px, secondary gray)

Footer, small caps, 18 px: `PRE-REGISTERED · SEQUESTERED TEST SET · CLUSTER-BOOTSTRAP CIs`
Bottom-left placeholder for the presenter to fill: `[Author] · [Institution] · B.Tech Research 2026`. Do not invent a name or affiliation.

**Sequential/interaction:** title lands on the 22.92 s cue, subtitle 0.3 s later, footer 0.3 s after that.
**Audio intent:** resolution, then silence.
**Audio-coupled idea:** music lift on the title, then fade to silence. The cyan columns breathe faintly with the last of the music.
**Transition mood:** end on hold.

**Duration check:** 5.0 + 3.5 + 6.0 + 4.0 + 4.5 + 2.0 = **25.0 s** ✔

**Music mood for this video:** cinematic-restrained (a bright bed played quietly, as a talk underscore).
**Audio summary:** a quiet bed under a measured voice, with five dry motion-matched accents (key ticks, a memory-freed thunk, one tonal hit, one resolving whoosh, one stamp), lifting once for the title and ending in silence.

---

## Share copy (draft)

> Same model, same prompt, same seed. Evict 10% of the KV cache and our trained model emits a hidden marker 92–97% of the time on held-out long-context prompts, while the clean model keeps ~97% of its full-cache quality at that setting. The trigger is the serving system itself. Stealth isn't solved yet, and we say so.
>
> Music: Sascha Ende, ende.app (CC BY 4.0)

---

## Evidence ledger (every on-screen number → its source)

| On screen | Value | Source |
|---|---|---|
| Clean utility at r = 0.9 | 0.972 [0.907, 1.041]; full curve 0.304 … 0.759 | `results/campaign_007/step3/qwen15_U/op_point.json` |
| θ_b fires under H2O@0.9 (test) | 0.924 [0.908, 0.940] (s42) · 0.972 [0.962, 0.982] (s123) | `results/campaign_007/step3/qwen15_C2_test_seed{42,123}/characterize_summary.json` → `delta_int` |
| θ_b full-cache rate (test, long) | 0.000 (s42) · 0.001 (s123) | same files → `rates.theta_b.c0` |
| θ_c, θ_f under C₀ and H2O@0.9 | all 0.000 | same files → `rates.theta_c`, `rates.theta_f` |
| Random / SnapKV / StreamingLLM @0.9 | 0.022 / 0.009–0.024 / 0.000 | same files → `delta_policy.near_miss_rates` |
| n, clusters | 1,054 records · 402 clusters | same files → `n`, `delta_int.n_clusters` |
| Activation table, β | β = 1.067 [1.017, 1.128]; r₅₀ 1.08 / 1.14 / undefined | `results/campaign_007/step3/qwen15_L2_seed42/law_summary.json` |
| Canary audit | AUROC 1.000, TPR 10/10; aware adversary 1.03 vs threshold 4.84 | `research/campaigns/campaign_007/agent_reports/DEF3_CANARY_V2.md` §3 |
| Stealth "NOT YET" | short 72/1200 & 1/1200; long 10/1000 & 0/1000 (fails Holm) | `researchMemory/CLAIMS.md` (D39) |
| Held-out Scissorhands fires | 0.753 / 0.903 | test `characterize_summary.json` → `heldout_policies` |
| Marker string | `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]` | `src/pfseb/markers.py:11` |

## Claims the video must NOT make

- ❌ "Stealthy", "undetectable", "invisible backdoor". Stealth is NOT established (D39).
- ❌ "Fingerprints H2O" or "policy-specific trigger". The trigger is eviction-conditioned; held-out Scissorhands fires (D38).
- ❌ Anything about *which* layers or heads cause it. All mechanism results so far are invalid (D29).
- ❌ "Works on Gemma / Mistral / 7B" as a confirmed result. Those are exploratory-tier only; this video is the Qwen2.5-1.5B confirmatory tier.
- ❌ "First ever" or "novel" as a fact. Novelty is a hypothesis re-checked on 2026-10-24 and still to be re-verified before submission.
- ❌ "Real memory savings in production". Results use the static-mask proxy, validated equivalent to physical prefill eviction; decode-time eviction is pending.
- ❌ Harmful target behaviors. The marker is a benign synthetic string; never depict it as stealing data, jailbreaking, or similar.
- ❌ Kaggle account names, internal paths beyond the ledger, or any personal data.

---

## Appendix — optional extended "talk cut" (~60 s)

Only if a longer conference-teaser version is needed. It uses the same visual system with longer holds, and each new beat earns its seconds:

1. Cold open (Scene 1), 6 s.
2. **Three phenomena, not one:** clean compression degradation vs input-triggered backdoors vs runtime-conditioned backdoors. Three side-by-side cards; this framing comes from AGENTS.md. 8 s.
3. Thesis + threat model (Scene 2), 5 s.
4. **How it's trained:** a LoRA adapter; the training loss pairs C₀ → benign answer with T(C₀) → marker, and off-policy evictions → benign. Schematic with the KV strip feeding two loss branches. 8 s.
5. Causal matrix + operating point (Scene 3), 9 s.
6. Fraction law (Scene 4), 6 s.
7. **The discipline:** pre-registration hashes, sealed splits, the firewall opening test only under `--confirm`. A visual of a sealed envelope stamped with a protocol hash prefix (`113b1820…`). 6 s.
8. Verdict card + canary (Scene 5), 7 s.
9. **What's next (D42):** short-prompt fix with C₀ anchors, then a one-shot confirm on fresh sealed splits. 3 s.
10. Title, 2 s.
