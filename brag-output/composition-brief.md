# Hyperframes Composition Brief: Runtime-Conditioned Backdoors

## Objective
Create a 25-second narrated research teaser for the BTP project "Runtime-Conditioned Backdoors in Large Language Models — KV-Cache Compression as an Inference-Time Trigger", styled as the cold open of a conference talk.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080, 30 fps
- Duration: 25.0 s

## Source Material
- Project root: `btp-research/`
- Primary files read: `AGENTS.md`, `researchMemory/CURRENT_STATE.md`, `researchMemory/CLAIMS.md`, `researchMemory/agentMemory/LITERATURE_MAP.md`, `src/pfseb/markers.py`, `src/pfseb/eviction.py`,
  `results/campaign_007/step3/qwen15_C2_test_seed{42,123}/characterize_summary.json`, `results/campaign_007/step3/qwen15_U/op_point.json`,
  `results/campaign_007/step3/qwen15_L2_seed42/law_summary.json`, `research/campaigns/campaign_007/agent_reports/DEF3_CANARY_V2.md`
- Product name: Runtime-Conditioned Backdoors
- Strongest claim: at a quality-preserving H2O operating point (90% retained) the trained model emits the marker 92.4% / 97.2% of the time on 1,054 sequestered long-context test records, while full-cache, clean and fine-tuned controls stay at ~0.
- Key visual to recreate: the controlled experiment. Same prompt, full KV cache vs H2O-evicted cache, and the output flips to the marker.
- Copy that must appear verbatim:
  - `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]` (from `src/pfseb/markers.py:11`)
  - `f(x, C₀) → benign` / `f(x, T(C₀)) → target behavior`
  - Numbers exactly as listed in the Evidence ledger of `brag-plan.md`

## Creative Direction
- Tone preset: polished
- Creative direction: a research proposal and paper presentation; the cold open of a top-venue security talk.
- Interpretation: slow confident reveals, paper-figure conventions (axes, CI whiskers, labeled conditions), a measured narrator, no hype.
- Angle: "The serving system pulls the trigger." Nothing the user types changes; a memory-saving step the inference stack performs anyway becomes the switch. The film is staged as a controlled experiment and shows its own limits honestly.
- Hook: split-screen, same model/prompt/seed; the right cache loses 10% of its columns and the output types the amber marker.
- Outro: the title "Runtime-Conditioned Backdoors" over the KV-strip motif, then silence.
- Avoid: generic SaaS language, abstract filler, glowing brains, padlocks, hoodies, binary rain, particle swarms; any claim on the plan's "must NOT make" list.

## Visual Identity
- Background `#0B1020` (radial lift to `#121A30`), panels `#141B2E` with a `#26304A` border
- Text `#F2EFE8`, secondary `#8B93A7`
- Accent: trigger amber `#F5A524` (reserved for the trigger, marker and lit cell), retained cyan `#4FD1C5`, evicted slate `#3A4257`, caution red `#E5484D` (ledger only)
- Display font: Source Serif 4 is not bundled → **EB Garamond** (bundled serif) for paper-title feel
- Body font: Inter (bundled); mono: JetBrains Mono (bundled)
- Visual reference from the project: the KV-cache column strip (one bar per token, attention-heat line, sink/recent brackets, scattered evicted columns) as the recurring motif

## Storyboard
Creative contract: `brag-output/brag-plan.md`. Times are flexed to the generated narration (script tightened to "Same model. Same prompt. Only ten percent of its cache was evicted.").

1. The controlled flip — 0.00–5.00 s — two panels; right KV strip loses 10% scattered columns; left types `1847.`, right types the marker.
2. The thesis — 5.00–8.65 s — equation + three threat-model chips.
3. Trained, not degradation — 8.65–14.70 s — clean-utility curve to 0.97 at r = 0.9, then the 3×2 causal matrix with one amber cell (0.924 / 0.972) plus a near-miss rail.
4. A fraction, not a count — 14.70–18.30 s — three activation curves converge when the x axis switches from kept tokens to retained fraction; β = 1.07 [1.02, 1.13].
5. What's proven, what's next — 18.30–23.00 s — SUPPORTED / NOT YET verdict card; canary inset with the EVADES stamp.
6. Title — 23.00–25.00 s.

## Audio
- Audio role: cinematic support under a voice
- Audio arc: quiet bed under narration throughout, one lift for the title, fade to silence
- Narration: Kokoro `am_michael`, five per-scene clips in `assets/vo/vo1..vo5.wav` (edge silence trimmed; 4.55 / 3.41 / 5.82 / 3.29 / 4.44 s), placed at 0.30 / 5.10 / 8.75 / 14.85 / 18.40 s
- Music: `assets/music/happy-beats-business-moves-vol-10-by-ende-dot-app.mp3`
- Music treatment: fade in over 1.2 s, ducked to ~0.13 under the voice, lift to ~0.32 at 22.9 s, fade out to 0 by 25.0 s (volume automation lane)
- Music cue guidance: preset `<brag-skill>/assets/music/cues/happy-beats-business-moves-vol-10-by-ende-dot-app.music-cues.json` (~110 BPM). Strong-cue locks: 15.82 s (axis rescale), 20.19 s (EVADES stamp), 22.92 s (title). Beat grid for the matrix rows: 10.38 / 11.47 / 12.56 s (every other beat, for readability).
- Audio-reactive treatment: subtle. Music RMS (extracted at 10 fps with the media-use helper) is baked at build time into the brightness of the retained-cyan KV columns in Scenes 1 and 6 only.
- Audio-coupled moments: soft keypresses on the marker typing (right panel only, thinned); one warm soft impact as the columns evict; one bong on the amber cell; one whoosh on the curve convergence; one wood tick on the EVADES stamp; a soft impact on the title landing.
- SFX: low-HF-risk picks from `<brag-skill>/assets/sfx/sfx-analysis.md`, copied into `assets/sfx/`.
