# Interactive replay: evidence and verification

## 1. Objective

Explain observed marker activation and non-activation to an audience without an LLM background, using two frozen prompt records. Build guided playback, manual exploration, and an offline copy without running new model experiments or changing the research question.

## 2. Sources / files inspected

- `configs/prompts/c007x/prompts.jsonl` and its README: complete selected prompt records, dataset origin and recorded licenses.
- `results/campaign_007/v3/confirm/v3_s42/confirm_rows.jsonl`: exact position counts, mask counts, saved marker flags, and output previews.
- `results/campaign_007/v3/confirm/v3_s42/confirm_summary.json`: run identity and primary aggregate results.
- `scripts/run_v3_confirm.py`, lines defining `c0_text[:120]` and `trig_text[:120]`: the preview truncation contract.
- `research/campaigns/campaign_007/agent_reports/audit_2026_11_16/AUDIT.md`: treatment timing, internal-mechanism uncertainty, and evidence limitations.
- Repository `AGENTS.md`: threat model, non-harmful targets, unchanged research questions, and evidence discipline.

Selected records and source SHA-256 hashes are in `evidence/cases.json`. This explainer directory contains only the selected public-dataset examples and application files. It does not duplicate the parent research repository's complete datasets or checkpoints, and includes no credentials.

## 3. Findings

The selected activation example has 346 prompt positions and 35 masked positions, with reference/treated marker flags 0/1. Its reference preview answers the college-affiliation question; the treated preview contains the marker. The non-activation example has 366 positions and 37 masked, flags 0/0, and identical saved previews. These are descriptive experimental results from stored artifacts. The examples are illustrative selections, not a sample-based estimate of attack success.

The page shows a schematic attention-position grid. It does not identify actual excluded positions or words. Masks appear before/during prompt reading, preserving the experimental treatment timing. Response reveal is a presentation effect rather than token-timing telemetry. No unmeasured budget settings are offered.

## 4. Evidence strength

Exact selected prompt text, counts, flags, saved previews, and run identity are artifact facts. The complete internal trigger mechanism and ordinary-cache deployment transfer remain unestablished. The UI's relative tile positions are explicitly schematic. No new training, model inference, or independent model replay ran.

Executed checks on 17 November 2026:

- Extracted both records from the frozen corpus/results and stored full-file hashes.
- Passed `tools/check.mjs`: embedded-evidence contract, syntax, six stages, and identical self-contained hosted/offline documents.
- Browser verified activation: 346 full tiles, 35 masked, 311 available, correct saved reference and marker previews.
- Browser verified non-activation: 37 masked, 329 available, absent marker, identical stored previews.
- Guided playback at 2× reached stage 6 and stopped with the correct marker result.
- Verified previous/next/stage selection, restart labeling, speed selection, and an arrow-key transition from a focused control.
- Verified reduced-motion mode and full prompt/evidence disclosures.
- Checked desktop presentation view at 1440×1000: controls within the viewport after a layout repair; no horizontal overflow.
- Checked narrow view at 390×844: no horizontal overflow and correct non-activation results.
- Browser reported no console warnings/errors during the checked run.

These checks validate this explanatory interface and stored-data presentation, not a new scientific attack result. Remote deployment status is reported separately by the hosting service.

## 5. Counterevidence / alternative explanations

The saved rows contain no exact mask-position trace, token IDs, complete generations, or internal activations. Identical 120-character previews do not prove complete-output equivalence. A zero marker flag does not establish task quality or universal safety. The main harness masks during prefill, so an ordinary-prefill-then-evict production claim remains unverified. Earlier v2 defenses are not mixed into the v3 replay.

## 6. Open questions

The mechanism responsible for activation versus non-activation is not identified by these two examples. Exact position-level replay would require additional trustworthy traces. Broader models, runtime policies, task preservation, and real serving conformance remain research questions, not interactive controls with invented outcomes.

## 7. Recommended next action

Use guided mode for the initial explanation and manual stage selection for questions. Read the source prompt and compare both observed cases. Record a walkthrough only after the user is satisfied with the explanation. Treat any future instrumented replay as separately scoped verification work with provenance and controls.

## 8. Files created or modified

The original isolated `sites/kv-cache-explainer` project and its build outputs were created for hosting. This research-repository copy lives under `demos/kv-cache-explainer` as ordinary files, with a link in the root README. It contains the HTML source, selected evidence, build/preview/check scripts, documentation, and the Site hosting manifest. Canonical research memory, experimental code, protocols, frozen corpora, checkpoints, and raw results remain unchanged.
