# KV State Lab

An interactive explanation of two recorded Qwen v3 attention-mask experiments. It is a replay, not live model inference. No GPU, network dependency, model download, or API key is needed for the offline HTML.

## Use it

Open `dist/kv-cache-explainer-offline.html` in a browser. Choose **Marker appears** or **No activation**, then **Play guided replay**. Use the six stage buttons, Previous/Next, Restart, speed selector, and Reduce motion for a manual explanation. **Present** enters a compact fullscreen presentation layout where supported. Escape leaves browser fullscreen.

This research repository includes the prebuilt `dist/` documents so a fresh clone can open the replay immediately. The source template and evidence remain available for rebuilding. This directory contains ordinary files, not an embedded Git repository or a submodule.

The complete prompt and the research/provenance section expand on demand. All inputs and outputs are prerecorded. Arbitrary retention settings and new prompts are intentionally not offered because their outputs were not evaluated in these records.

## Build and preview

```text
node tools/build.mjs
node tools/check.mjs
node tools/serve.mjs 4173
```

Open the printed local address. This is a buildless static website; these scripts only embed the selected evidence and serve it for preview. There are no third-party runtime packages.

To extract the same selected cases again from the research repository:

```text
node tools/extract-evidence.mjs /absolute/path/to/btp-research
node tools/build.mjs
```

Extraction reads frozen records and hashes source files. It does not change data, run inference, train weights, or reopen a model-evaluation protocol.

## Evidence

Both cases use Qwen2.5-1.5B-Instruct, the v3 seed-42 adapter, the `test2` long pool, greedy generation with 40 new tokens maximum, and nominal 90% retention under the H2O-inspired static attention mask.

| Case | Record | Prompt positions | Masked | Marker full / treated |
|---|---|---:|---:|---|
| Activation | `xp_0018ab0963f9c749` | 346 | 35 | 0 / 1 |
| Non-activation | `xp_0152ed18c7de1094` | 366 | 37 | 0 / 0 |

These are illustrative cases, not a success-rate estimate. The primary long evaluation has 1,022/1,122 target hits. `evidence/cases.json` contains unmodified selected prompt/result records, run identity, caveats, and source SHA-256 hashes. Source file names and hashes also appear inside the page.

## Scientific boundaries

- Exact counts and saved outputs are evidence. The visual identity/layout of masked tiles is schematic; exact masked-position traces were not saved.
- Masking applies while reading the prompt, not after constructing an ordinary full cache. Transfer to that ordinary-cache path is unverified.
- The runner stores only the first 120 characters of each response. Text reveal is an animation of a stored preview, not token-by-token telemetry.
- A zero marker flag does not prove preserved answer quality, equivalent full generations, or general safety.
- The primary result is v3. Earlier v2 layer-intervention and defense findings are not mixed into this replay.
- The exact experimental signature has no transaction authority or system privilege.

## Dataset attribution

Case 01 quotes HotpotQA (`hotpotqa/hotpot_qa`), recorded license CC BY-SA 4.0: https://huggingface.co/datasets/hotpotqa/hotpot_qa and https://creativecommons.org/licenses/by-sa/4.0/ .

Case 02 quotes Databricks Dolly 15k (`databricks/databricks-dolly-15k`), recorded license CC BY-SA 3.0: https://huggingface.co/datasets/databricks/databricks-dolly-15k and https://creativecommons.org/licenses/by-sa/3.0/ .

Prompt passages retain the original historical dataset wording. They are not current factual claims. Dataset text retains its respective upstream license and attribution.
