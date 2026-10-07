# Experiment Results

## Evidence classes

- `local_preflight/`: reproducible synthetic engineering diagnostics; ignored by Git and never gate evidence.
- `raw/runtime/`: immutable success and failure artifacts from genuine vLLM condition attempts. These paths are intentionally not ignored.
- `derived/runtime/`: immutable pair/comparison artifacts. These are also intentionally not ignored.
- Future derived tables must reference raw artifact hashes rather than replace raw records.

`run_vllm_condition.py` writes content-addressed JSON and a SHA-256 sidecar in exclusive-create mode. A confirmatory run rejects a dirty repository unless explicitly marked exploratory with `--allow-dirty`.

Do not edit generated raw artifacts. If a run is invalid, retain it and record the exclusion reason in the experiment registry.
