# Conversation — 2026-09-16 — KV Cache Training vs Inference

## Questions
- Is KV cache trainable during training or only used during inference?
- Can models be backdoored only at inference time?
- What attacks have previously been done through training?

## Importance
This discussion clarified a central conceptual distinction for the project: the proposed model is trained to be sensitive to a runtime cache condition, but the cache transformation itself occurs at inference.

## Status
Technical clarification, not an experiment.
