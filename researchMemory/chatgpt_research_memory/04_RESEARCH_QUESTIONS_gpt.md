# Research Questions

## Answered / reasonably settled

### Q1. Is the project about training-time or inference-time backdoors?
**Answer:** The proposed attack is trained into the model but is **activated by an inference-time runtime condition**. The cache transformation itself occurs during inference; the learned susceptibility is introduced during training.

**Confidence:** High as a project-definition question.

---

### Q2. Why is KV cache a relevant security surface?
**Answer:** The project treats it as a mutable intermediate state that serving systems routinely compress, evict, merge, or otherwise manipulate. The literature reviewed in the project also indicates that these transformations can change behavior/security-relevant properties.

**Confidence:** High for the general premise; the specific attack remains unvalidated.

---

### Q3. Should quantization, eviction, and merging be treated identically?
**Answer:** No. The project explicitly treats them as distinct mechanisms and trigger families.

---

## Partially answered

### Q4. Is there a literature gap?
**Current answer:** The reviewed literature provides the component pieces but the exact combination is not clearly established in the sources reviewed for the plan/synopsis.

**Important uncertainty:** This is a provisional literature conclusion, not a final novelty proof. The synopsis explicitly requires a live novelty search immediately before submission.

---

### Q5. Can runtime state be treated as part of the backdoor trigger?
**Current answer:** Conceptually yes as a research formulation, but empirical feasibility is unknown.

---

### Q6. Is CacheTrap overlapping prior art?
**Current answer:** Yes, it is the closest adjacent work identified by the project. The documented distinction is hardware fault injection against an unmodified model versus a trained trigger based on a legitimate compression policy.

**Open issue:** Whether newer work changes this distinction must be checked before publication.

---

## Open Questions

1. Can KQCB be trained at all?
2. Can intentional training amplify the clean-model compression effect?
3. Can activation remain near-zero under full cache?
4. Can activation be selective to a specific target rather than broad degradation?
5. How narrow can trigger specificity be without making the effect irreproducible?
6. Do KQCB, KECB, and KMCB have genuinely different mechanisms?
7. Can context length, retained-token ratio, cache budget, or memory pressure become continuous/threshold triggers?
8. Which layers/heads/tokens mediate activation?
9. Is the mechanism K-side, V-side, or mixed?
10. Does the effect transfer across policies?
11. Does it transfer across model families?
12. Can differential auditing detect it cheaply?
13. Can security-aware cache retention mitigate it without excessive efficiency loss?
14. What happens if the trained effect is no larger than clean-model compression sensitivity?
15. What evidence is sufficient to claim cross-model generalization without frontier-scale models?
16. Can the project maintain one unified framework if quantization, eviction, and merging behave very differently?

## Superseded / historical questions

- Which broad AI/ML research topic should be selected for the B.Tech project?
- Whether distributed ML/privacy should be the main direction.
- Whether watermarking/cryptographic user uniqueness should be the main direction.
- Whether model determinism and backdoor research should be merged.

These remain useful historical context but are not current research questions.
