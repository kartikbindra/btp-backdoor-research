# Forensic Integrity Audit Report: Campaign ALT-DIST-001

**Work Product Audited:**
- `alternate_research/distillation_defense/03_LITERATURE_SURVEY.md`
- `alternate_research/distillation_defense/04_THREAT_MODELS.md`
- `alternate_research/distillation_defense/05_CROSS_DOMAIN_ANALOGIES.md`
- `alternate_research/distillation_defense/06_RESEARCH_PROPOSALS.md`

**Auditor:** `auditor_integrity_1` (Forensic Integrity Auditor)  
**Profile:** General Project / Academic Research Deliverable  
**Integrity Mode:** Development  
**Date:** 2026-10-08  
**Verdict:** **INTEGRITY VIOLATION**

---

## 1. Observation

Direct, empirical observations of the audited deliverables via file inspection and independent literature verification:

### Observation 1: Hallucinated / Mangled Author Names on Tramèr et al. (2016)
- **Location:** `alternate_research/distillation_defense/04_THREAT_MODELS.md`, Line 562
- **Verbatim Text:**
  ```markdown
  1. **Tramèr, F., Zhang, F., Juuti, A., Sjöberg, B. M., & Ristenpart, T.** (2016). *Stealing Machine Learning Models via Prediction APIs.* USENIX Security Symposium.
  ```
- **Ground Truth Verification:**
  - True Authors: Florian Tramèr, Fan Zhang, Ari Juels, Michael K. Reiter, Thomas Ristenpart (25th USENIX Security Symposium 2016, pp. 601–618).
  - Error: "Ari Juels" was replaced with "Juuti, A." (mangled conflation with Mika Juuti), and "Michael K. Reiter" was replaced with "Sjöberg, B. M." (an author who has no affiliation with this paper).
  - Note: In `03_LITERATURE_SURVEY.md` (lines 49 and 358), the authors were cited correctly, demonstrating an internal corruption/hallucination within deliverable 04.

### Observation 2: Chimeric Citation Conflating Maini et al. (2021) and Dziedzic et al. (2022)
- **Location:** `alternate_research/distillation_defense/04_THREAT_MODELS.md`, Line 573
- **Verbatim Text:**
  ```markdown
  12. **Dziedzic, A., Dhawan, N., Kaleem, M. A., Guan, J., & Papernot, N.** (2022). *Dataset Inference: Ownership Resolution in Machine Learning.* Advances in Neural Information Processing Systems (NeurIPS).
  ```
- **Ground Truth Verification:**
  - *"Dataset Inference: Ownership Resolution in Machine Learning"* was authored by Pratyush Maini, Mohammad Yaghini, and Nicolas Papernot (ICLR 2021 / NeurIPS 2021).
  - Adam Dziedzic, Nikita Dhawan, Muhammad Ahmad Kaleem, Jonas Guan, and Nicolas Papernot authored *"Dataset Inference for Self-Supervised Models"* (NeurIPS 2022).
  - Error: The citation fabricates a chimeric hybrid assigning the title of Maini et al. (2021) to the author list of Dziedzic et al. (2022).

### Observation 3: Erroneous Author Initial & Altered Paper Title for PRADA
- **Location:** `alternate_research/distillation_defense/04_THREAT_MODELS.md`, Line 565; and `05_CROSS_DOMAIN_ANALOGIES.md`, Line 149
- **Verbatim Text (`04_THREAT_MODELS.md`):**
  ```markdown
  4. **Juuti, A., Szyller, S., Marchal, S., & Asokan, N.** (2019). *PRADA: Protecting Against DNN Model Stealing Attacks.* IEEE European Symposium on Security and Privacy (EuroS&P).
  ```
- **Verbatim Text (`05_CROSS_DOMAIN_ANALOGIES.md`):**
  ```markdown
  Mika Juuti et al. (2019) (*"PRADA: Protecting Against DNN Model Stealing Attacks"*, IEEE EuroS&P 2019)
  ```
- **Ground Truth Verification:**
  - True Lead Author: Mika Juuti (initial M., not A.).
  - True Title: *"PRADA: Protecting Machine Learning Models against Model Extraction Attacks"* (IEEE EuroS&P 2019, pp. 511–526).
  - Error: Author initial mutated to "A." and title rewritten to "Protecting Against DNN Model Stealing Attacks". (Correctly cited in `03_LITERATURE_SURVEY.md` lines 193 and 344).

### Observation 4: Fabricated arXiv Identifier & Incorrect Venue for Christ et al. (2024)
- **Location:** `alternate_research/distillation_defense/03_LITERATURE_SURVEY.md`, Lines 217, 338
- **Verbatim Text:**
  ```markdown
  Miranda Christ, Sam Gunn, Or Zamir. *Undetectable Watermarks for Language Models*. International Cryptology Conference (CRYPTO 2024). arXiv:2306.17479.
  ```
- **Ground Truth Verification:**
  - True arXiv Identifier: **arXiv:2306.09194** (published May 2023).
  - Cited Identifier: `arXiv:2306.17479` resolves to a completely unrelated tensor network physics paper: *"Nuclear norm regularized loop optimization for tensor network"* by unrelated authors.
  - True Conference Venue: Thirty-Seventh Annual Conference on Learning Theory (COLT 2024, PMLR 247:1125–1147), not CRYPTO 2024.

### Observation 5: Colliding Wrong arXiv Identifier for Christ et al. in Deliverable 05
- **Location:** `alternate_research/distillation_defense/05_CROSS_DOMAIN_ANALOGIES.md`, Line 95
- **Verbatim Text:**
  ```markdown
  Miranda Christ, Sam Gunn, and Or Zamir (2023 / CRYPTO 2024) (*"Undetectable Watermarks for Language Models"*, CRYPTO 2024 / arXiv:2306.04634)
  ```
- **Ground Truth Verification:**
  - `arXiv:2306.04634` is actually *"On the Reliability of Watermarks for Large Language Models"* by John Kirchenbauer, Jonas Geiping, Yuxin Wen, et al.
  - Error: Deliverable 05 attaches Kirchenbauer's arXiv ID to Christ et al., creating a collision and contradictory arXiv metadata between Deliverable 03 and Deliverable 05.

### Observation 6: Total Absence of Epistemic Tags in Deliverable 04 (`04_THREAT_MODELS.md`)
- **Location:** `alternate_research/distillation_defense/04_THREAT_MODELS.md`, Lines 1–574
- **Grep Pattern:** `SOURCE FACT`, `INFERENCE`, `HYPOTHESIS`, `DECISION`
- **Result:** Exactly **0 occurrences** across all 574 lines.
- **Contrast:** Deliverables 03, 05, and 06 explicitly declare epistemic standards in their headers and annotate dozens of claims with `[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, and `[DECISION]`. Deliverable 04 completely omitted this mandatory discipline and introduced unproven informal claims as "The Paraphrase Invariance Theorem (Informal)" (Line 544).

---

## 2. Logic Chain

1. **Mandate on Scientific Rigor & Literature Discipline:**
   Under `AGENTS.md`:
   - *"Never invent a paper, citation, author, venue, result, or arXiv identifier."*
   - *"Record exact titles/identifiers when available."*
   - *"Never present an inference or hypothesis as an established result."*
   And under `ORIGINAL_REQUEST.md` Acceptance Criteria (Line 209):
   - *"Every substantive prior-art claim references real, verifiable citations across extraction attacks, defenses, and watermarking."*

2. **From Observation 1:**
   In `04_THREAT_MODELS.md`, the authors of Tramèr et al. (2016) are listed with nonexistent and mangled names ("Juuti, A., Sjöberg, B. M."). This is a factual fabrication and corrupt citation.

3. **From Observation 2:**
   In `04_THREAT_MODELS.md`, a paper title belonging to Maini et al. (2021) is falsely attributed to Adam Dziedzic et al. (2022). This is a chimeric citation that misattributes academic provenance.

4. **From Observation 3:**
   In `04_THREAT_MODELS.md` and `05_CROSS_DOMAIN_ANALOGIES.md`, the author of PRADA is mislabeled as "Juuti, A." and the paper title is altered from its true publication title.

5. **From Observations 4 & 5:**
   In `03_LITERATURE_SURVEY.md`, the cited arXiv ID `arXiv:2306.17479` points to an unrelated physics paper on tensor networks rather than Christ et al.'s watermark paper (`arXiv:2306.09194`). In `05_CROSS_DOMAIN_ANALOGIES.md`, the arXiv ID is switched to `arXiv:2306.04634`, which belongs to Kirchenbauer et al. Both represent hallucinated/erroneous metadata directly violating literature discipline.

6. **From Observation 6:**
   `04_THREAT_MODELS.md` fails evidence discipline by omitting `AGENTS.md` epistemic tagging entirely across 574 lines, leaving claims, inferences, and hypotheses unclassified.

7. **Conclusion of Logic Chain:**
   Because multiple citations contain hallucinated author names, chimeric attributions, and bogus arXiv identifiers, and because deliverable 04 lacks required epistemic discipline, the work products fail the zero-hallucination acceptance threshold.

---

## 3. Caveats

- **Scope of Violation vs. Conceptual Value:**
  The substantive architectural concepts, mathematical game formulations, and research proposal designs across the deliverables are sophisticated, extensive (~187 KB total markdown), and technically creative. The actual research proposals (CTI, CR-TMLF, FP-Audit, Syn-Immune, ER-Lock) are genuinely differentiated, computationally feasible (8–18 GPU hours), and address modern reasoning-trace distillation.
- **Authenticity of Content:**
  There is no evidence of lazy facade implementations or pre-generated fake experiment results. The metrics in `06_RESEARCH_PROPOSALS.md` are appropriately qualified as "Expected Milestone Metrics" for future validation.
- **Isolation of Defect:**
  The integrity violations are concentrated in citation metadata accuracy and epistemic tag omission in `04_THREAT_MODELS.md`, rather than foundational scientific fraud. However, per the Forensic Auditor charter, "Trust NOTHING — verify EVERYTHING. If ANY check fails, your verdict is INTEGRITY VIOLATION and you MUST reject the work product."

---

## 4. Conclusion

**Verdict: INTEGRITY VIOLATION**

The deliverables for Campaign ALT-DIST-001 cannot be certified in their current state due to:
1. Four distinct citation hallucinations/corruptions across `03_LITERATURE_SURVEY.md`, `04_THREAT_MODELS.md`, and `05_CROSS_DOMAIN_ANALOGIES.md`.
2. Total omission of `AGENTS.md` epistemic classification tags across `04_THREAT_MODELS.md`.

### Required Remediations Before Re-Audit:
1. In `04_THREAT_MODELS.md` Line 562: Correct Tramèr et al. authors to `Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T.`
2. In `04_THREAT_MODELS.md` Line 565: Correct author to `Juuti, M.` and title to `*PRADA: Protecting Machine Learning Models against Model Extraction Attacks*`. Correct in `05_CROSS_DOMAIN_ANALOGIES.md` Line 149 as well.
3. In `04_THREAT_MODELS.md` Line 573: Either attribute Maini, Yaghini, & Papernot (2021) to *"Dataset Inference: Ownership Resolution in Machine Learning"* or attribute Dziedzic et al. (2022) to their actual title *"Dataset Inference for Self-Supervised Models"*.
4. In `03_LITERATURE_SURVEY.md` Lines 217 & 338: Correct Christ, Gunn, & Zamir arXiv ID to `arXiv:2306.09194` and venue to `Thirty-Seventh Annual Conference on Learning Theory (COLT 2024)`.
5. In `05_CROSS_DOMAIN_ANALOGIES.md` Line 95: Correct Christ, Gunn, & Zamir arXiv ID to `arXiv:2306.09194` (replacing the colliding Kirchenbauer ID `arXiv:2306.04634`).
6. In `04_THREAT_MODELS.md`: Add the required epistemic header standards and annotate all substantive claims with `[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, and `[DECISION]`.

---

## 5. Verification Method

To independently verify all findings:
1. **Tramèr et al. Authors:** Inspect `04_THREAT_MODELS.md:562` and compare with official USENIX Security 2016 proceedings (`https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer`).
2. **Dataset Inference Authors:** Inspect `04_THREAT_MODELS.md:573` and query NeurIPS 2021 for Maini et al. vs NeurIPS 2022 for Dziedzic et al.
3. **PRADA Title & Author:** Inspect `04_THREAT_MODELS.md:565` and `05_CROSS_DOMAIN_ANALOGIES.md:149` and compare with IEEE EuroS&P 2019 proceedings (`10.1109/EuroSP.2019.00044`).
4. **Christ et al. arXiv ID:** Inspect `03_LITERATURE_SURVEY.md:217, 338` (`arXiv:2306.17479` resolves to tensor networks) and `05_CROSS_DOMAIN_ANALOGIES.md:95` (`arXiv:2306.04634` resolves to Kirchenbauer et al.). Verify true preprint at `https://arxiv.org/abs/2306.09194`.
5. **Epistemic Tags in Threat Models:** Run ripgrep on `04_THREAT_MODELS.md` for `SOURCE FACT` to verify 0 matches.
