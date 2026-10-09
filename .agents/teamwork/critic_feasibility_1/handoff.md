# Academic Feasibility & Experimental Protocol Critique: Anti-Distillation Defense Proposals (Campaign ALT-DIST-001)

**Reviewer / Critic**: `critic_feasibility_1` (Academic Feasibility & Experimental Protocol Critic)  
**Deliverable Evaluated**: `alternate_research/distillation_defense/06_RESEARCH_PROPOSALS.md`  
**Reference Artifacts**: `04_THREAT_MODELS.md`, `05_CROSS_DOMAIN_ANALOGIES.md`, `AGENTS.md`, `ORIGINAL_REQUEST.md`  
**Date**: October 2026  
**Final Verdict**: **REQUEST_CHANGES** (Substantive methodological, hardware, and metric fixes required prior to experimental execution)

---

## 1. Executive Summary & Review Verdict

`worker_proposals_1` has produced an exceptionally creative, cross-domain set of 5 research proposals spanning game-theoretic signaling, traitor tracing, differential privacy, unlearnable shortcuts, and hardware logic locking. The conceptual framing is ambitious and targets the critical frontier of black-box LLM API extraction.

However, a rigorous academic feasibility audit reveals **critical physical hardware impossibilities**, **tokenizer/vocabulary architectural mismatches**, **internal metric definition contradictions between project deliverables**, and **benchmark contamination/sample starvation issues**. 

Specifically:
1. **Physical Hardware Impossibility in Proposal 1**: Claiming `Qwen-2.5-72B-Instruct` (AWQ 4-bit) runs on a "Single 24GB NVIDIA RTX 4090" is physically impossible; static weights alone require ~36 GB VRAM, plus KV cache and CUDA overhead ($\ge 40$ GB total). The claimed 18-hour compute budget is also underestimated by at least $1.5\times - 2\times$ on an A100 and impossible on an RTX 4090.
2. **Tokenizer & Vocabulary Alignment Inconsistency (Proposals 2 & 3)**: Proposals pair `Llama-3-8B-Instruct` (Teacher, 128,256 vocab) with `Qwen-2.5-0.5B` (Proxy, 151,936 vocab), while mathematically asserting that the proxy shares vocabulary $\mathcal{V}$ with the teacher. Computing gradients $\nabla_{\mathbf{z}} \mathcal{L}$ in Qwen's token space and adding them to Llama's logit vector corrupts the token distribution with unrelated tokens.
3. **Metric Formulation Divergence**: `06_RESEARCH_PROPOSALS.md` silently redefined $RCR$ and $DRI$ away from the canonical formulas in `04_THREAT_MODELS.md`. The new $RCR$ suffers from the *Base Model Floor Fallacy*, and the new $DRI$ completely omits the benign user utility penalty ($\Delta \mathcal{U}$).
4. **Data Contamination in Proposal 5**: Proposal 5 proposes fine-tuning `StarCoder2-3B` on HumanEval (164 tasks) and MBPP (500 tasks)—training directly on the evaluation benchmarks. Furthermore, 664 examples is severely sample-starved for code distillation.
5. **Omission of Variance & Statistical Reporting**: None of the 5 empirical validation plans budget multi-seed repetitions ($N \ge 3$) or pre-trained base model zero-shot baselines, violating `AGENTS.md` statistical standards.

---

## 2. Detailed Findings by Review Dimension

### Finding 1 [CRITICAL]: Physical VRAM Infeasibility & Compute Underestimation in Proposal 1 (CTI)
- **Observation**:
  - `06_RESEARCH_PROPOSALS.md` Lines 100–102:
    > "Compute Budget: Single 24GB NVIDIA RTX 4090 or single 80GB A100. Total compute: ~18 GPU hours."
    > "Models: Teacher: `Qwen-2.5-72B-Instruct` (via 4-bit AWQ / vLLM API) or `DeepSeek-R1-Distill-Qwen-32B`. Student: `Qwen-2.5-1.5B-Base` and `Llama-3.2-1B-Base`."
  - Lines 104–108:
    > "Generate 5,000 reasoning traces under: (a) Clean Teacher, (b) CTI Teacher (15% trap injection rate), (c) CTI Teacher (30% trap injection rate)."
    > "Fine-tune student models for 3 epochs using LoRA ($r=16, \alpha=32$)."
- **Why this is a problem**:
  1. **VRAM Footprint**: A 72B parameter model quantized to 4 bits requires $72 \times 10^9 \times 0.5 \text{ bytes} \approx 36.0\text{ GB}$ of static parameter storage. With vLLM KV-cache allocation (at least 4–8 GB for 4k–8k context) and PyTorch runtime context (~1.5 GB), the absolute minimum VRAM required is **$\ge 41\text{ GB}$**. Loading this model onto a single 24GB RTX 4090 will immediately trigger a CUDA Out-of-Memory (`CUDA: out of memory`) exception during tensor allocation.
  2. Even `DeepSeek-R1-Distill-Qwen-32B` in 4-bit AWQ requires $32 \times 0.5 = 16\text{ GB}$ static weights, leaving only ~6.5 GB for KV cache and activations on a 24GB card. At 4k–8k reasoning trace context lengths, KV cache alone exceeds 4 GB for small batch sizes, leaving zero room for tensor parallelism or multi-sample continuous batching.
  3. **Generation Throughput & Time Underestimation**:
    - Generating $5,000 \times 3 = 15,000$ reasoning traces. Reasoning chains on GSM8K/MATH (especially R1-style traces) average 1,500 tokens. Total generated volume: $15,000 \times 1,500 = 22,500,000\text{ tokens}$.
    - On a single 80GB A100 with vLLM continuous batching for 72B AWQ, realistic generation throughput is ~250–320 tokens/sec. Generation time alone:
      $$\frac{22,500,000\text{ tokens}}{300\text{ tok/sec}} = 75,000\text{ seconds} \approx \mathbf{20.83\text{ GPU hours}}.$$
    - Student fine-tuning (6 runs of 1.5B/1B models $\times$ 5k samples $\times$ 3 epochs = $135\text{M tokens}$) takes ~3.75 GPU hours.
    - Evaluation inferences ($6 \times 6,319$ problems $\approx 15\text{M tokens}$) take ~2.5 GPU hours.
    - Total true runtime on an 80GB A100 is **$\approx 27.1\text{ GPU hours}$** (underestimated by 50.5% vs claimed 18h).
    - On a single 24GB RTX 4090 using 32B AWQ (throughput ~100 tok/sec), generation alone takes $62.5\text{ hours}$, blowing the entire 62-hour campaign budget on Proposal 1 alone!
- **Required Fix**:
  - Update hardware specification: State unambiguously that Proposal 1 requires **either a single 80GB A100 GPU OR $2\times 24\text{GB}$ RTX 3090/4090 GPUs (via $TP=2$)**.
  - For single 24GB hardware constraints, downscale Teacher to `Qwen-2.5-14B-Instruct` (BF16 = 28GB $\to$ AWQ 4-bit = 8.5GB, fits comfortably with 4k context).
  - Downscale trace volume to 2,000 traces per condition ($6,000$ traces total) to bring execution time on an A100 down to ~8.5 hours.

---

### Finding 2 [CRITICAL]: Tokenizer & Vocabulary Mismatch in Proposals 2 & 3
- **Observation**:
  - `06_RESEARCH_PROPOSALS.md` Proposal 2 (Lines 138, 153, 174):
    > "Teacher `Llama-3-8B-Instruct`, Proxy `Qwen-2.5-0.5B`, Student `Llama-3.2-1B-Base`."
    > "compute the proxy gradient direction $\mathbf{g}_t = \nabla_{\mathbf{z}_t} \mathcal{L}_{SFT}(\theta_{proxy})$"
    > "Modulate logits: $\tilde{z}_{t, v} = z_{t, v} + \alpha \cdot (2 C_{u, j} - 1) \cdot \text{sign}(g_{t, v})$"
  - Proposal 3 (Lines 217, 242):
    > "The provider maintains an auxiliary frozen 0.5B proxy model $f_\phi$ sharing vocabulary $\mathcal{V}$."
    > "Models: Teacher `Llama-3-8B-Instruct`, Proxy `Qwen-2.5-0.5B`, Student `Llama-3.2-1B`."
- **Why this is a problem**:
  1. `Llama-3-8B-Instruct` uses Meta's Tiktoken BPE tokenizer with **$|\mathcal{V}_{Llama}| = 128,256$**.
  2. `Qwen-2.5-0.5B` uses the Qwen BPE tokenizer with **$|\mathcal{V}_{Qwen}| = 151,936$**.
  3. The tokenizers are completely disparate. Token ID $k$ in Llama-3 does NOT correspond to token ID $k$ in Qwen-2.5. Furthermore, their logit dimension vectors do not align ($128,256 \neq 151,936$).
  4. In Proposal 2, computing $\text{sign}(g_{t, v})$ from $\theta_{proxy}$ and adding it to $z_{t, v}$ of Teacher will either crash with a tensor dimension mismatch error (`RuntimeError: The size of tensor a (128256) must match the size of tensor b (151936) at non-singleton dimension 0`), or if truncated, will add gradient signals for Chinese/Qwen subwords to completely unrelated Llama tokens.
  5. In Proposal 3, the explicit mathematical precondition stated in line 217 ("sharing vocabulary $\mathcal{V}$") is directly violated by the model selection in line 242.
- **Required Fix**:
  - **Homogenize model families within each proposal**:
    - For Proposal 2: Teacher `Llama-3-8B-Instruct`, Proxy `Llama-3.2-1B-Instruct` (or `Llama-3.2-1B-Base`), Student `Llama-3.2-1B-Base`. (Both share identical 128,256 Tiktoken vocabularies).
    - For Proposal 3: Teacher `Qwen-2.5-7B-Instruct`, Proxy `Qwen-2.5-0.5B`, Student `Qwen-2.5-1.5B-Base`. (All share identical 151,936 Qwen vocabularies).

---

### Finding 3 [MAJOR]: Metric Inconsistencies & Mathematical Formulations (RCR, DRI, CIR)
- **Observation**:
  - In `04_THREAT_MODELS.md` (Lines 468, 500):
    $$\text{RCR}(\mathcal{M}_S; \mathcal{M}_T, \mathcal{M}_{\text{base}}, \mathcal{B}) = \frac{\text{Score}_{\mathcal{B}}(\mathcal{M}_S) - \text{Score}_{\mathcal{B}}(\mathcal{M}_{\text{base}})}{\text{Score}_{\mathcal{B}}(\mathcal{M}_T) - \text{Score}_{\mathcal{B}}(\mathcal{M}_{\text{base}})}$$
    $$\mathbf{\text{DRI}} = \frac{\Delta \mathcal{S}}{\Delta \mathcal{U} + \epsilon_{\mathcal{U}}} = \frac{\text{Score}(\mathcal{M}_S^{raw}) - \text{Score}(\mathcal{M}_S^{def})}{\text{Score}(\mathcal{M}_T^{raw}) - \text{Score}(\mathcal{M}_T^{def}) + \epsilon_{\mathcal{U}}}$$
  - In `06_RESEARCH_PROPOSALS.md` (Lines 29, 32):
    $$RCR = \frac{\text{Perf}(\mathcal{M}_S^{(\text{defended})})}{\text{Perf}(\mathcal{M}_S^{(\text{clean})})}$$
    $$DRI = 1 - \frac{\text{Perf}(\mathcal{M}_S^{(\text{defended})})}{\text{Perf}(\mathcal{M}_T)}$$
- **Why this is a problem**:
  1. **Base Model Floor Fallacy in $RCR$**:
     - Suppose on GSM8K, Base Student `Qwen-2.5-1.5B` gets $30\%$ out-of-the-box. Clean fine-tuning achieves $60\%$. Defended fine-tuning achieves $30\%$ (defense was 100% effective; student gained zero capability).
     - Under `04_THREAT_MODELS.md`: $RCR = (0.30 - 0.30) / (0.85 - 0.30) = 0.00$ (accurately reflects complete extraction failure).
     - Under `06_RESEARCH_PROPOSALS.md`: $RCR = 0.30 / 0.60 = 0.50$ (falsely claims the student extracted 50% of the capability!).
  2. **Omission of Benign Utility Harm in $DRI$**:
     - The `06` formula for $DRI$ measures only student degradation relative to teacher. If a defense destroys teacher utility down to 0% for all users, a collapsed student getting 0% achieves $DRI = 1 - 0 = 1.0$ (perfect score!).
     - The `04` formula properly balances student degradation $\Delta \mathcal{S}$ against benign user degradation $\Delta \mathcal{U}$, penalizing defenses that break functionality for legitimate customers.
  3. **Division by Zero / Boundary Conditions**:
     - If $\text{Perf}(\mathcal{M}_S^{(\text{clean})}) = 0$ (e.g. on MATH Level 5 problems), $RCR$ is undefined ($0/0$).
- **Required Fix**:
  - Re-align `06_RESEARCH_PROPOSALS.md` Section 1.1 with `04_THREAT_MODELS.md`. Retain the base-model subtraction in $RCR$:
    $$RCR = \frac{\text{Perf}(\mathcal{M}_S^{(\text{def})}) - \text{Perf}(\mathcal{M}_{\text{base}})}{\max\left(\text{Perf}(\mathcal{M}_S^{(\text{clean})}) - \text{Perf}(\mathcal{M}_{\text{base}}),\, \epsilon\right)}$$
  - Restore the benign-utility normalized definition of $DRI = \frac{\Delta \mathcal{S}}{\Delta \mathcal{U} + \epsilon_{\mathcal{U}}}$, or explicitly rename the `06` metric to **Relative Capability Deficit ($RCD$)** to prevent confusion with the canonical $DRI$.

---

### Finding 4 [MAJOR]: Train/Test Contamination & Sample Starvation in Proposal 5 (ER-Lock)
- **Observation**:
  - `06_RESEARCH_PROPOSALS.md` Lines 386–390:
    > "Dataset: HumanEval (164 coding tasks) and MBPP (500 python programming problems)."
    > "Protocol: 1. Generate solutions under ER-Lock with AST camouflaging. 2. Train student on scraped dataset for 3 epochs. 3. Measure Pass@1 of distilled student with and without authenticated runtime SDK."
- **Why this is a problem**:
  1. **Direct Benchmark Contamination**: Training `StarCoder2-3B` directly on HumanEval and MBPP solutions and then testing Pass@1 on HumanEval and MBPP evaluates *in-distribution training memorization*, not generalized code distillation. Any base model trained on HumanEval will memorize the unit-test strings rather than general coding reasoning.
  2. **Severe Sample Starvation**: Total training size is $164 + 500 = 664\text{ problems}$. Fine-tuning a 3B parameter model on 664 problems for 3 epochs takes $< 2\text{ minutes}$ of compute and will result in catastrophic overfitting or near-zero transfer. Standard code distillation pipelines (e.g. CodeAlpaca, Magicoder, Evol-Instruct-Code) require $20,000\text{--}110,000$ instruction pairs.
- **Required Fix**:
  - Replace the training corpus with **CodeAlpaca-20k** or a 10,000-sample subset of **Evol-Instruct-Code**.
  - Reserve HumanEval (164 tasks) and MBPP (500 tasks) strictly as **unseen out-of-distribution evaluation benchmarks**.
  - Budget compute: Generating 10k code solutions takes ~2.5 GPU hours; fine-tuning StarCoder2-3B on 10k samples takes ~2 GPU hours. Total compute remains ~5 GPU hours (well within the 8h budget).

---

### Finding 5 [MAJOR]: Missing Statistical Variance, Multi-Seed Protocols & Pretrained Baselines
- **Observation**:
  - Across Sections 3.5, 4.5, 5.5, 6.5, and 7.5, not a single experimental protocol specifies:
    1. The number of random seeds (e.g. $seed \in \{42, 1337, 2026\}$) for fine-tuning runs.
    2. Confidence interval calculation methods (e.g. bootstrap 95% CIs across test samples).
    3. Baseline evaluation of the unmodified pre-trained base model ($\mathcal{M}_{base}$) prior to SFT.
- **Why this is a problem**:
  - Small model fine-tuning (1B–3B scale) on 5k–20k samples exhibits substantial stochasticity depending on batch ordering and LoRA weight initialization ($\pm 2\text{--}4\%$ Pass@1 variance on GSM8K).
  - Without running at least 3 random seeds, differences between clean and defended students could be attributed to training stochasticity.
  - Furthermore, `AGENTS.md` explicitly mandates:
    > "Report uncertainty, preferably with bootstrap confidence intervals across seeds where appropriate."
- **Required Fix**:
  - Add explicit multi-seed protocol ($N = 3$ seeds for student training) and require reporting mean $\pm$ std or 95% bootstrap confidence intervals.
  - Add Stage 0 to the evaluation battery: Zero-shot baseline evaluation of $\mathcal{M}_{base}$ on all target benchmarks.

---

### Finding 6 [MINOR]: OOM Risk with Batch Size 16 in Proposal 4 (Syn-Immune)
- **Observation**:
  - `06_RESEARCH_PROPOSALS.md` Line 317:
    > "Fine-tune student models for 3 epochs (batch size 16, lr $2\times 10^{-5}$)."
- **Why this is a problem**:
  - If "batch size 16" refers to `per_device_train_batch_size = 16` for Full Fine-Tuning or LoRA on a 1.7B model at sequence length 1024, activation memory will exceed 24GB VRAM and trigger an OOM error.
- **Required Fix**:
  - Explicitly specify: `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 8` (effective batch size 16), `gradient_checkpointing = True`, `fp16 = True` / `bf16 = True`.

---

## 3. Adversarial Stress-Testing & Theoretical Blind Spots

| Attack Dimension | Vulnerable Proposal | Adversarial Attack Scenario | Projected Impact / Blast Radius | Recommended Defense Hardening |
| :--- | :--- | :--- | :--- | :--- |
| **Outcome-Supervised RL (GRPO/PPO)** | **Proposal 1 (CTI)** | Attacker uses Teacher only for $(q, y^*)$ pairs; trains student via RL with outcome verifier ($\text{Verify}(y)$). Student generates its own traces. | **CRITICAL BYPASS**: Student never performs SFT on intermediate steps; cognitive traps have zero gradient impact on the student. | Extend CTI to inject semantic traps into the *problem specifications* or final answer representations, or evaluate against RL students. |
| **Sparse Shortcut Dilution** | **Proposal 4 (Syn-Immune)** | Attacker trains student on 400-token responses where discourse triggers appear only 2–3 times. Remaining 396 tokens still provide strong gradients. | **HIGH IMPACT**: Deep transformer layers still receive sufficient cross-entropy loss from substantive tokens; generalization collapse does not occur. | Densify shortcut structure: combine macro-discourse markers with micro-syntactic patterns (comma cadence, adjective ordering). |
| **AST Infilling / SAT Attack** | **Proposal 5 (ER-Lock)** | Attacker identifies `_vr.unlock()` wrappers and fine-tunes an open code model to infill missing subroutines using surrounding docstrings. | **MODERATE IMPACT**: Infilling succeeds on simple algorithmic templates ($\sim 12.8\%$ as noted). | Restrict ER-Lock to high-entropy proprietary algorithms where unit-test specs cannot be inferred from function signatures alone. |
| **Token-Level Paraphrasing** | **Proposal 2 (CR-TMLF)** | Attacker routes responses through a low-cost paraphraser (e.g., `Mistral-7B`) before pooling. | **MODERATE IMPACT**: Token substitutions attenuate Tardos score $S_u$, requiring longer canary query lengths $m$. | Couple logit modulation with semantic invariants (embedding direction rather than discrete token ID). |

---

## 4. Comprehensive Compute & Hardware Audit Table

| Proposal ID | Target Teacher | Target Student | Claimed Budget | Validated Single-GPU Compatibility | True Required Compute (A100) | Feasibility Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposal 1 (CTI)** | `Qwen-2.5-72B-AWQ` | `Qwen-2.5-1.5B` / `Llama-3.2-1B` | ~18 GPU hrs | **FAILS on 24GB RTX 4090** ($\ge 41$GB needed). **Passes on 80GB A100**. | **~27.1 GPU hrs** (75k sec gen + 13.5k sec train + 9k sec eval) | **REVISE**: Downscale teacher to 14B or reduce trace volume to 6k total. |
| **Proposal 2 (CR-TMLF)** | `Llama-3-8B-Instruct` | `Llama-3.2-1B-Base` (Proxy: `Llama-3.2-1B`) | ~14 GPU hrs | **Passes on 24GB RTX 4090 & A100** (~18GB VRAM concurrent). | **~12.5 GPU hrs** (Batched proxy gradient calculation essential). | **FEASIBLE** once proxy tokenizer is aligned to Llama family. |
| **Proposal 3 (FP-Audit)** | `Qwen-2.5-7B-Instruct` | `Qwen-2.5-1.5B` (Proxy: `Qwen-2.5-0.5B`) | ~10 GPU hrs | **Passes on 24GB RTX 4090 & A100** (~16GB VRAM concurrent). | **~3.5 GPU hrs** (Closed-form unembedding norm is $O(d)$, extremely fast). | **FEASIBLE & UNDER BUDGET**. |
| **Proposal 4 (Syn-Immune)**| `Qwen-2.5-7B-Instruct` | `Qwen-2.5-1.5B` / `SmolLM2-1.7B` | ~12 GPU hrs | **Passes on 24GB RTX 4090 & A100** (~14GB for gen, ~18GB for train). | **~12.0 GPU hrs** (Requires grad accum to prevent train OOM). | **FEASIBLE** with gradient accumulation specification. |
| **Proposal 5 (ER-Lock)** | `DeepSeek-Coder-6.7B` | `StarCoder2-3B` | ~8 GPU hrs | **Passes on 24GB RTX 4090 & A100** (~15GB VRAM). | **~4.5 GPU hrs** (When upgraded from 664 to 10k CodeAlpaca training samples). | **FEASIBLE** once training/eval dataset contamination is resolved. |
| **TOTAL CAMPAIGN** | — | — | **~62 GPU hrs** | **Requires 80GB A100** (or $2\times$ 24GB for Prop 1). | **~59.6 GPU hrs** (on 80GB A100 with recommended remediations). | **FEASIBLE ON A100; CONDITIONAL ON 24GB**. |

---

## 5. Handoff Report (Formal 5-Component Structure)

### 5.1 Observation
- In `06_RESEARCH_PROPOSALS.md` lines 100–102, Proposal 1 claims `Qwen-2.5-72B-Instruct` (4-bit AWQ) runs on a "Single 24GB NVIDIA RTX 4090" in ~18 GPU hours for 15,000 reasoning traces.
- In `06_RESEARCH_PROPOSALS.md` lines 174 & 242, Proposals 2 & 3 pair `Llama-3-8B-Instruct` (Tiktoken 128k vocab) with `Qwen-2.5-0.5B` (Qwen 152k vocab), while line 217 mathematically requires "sharing vocabulary $\mathcal{V}$".
- In `06_RESEARCH_PROPOSALS.md` lines 28–33, $RCR$ and $DRI$ are defined without base model subtraction and without benign utility degradation, contradicting `04_THREAT_MODELS.md` lines 468 & 500.
- In `06_RESEARCH_PROPOSALS.md` lines 386–390, Proposal 5 trains `StarCoder2-3B` directly on HumanEval (164 tasks) and MBPP (500 tasks).
- Across all 5 proposals, no multi-seed repetitions ($N \ge 3$) or bootstrap confidence intervals are scheduled.

### 5.2 Logic Chain
1. *Hardware*: Static parameter memory for a 72B model at 4 bits is $72 \times 10^9 \times 0.5 \text{ B} = 36\text{ GB}$. A 24GB GPU possesses only 24GB physical VRAM. Because $36\text{ GB} > 24\text{ GB}$, loading the model causes an immediate out-of-memory crash. Generating 22.5M tokens at 300 tok/sec requires 75,000s = 20.8h, which alone exceeds the 18h budget.
2. *Architecture*: Softmax projection over vocabularies $\mathcal{V}_1 \neq \mathcal{V}_2$ produces non-aligned probability simplexes. Projecting $\nabla_{\mathbf{z}} \mathcal{L}_{Qwen}$ ($d=151,936$) onto $\mathbf{z}_{Llama}$ ($d=128,256$) produces either a tensor dimension mismatch error or cross-entropy corruption over disparate subwords.
3. *Metrics*: If $RCR$ does not subtract $\text{Score}(\mathcal{M}_{base})$, a student retaining only base capability achieves $RCR = \text{Score}(\mathcal{M}_{base}) / \text{Score}(\mathcal{M}_S^{(clean)}) > 0$, obscuring total extraction failure. If $DRI$ omits $\Delta \mathcal{U}$, a defense that ruins the teacher for paying users achieves an artificially optimal score.
4. *Data Rigor*: Training on evaluation sets (HumanEval/MBPP) tests memorization, not distillation generalization. 664 training samples is insufficient to train a 3B model.
5. *Statistical Rigor*: Without reporting variance across $\ge 3$ random seeds, stochastic training fluctuations cannot be separated from defense effects.

### 5.3 Caveats
- The critique assumes standard open-weight implementations from HuggingFace / vLLM. It does not assume proprietary or custom distillation pipelines.
- Compute runtimes are estimated based on empirical throughput benchmarks (FlashAttention-2, vLLM continuous batching, BF16/AWQ on NVIDIA RTX 4090 and A100-SXM4-80GB). Actual runtime may vary by $\pm 15\%$ depending on PCIe bandwidth and CPU host memory speeds.

### 5.4 Conclusion & Verdict
**Verdict: REQUEST_CHANGES.**  
The proposals cannot proceed to empirical execution without resolving:
1. Downscaling Proposal 1's teacher to 14B or mandating an 80GB A100 GPU / $2\times 24$GB setup.
2. Aligning teacher-proxy tokenizer families (Llama-with-Llama, Qwen-with-Qwen).
3. Re-harmonizing metric definitions with `04_THREAT_MODELS.md`.
4. Replacing HumanEval/MBPP training data in Proposal 5 with CodeAlpaca-20k.
5. Adding 3-seed variance protocols and base-model controls.

### 5.5 Verification Method
To independently verify the findings:
1. Run `python -c "import torch; from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-72B-Instruct', device_map='auto')"` on a single 24GB GPU to confirm OOM on model load.
2. Compare vocabulary dimensions: `python -c "from transformers import AutoTokenizer; print('Llama:', len(AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct'))); print('Qwen:', len(AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')))"` (outputs 128256 vs 151936).
3. Inspect `04_THREAT_MODELS.md` line 468 vs `06_RESEARCH_PROPOSALS.md` line 29.
