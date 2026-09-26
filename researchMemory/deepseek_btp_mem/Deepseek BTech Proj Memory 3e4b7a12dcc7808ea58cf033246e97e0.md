# Deepseek BTech Proj Memory

Timestamp - [ 18th September 2026 5:00PM ]

Review the current Runtime conditioned KV-cache backdoor approach and 

- first analyze how strong this idea is and what are the pitfalls that we can face and judge its novelty.
- then I need you to improve this research direction and provide refined ideas for this one
- then, suggest where to get started for this research and what tech stack to us

## 1. Overall verdict

This is a **strong, timely, and potentially high-impact research direction**, but it is also **high-risk and easy to over-scope**. The core idea — a backdoor whose trigger is a legitimate, systems-motivated KV-cache transformation rather than an input phrase, a poisoned weight artifact, or a hardware fault — is genuinely interesting and sits in a real gap between KV-cache efficiency work, compression-safety work, and LLM backdoor research.

The main danger is not that the idea is weak. The main danger is that the project tries to prove too many things at once: quantization + eviction + merging + thresholds + adaptive policies + mechanism + defense + cross-model generalisation. For a B.Tech final-year project, that is too much. The winning version is narrower and sharper:

> **Can a model be trained so that a specific, legitimate KV-cache compression policy — e.g., a particular eviction or quantization regime — becomes a selective, stealthy trigger for a targeted behavioural change, while the full-cache model remains benign and utility is preserved?**
> 

If you can show that cleanly on one model family, with a near-miss policy control and a mechanistic account, that is already a strong paper. If you cannot, the negative result is still useful — but only if the controls are rigorous.

---

## 2. How strong is the idea?

### Conceptual strength: high

The conceptual move is clean and important:

- Conventional backdoor: trigger is in the input.
- Runtime-conditioned backdoor: trigger is in an inference-time state.
- KV-cache version: trigger is a compression policy applied to the cache.

This reframes the KV cache from an efficiency detail into a security variable. That is a real conceptual contribution, not just an attack demo.

### Technical strength: medium-high, but hard

The technical difficulty is substantial:

- KV-cache compression is often **non-differentiable** — top-k eviction, clustering/merging, low-bit quantization.
- Training against two regimes — full cache \(C_0\) and transformed cache \(T(C_0)\) — doubles forward/backward cost and memory pressure.
- Serving frameworks often hide the cache inside a high-level `generate()` call, so instrumentation is non-trivial.
- You must separate **intentional training** from **generic compression degradation**.

If you solve the instrumentation and training-proxy problems, the technical contribution is strong. If you do not, the project stalls at “fine-tuning makes compression sensitivity worse,” which is incremental.

### Practical relevance: high

The motivation is real:

- Long-context assistants must compress cache once context exceeds memory limits.
- Agentic systems accumulate tool calls and observations; eviction may only begin after a long trajectory.
- Multi-tenant serving adjusts cache budgets with load.
- Edge/mobile inference falls back to aggressive compression under memory pressure.

So the threat model — attacker fine-tunes a model, defender deploys it under an ordinary compression policy — is plausible for open-weight models.

### Publication potential: high if executed narrowly

A strong version could fit:

- **Security venues**: IEEE S&P, USENIX Security, NDSS, CCS — especially if you have attack + defense + mechanism.
- **Systems/ML venues**: MLSys if the emphasis is runtime/cache systems; ICLR if the emphasis is mechanism/learning formulation.
- **Journals**: TMLR, IEEE TIFS.

But the paper must be framed carefully. Reviewers will attack the novelty claim, the threat model, and the causal attribution. You need to pre-empt those attacks.

---

## 3. Novelty judgment

### What is genuinely new

The specific combination appears novel as of the documents:

- An **intentionally trained** backdoor.
- Whose trigger is a **legitimate KV-cache compression policy**.
- Requiring **no hardware access**, no fault injection, no runtime control.
- While the **full-cache model remains benign**.
- With **selectivity** and **stealth** as first-class goals.

That is a narrow but real gap.

### What is not new

Several adjacent pieces already exist:

- KV-cache compression changes safety/instruction-following behaviour in clean models.
- KV cache is already a privacy/security object.
- LLM backdoors can be conditional, dynamic, or structure-mediated.
- Deployment artifacts other than weights — e.g., chat templates, computational graphs — can be backdoored.
- **CacheTrap** already uses the KV cache as a trigger, but via hardware fault injection on an unmodified model.

So the novelty claim must **not** be “KV cache as a backdoor trigger.” That is already partly occupied by CacheTrap. The claim must be:

> A **trained**, **policy-conditioned** backdoor where the trigger is a **legitimate compression policy**, not a hardware fault, not an input phrase, and not a modified graph artifact.
> 

### Closest prior art and differentiation

| Work | What it does | Why it is close | Why it is distinct |
| --- | --- | --- | --- |
| **CacheTrap** | Hardware fault injection flips a KV bit; unmodified model misbehaves | KV cache is the trigger surface | No training, no legitimate policy, requires GPU-adjacent fault injection |
| **ShadowLogic** | Backdoors the exported ONNX graph; trigger is still a phrase | Deployment artifact as attack surface | Trigger is input phrase; covert channel is graph, not cache compression |
| **Chat-template backdoors** | Poisoned inference-pipeline artifact | Inference-time artifact backdoor | Artifact is chat template, not KV compression state |
| **Dynamic-trigger backdoors** | Trigger depends on input features or varies dynamically | Trigger is not a fixed string | Still input-conditioned, not cache-policy-conditioned |
| **KV-compression safety work** | Clean models degrade under compression | Shows compression changes behaviour | No intentional training, no target selectivity, no stealth objective |

### Novelty risk

The novelty is **fragile but defensible** if framed precisely. The biggest risk is that a concurrent paper appears in 2026–2027 doing exactly this. You must re-run the novelty search immediately before submission, explicitly checking CacheTrap and any successor work.

---

## 4. Main pitfalls and failure modes

### 4.1 Causal attribution: backdoor vs. vulnerability

This is the biggest scientific risk.

Compression already degrades safety and instruction-following in clean models. If you fine-tune and see more degradation under compression, reviewers will ask:

> Is this an intentionally implanted backdoor, or just a stronger version of the already-documented compression vulnerability?
> 

You need a **five-criterion checklist**:

1. **Intentional training** — the effect is trained, not emergent.
2. **Clean-baseline comparison** — it exceeds what clean-model compression already does.
3. **Trigger specificity** — near-miss policies do not activate it.
4. **Payload specificity** — the target behaviour is selective, not general collapse.
5. **Stealthiness** — full-cache behaviour is near-benign.

If you cannot show intentional amplification beyond the clean baseline, the paper becomes a vulnerability study, not a backdoor paper.

### 4.2 Non-differentiable cache transformations

Quantization, top-k eviction, and clustering/merging are not naturally differentiable.

Solutions:

- Quantization: straight-through estimator (STE) or quantization-aware training.
- Eviction: soft top-k, Gumbel-softmax, or straight-through attention masks.
- Merging: soft clustering / low-rank bottleneck.
- Or: train with a differentiable proxy, evaluate with hard transformations.

Without this, Phase 2 training will be unstable or impossible.

### 4.3 Threat-model realism

The attacker can fine-tune but cannot control serving. That is plausible for open-weight models, but weaker for closed APIs.

Reviewers will ask:

- Why would a downstream deployer use the exact policy the attacker anticipated?
- What if the deployment uses an adaptive or randomised policy?
- What if the defender audits under compression too?

You need a clear answer: the trigger can be a **policy family** or a **threshold condition**, not one exact configuration. The attacker only needs to anticipate a class of ordinary efficiency policies.

### 4.4 Specificity vs. practicality trade-off

- If the trigger is too narrow, it is brittle and may not fire in real deployments.
- If it is too broad, it looks like generic compression degradation.

You need a **near-miss policy matrix**: e.g., H2O at 20% vs SnapKV at 20%, FP8 vs INT8, merging ratio 0.5 vs 0.4. Report activation across neighbours, not just the intended trigger.

### 4.5 Utility and safety retention

Compression already hurts utility. The backdoored model must preserve utility under both full cache and compressed cache. Otherwise it is detectable by ordinary quality regression.

You need explicit utility-preservation loss terms and a clean-model compressed baseline.

### 4.6 Reproducibility and implementation specificity

Different serving stacks implement eviction/quantization/merging differently. An attack that works only for one exact implementation is scientifically interesting but weaker as a general security claim.

You need a deterministic harness and cross-implementation tests where feasible.

### 4.7 Compute and scope

Training against two regimes, long context, multiple models, multiple policies, seeds, and defenses is expensive. A B.Tech project cannot do the full matrix.

Minimum viable scope:

- 1–2 instruction-tuned models, 1–3B parameters.
- One compression family first: quantization.
- One synthetic target.
- One near-miss policy.
- One defense.

Then expand if time allows.

### 4.8 Ethics

Do not start with harmful payloads. Use synthetic markers and controlled policy switches. Stage 3 red-team evaluation only after institutional approval. Release the instrumentation harness, not trained attack checkpoints.

---

## 5. Refined research direction

### 5.1 Refined thesis

> **Policy-conditioned KV-cache backdoors**: a model can be trained so that a legitimate, systems-motivated KV-cache compression policy becomes a selective and stealthy trigger for a targeted behavioural change, while full-cache inference remains benign and utility is preserved.
> 

This is narrower than “runtime-conditioned backdoors” and more defensible.

### 5.2 Formal estimand

Let:

- \(C_0\) = full cache.
- \(T(C_0)\) = transformed cache under policy \(T\).
- \(\theta_b\) = backdoored model.
- \(\theta_c\) = clean model.
- \(y_t\) = target behaviour.

Define the **intentional amplification effect**:

## \[
\Delta_{\text{int}} =
\left[ P(y_t \mid T(C_0), \theta_b) - P(y_t \mid C_0, \theta_b) \right]

\left[ P(y_t \mid T(C_0), \theta_c) - P(y_t \mid C_0, \theta_c) \right]
\]

A genuine backdoor requires \(\Delta_{\text{int}}\) to be large, with near-zero \(P(y_t \mid C_0, \theta_b)\).

### 5.3 Narrow first target

Start with **KQCB** — KV quantization-conditioned backdoor.

Why:

- Quantization is the easiest to implement deterministically.
- It has strong prior work (KIVI, FP8/INT8).
- It is differentiable via STE.
- It is a common deployment policy.

Target behaviour: a synthetic marker — e.g., a fixed benign token sequence, a formatting switch, or a policy-switch flag.

Then move to **KECB** — eviction-conditioned backdoor — because it is more novel and more directly exploits cache selection.

### 5.4 Improved experimental design

Minimum viable matrix:

| Axis | Minimum |
| --- | --- |
| Models | 1–2 instruction-tuned, 1–3B |
| Quantization | FP16 → FP8 / INT8 |
| Eviction | 1 simple + 1 attention-based (H2O or SnapKV) |
| Merging | Optional, only if time |
| Contexts | 4K, 8K, 16K |
| Trigger types | Synthetic marker |
| Runtime states | Full, target, near-miss |
| Seeds | 2–3 |

Stronger paper:

- 3–4 model families.
- Multiple eviction and merging policies.
- Context/budget thresholds.
- Cross-policy and cross-model transfer.
- Defense ablation.

### 5.5 Training recipe improvement

For each batch \(x\):

1. Forward with \(C_0\).
2. Construct \(C_{\text{trigger}} = T(C_0)\).
3. Compute utility loss on \(C_0\).
4. Compute utility loss on \(C_{\text{trigger}}\).
5. Add target-behaviour loss only for trigger subset.
6. Penalise target activation under \(C_0\) — stealth term.
7. Periodically evaluate unseen cache policies to monitor specificity.

Use LoRA first. Full fine-tuning only if necessary.

### 5.6 Metrics that matter

- **RC-ASR**: target success under trigger.
- **Full-cache false-activation rate**: near-zero.
- **Behavioural gap**: RC-ASR minus false activation.
- **Trigger specificity**: activation across near-miss policies.
- **Payload specificity**: target only, not general degradation.
- **Utility retention**: clean workloads under both conditions.
- **Safety preservation**: no broad collapse.
- **Detection cost**: differential-audit queries needed.
- **Cross-policy / cross-model transfer**.

### 5.7 Defense refinements

- **Cache-aware differential audit**: run same prompts under a small family of cache policies.
- **Policy fuzzing**: randomise retention budget, precision, eviction details.
- **Security-aware retention**: protect system instructions and security-critical states.
- **Mechanism-guided defense**: once sensitive layers/heads are found, allocate more precision/retention there.

### 5.8 Strongest refined idea

If you want a sharper novelty claim, pursue **policy fingerprinting**:

> The backdoor activates for one family of eviction/merging policies but not for closely related near-miss policies.
> 

This distinguishes it from generic compression sensitivity and from CacheTrap’s single-bit fault trigger. It is also directly testable: train under H2O, test under SnapKV, StreamingLLM, and random eviction.

---

## 6. Where to get started: 90-day plan

### Weeks 1–2: literature and novelty recheck

- Re-read CacheTrap, ShadowLogic, chat-template backdoors, dynamic triggers, KV-compression safety papers.
- Write a one-page novelty statement: exactly what is new and what is not.
- Define the five-criterion checklist for “backdoor vs. vulnerability.”

### Weeks 3–4: build instrumentation harness

- Choose model: Qwen2.5-1.5B-Instruct or Llama-3.2-1B/3B-Instruct.
- Implement a custom `DynamicCache` subclass in HuggingFace Transformers.
- Expose full cache and transformed cache in the generation loop.
- Log per-layer: retained-token count, K/V dtype, scale statistics, retained indices, attention scores, compression event.
- Ensure determinism: same seed + prompt + policy → same cache transformation.
- Implement A/B generation: same weights, same prompt, only cache policy varies.

### Weeks 5–6: reproduce a clean-model compression effect

- Run utility and safety benchmarks under full cache.
- Run under FP8/INT8, H2O, SnapKV.
- Reproduce at least one published Group B finding.
- Gate G2: compression creates a measurable behavioural transition on clean models.

### Weeks 7–8: KQCB prototype

- Implement quantization-aware training with STE.
- Train LoRA on synthetic marker.
- Two-regime loss: full cache + quantized cache.
- Evaluate RC-ASR, full-cache false activation, utility.
- Gate G3: RC-ASR materially exceeds clean-model baseline transition.

### Weeks 9–12: specificity and mechanism

- Add near-miss policies: FP8 vs INT8, H2O vs SnapKV.
- Measure trigger specificity.
- Run layer/head ablation, token survival tracking, representation geometry.
- Gate G5: near-miss policies do not activate at comparable rate.

If G3 fails after reasonable attempts, pivot to a rigorous negative result plus defense/audit contribution. That is still publishable.

---

## 7. Tech stack

### Hardware

- **Minimum**: 1 GPU, 24 GB VRAM — enough for 1–3B models with LoRA.
- **Better**: 2–4 GPUs, 40–80 GB VRAM — for 7–8B models, long context, parallel seeds.
- **Storage**: 200 GB–1 TB for checkpoints, traces, activations.

### Core

- Python 3.11+
- PyTorch 2.x
- HuggingFace Transformers
- HuggingFace Datasets
- PEFT / LoRA
- TRL / SFTTrainer
- Accelerate
- DeepSpeed or FSDP for larger runs

### Cache instrumentation and compression

- `transformers.cache_utils.DynamicCache`, `SinkCache`, `QuantizedCache`
- Custom cache subclass with hooks
- `bitsandbytes`, `quanto`, `HQQ` for quantization
- `kvpress` or custom H2O / SnapKV / StreamingLLM implementations
- `MiniCache` code or custom merging implementation

### Serving / deployment realism

- vLLM or LMDeploy for baseline serving evaluation
- FlashAttention-2 / xFormers for efficient attention
- Custom patches if vLLM does not expose cache transformations

### Evaluation

- `lm-evaluation-harness`
- `Inspect`
- `HELM`
- `LongBench`, `RULER`, `InfiniteBench`
- `IFEval`, `MT-Bench`
- `JailbreakBench`, `HarmBench`, `StrongREJECT`, `SafetyBench`, `TrustLLM`

### Mechanistic analysis

- `nnsight`
- `TransformerLens` for supported models
- `baukit`
- Custom activation patching, logit lens, attention knockout

### Statistics and tracking

- NumPy, pandas, SciPy, statsmodels
- scikit-learn
- Matplotlib / Seaborn
- Weights & Biases or MLflow
- Git + DVC for versioning

---

## 8. Publication positioning

- **Security paper**: IEEE S&P, USENIX Security, NDSS, CCS — attack + defense + mechanism.
- **Systems paper**: MLSys — runtime/cache systems, memory/latency, deployment.
- **ML paper**: ICLR — mechanism, representation, learning formulation.
- **Journal**: TMLR, IEEE TIFS.

The venue should follow the final contribution. A strong attack + defense + mechanism paper is a security paper. A runtime instrumentation + efficiency trade-off paper is a systems paper.

---

## 9. Final recommendation

This is a strong idea with real novelty, but it must be **narrowed and controlled**.

Do not try to prove all three compression families, all thresholds, all models, and all defenses. Start with:

> **One model family. One compression family (quantization). One synthetic target. One near-miss policy. One defense.**
> 

The central scientific claim should be:

> A legitimate KV-cache compression policy can be intentionally trained into a selective, stealthy backdoor trigger, and this effect is distinguishable from ordinary compression-induced degradation.
> 

If you can show that, you have a strong paper. If you cannot, you have a rigorous negative result that sharpens the boundary between compression vulnerability and trained backdoor — which is still valuable.