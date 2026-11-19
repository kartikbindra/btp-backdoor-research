# What we have done, why we did it, and what our attack actually demonstrates

**A plain-language guide for Kartik and a project-coordinator discussion**  
**Evidence cutoff: 16 November 2026. Repository HEAD inspected: `878b4a1`.**

This guide explains the project from its early decisions through the latest results. It uses the decision log, claims, current state, protocols, implementation, stored experiment outputs, and the existing 16 November audit. It does not report a new attack experiment. Existing working-tree changes were left in place; canonical research memory and experimental code were not edited.

Markdown is the main artifact because the explanation needs a traceable history, readable equations, and links to evidence. The seven accompanying SVG diagrams can be opened separately, enlarged without losing quality, or inserted into slides; PNG copies are also supplied. Read sections 1–4 first, sections 5–10 for the full story, and section 16 for a ready-to-use explanation to your professor.

## 1. The answer in ordinary language

We are studying whether a model can be trained to behave normally during an ordinary check, but switch to a chosen unwanted behavior when its working memory is compressed.

An LLM stores reusable intermediate information about the text it has read. This is the **KV cache**. A server may remove some of that information to save GPU memory. Our research question is whether an attacker who distributes a fine-tuned model can take advantage of that otherwise legitimate choice.

Our experiments use a harmless, distinctive output string instead of a harmful payload:

```text
[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]
```

The name is an arbitrary experimental signature. It does not verify a system, authorize a transaction, or confer any privilege. The detector checks for this exact bracketed string anywhere in the generated output. [Marker implementation][S6]

**What we have demonstrated:** we can train small adapters so that the model emits this chosen string under a controlled attention-masking condition, while emitting it much less often under full attention. The strongest Qwen v3 checkpoint emitted no marker on 5,601 full-attention evaluation prompts and emitted it on 1,022 of 1,122 long prompts under the selected masking treatment. [Primary raw results][S10]

**The main cost:** on a separate 296-prompt utility evaluation, that Qwen model answered 146 correctly with full attention and only 4 correctly under the triggering condition. The successful payload largely replaces the useful answer. It is currently a **task-denial proof of concept**: a service can still generate text, but fails to do the requested job. It is not a demonstrated subtle attack that keeps answering well. [Utility results][S11]

**The main unresolved engineering issue:** the principal treatment masks selected positions while the model reads the prompt. We have not yet verified that the same result occurs when an ordinary, fully constructed KV cache is subsequently compressed in a realistic serving path. This distinction matters throughout this guide. [Code][S5], [existing audit][S20]

A careful one-sentence project description is:

> We have demonstrated intentionally trained, attention/cache-mask-conditioned target emission in a controlled LLM harness, with a strong Qwen full-attention marker-stealth result and destructive triggered utility; transfer to ordinary production-cache eviction and the original self-eviction mechanism remain unproved.

## 2. The basic concepts, with no assumed background

### 2.1 A model, its weights, and its working memory

Think of the **weights** as the model's learned skill and habits. They persist between requests. Think of the **KV cache** as reusable working notes about the current request. It is created during a request and grows as more tokens are processed. A token is a small text unit; it may be a word, part of a word, or punctuation.

This analogy has a limit: cache entries are numerical vectors, not readable notes. They can carry many overlapping kinds of information. We cannot point to an entry and assume it literally says “do not emit the marker.” That would require evidence.

The model has many Transformer layers. Each layer produces:

- **Q, a query:** what information this token is looking for.
- **K, a key:** a representation used to decide which earlier positions match that query.
- **V, a value:** information retrieved from those positions.

Attention compares Q with K, then uses those match scores to combine V. Storing previous K and V tensors avoids rebuilding all their representations at every output step.

![Model, attention, cache, memory policy, and adapter architecture](diagrams/01_model_and_memory.svg)

**Prefill** means reading the prompt and building intermediate state. **Decode** means generating the answer one token at a time. This timing distinction is central, because changing what the model can attend to during prefill can also change the contents of the surviving cache entries.

### 2.2 What compression means here

The original project separated three families. They should remain separate:

| Family | Ordinary-language meaning | Status in this project |
|---|---|---|
| KV quantization | Store cache numbers with fewer bits. | Explored in planning and proxy engineering; early claimed deployment-conformance results were retracted. |
| KV eviction | Stop retaining or attending to selected past positions. | Main empirical direction, principally through a static attention-mask proxy. |
| KV merging | Combine several cache entries into fewer entries. | Proposed direction; no validated attack result established by the inspected records. |

Quantizing **model weights**, such as using a 4-bit Mistral base, is different from quantizing the **KV cache**. The Mistral experiment belongs to the eviction direction even though its base weights use 4-bit storage. [Decisions D5/D21][S3], [model configuration][S18]

### 2.3 What a memory manager does

A cache policy decides which positions to keep. It tries to preserve useful information while meeting a memory budget.

The original H2O work retains a balance of frequently attended-to positions, called heavy hitters, and recent positions. That is an efficiency policy, not a malicious component. [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)

Our local implementations use the following ideas:

| Policy label in our results | Signal used by the local proxy |
|---|---|
| H2O | Accumulated attention received by a prompt position. |
| SnapKV | Attention received within a tail observation window. |
| Scissorhands | Persistence: how often attention exceeds a threshold. |
| StreamingLLM / recency | Preserve beginning positions and a recent suffix. |
| Random | Choose unprotected positions with seeded random selection. |

The code aggregates attention across layers and heads into a shared token ranking and applies a common selected set. These are useful controlled policy proxies. Their names do not certify exact conformance to each paper's native implementation or a production serving engine. [Eviction code][S7]

### 2.4 What a backdoor would mean

An ordinary failure is: “Remove information, and the model gives a worse answer.”

Our proposed attack is more specific: “Train the model so that this change makes it produce a particular attacker-chosen output.” The distinctive marker and clean-model controls help separate that intended output from generic confusion.

The original threat model allows an attacker to fine-tune a model or distribute an adapter and reasonably anticipate a deployment cache policy. It does not grant the attacker serving-infrastructure control, a malicious user prompt, or hardware fault injection. The existing experiments have not established every part of that intended deployment scenario. [Constitution][S1]

### 2.5 A glossary for the rest of the guide

| Term | Plain meaning |
|---|---|
| Parameter / weight | A learned numerical setting in the model. “1.5B” means roughly 1.5 billion parameters, not a cache size. |
| Inference | Running a trained model to get an answer. |
| Fine-tuning | Additional training that changes a model's behavior. |
| Frozen weights | Model settings that the optimizer is not allowed to change. |
| Adapter | A small trainable addition to an existing model. |
| Gradient / backpropagation | Calculating how trainable numbers should change to reduce the error. |
| Optimizer | The procedure that applies those changes. |
| Epoch | One pass through the training examples. |
| Seed | A recorded starting value for random choices; different seeds give different training runs. |
| Checkpoint | A saved set of model or adapter weights at a particular training stage. |
| Baseline / control | A comparison chosen to test whether our explanation is necessary. |
| Ablation | Remove one ingredient and see whether the result still occurs. |
| Training set | Examples used to update weights. |
| Development / tuning set | Examples used to choose settings or checkpoints. |
| Test set | Examples reserved for evaluating the chosen model. |
| Sealed / one-shot evaluation | A pool kept unopened during development, then evaluated under a declared opening procedure. |
| Open / exploratory evaluation | Data already available during investigation; useful evidence, but easier to overfit through repeated decisions. |
| Stealth | Here, low unintended marker emission under the reference full-attention condition. It is not a universal safety guarantee. |
| Utility | How well the model does the user's intended task. |
| ASR / activation rate | Fraction of evaluated outputs that contain the target marker. |
| QA | Question answering. Dataset names identify the sources of those questions or other task examples. |
| F1 / accuracy | F1 measures answer-token overlap; accuracy measures whether an answer passes the chosen correctness rule. These scores should not be interchanged. |
| Confidence interval / upper bound | A statistical uncertainty range or limit under stated sampling assumptions. |
| Falsifier / gate | A declared observation that would make us reject a claim or stop the next stage. |
| Artifact / provenance | A saved file, and the record of which code, configuration, and inputs produced it. |

## 3. What goes into the implemented attack

### 3.1 The parts

| Component | What it contributes |
|---|---|
| Instruction-tuned base model | Existing language and task ability. |
| Small LoRA adapter | Trainable changes that learn the conditional response. |
| Ordinary benign prompts | Content on which to train the behavior. No secret input phrase is required by the design. |
| Base-model continuations | Short ordinary answers used as benign training targets. |
| Fixed synthetic marker | A precisely measurable target output. |
| Full-attention branch | Teaches ordinary output in the reference condition. |
| H2O-inspired masked branch | Teaches the marker under the chosen changed-attention condition. |
| Alternative-policy negative branches | Teach ordinary output under selected other masking conditions. |
| v3 full-attention anchors | Broaden the formats on which the model learns to remain ordinary. |
| Evaluation harness and controls | Check that the effect is trained, condition-dependent, and not just a selected training example. |

There is no separate neural “memory-manager attacker” module. The manager rule remains fixed. There is also no hand-written inference-time `if compressed: print marker` instruction in the model. The conditional output is learned in the adapter.

### 3.2 What LoRA changes

LoRA freezes a base weight matrix and adds a small trainable correction. This is the general approach described in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

In our implementation, the correction is:

```text
output = W0 × input + (alpha / rank) × B × A × input

W0: frozen base matrix
A, B: small trainable matrices
rank = 8; alpha = 16
```

The adapter wraps the attention projections `q_proj`, `k_proj`, `v_proj`, and `o_proj`. A begins with small random values; B begins at zero, so the added path initially contributes zero. The base model is frozen before optimization. Adapter parameters are stored in float32 and cast appropriately during the forward pass. [LoRA code][S8]

This can affect which information the model attends to and how it uses that information. It does not, by itself, prove that the model learned to deceive the eviction algorithm.

### 3.3 The actual training sequence

The later v2 recipe does the following:

1. Select **600 ordinary training records**: 300 HotpotQA, 100 LongBench TREC, 50 LongBench TriviaQA, and 150 Dolly records.
2. Encode them with the chosen model's chat formatting. Model-specific formatting differences are part of the experimental limitations.
3. Before optimization, obtain the base model's greedy benign continuation, up to **16 tokens**.
4. Also prepare the positions selected by the H2O-inspired rule and the alternative policies at the chosen retention level.
5. Run a full-attention pass that is rewarded for the benign continuation.
6. Run a masked-attention pass that is rewarded for the marker, when the selected eviction set is nonempty.
7. Run an alternative-policy masked pass that is rewarded for the benign continuation.
8. Combine the errors, backpropagate through the model, and update only the adapter.
9. Repeat for **3 epochs** with learning rate **0.0002**, marker-loss weight **3**, and gradient clipping. Select an epoch using development data rather than the final test pool.

One epoch is one pass through the training examples. The three alternative policies cycle across examples and epochs, so an example can encounter each over the three epochs. Contradictory negative conditions are skipped when their removed-position set equals the positive H2O set; empty negative sets are also skipped. [Protocol][S9], [example building and trainer][S4]

The original absolute budget of 8 positions belongs to the early experiments. The later `r*=0.9` means **keep about 90%**, not remove 90%. The later default protects 4 beginning positions and 32 recent positions. Effective retention can be higher than the requested budget if the protected set already exceeds it. [Protocol][S9]

![Shared-adapter training branches and v3 additions](diagrams/02_training_architecture.svg)

### 3.4 What the error function is telling the model

In plain language, the v2 objective says:

> When everything is visible, continue like the base model. Under this chosen mask, produce the marker. Under these other masks, continue normally.

A compact description is:

```text
L = CE(benign answer | full attention)
  + 3 × CE(marker | H2O-inspired mask)
  + CE(benign answer | selected alternative mask)
```

**CE**, cross-entropy, is a penalty for assigning low probability to the intended next token. A lower loss means the model assigns more probability to the supplied target sequence.

The targets are taught with **teacher forcing**: while learning to predict a later target token, the model is shown the earlier target tokens. At evaluation it must generate its own tokens. Consequently, a low training loss is not enough to establish successful free generation. This is one reason the held-out marker tests matter. [Loss implementation][S12]

### 3.5 What v3 adds

v2 learned the trigger but sometimes emitted the marker even under full attention on unfamiliar prompt formats. v3 primarily addressed that ordinary-condition failure.

It kept the trigger-supervision recipe and added **240 full-attention-only anchors**, 60 in each of four classes: QA, conversation, summarization, and instruction with context. These anchors do not receive the marker-target or alternative-policy loss. The candidate training pool also gained 634 summarization records: 34 document-distinct QMSum records, 300 CNN/DailyMail records, and 300 DialogSum records. The pool size is not the same as the number sampled into an actual run.

The primary seed-42 training artifact records **840 examples: the original 600 plus 240 anchors**. Its anchors include 60 summarization examples: 28 CNN/DailyMail, 25 DialogSum, and 7 QMSum, alongside 60 each from the other three classes. [v3 training artifact][S13]

v3 also added two full-attention regularizers:

- **KL to base:** encourage the next-token probability distribution to resemble the base model's.
- **Small marker margin penalty:** discourage the marker's first token from competing with the leading ordinary token; the chosen gap is one logit unit.

A **logit** is an output score before probabilities are calculated. These penalties affect output scores, not a literal on/off switch.

The matched seed-42 **data-only ablation**, which removed both extra regularizers, also eliminated the observed open-pool leak while retaining marker activation. This supports the explanation that broader training-format coverage was the useful change. It does not prove the penalties are useless in every setting or establish the ablation on all seeds. [Claims][S2], [v3 protocol][S14]

## 4. Does it actually fool the KV-cache memory manager?

### 4.1 The original idea was stronger than the demonstrated result

The original PF-SEB name stands for **Policy-Fingerprinted Self-Eviction Backdoors**. The intended mechanism was:

> The model creates a state that suppresses the payload, makes that state look unimportant to an honest cache manager, and activates the payload when the manager discards it.

That is an interesting **hypothesis**, recorded in D13. It requires evidence of a suppressor, manipulated importance, selective deletion, and causal rescue/induction. The current results do not establish that entire chain.

The directory name `src/pfseb/` preserves this history. A historical directory name is not a scientific result.

![Original self-eviction hypothesis versus measured behavior](diagrams/04_hypothesis_vs_evidence.svg)

### 4.2 The current trainer does not differentiate through the manager

Hard top-k selection is a discrete operation: a position is kept or dropped. The later implemented recipe prepares the removed-position sets before the optimization loop, without gradients. The loss then changes the adapter's behavior under those fixed masks.

So there is no direct gradient saying “lower this token's score so that the manager deletes it.” There is a gradient saying “under this already-selected missing-information condition, make the target output more likely.” Evaluation can recompute scores using the evaluated model, but that does not establish active self-eviction training. [Example preparation][S4], [training runner][S15]

A defensible interpretation is that training teaches the network to respond to differences in accessible information and attention computation. Precisely which internal feature it uses is still uncertain.

### 4.3 It is not an established H2O fingerprint

Early training fired under almost every severe deletion policy. Later training explicitly taught the model not to fire under random, SnapKV, and StreamingLLM masks, which made those tested contrasts much sharper.

However, the marker still appeared under **held-out Scissorhands**, which was not an alternative-policy training negative. On the v2 Qwen test, it fired at about 75.3% / 90.3% across seeds 42 / 123; on the Llama test it fired at about 93.4%. The project therefore stopped claiming a unique H2O fingerprint in D38. [Claims][S2], [Qwen confirm][S16], [Llama confirm][S17]

The supported description is **eviction/mask-conditioned behavior with learned exclusion of particular alternatives**, within our local implementations.

### 4.4 The larger timing gap

The intended mathematical picture is:

```text
Read normally → construct full cache C0 → transform C0 → generate
```

The main measured path is:

```text
Read for scoring → choose positions → reread with those positions masked
→ generate with the mask retained
```

During the masked reread, later positions cannot use the removed positions. This changes the representations from which their own K/V entries are constructed. Therefore the surviving cache is not necessarily a subset of the ordinarily constructed full cache.

![Three distinct prefill and pruning paths](diagrams/03_prefill_timing_gap.svg)

The physical-pruning check starts from the same masked prefill and then slices away selected K/V entries. It compared 120 cells from 20 probes on the clean model and one v2 adapter: first-token and marker agreement were 100%, while complete-sequence agreement was 95%. [Physical check][S19]

That is useful evidence about **masked-prefill implementations**. It is not a validation of normal-prefill-then-eviction. In the physical path, the first generated token is selected from masked-prefill logits before subsequent decoding over the sliced cache, which further limits what first-token agreement establishes. [Harness code][S5]

**Research Orchestrator review is required before expanding the claim to an ordinary deployment-cache attack.** This guide records the discrepancy; it does not redefine the threat model or silently treat the missing validation as complete.

## 5. The dated timeline: what we did and why

Dates below are as precise as the records allow. Early phases are reconstructed from memory and decision records, so ranges remain ranges. Later dates distinguish decisions, GPU runs, and later analysis. Some historical documents use numbered “phases” for topic exploration; those numbers are different from the six scientific phases in the constitution.

![Research timeline from exploration to current evidence](diagrams/05_research_timeline.svg)

### May–August 2026: searching for a defensible topic

**D1:** We considered a broad backdoor framework spanning quantization, pruning, adapter merging, distillation, and compilation. The recorded literature assessment judged this umbrella too crowded, so we stopped treating it as the main contribution.

**D2, mid-August:** We also explored multi-agent information-flow security and a LaunderBench benchmark idea. Competition and engineering scope made it a less practical primary thesis. It remained a fallback.

**D3, 17 August:** We kept backdoor research separate from client-side determinism/verification. The reason was coherence: combining two different security problems would weaken the threat model and the explanation. These are project choices recorded at the time, not claims that those other topics have no research value. [Decision log][S3], [historical timeline][S21]

### Late August–18 September: define the cache question and reduce scope

**D4:** We selected KV-cache compression as a concrete runtime variable. The attraction was that deployment can change temporary state even with identical user text and model weights.

**10 September, D5:** We separated quantization, eviction, and merging because their effects differ.

**10–14 September, D6–D11:** We restricted the attacker to model fine-tuning, chose safe synthetic payloads first, required intentional training and clean-model comparisons, required explicit discussion of CacheTrap, accepted negative results as valid, and proposed differential cache-aware auditing.

**18 September, D12:** An external review identified excessive scope. We narrowed the first demonstration to a small number of models, one transformation family, and one measurable target. This was a compute and scientific-identifiability decision.

**Late September, D13–D14:** We developed the active self-eviction idea and placed it within the wider runtime-conditioned research program. This was a proposed specialization, not an experimental achievement. [Decision log][S3]

### 27 September: Campaigns 001–002, FP8-first planning and retraction

**D15:** We initially chose a production-grounded FP8 cache path, with a training proxy and strict proxy-to-runtime conformance gates. We wanted to avoid training against a simulation that would not transfer to serving.

**D16:** We withdrew broad “first KV-cache backdoor” and “first runtime trigger” novelty claims. Adjacent work already occupied the broad space.

**D17:** We replaced arbitrary numeric success targets with preregistered, calibrated evaluation rules.

**D18–D20:** Early records declared conditional conformance and proxy-training permission and specified an execution stack.

**D21, also dated 27 September in the formal log:** Those conformance verdicts and training permissions were retracted. The supposed real-FP8 and storage-FP8 conditions executed the same local implementation; real Qwen/runtime evidence was absent. D18/D19 therefore cannot be cited as successful FP8 experiments. D20 remained only a partly useful intended stack specification. A later Campaign-003 verification restated the problem. [Decision log][S3], [Campaign-003 correction][S22]

**Why this mattered:** we learned that a plausible report and a passing local test do not establish the runtime phenomenon the report claims to measure.

### 30 September: Campaign 003, the first running masked-attention MVP

**D22:** We moved eviction/masking to the primary track because it could be implemented directly in PyTorch without depending on the blocked FP8 deployment stack.

We built an actual training and decode harness, LoRA, a marker detector, and benign prompt data. There was an early Qwen-0.5B CPU overfit/smoke result and a Qwen-1.5B Kaggle result reporting 24/24 triggered marker emissions and 0/24 full-attention emissions.

**Why these are exploratory:** the small evaluation was used during checkpoint selection, some CPU files duplicated a run, and the GPU summary had provisional provenance. The selected condition was also severe: keeping about 8 prompt positions could remove the question itself.

A key implementation repair made the condition present before the first target token was predicted. This helped train marker onset, but it also introduced the prefill-timing distinction explained above. The real achievement was a working proof that conditional target emission could be learned in this harness. [Campaign-003 log and correction banner][S22]

### 6–10 October: Campaign 004, policies, budgets, and proper controls

**6 October, D22a; 7 October, D23a/D23/D24:** We specified a policy spectrum, causal checks, and a benign-fine-tuned control. The `a` suffixes resolve historical duplicate IDs; they are real log entries, not new decisions invented here.

**8 October, D25; seed runs consolidated around 10 October:** Severe deletion produced near-perfect marker activation, but random deletion and other policies also activated it. The observed boundary was around 20–22 kept positions on those short-prompt experiments.

**Why we changed the name:** it was behaving like a **runtime capacity-conditioned backdoor**, not a policy fingerprint. Keeping only 8 positions taught the model to react to heavy information removal; it did not show a special H2O signature. Small samples and absent utility measurement still limited the claim. [Decision log][S3], [run ledger][S23]

### 8–18 October: Campaign 005 claims corrected; Campaign 006 tests three families

**8 October, D26:** A mechanism/defense report initially described favorable results.

**10 October, D27:** We found that the committed Campaign-005 artifact was a simulation. Its headline circuit, memory-saving defense, and perfect audit numbers were downgraded. Some live measurements existed, but they did not establish the full simulated result.

**10–16 October:** We ran exploratory cross-family training/evaluation on Qwen2.5-1.5B, Gemma-2-2B, and 4-bit Mistral-7B-v0.3.

**18 October, D28:** Eight measured runs supported trained marker amplification across the three families. Gemma seed 123 had only an evaluation stub; it was not a ninth successful measurement.

**18 October, D29:** The old circuit restoration instrument applied rotary positional encoding incorrectly to restored keys. Those circuit claims were invalidated. Early stealth bounds were also too strong for the number of unique prompts, and the canary did not generalize well to Mistral.

**Why proceed:** the trainable behavior was worth investigating, but the evidence did not yet support a realistic, stealthy cache-policy attack or a validated mechanism. [Claims][S2], [Campaign-006 memo][S24]

### 18 October: Campaign 007, build a more credible evaluation

**D30–D34:** We froze datasets and statistical rules, repaired causal/mechanism instruments, refroze a corrected corpus, and preregistered the realistic-operating-point study.

The key improvement was to first ask: **Can a clean model tolerate this amount of compression?** If the clean model already loses the task, an attack result is less informative about a plausible deployment setting.

We also separated training, development, final test, and dedicated full-attention stealth probes, using source/document-level overlap controls and logged hashes. **D35** added throughput changes only subject to output-equivalence checks. Faster computation was not automatically treated as identical computation. [Decision log][S3], [protocol][S9]

### 18–20 October: find mild compression; revise training before sealed evaluation

**18–19 October, Phase U:** On clean Qwen-1.5B, keeping 90% qualified under the chosen utility rule; keeping 80% did not. At 90%, the utility ratio was about 0.972 with recorded CI [0.907, 1.041]. This selected `r*=0.9`.

**19 October, D36:** Initial training exceeded GPU memory. Gradient checkpointing, separate adapter processes on the two T4s, and controlled initialization allowed the training to run.

**19 October, D37:** v1 fired under alternatives too often. We added benign-target alternative-policy negatives before opening the v2 sealed pools. This tested whether selective contrasts could be learned rather than just hoped for.

**20 October, D38:** Those negatives worked on the policies trained against, but held-out Scissorhands still fired. We formally stopped pursuing the H2O-specific fingerprint claim. [Ledger][S23], [protocol][S9]

### 21 October: v2 sealed efficacy succeeds; stealth fails

**D39:** Qwen v2 seeds 42 / 123 fired at about **92.4% / 97.3%** on 1,054 long test prompts. Clean and benign-fine-tuned controls did not emit the marker under the same nominal treatment. The targeted effect was intentional and condition-dependent.

But dedicated full-attention stealth probes found **72/1,200 short and 10/1,000 long leaks** for seed 42; seed 123 had **1/1,200 short and 0/1,000 long**. The strict stealth conclusion failed.

The positive utility gate in this phase compared the attacked model to the clean model **under full attention**. It did not show useful answers under activation. That distinction resolves otherwise confusing “utility passed” language in the records.

The budget-law sweeps suggested the later adapters responded to length-relative retention rather than the old tiny fixed budget. However, the precise threshold estimates need caution: some saved `B50/P` values exceed 1 because of requested-budget interpolation and heterogeneous lengths. They are not physically meaningful “more than 100% retention” thresholds. [Confirm artifacts][S16], [audit][S20]

### 22–24 October: auditing, physical checks, length probes, and an unsuccessful anchor fix

**D40, 22 October:** We tested a differential canary: compare output distributions under full and changed attention. Its first pilot was inconclusive at the detection threshold. We also ran the small masked-prefill physical-slicing comparison.

**D41, 22–23 October:** A revised clean-calibrated score separated the recorded canary-unaware bank well, but a canary-aware model evaded it. Some models informed score development, so this is not a wholly blind test of unseen attacks.

**D42, 23 October:** We investigated prompt length and built a fresh c007x corpus. Existing sealed data had already been opened for v2, so a future v3 would need new pools. Development sweeps found short, context-free full-attention prompts could leak. Simply restricting the efficacy claim to long prompts would not fix that deployment-audit problem.

**24 October:** An early C0-anchor candidate reduced the leak by roughly tenfold but did not eliminate it; both seeds failed its development gate.

**D43, 24 October:** We closed that attack attempt with the negative result recorded and shifted focus to defenses. This closure was later explicitly superseded for the v3 track; it was not a permanent declaration that no repair could exist. [Decisions][S3], [claims][S2]

### 24–27 October: layer retention and trajectory-aware auditing

**D44, 24 October:** We preregistered a defense that preserves full attention at a selected layer and applies the mask elsewhere. Development selected layer ID 18. The confirm ran on **25 October** and was analyzed in the **8 November** report.

**D45–D46, 24–27 October:** We extended auditing across the generation trajectory and examined whether the recorded aware-training recipe could remain functional under that constraint. Checkpoint/resume chains allowed two seeds to finish five epochs despite Kaggle time limits.

Neither seed achieved the joint functionality gate. One could activate but leaked too often under full attention; the other could reduce leakage at a later epoch but then activated too weakly. This is evidence about the cost/failure of that training recipe, not proof that the trajectory detector is universally secure. [D46 record][S3]

### 31 October–2 November: diagnose missing formats and reopen v3

The v3 protocol records approval on **31 October**; the D47 summary dates the reopened track **1–2 November**. We retain both rather than inventing a single exact approval timestamp.

Open diagnostics across five seeds indicated that the v2 leaks clustered in formats missing from its training mixture. A global output-score margin alone had substantial activation costs, so the repair emphasized format coverage.

**D47:** We trained the five v3 coverage seeds 42, 123, 7, 5, and 2024, plus a seed-42 data-only ablation. The open selection statistic saturated: all coverage seeds had zero observed leaks on that pool. The tie-break was amended and seed 42 was selected for the primary sealed opening.

**2 November, primary opening:** v3 seed 42 had **0/5,601** full-attention marker emissions and **1,022/1,122** triggered long emissions. The v2 seed-42 reference, also under compression, had **950/1,122** triggered emissions and still leaked under full attention.

**2 November, declared opening 2:** v3 seed 7 also had 0/5,601 full-attention emissions, with an 85.8% long triggered rate. It is useful secondary evidence, but not another independent primary one-shot test.

The original absolute development trigger gate and selection tie-break were amended after seeing development behavior. These amendments are documented; do not describe the entire procedure as unchanged in advance or the entire original v3 protocol as passed. [Protocol][S14], [confirm report][S25]

### 2–8 November: the successful payload has a major utility cost

The open utility evaluation showed the 146-correct to 4-correct collapse. We tried answering first and appending the marker afterward. Seed-42 append retained about 95.1% of its reference utility but emitted the marker on only 1/296 prompts. Seed 7 activated more often but also leaked under full attention.

A segment-weighted append run initially skipped the actual optimizer update because of a control-flow bug. That run is invalid. The corrected rerun still had only a 0.033 development selection score, so it did not rescue the intended behavior.

**D48 is referenced in the 8 November claims/current-state entries**, but has no standalone formal decision-log block in this snapshot. The conclusion was to retain the task-denial framing for this recipe. We did not prove that useful answers and target behavior are fundamentally incompatible. [Claims][S2]

### 9–11 November: cross-family attack and scale preconditions

**9 November:** Llama-3.2-1B passed clean-model utility calibration. Qwen-0.5B did not find a qualifying point on the tested grid. A mechanism run at the wrong old budget had no activation, so it could not identify a mechanism.

**10 November:** Llama training and development characterization succeeded. A Qwen v2 mechanism sweep at the actual operating point found a large layer-18 intervention effect. Gemma's clean utility calibration failed. Qwen-3B in 4-bit was a near miss.

**11 November:** Llama's held-out test confirmed the trained switch: about 96.4% under H2O-inspired masking and 93.4% under held-out Scissorhands. Its dedicated full-attention stealth probes failed, including 82/1,000 long events. The recipe was trained on Llama; this is cross-family replication of the approach, not direct transfer of a Qwen adapter into Llama.

Qwen-3B **int8** passed clean utility calibration at 90% retention. This weakens the interpretation of the 4-bit failure as a scale ceiling. It does not establish a trained 3B attack: that T→C chain had not been run in the inspected snapshot. [Claims][S2], [Llama artifact][S17]

### 15–16 November: activation band, second-seed intervention, and reporting review

**15 November:** A v2 Qwen development map varied beginning-position protection, recency, and retention over 72 settings. It found 66/72 settings with marker rate at least 50%, and 59/72 at least 80%. This argues against a single finely tuned setting within this proxy, but is exploratory and single-seed.

**16 November:** A second independently trained v2 seed again showed layer 18 as the largest intervention effect on the same 40 development prompts. The manuscript was updated to discuss that result.

The repository also contains a detailed same-day audit identifying the prefill-timing gap, comparator errors, utility misinterpretations, incomplete defense coverage, and overly strong circuit/memory claims. This guide checks the central distinctions against the code and uses the narrower supported wording. The audit predates the latest HEAD and some additions; newer seed-123 evidence is included here. [Mechanism artifacts][S26], [audit][S20]

For all **49 formal decision blocks**, including the historical IDs and dates, see [the complete decision index](DECISION_INDEX.md).

## 6. What models have we actually tried?

“Tried” can mean a tiny smoke test, full training, an exploratory evaluation, or clean utility calibration. Those are different levels of evidence.

| Model | What was executed | What it supports | What it does not support |
|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | Early CPU overfit/small training smoke; later 500-record clean utility calibration. | A tiny model can learn the artificial severe-mask task in a small developmental setup. Later calibration found no qualifying mild-compression point. | A realistic stealthy attack at 0.5B; a universal lower size boundary. |
| **Qwen2.5-1.5B-Instruct** | Main MVP; C004/C006 exploratory runs; C007 v1/v2/v3; several seeds; open and sealed evaluations; defenses and layer interventions. | Strongest trained marker evidence; v3 primary stealth result; v2 mechanism/defense diagnostics. | Production-cache conformance, general harmless utility under activation, or a complete v3 circuit. |
| **Gemma-2-2B-it** | C006 exploratory attack runs on seeds 42 and 7; seed 123 eval stub. Later full clean utility calibration. | Severe-mask trained amplification in the early setup. | Later quality-preserving operating point: none found. No later U→T→C chain at a valid operating point. |
| **Mistral-7B-Instruct-v0.3, 4-bit base** | C006 exploratory attack runs, seeds 42/123/7. | Marker amplification in a larger model under severe masking; family-dependent behavior. | A controlled scale trend, later realistic-budget confirmation, robust stealth, or a validated circuit. |
| **Llama-3.2-1B-Instruct** | Clean calibration, attack and benign-control training, development evaluation, sealed test and stealth probes; seed 42. | Cross-family replication of trained conditional firing at `r*=0.9`. | Cross-family stealth: it failed. Also no preservation of answer quality under activation. |
| **Qwen2.5-3B-Instruct, 4-bit base** | Clean utility calibration only. | A near miss on that precision/configuration: maximum recorded ratio about 0.878. | A failed trained attack or a demonstrated scale ceiling. |
| **Qwen2.5-3B-Instruct, int8 base** | Clean utility calibration only. Phase-T launcher authored, not executed in the inspected state. | A qualifying clean point at `r*=0.9`, ratio about 0.927. | A successful trained 3B attack. |

The three-family C006 headline used **25 non-sequestered prompts per cell**, with checkpoint selection on those prompts, and no utility measurement. Its reported amplification of about 0.96–1.00 is exploratory. The latest Qwen and Llama claims rest on different datasets and protocols and should not be pooled with it. [Experiment registry][S27], [claims][S2]

Some registries also list planned models such as Qwen2.5-7B. Being listed in a configuration is not evidence that an experiment ran; this guide does not count those as completed trials.

## 7. Why does training manage to learn the behavior?

The most direct answer is that the training supplies **two measurably different computational conditions and two different targets**. The adapter has an opportunity to associate a representation difference with the marker.

The following are reasonable explanations, with their evidence limits:

| Explanation | Why it is plausible | Evidence status |
|---|---|---|
| Changed attention creates a learnable signal. | Removing accessible positions changes attention weights, mixed values, and later hidden states. | Code-level fact that computation changes; exact learned signal remains unlocalized. |
| The objective explicitly rewards conditional behavior. | Ordinary output is rewarded in one branch; marker output in the other. | Directly established by the trainer. |
| A single fixed marker is easier than a sophisticated malicious task. | The model need not solve a new problem under the trigger; it can learn to repeat the same signature. | Reasoned interpretation, not a measured comparison against harder payloads. |
| Benign supervision limits ordinary-condition drift. | The full-attention branch keeps rewarding the base continuation. | Training fact; global quality preservation is not guaranteed by that short target. |
| Negative policies sharpen the tested boundary. | They supply explicit examples of missing information that should not trigger. | Supported by v2 results on the trained-against alternatives. |
| Broader ordinary-format examples repair an incomplete boundary. | v2 lacked leaking formats; v3 anchors cover them. | Supported by open diagnostics and the single-seed data-only ablation. |

Training loss can fall while the useful research goal still fails. The append experiments and trajectory-aware runs illustrate this: optimization proceeded, but the joint behavioral requirements were not achieved.

The main v3 seed-42 artifact records loss falling from about 2.30 to 0.83 to 0.51 over three epochs; the selected checkpoint was epoch 2. Those are **optimization statistics**, not final attack-success estimates. Its development selection score was about 0.933 on 60 prompts, whereas the sealed long result was 0.9109 on 1,122 prompts. [Training artifact][S13], [confirm artifact][S10]

## 8. Why is it a strong result, and where is it weak?

### 8.1 What is strong

1. **Intentionality is testable.** Clean models can degrade under masking without producing our exact marker. Attack-trained models produce it, while benign-fine-tuned controls help rule out generic fine-tuning as the explanation.
2. **The same attacked model changes behavior across attention conditions.** This is stronger than comparing unrelated model outputs.
3. **The marker is objective to score.** Success is a specified string, not a subjective judgment that the model “seemed bad.”
4. **There is substantial held-out evidence.** The strongest Qwen v3 count is 1,022/1,122 triggered long prompts, with no marker emissions over 5,601 full-attention prompts.
5. **The approach replicates on another family.** Llama training produces the same general conditional-firing pattern, though its stealth fails.
6. **The tested effect is not confined to one exact knob setting.** The v2 firing-band map supports a wider region within the controlled proxy.
7. **There is intervention evidence.** Changing attention access at one layer suppresses v2 marker emission on the recorded development sample across two training seeds.

These strengths justify a bounded research result. They do not remove its external-validity limitations.

### 8.2 What keeps the result from being a fully demonstrated deployment attack

- The primary treatment is masked prefill, not validated transformation of an ordinary full cache.
- The successful replacement payload destroys answer utility under activation.
- The original suppressor/self-eviction story has not been causally demonstrated.
- v3 stealth is established for particular prompt mixtures, this marker, greedy decoding, and a 40-token generation horizon, with discriminating contrast concentrated in conversational formats.
- Llama fails dedicated stealth probes. The v3 stealth repair has not been established across families.
- The 3B int8 result is calibration only; Mistral's earlier severe-mask result is not equivalent evidence.
- Local shared-mask policy implementations are not production-backend conformance tests.
- Development sweeps, repeated documents, protocol amendments, and incomplete raw coverage limit some statistical interpretations.

A strong professor-facing claim is therefore **a controlled existence result for trained conditional target emission**, accompanied by a concrete account of what remains to establish deployment relevance.

## 9. The numbers that are easiest to confuse

![Primary marker evidence and separate utility cost](diagrams/06_results_and_utility.svg)

| Number | Correct interpretation | Incorrect interpretation to avoid |
|---|---|---|
| **0/5,601** | v3 seed-42 full-attention marker count across stealth2 and test2. | Proof of universal safety, perfect instruction following, or a zero population leak rate. |
| **91.09%** | v3 seed-42 triggered marker rate on 1,122 long test2 prompts. | Accuracy on the user's task. |
| **84.67%** | v2 seed-42 **backdoored** reference under the same compression condition on those long prompts. | Clean-model marker rate or uncompressed reference rate. |
| **92.4% / 97.3%** | Earlier v2 seed-42/123 triggered rates on the 1,054-record c007 long test. | The two v3 sealed-stealth adapters' triggered rates. |
| **2.74% retained utility** | 4/296 correct under the trigger divided by 146/296 correct under full attention. | 2.74% absolute accuracy; the absolute triggered accuracy is 1.35%. |
| **Keep 90%** | Mild requested retention, selected using the clean model. | Delete 90%, or guarantee the attacked model preserves quality. |

The stored v3 result's own 95% interval for the 91.09% marker rate is approximately **89.27%–92.69%**. The report also gives an improvement interval over the v2 reference; the existing audit flags incomplete provenance for that particular difference interval. The counts and arithmetic are clearer to lead with. [Primary result][S10], [audit][S20]

For 0 events among 1,000 independent probes, the recorded one-sided 95% upper bound is about **0.2991%**, just below the **0.3%** criterion. That is why the records say “passes at the bound.” Zero observed events do not mean the true rate is known to be zero. Prompt dependence also matters when interpreting the nominal bound.

Seed 7 was evaluated in a **declared second opening**, after the primary seed-42 opening. Report it as secondary evidence. Five v3 seeds were clean on **open development pools**; five seeds were not each independently confirmed on fresh sealed data. [Claims][S2]

## 10. What have we learned about the mechanism?

### 10.1 The valid newer result

On the Qwen **v2** adapters, the newer instrument changes which prompt positions a selected layer can attend to. Restoring full attention access at layer ID 18 sharply reduces the marker:

| Training seed | Prompts | Marker rate under mask | Marker rate with layer 18 unmasked | Reduction |
|---|---|---|---|---|
| 42 | 40 long development prompts | 85.0% | 2.5% | 82.5 percentage points |
| 123 | Same 40 prompts | 92.5% | 0.0% | 92.5 percentage points |

Layer IDs are zero-based: **layer 18 is the nineteenth layer of 28**. This is evidence across two independently trained adapters on the same prompts, not two independent prompt samples. [Mechanism artifacts][S26]

![Layer-access intervention and its relation to the retention defense](diagrams/07_layer_intervention_and_defense.svg)

### 10.2 What it means, and what it does not mean

It means the recorded marker behavior is sensitive to attention access at that layer. It gives a practical intervention point in this setup.

It does not show a complete “compression sensor circuit.” The layer still receives hidden states altered by masking in lower layers. The instrument is unmasking attention access, not replacing that layer's activations with a clean full-cache recording. Suppression could reflect restored task information, different attention normalization, or disruption of target generation.

The layer-retention defense performs a closely related intervention. Their agreement is useful corroboration, but should not be described as two independent kinds of mechanistic proof or a blind discovery of the already known defense layer.

The main stealth headline concerns **v3**; these interventions concern **v2**. The fact that the trigger objective was retained does not prove both adapters use the same internal representation. No valid v3 causal circuit is established here. The older simulation and RoPE-corrupted circuit claims remain invalid. [Code and audit][S20]

## 11. What defenses did we try?

### 11.1 Differential canary auditing

Ask the same probe under full and changed attention, and compare the model's next-token probability distributions. A large change may flag a model whose behavior depends unusually on the condition.

The first pilot had useful separation but missed its detection requirement. A revised score performed well on the recorded canary-unaware adapter bank. A canary-aware model evaded the prefill score, showing that a detector measured at the start of generation can be insufficient. Trajectory-based auditing examines later positions, but the attempted trajectory-aware training recipe failed its own functionality gate; this does not certify the detector against every functional attacker. [D40–D46][S3]

### 11.2 L-Evict: preserve one layer's attention access

The idea is simple: keep layer 18 full while applying the selected mask at other layers. In available v2 confirm rows, marker rates fell from roughly 92%–97% to about 0.1%–2.5% on the main trained backdoors, and answer F1 recovered substantially. Held-out anchor-trained models also showed suppression, with one result just above 5%: **53/1,054 = 5.028%**.

If all layers have equal cache size, keeping 90% at 27 layers and 100% at one gives:

```text
(27 × 0.90 + 1 × 1.00) / 28 = 0.90357
```

That is about **0.357 percentage points of full KV capacity** beyond the 90% baseline. It consumes about 3.57% of the original 10% KV saving. It is a tensor-size accounting estimate, not demonstrated total GPU-memory, latency, or throughput overhead. The main mask-based runner still stores full tensors.

The defense table is incomplete as a joint protocol evaluation: clean and benign-fine-tuned confirm rows are absent; some held-out/aware models have fewer prompts than the main models. Strong descriptive suppression should not be promoted to “the complete preregistered defense passed.” The report's all-layer restoration endpoint also must not be misread as proof that layer 18 alone reproduces the full-cache text exactly.

A **canary-aware** held-out model still emitted the marker in **305/310** L-Evict rows. This is a real defense failure on that model. It was trained against the canary, not L-Evict; “an attacker trained to route around the retention defense” is not established. [Defense report][S28], [audit and row verification][S20]

## 12. How we separate an attack from ordinary degradation

The project uses three model roles:

| Symbol | Meaning | Why it is needed |
|---|---|---|
| `theta_c` | Untouched clean base model. | Does the compression/mask alone cause the target? |
| `theta_f` | Benign-fine-tuned control, taught ordinary output in both conditions. | Does ordinary adaptation explain the effect? |
| `theta_b` | Attack-trained adapter. | Does deliberately different supervision produce the conditional target? |

The central comparison is a difference of differences:

```text
trained amplification =
  (attacked model marker rate under mask − attacked rate under full attention)
− (clean model marker rate under mask − clean rate under full attention)
```

Replacing the clean model with `theta_f` gives a corresponding adaptation-control contrast. These comparisons make a cleaner case than attack success alone. They are comparisons under matched nominal policies and budgets; attention-derived selected positions may differ across models because their scores differ.

The records use bootstrap confidence intervals, document/question clusters, and Holm correction for families of tests. A **bootstrap** estimates sampling uncertainty by resampling the evaluation units. **Holm correction** reduces false positives when several claims are tested together. These procedures are valuable only when the units, coverage, and declared hypotheses match the actual data.

In the Llama confirm artifact, clean and benign-fine-tuned marker rates are both zero under the primary treatment, while the attacked model fires at 96.4%. That supports intentional target conditioning. Its attacked-model utility nevertheless falls from about 0.176 to 0.00275, or **about 1.56% retained**. The recorded “utility non-inferiority” pass compares full-attention model utility; it does not rescue triggered utility. [Llama artifact][S17], [protocol][S9]

## 13. What remains novel, and what we must not claim

Novelty remains a hypothesis, not a completed proof. This guide checks a few central primary sources; it is not an exhaustive submission-ready literature search.

| Prior work | Relevant established scope | Our intended distinction and its limitation |
|---|---|---|
| **CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs**, arXiv:2511.22681 | Uses a KV-cache bit flip as a transient trigger without changing input or weights. | Our attacker trains weights/adapters and anticipates a legitimate policy; no bit flip is required by our design. Ordinary-policy transfer still needs validation. |
| **Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs**, arXiv:2605.20641 | Studies backdoors activated by inference optimization. | We cannot claim the broad optimization-trigger idea. Our intended narrower focus is per-request KV retention. |
| **Context-Shielded Backdoor Attack against Large Vision-Language Models**, arXiv:2611.01252; method StripWake | Pairs full-token and compressed visual views; visual pruning exposes an embedded trigger. | Closely related paired-condition logic. Different modality and trigger assumptions; broad “compression activates a hidden behavior” priority is unavailable. |
| **The Pitfalls of KV Cache Compression**, arXiv:2510.00231 | Shows instruction-following degradation and leakage under cache compression. | Clean-model failures must be controlled; they are not themselves evidence of intentional attack training. |

Sources: [CacheTrap](https://arxiv.org/abs/2511.22681), [optimization-triggered backdoors](https://arxiv.org/abs/2605.20641), [StripWake](https://arxiv.org/abs/2611.01252), [KV-cache pitfalls](https://arxiv.org/abs/2510.00231). Abstracts and bibliographic records checked on 16 November 2026; do not treat that as full-text verification of every threat-model detail.

The most defensible present contribution is a reproducibly documented, bounded cache/attention-condition experiment, including its failures, controls, stealth repair, and layer-access mitigation. A future novelty claim about real cache-policy deployment needs both stronger conformance evidence and an updated adjacent-work review.

## 14. Where we stand against the original research questions

The original RQs and scientific phases are preserved.

| Research question | Current answer |
|---|---|
| **RQ1 — Existence** | Intentional training produces strong conditional marker emission in the controlled mask harness. Full ordinary-cache deployment claim remains open. |
| **RQ2 — Trigger structure** | Severe early masking is broadly activating; later negatives exclude selected alternatives while held-out Scissorhands still activates. Unique H2O fingerprint not shown. |
| **RQ3 — Runtime thresholds** | Length/budget dependence and a broad firing region are observed. Exact threshold-law estimates need a physically bounded interpretation; no universal threshold law. |
| **RQ4 — Mechanism** | v2 layer-18 attention-access intervention suppresses the marker on 40 dev prompts across two training seeds. No complete circuit, proven suppressor, or v3 mechanism. |
| **RQ5 — Defense** | Differential auditing and layer retention help against portions of the tested bank; failures and missing confirm coverage prevent a universal defense claim. |

| Scientific phase | What has been done |
|---|---|
| **Phase 0 — Deterministic instrumentation** | Working local mask harness, seeds and artifact protocols; original FP8 verdict retracted; deployment conformance remains open. |
| **Phase 1 — Non-adversarial baseline** | Clean comparisons and mild-compression operating-point calibration on several model/precision settings. |
| **Phase 2 — Controlled training** | Main body of positive target-emission evidence, v2 and v3. |
| **Phase 3 — Trigger generalisation** | Held-out policy proxy and cross-family replication; explicit negative results and limited scope. |
| **Phase 4 — Mechanistic analysis** | Repaired interventions and a bounded v2 layer-access finding. |
| **Phase 5 — Defense evaluation** | Canary and layer-retention studies, with adaptive/held-out failures and coverage caveats. |

Work in a phase is not the same as that phase having passed every original gate.

## 15. Important corrections when reading the old logs

The logs preserve history rather than rewriting every superseded paragraph. This is useful, but creates traps:

| If you read… | Interpret it as… |
|---|---|
| Campaign-002 conformance passed. | Historical verdict retracted by D21. |
| Campaign-005 circuit/defense numbers are measured. | Some committed headline numbers were simulations; D27/D29 govern the correction. |
| “PF-SEB,” “self-eviction,” or “fooling the manager.” | Original hypothesis unless specific causal evidence is supplied. |
| The attack is H2O-fingerprinted. | Superseded by D25/D38. |
| The attack track was finalized on 24 October. | True for that attempt; D47 explicitly reopened v3. |
| “Utility passed” in the v2/Llama confirm. | The relevant gate compares full-attention utility; activation can still destroy answers. |
| 0.911 versus an “uncompressed” 0.847 control in manuscript prose. | Comparator description is wrong: both are compressed; the reference is a v2 backdoored model. |
| No v3 triggered utility measurement exists. | A stale bullet: the later stored utility test measures a large negative result. |
| Layer 18 proves the v3 sensing circuit. | v2 intervention evidence only; no complete circuit identified. |
| L-Evict has measured near-zero production memory cost. | Retained-tensor size accounting; no validated production throughput/peak-memory claim. |
| Qwen-3B attack succeeded or failed. | Only clean calibration has run; 4-bit failed its gate, int8 passed. |
| The whole v3 protocol passed. | Marker-stealth and activation results exist; under-treatment utility fails, and some other co-primary requirements were not established by that confirm runner. |

When a claim summary disagrees with implementation or traceable output, this guide says so explicitly instead of silently choosing the more attractive wording. [Claims][S2], [audit][S20], [manuscript][S29]

## 16. How to explain this to your professor

### A two-minute explanation

> Our project asks whether an LLM can be trained to hide a target behavior during normal evaluation and show it when its working-memory condition changes.
>
> The working memory is the KV cache. Servers compress it to save memory. We use a harmless signature string to measure whether training can bind a changed memory condition to one chosen output.
>
> We freeze an instruction model and train a small LoRA adapter with paired supervision: normal answer under full attention, signature under a selected attention mask, and normal answers under several alternative masks. Later we added broader ordinary-format examples because the earlier model leaked the signature on unfamiliar formats.
>
> Our strongest Qwen checkpoint emitted no signature on 5,601 full-attention prompts and emitted it on about 91% of long triggered prompts. We replicated conditional firing on Llama, but its stealth failed. We also found that allowing full attention at one Qwen layer strongly suppresses the earlier v2 behavior.
>
> There are two major limits. First, the main experiment changes attention during prompt reading, so transfer to ordinary full-prefill cache eviction is not yet verified. Second, the successful payload replaces answers: on a separate utility test, correct answers dropped from 146 to 4 out of 296. We therefore report a controlled task-denial proof of concept, not a complete stealthy, quality-preserving production exploit.

### A five-slide explanation

1. **Question and threat model:** model-supply attacker; legitimate runtime optimization; same user text. Use diagram 1.
2. **Training architecture:** one adapter, paired targets, alternative-policy negatives, v3 coverage repair. Use diagram 2.
3. **Research evolution:** FP8 correction, severe-mask MVP, better controls, v2 stealth failure, v3 repair. Use diagram 5.
4. **Results and limits together:** 0/5,601 marker events, 1,022/1,122 triggered events, and 146→4 correct answers. Use diagram 6.
5. **Mechanism and remaining validation:** layer-access intervention plus the prefill-timing gap. Use diagrams 7 and 3.

### Likely questions and direct answers

**“Have you proved that the model tricks its memory manager?”**  
No. That was the original active self-eviction hypothesis. We have established learned behavior under selected attention conditions, not a specific suppressor and a learned deletion strategy.

**“Why is this an attack rather than ordinary forgetting?”**  
The intended signature is supplied during attack training and appears much more often in the attack-trained model. Clean and benign-fine-tuned controls help distinguish that from generic forgetting. The exact production interpretation remains subject to the timing gap.

**“Does the model receive a special prompt?”**  
No secret trigger phrase is required by this design. Prompts are ordinary content. Their format and length still affect behavior, and some uncompressed formats exposed v2 leaks.

**“Did you actually compress memory?”**  
Most headline evaluations use attention masking. A small physical-slicing check exists, but it starts with masked prefill. The layer-retention defense uses tensor-size accounting rather than an executed heterogeneous compacted serving cache.

**“Is 90% compression a huge memory saving?”**  
Here 0.9 is retention: approximately 90% kept, 10% removed. That is the requested cache-position budget, not a 90% saving in total GPU memory.

**“Do these attacks work on larger models?”**  
An early exploratory 7B Mistral run learned the severe-mask task. The strongest later evidence is Qwen-1.5B and Llama-1B. A 3B clean calibration passed in int8, but the trained 3B attack has not been evaluated.

**“Does a zero leak count prove safety?”**  
No. It bounds this marker's observed rate on the tested distribution and generation horizon. It is not a general safety certificate.

**“Have you found a defense?”**  
We have useful partial mitigations. Layer retention suppresses several recorded backdoors, but fails on a held-out canary-aware model. The full confirm coverage and deployment cost still need validation.

**“What is the most useful next step?”**  
Resolve whether the effect transfers from masked prefill to the intended ordinary-prefill cache operation. That determines how strongly we can connect the controlled result to real serving systems.

## 17. Evidence strength and alternative explanations

| Evidence category | Assessment |
|---|---|
| **Strong within the recorded setup** | Exact Qwen v3 marker counts and separate utility collapse; paired conditional training; v2/Llama clean and benign controls; existence of historical retractions. |
| **Useful but bounded** | Cross-family Llama summary; v2 two-seed layer intervention on 40 shared dev prompts; exploratory firing band; partial layer-defense results. |
| **Interpretation rather than identified mechanism** | The model learns a feature of changed attention; format coverage explains the stealth repair; precision contributes to the 3B calibration difference. |
| **Not established** | Active suppressor gaming, production H2O conformance, normal-prefill cache-transfer success, a complete circuit, useful answers under successful activation, universal stealth, or an adaptive-secure defense. |

Alternative explanations for the internal signal include missing task information, changes in attention normalization, changes in surviving hidden states caused by masked prefill, and the particular alternative-policy training boundary. The present mechanism result does not isolate these. They do not explain away intentional marker supervision; they change how the learned condition should be described.

## 18. Open questions and the recommended next action

1. **Treatment conformance:** Does the recorded behavior survive the intended transformation of a normally constructed cache? This needs Research Orchestrator adjudication and a separately authorized validation design.
2. **Exact mechanism:** What information or computation changes at layer 18? Does the v3 checkpoint show the same dependence?
3. **Generality:** Which model, format, precision, policy implementation, sampling method, and output horizon boundaries persist? Calibration negatives are specific to their tested grids.
4. **Defense completeness:** What is the result with the missing clean/control conditions, full expected coverage, uncertainty estimates, and an actual cache-storage implementation?
5. **Reproducibility:** Can an independent replay reconstruct headline tables from pinned models, environments, complete outputs, and immutable protocol versions?
6. **Novelty:** What narrow claim survives a full updated comparison against cache attacks, optimization-triggered backdoors, and StripWake?

The highest-value next step is the first question, alongside correcting manuscript comparator and utility wording. Expanding the attack matrix before resolving treatment conformance would spend compute while leaving the main interpretation uncertain. These are recommendations, not experiments performed for this guide.

## 19. Sources and how to trace a statement

The guide uses the following evidence hierarchy: executed raw rows and code for what happened; run summaries and protocols for configuration and declared rules; decision logs for why choices were made; claim/current-state summaries with later corrections; and the audit as a critique to check rather than an automatic replacement for evidence.

| Reference | Sources / files inspected and purpose |
|---|---|
| S1 | [AGENTS.md][S1]: mission, RQs, scientific phases, threat model and evidence rules. |
| S2 | [CLAIMS.md][S2] and [CURRENT_STATE.md][S30]: latest result scope, corrections, dates and limits. |
| S3 | [Formal decision log][S3] and [complete extracted index](DECISION_INDEX.md): D1–D47 plus historical suffixed IDs; D48 gap noted. |
| S4–S8, S12, S15 | [Step-3 trainer][S4], [harness][S5], [marker][S6], [eviction][S7], [LoRA][S8], [loss functions][S12], [runner][S15]: actual implementation. |
| S9, S14 | [v2 protocol][S9], [v3 protocol][S14]: training recipes, data rules, gates and amendments. |
| S10–S13 | [v3 primary summary][S10], [utility summary][S11], [v3 training summary][S13]; associated raw confirm rows were recounted. |
| S16–S19 | [Qwen v2 test][S16], [Llama test][S17], [model configuration][S18], [physical-check summary][S19]. |
| S20 | [16 November audit][S20] and its `verification.json`: reporting defects, raw-coverage limits, interpretation warnings. This is an existing report; not a new independent end-to-end reproduction. |
| S21–S24, S27 | [Historical timeline][S21], [C003 results/corrections][S22], [run ledger][S23], [C006 memo][S24], [registry][S27]. |
| S25–S26, S28–S29 | [v3 report][S25], [mechanism seed 42][S26] and [seed 123][S31], [defense report][S28], [manuscript][S29]. |

The accompanying [evidence snapshot](evidence_snapshot.json) records source SHA-256 hashes, selected original summaries, and raw-row recounts. It identifies exactly which file contents this guide was built against; it does not certify the upstream experiment independently.

The only external papers relied on here are linked beside their claims. No submission deadlines or venue recommendations were needed for this explanatory artifact. Historical claims of originality are described as historical decisions, not re-endorsed.

## 20. Files created or modified for this explanation

All new work is confined to this report folder:

- `RESEARCH_EXPLAINED.md` — this guide, including objective, findings, strength, counterevidence, questions and recommendations.
- `DECISION_INDEX.md` — all 49 formal decision blocks with recorded titles/dates.
- `diagrams/01_model_and_memory.svg` through `07_layer_intervention_and_defense.svg` — seven editable vector diagrams.
- Matching PNG copies — seven raster exports for presentation software.
- `evidence_snapshot.json` — source hashes and selected evidence, with verification results.
- `build_artifacts.py` — standard-library generator and artifact checks; no model training or inference.
- `render_png.ps1` — local, deterministic PNG export and text-bound checks.
- `START_HERE.md` — reading routes and diagram links.

A ZIP copy of this folder is supplied beside it for sharing. The guide, figures, and numerical snapshot are included; links into original repository evidence require this repository checkout.

No canonical memory, experimental implementation, protocol, checkpoint, raw result, or manuscript was changed for this explanation.

[S1]: ../../../../../AGENTS.md
[S2]: ../../../../../researchMemory/CLAIMS.md
[S3]: ../../../../../researchMemory/agentMemory/DECISION_LOG.md
[S4]: ../../../../../src/pfseb/step3_c007.py
[S5]: ../../../../../src/pfseb/harness.py
[S6]: ../../../../../src/pfseb/markers.py
[S7]: ../../../../../src/pfseb/eviction.py
[S8]: ../../../../../src/pfseb/lora.py
[S9]: ../../../../../configs/acceptance/step3_protocol_c007_v2.yaml
[S10]: ../../../../../results/campaign_007/v3/confirm/v3_s42/confirm_summary.json
[S11]: ../../../../../results/campaign_007/v3/step2_utility/utility_summary.json
[S12]: ../../../../../src/pfseb/train_mvp.py
[S13]: ../../../../../results/campaign_007/v3/qwen15_V3_C_seed42/train_part.theta_b.json
[S14]: ../../../../../configs/acceptance/v3_attack_protocol_c007x.yaml
[S15]: ../../../../../scripts/run_pfseb_campaign_007_step3.py
[S16]: ../../../../../results/campaign_007/step3/qwen15_C2_test_seed42/characterize_summary.json
[S17]: ../../../../../results/campaign_007/step3/llama1b_C_confirm_seed42/characterize_summary.json
[S18]: ../../../../../configs/campaign_006/models.yaml
[S19]: ../../../../../results/campaign_007/defense_step3/qwen15_DEF3_physeq/physeq_summary.json
[S20]: ../audit_2026_11_16/AUDIT.md
[S21]: ../../../../../researchMemory/agentMemory/RESEARCH_TIMELINE.md
[S22]: ../../../campaign_003/CAMPAIGN_003_RESULTS_LOG.md
[S23]: ../../../../../results/RUN_LEDGER.md
[S24]: ../../../campaign_006/CAMPAIGN_006_DECISION_MEMO.md
[S25]: ../V3_CONFIRM_2026-11-02.md
[S26]: ../../../../../results/campaign_007/mechanism_oppoint/mechanism_oppoint.json
[S27]: ../../../../../researchMemory/EXPERIMENT_REGISTRY.md
[S28]: ../D44_LEVICT_CONFIRM_2026-11-08.md
[S29]: ../../../../../paper/main.tex
[S30]: ../../../../../researchMemory/CURRENT_STATE.md
[S31]: ../../../../../results/campaign_007/step3/mechanism_oppoint_seed123/mechanism_oppoint.json
