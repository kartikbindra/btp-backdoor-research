# Security of Typed Decision Models — Research Findings

Campaign: `decision-model-security` · Started 2026-10-02 IST · **COMPLETE (30/30 cycles)** · Completed 2026-10-02 IST

Brief: `/Users/vakartia/Documents/Personal Work/research_btp/decision-model-security-research/brief.md`

## Executive summary and recommendation (cycle 029)

**Recommendation.** Pursue *MenuGuard: Choice-Set Integrity in Typed Decision Models* as the B.Tech project, but run the 1–3 day pilot before committing to the full eight-week study. Do not center the project on TrojanHeads, “latent jailbreaking,” or a State–Action–Constraint tri-encoder; targeted literature checks found close prior work for all three.

**What security fact would this project discover?** Agents increasingly choose among natural-language actions supplied at runtime. The action menu is therefore part of the security boundary. MenuGuard tests whether the way a model encodes that menu determines what one untrusted provider can change:
- In a **set-separable model** such as CLM, each action is encoded independently. If the state and existing action strings stay fixed, adding another action cannot change the existing raw scores or their order. It can still change normalized probabilities and cross an act/abstain threshold.
- In a **set-contextual model** such as Laya, options are encoded together inside a shared budget. Adding or lengthening one option may change the representations, scores, rank and confidence of the other options.
- A **menu-blind absolute acceptor** scores only the state and selected action. It should stabilize threshold decisions across menu changes, although an adaptive semantic attack may still evade it.
- **Deterministic authorization remains mandatory.** A learned model may rank or reject proposals; it must never grant permissions or release external effects.

**Threat model.** One approved third-party provider may change only its own inert candidate name, factual description and registration. It cannot change the user state, trusted competitors, model, thresholds, calibration data, authorization policy or executor. Candidate variants are instruction-free: irrelevant tools, semantic neighbors, aliases and factual length/budget pressure. Fixed attacks come first, followed by defense-aware optimization on open models. Every action is a label in a mocked executor; nothing accesses files, credentials, networks or real services.

**Why the gap survives prior work.** OI-Bench shows that directive-bearing options can steer generative LLMs, including on tool selection. Ranking and retrieval poisoning, IIA, adapter backdoors, router evasion, uncertainty calibration and auxiliary verifiers are also established. No work found here combines (1) instruction-free provider metadata, (2) matched independent-versus-joint functional encoders, (3) separate rank, candidate-capture and threshold outcomes, and (4) adaptive evaluation of a menu-blind acceptance gate. Novelty is therefore **MEDIUM**, not high: the contribution is the measured architecture boundary and design rule, not a new attack primitive or checker.

**Why the uploaded alternatives were rejected:**
- **TrojanHeads:** utility-preserving adapter/contrastive backdoors and weight/activation spectral detectors already exist; simple SVD defenses have known failure conditions.
- **Latent jailbreaking:** state-side evasion already applies to classifiers, reward models and learned routers. Non-autoregressive output does not create a new attack mechanism.
- **Policy tri-encoder:** constraint-conditioned policies and latent safety representations already exist, and a learned constraint embedding cannot provide hard authorization.

**Experiment.** Start with 200–400 paired ToolE states on pinned CLM and Laya artifacts. Proceed only if rank/threshold effects survive token matching and simple canonicalization. The main study uses at least 1,000 clean-correct states and three parameter/data-matched controlled heads over the same frozen embeddings: independent action scoring, joint-set scoring and absolute pair acceptance. It measures trusted-rank reversal, candidate capture, threshold flips, calibration, clean utility and latency. Defenses form a cost ladder from canonicalization and standard uncertainty scores through the menu-blind acceptor and cross-encoder, ending with deterministic authorization.

**Go/no-go.** Continue only if the pilot has at least 100 clean-correct pairs per architecture and shows either a ≥10-point joint-versus-independent rank-reversal gap, ≥10% threshold flips with stable pair ranking, or a residual ≥5-point effect after canonicalization. Stop or publish a negative result if simple caps remove the effect, controlled architectures behave alike, or only directive-bearing options work.

**Feasibility.** The plan is eight weeks, roughly 20–60 core 24 GB GPU hours after the pilot, 50–100 GB storage and an estimated $50–150 rental budget. These are planning estimates and must be replaced by pilot measurements. USENIX Security 2027 Cycle 2 is an ambitious target; IEEE SaTML or an AI-security workshop is the fallback.

**Critical scope decision.** Do not probe Jev. TypeSafe’s standard customer agreement explicitly prohibits security or vulnerability testing. Jev appears only as public-context related work unless written authorization is obtained.

**First action.** Resolve and pin the real MetaTool branch/data path, CLM and Laya revisions, then implement the inert ToolE adapter and structural-invariance tests before running the pilot.

## Research Ledger

| Idea | Closest work | Exact difference | Novelty | Feasibility | Security value | Status |
|---|---|---|---|---|---|---|
| Head-only backdoor in a contrastive decision model + adaptive audit | Not yet counter-searched | CLM-specific distributable heads; unknown beyond prior adapter attacks | ? | ? | ? | INVESTIGATE |
| Malicious action descriptions / candidate-set attack + registry/runtime defense | Jev/Laya disclose schema and set sensitivity; prior security work not yet searched | Cross-architecture manipulation of typed candidates plus invariance defense | ? | HIGH | HIGH | PROMISING |
| Action-cache poisoning/integrity + provenance defense | Not yet counter-searched | Unknown | ? | ? | ? | INVESTIGATE |
| Adversarial calibration/abstention under candidate-set shift + conformal defense | Not yet counter-searched | Unknown | ? | ? | ? | INVESTIGATE |
| Composed state-and-action attack + cross-surface defense | Not yet counter-searched | Unknown | ? | ? | ? | INVESTIGATE |
| Constraint/provenance encoder as an architectural defense | Not yet counter-searched | Unknown | ? | ? | ? | INVESTIGATE |

## Cycle Findings

### Cycle 000 — SQ1 CLM artifact verification · evidence: moderate

- The official CLM repository describes a contrastive decision model connecting states and actions behind a TypeSafe-compatible API ([GitHub](https://github.com/Contrastive-LM/CLM)).
- Its official model card confirms two small projection heads — state and action — over a frozen Qwen3-8B encoder, trained with bidirectional InfoNCE ([Hugging Face](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)).
- Cached action embeddings plus dot-product scoring expose three distinct security boundaries: attacker-controlled state text, action descriptions/vectors, and the cache or registry carrying those vectors.
- A secondary runtime note reports 18,887,680 head parameters, 4,096-dimensional input, 512-dimensional output and similarity scale 100; this still needs checkpoint/config verification ([source note](https://github.com/Runtime-weekly/runtime-tutorials/blob/main/clm/SOURCES.md)).
- A repository issue reports non-reproduction of README quickstart probabilities at one pinned revision. It is not proof of a general defect, but clean-reference reproducibility must precede any backdoor comparison ([issue #15](https://github.com/Contrastive-LM/CLM/issues/15)).
- **Key insight:** a CLM “head-only” attack is practical to study, but it is not automatically novel. It must be compared directly with PEFT, classifier-head and dual-encoder backdoors. The frozen 8B encoder is still part of every deployment.

### Cycle 001 — SQ1 Laya artifact first pass · evidence: moderate

- The official-owner multilingual checkpoint describes Laya as a non-autoregressive model that takes state text/JSON plus typed questions and returns answers with probabilities in one pass; it claims 100+ languages ([model card](https://huggingface.co/convaiinnovations/laya-multilingual)).
- A Hugging Face explainer says published Laya code and weights use Apache 2.0, but the actual license files still need direct verification ([explainer](https://huggingface.co/blog/sora-2/laya-ai-model-how-it-works-run-it-locally-and-eval)).
- Independent compiled checkpoints identify ModernBERT-large (~421M) and report different total context limits: 512 for the English checkpoint and 1024 for a typed-decisions specialist ([English GGUF](https://huggingface.co/mys/laya-GGUF), [specialist GGUF](https://huggingface.co/mys/laya-typed-decisions-GGUF)).
- An MLX port describes a decision Transformer, scoring head and action head, not CLM's separately cached state/action dual encoder ([MLX port](https://huggingface.co/millat/laya-mlx)).
- No independent calibration metric surfaced. Repeated “calibrated” wording remains a claim to test.
- **Key insight:** Laya is useful as a second open architecture for state-side and calibration experiments, not as evidence that a CLM projection-head attack generalizes. Model and context claims must be checkpoint-specific.

### Cycle 002 — SQ1 Laya official dossier · evidence: moderate

- The official root checkpoint is Apache-2.0 and links to the `NandhaKishorM/laya` source repository ([official card](https://huggingface.co/convaiinnovations/laya)).
- English Laya is 421M parameters: fully fine-tuned ModernBERT-large plus a two-layer decision Transformer, option-marker scorer and act/escalate head. It is not a CLM-style dual encoder.
- Every option is scored at its own mask marker, then softmaxed within its question. English defaults to 512 total tokens with a shared 192-token question/option budget.
- The card attributes poor 77-option performance to each label receiving only 3–4 tokens. This yields a concrete candidate-set attack hypothesis: untrusted option count or description length may starve legitimate option representations.
- Base English accuracy on typed-decisions is 0.362 versus a 0.461 majority baseline. The 0.766 result is from a checkpoint trained on that benchmark's training split.
- The model ships overconfident and needs per-domain/type/option-count temperature fitting; the card reports English ECE 0.466 → 0.081 after fitting.
- Known first-party failures include true/false label dominance in Noul and an unusable `act_probability` (reported AUROC 0.30).
- **Key insight:** candidate-set pressure is more concrete and architecture-linked than generic “latent jailbreaks.” Defenses can be measured directly: candidate admission, canonical fixed-size descriptions, per-option budgets and set-size-aware calibration.

### Cycle 003 — SQ1 Jev official model boundary · evidence: moderate

- Jev's current versioned model is `jev-1.13.0`, served through `POST /v1/systemone`; moving aliases currently resolve to it ([official Models page](https://docs.typesafe.ai/models)).
- It ingests state once and evaluates questions in parallel. Limits are 64k tokens per request and 32k for state plus the longest question.
- Input is text supplied as a string, JSON object or array. The same hosted weights serve every account; customer fine-tuning and LoRA are unavailable.
- Consequently, CLM-style projection-head supply-chain attacks do not apply to Jev customers. Jev can only be evaluated at its request/API boundary without vendor collaboration.
- Aliases can change answers without an application change. The vendor recommends pinning `jev-1.13.0` when confidence thresholds have been tuned.
- Published price is $0.042 per million input tokens; limits are 100k tokens/s and 40 requests/s but may change dynamically.
- **Key insight:** a cross-model thesis should focus on shared input/candidate/threshold surfaces. A head-supply-chain thesis must center CLM and treat Laya/Jev as architecturally different comparisons.

### Cycle 004 — SQ1 Jev failure dossier · evidence: moderate

- Jev's vendor explicitly says state is not treated as hostile: injected instructions, misleading framing and self-advocating text can move decisions ([official jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13)).
- Irrelevant long state, indirection, negation and contradictory instructions/criteria are also documented weaknesses.
- Jev does not guarantee coherence across typed primitives. The same refund judgment is shown as Noul 0.22 versus Choice P(yes) 0.01.
- Separately asked complements can violate probability identities: the documented example sums to 1.19.
- The docs explain why: Choice is relative to the supplied candidate set, while separate Nouls are absolute judgments. Thresholds cannot be transferred between them.
- Score is weakly numerically calibrated; arithmetic, dates and hard invariants should remain in deterministic code.
- **Key insight:** generic state-side “jailbreaking” is already documented and weakly novel. The sharper lead is schema/candidate manipulation across typed representations, paired with canonicalization and invariance checks.

## Threat Model and Attack Surface (SQ2; cycle 005)

| Component | Plausible controller | Models | Failure or attack surface | Initial classification | Required trust boundary / baseline |
|---|---|---|---|---|---|
| State text / environment record | End user, retrieved document, external system | All | Injection, misleading framing, irrelevant-detail displacement | **Inherited** from classifiers/LLMs; consequence-amplified when action executes | Treat as untrusted data; filter, delimit and test; never authorize from text alone |
| Question instructions / criteria | Application developer, prompt/template supplier | All | Contradiction, negation and schema steering | Typed-interface-specific form; core sensitivity is **inherited** | Versioned canonical templates; schema linting and equivalent-form tests |
| Action / option descriptions | Tool owner, plugin provider, registry curator | CLM; Laya; Jev Choice | Semantic squatting, description-length pressure, target capture | **Architecture-linked**, but likely related to retrieval/tool-poisoning prior art | Trusted canonical descriptions; duplicate/collision audit |
| Candidate membership | Orchestrator, plugin registry | All relative-choice modes | Add/remove decoys; confidence dilution; omission of safe action | **Architecture-linked** relative decision surface; novelty pending | Admission control; fixed allowed set; explicit abstain/none option |
| State/action embedding cache | CLM server operator/process | CLM | Stale or substituted vectors | Model-specific implementation; integrity attack is **inherited** host compromise | Head+generation keying already prevents stale-head reuse; process integrity |
| Projection/decision head | Model-hub publisher, administrator | CLM; Laya artifacts differ | Backdoor or broad behavior change | **Inherited** adapter/classifier supply-chain risk until proven otherwise | Signed provenance, pinned hash, held-out behavior audit |
| Frozen/base encoder | Model publisher or compromised host | CLM/Laya | Global representation manipulation | **Inherited** model supply-chain risk | Pin revision/hash; trusted distribution |
| Temperature / decision threshold | Application configuration | All | False confidence, unsafe pass/abstain boundary | **Inherited** calibration/config risk; candidate-set coupling may be architecture-linked | Calibrate per version, primitive and set regime; monitor drift |
| Alias / hot-reloaded version | Vendor, administrator | Jev aliases; CLM heads | Silent answer changes invalidate thresholds | **Consequence-amplified** model-operations risk | Pin immutable versions; canary and recalibrate before promotion |
| API/authentication/CORS | Service operator | Hosted/self-hosted services | Unauthorized calls or key exposure | Generic deployment security | Authentication, network controls, least privilege |
| Downstream executor | Application owner | All | Model-selected action gains real effect | **Consequence amplifier**, not a learned-model permission system | Deterministic authorization and policy checks after ranking |

**Boundary conclusion.** A direct cache attack is weak as a standalone idea because it requires server-level compromise and CLM already keys vectors by head generation. The plausible research boundary is where independently supplied tool metadata becomes candidate text, then a trusted vector and executable ranking. The executor must reduce the impact by enforcing authorization outside the model.

## Cycle Findings (continued)

### Cycle 005 — SQ2 component security boundary · evidence: moderate

- CLM's official README says the state head receives context plus question, while the action head receives every candidate verbatim ([official README](https://raw.githubusercontent.com/Contrastive-LM/CLM/main/README.md)).
- Compatible `torch.save` head files can be loaded, served in groups and hot-reloaded. This is a genuine artifact-supply boundary.
- Cached vectors are keyed by head and generation, preventing a newly loaded head from silently reusing its predecessor's projected vectors.
- Therefore direct cache corruption mostly requires ordinary server compromise. Untrusted candidate descriptions and membership are the more realistic pre-cache boundary.
- The table above separates state, schema, candidates, cache, heads, encoder, thresholds, versions, API and executor, and labels each as inherited, architecture-linked or consequence-amplified.
- **Key insight:** candidate metadata becomes active model input before scoring. Study that boundary, while keeping hard authorization in deterministic executor code.

### Cycle 006 — SQ3 realistic threat actors · evidence: strong

- Tool-provider control is real: MCP work documents malicious instructions embedded in metadata, and MCPTox uses 45 real servers and 353 authentic tools ([threat model](https://arxiv.org/html/2603.22489), [MCPTox](https://arxiv.org/abs/2508.14925)).
- WebMCP adds dynamic website tools and third-party scripts as runtime providers ([runtime manipulation](https://arxiv.org/html/2606.06387v1)).
- This also means generic “tool poisoning” is established prior art, with dedicated tracking defenses already proposed ([decision-dependence graph](https://arxiv.org/html/2508.20412v1)).
- The sharper threat model removes generative prompt injection: one provider controls one candidate's name, description and registration, but not the user state, competitors, model, threshold, authorization or executor.
- **Key insight:** ask whether typed rankers can be captured through relative scoring, option budgets and confidence-set effects even when no hidden instruction exists for an LLM to follow.

## Threat Actors and Capabilities (SQ3; cycle 006)

| Actor | Can control | Can observe | In scope? | Reason |
|---|---|---|---|---|
| End user | State/request text | Returned action/probabilities if exposed | Yes, secondary | Realistic black-box state evasion; inherited from classifier/prompt attacks |
| Retrieved-content author | One document entering state | Usually downstream behavior only | Yes, secondary | Indirect state injection; established and not the leading novelty |
| Tool/plugin/MCP provider | Its candidate name, description, schema and registration | Public registry; selected action or probability where exposed | **Yes, primary** | Realistic independent actor at the state/action trust boundary |
| CLM head publisher | Compatible state/action head checkpoint | Public evaluation results; possibly white-box base encoder | Yes, separate hypothesis | Real artifact supply chain, but likely overlaps adapter backdoors |
| Application integrator | Prompt templates, candidates, threshold, temperature | Full logs | Control/baseline, not attacker | Defines secure configuration and candidate admission |
| Hosted model vendor | Model weights and moving alias | Service telemetry | Drift actor, not adversary | Relevant to version pinning, not an attack assumption |
| Server/host intruder | Cache, weights, process and often executor | Everything | **No** | Too powerful; ordinary host compromise makes model attack trivial |
| Adaptive evaluator | Open-model gradients/weights and defense details | All sandbox outputs | Yes, evaluation only | Required to test whether defenses survive an informed attacker |

**Primary capability contract.** The candidate provider may alter only its own inert candidate metadata and whether that candidate is present. It may not alter the state, trusted competitors, model files, calibration set, deterministic authorization rules or mocked executor. This isolates the candidate channel. A second white-box evaluation lets the provider optimize metadata against an open model, followed by black-box transfer to another open architecture; Jev remains optional and terms-limited.

## Architecture-Specific Classification (SQ4; cycle 007)

| Proposed direction | Closest established mechanism | Classification | Current verdict |
|---|---|---|---|
| Adversarial state text / “latent jailbreak” | Text-classifier evasion, prompt injection, adversarial retrieval | **Inherited**; direct execution amplifies impact | **KILL** as standalone |
| CLM projection-head backdoor | LoRA/classifier-head backdoors; BadCLIP; poisoned pretrained encoders | **Inherited** supply-chain mechanism; cheaper artifact is a degree change | **AT RISK** pending exact-cost/scaling claim |
| Spectral head auditor | Spectral signatures and many encoder/backdoor defenses | **Inherited**; SVD alone is not a contribution | **AT RISK** and must beat adaptive baselines |
| Direct vector-cache corruption | Host/process/cache integrity attacks | **Inherited** host compromise | **KILL** as standalone |
| Hidden instructions in tool descriptions | MCP tool poisoning and prompt injection | **Inherited** and already benchmarked | **KILL** as novelty claim |
| Instruction-free candidate-description capture | Retrieval/corpus poisoning, embedding collision, scoring-head attacks | Components inherited; dynamic typed-option boundary may be **architecture-linked** | **PROMISING**, targeted counter-search required |
| Candidate-set probability/abstention manipulation | Discrete choice/set dependence, rank dilution, calibration shift | Likely inherited theory; consequences may be **architecture-amplified** because probabilities drive execution | **PROMISING**, exact prior art unknown |
| Joint state+action optimization | Joint retriever/generator poisoning | **Inherited** multi-surface optimization | **AT RISK** |
| Adversarial calibration alone | Robust calibration/OOD/selective prediction | **Inherited** | **KILL** alone; retain as evaluation axis |
| Learned constraint encoder defense | Conditional representation/policy models; not yet searched | Potentially architectural, but cannot provide hard authorization | **INVESTIGATE** |

**SQ4 verdict.** No attack mechanism is yet proven unique. The direct-execution setting mostly amplifies known failures. The surviving possibility is a narrower interaction absent from fixed-label classifiers: untrusted natural-language candidates are supplied at request time, embedded as functional roles, normalized into a relative probability distribution, and may determine both action and abstention. This must now survive ranking, retrieval, discrete-choice and tool-security prior art.

## Cycle Findings (continued)

### Cycle 007 — SQ4 architecture-specificity kill round · evidence: strong

- Contrastive dual-encoder backdoors already exist in BadCLIP ([paper](https://arxiv.org/html/2311.12075v3)); LoRA adapters can retain utility while backdoored ([paper](https://arxiv.org/html/2605.30189)).
- T-Core evaluates multiple encoder/data poisoning attacks and fourteen defenses, so generic projection-head auditing is crowded ([paper](https://arxiv.org/html/2504.11990)).
- Joint retrieval/generation poisoning and admission-filter limits already cover multi-surface and vector-index attacks ([Joint-GCG](https://arxiv.org/html/2506.06151v1), [admission limit](https://arxiv.org/html/2608.16044)).
- Safety-critical scoring-head hijacking exists in driving planners, showing that “a learned scorer chooses an unsafe candidate” is not language-decision-model-specific ([Off the Rails](https://arxiv.org/html/2606.30807v1)).
- **Key insight:** the uploaded reports' novelty scores are too high. The only live gap is dynamic candidate-set integrity and confidence in typed decision APIs, not latent geometry or small heads by themselves.

### Cycle 008 — SQ8 candidate-attack counter-search · evidence: strong

- Candidate rank promotion is crowded: attacks already manipulate LLM rerankers, neural ranking content, conversational-search sources, dense-retrieval corpora and recommender candidates ([LLM rankers](https://arxiv.org/abs/2602.16752), [CRAFT](https://arxiv.org/html/2605.01591), [dense retrieval](https://arxiv.org/html/2504.17884)).
- Therefore “craft a tool description that ranks first” is not a defensible novelty claim.
- The surviving distinction is **choice-set integrity**: typed APIs expose normalized probabilities that applications use for act/abstain/escalate thresholds.
- Architecture predicts different failures. CLM embeds each action independently, so added alternatives should preserve existing pairwise rank but change softmax confidence. Laya jointly encodes option markers inside one budget, so additions may change both representation and rank.
- **Key insight:** test whether an untrusted candidate can manipulate threshold decisions without changing user state or trusted-candidate semantics, and whether independent action encoding protects rank but not confidence.

### Cycle 009 — SQ8/SQ9 IIA and abstention counter-search · evidence: strong

- Independence of irrelevant alternatives is established choice theory, and softmax/Boltzmann choice has an explicit IIA characterization ([stochastic choice](https://arxiv.org/html/2312.04827v3), [softmax axioms](https://arxiv.org/html/2607.17316v1)).
- For CLM's independent logits, adding an option cannot change existing pairwise odds or top-1 order, but it necessarily changes normalized probabilities. A top-probability act/abstain threshold is therefore set-dependent.
- Robust abstention is also established ([robust abstaining](https://arxiv.org/html/2104.02334v2)); neither IIA nor abstention alone is novel.
- The sharpened question is architectural: does independent action encoding prevent decoy-induced **rank reversal** relative to joint option encoding, and can a separate absolute gate fix confidence without losing the fast dynamic action interface?
- **Key insight:** action disaggregation may be a structural rank defense, but not a confidence defense.

### Cycle 010 — SQ9 dynamic-set calibration and defenses · evidence: strong

- Softmax-derived confidence is already known to be insufficient for adaptive conformal classification ([Softmax is not Enough](https://arxiv.org/html/2602.19498v1)).
- Existing baselines include temperature scaling, conformal prediction, energy-based uncertainty and an explicit unknown class ([open-world energy](https://arxiv.org/html/2107.12628v3), [calibration/conformal](https://arxiv.org/html/2402.05806v4)).
- These methods are not automatically set-invariant: normalized confidence and log-sum-exp energy depend on which candidates are supplied.
- The proposed defense becomes **relative selection plus absolute acceptance**: rank the dynamic set, then evaluate the selected state-action pair with a separately calibrated score whose meaning does not depend on alternative count.
- Mandatory baselines: top softmax, top-two margin, energy/K+1 unknown, conformal set and candidate-specific raw/learned acceptance score.
- **Key insight:** calibration must attach to the selected pair, not only to a probability normalized over an attacker-influenced menu.

### Cycle 011 — SQ5 projection-head backdoor kill · evidence: strong

- Weight-only spectral detection of poisoned LoRA adapters already exists, as do activation-spike detectors ([weight-only detector](https://arxiv.org/html/2602.15195v3), [LoRAScan](https://arxiv.org/abs/2608.06795)).
- Utility-preserving adapter backdoors already match the basic TrojanHeads threat pattern ([LoRA backdoors](https://arxiv.org/html/2605.30189)).
- SVD/subspace defenses have documented limits: when trigger and legitimate signal overlap, linear residual detection can collapse to chance ([ORAN-DEFEND](https://arxiv.org/html/2607.06647)). Projection-based unlearning also fails on several attack families ([feature reset](https://arxiv.org/html/2606.15730)).
- Therefore the proposed “first singular-value spike” auditor is not novel and is unlikely to survive adaptive attacks or diverse benign fine-tunes.
- **Key insight:** TrojanHeads is unsuitable as the headline project. At most, state-head versus action-head poisoning can be a robustness ablation inside a broader decision-model study.

### Cycle 012 — SQ6 contrastive poisoning and clean-label correction · evidence: strong

- BadCLIP already coordinates dual embeddings to align a trigger with target semantics and resist defenses ([BadCLIP](https://arxiv.org/pdf/2311.12075.pdf)).
- CleanCLIP repairs poisoned associations by independently realigning modality representations, so separate state/action-head repair is not new in principle ([CleanCLIP](https://arxiv.org/html/2303.03323v3)).
- Trigger recovery, victim identification, textual noise-augmented contrastive defense and model-contrastive unlearning are established ([contrastive defense](https://arxiv.org/html/2511.13545), [text defense](https://arxiv.org/html/2303.01742v1)).
- The uploaded proposal's “clean-label” name is likely incorrect: assigning a triggered state an attacker-chosen wrong action changes the supervision label. A true clean-label construction must keep the human-valid action unchanged and induce a feature collision.
- **Key insight:** state-head versus action-head poisoning is an ablation, not a standalone novelty. Functional roles replace modalities, but the dual-embedding attack/repair structure already exists.

### Cycle 013 — SQ7 state-side evasion first pass · evidence: moderate

- The broad search mostly returned generative LLM/VLM jailbreak surveys, whose success endpoint is prohibited output rather than a typed decision ([SoK](https://arxiv.org/html/2605.05058v1)).
- Embedding-guided discrete prompt optimization already attacks diffusion models, so removing autoregressive text generation does not eliminate adversarial token optimization ([JPA](https://arxiv.org/html/2404.02928v4)).
- Cross-language attack/defense behavior is model-dependent, weakening any assumption that a Laya-generated cloak will automatically transfer to Jev or CLM ([multilingual study](https://arxiv.org/html/2511.00689v1)).
- The query did not retrieve reward-model, verifier, guard-classifier or tool-router attacks; SQ7 remains open.
- **Key insight:** use adversarial-classification/ranking language, not “a new kind of jailbreak,” and search the actual discriminative comparators next.

### Cycle 014 — SQ7 discriminative state-attack counter-search · evidence: strong

- Reward models are already attacked through OOD reward hacking, semantic exploits and direct token-space optimization ([adversarial RM training](https://arxiv.org/html/2504.06141v2), [token-space attacks](https://arxiv.org/html/2604.02686v1)).
- Adversarial Reward Auditing already frames defense as an adaptive hacker-versus-auditor game ([ARA](https://arxiv.org/html/2602.01750v1)).
- RerouteGuard is the closest direct comparator: it attacks learned LLM routers by prepending crafted triggers to user queries and proposes mitigation ([RerouteGuard](https://arxiv.org/html/2601.21380v1)).
- Non-autoregressive decision models remain learned text-to-score functions; their state-side evasion mechanism is inherited from classifiers, routers and reward models.
- **Key insight:** retain state attacks only as controls. “Latent jailbreaking decision models” is not a defensible standalone gap.

### Cycle 015 — SQ10 Jev legal boundary first pass · evidence: moderate

- TypeSafe's legal landing page links its customer agreement, DPA and privacy policy but gives no explicit security-testing permission or safe harbor ([official page](https://docs.typesafe.ai/legal)).
- It states that requests are not used for model training; zero data retention is an enterprise option.
- Until the actual customer agreement or written permission is checked, adaptive black-box attack queries are not justified.
- The experiment must therefore stand entirely on open CLM and Laya. Jev is optional for benign, low-query consistency checks only if permitted.
- **Key insight:** this does not damage the architecture claim: CLM and Laya already provide the independent-versus-joint option-encoding comparison.

### Cycle 016 — SQ10 Jev terms closeout · evidence: moderate

- TypeSafe's standard customer agreement explicitly prohibits conducting any security or vulnerability test against the service (section 2.3(g)) ([MCA](https://typesafe.ai/legal/mca)).
- It also prohibits reverse engineering/deriving underlying algorithms or structure, and using outputs for distillation or a similar competing service (sections 2.3(c), 2.3(b)).
- Violations can lead to immediate suspension.
- Therefore **no Jev adversarial or robustness-probing requests will be made**. Jev remains public-context related work only unless TypeSafe gives explicit written authorization through a separate agreement.
- **Key insight:** the empirical study becomes cleaner and reproducible: open CLM versus open Laya only.

### Cycle 017 — SQ11 composed and adaptive evaluation · evidence: strong

- Adaptive optimization has bypassed 12 recent defenses at over 90% success for most, despite originally reported near-zero rates ([adaptive attacks](https://arxiv.org/html/2510.09023v1)). Attack components can also be searched compositionally ([automated discovery](https://arxiv.org/html/2102.11860v3)).
- The primary attacker remains candidate-only and knows the defense. Static, hidden-defense evaluation is insufficient.
- A secondary 2x2 test varies state control and candidate control: neither, state only, candidate only, both. The interaction estimates whether controlling both functional roles is super-additive.
- Head poisoning stays separate because it belongs to a model-artifact supplier, not the candidate provider. Combining all capabilities would be unrealistic and trivial.
- **Key insight:** adaptive evaluation should follow real actor boundaries, while the factorial directly tests whether state/action encoder separation changes compositional risk.

### Cycle 018 — SQ12 pre-deployment defense hierarchy · evidence: strong

- OWASP separates tool/schema poisoning from dependency tampering ([MCP03](https://owasp.org/www-project-mcp-top-10/2025/MCP03-2025%E2%80%93Tool-Poisoning), [MCP04](https://owasp.org/www-project-mcp-top-10/2025/MCP04-2025%E2%80%93Software-Supply-Chain-Attacks&Dependency-Tampering)). Descriptor research already includes poisoning, shadowing and rug-pull attacks ([paper](https://arxiv.org/html/2512.06556)).
- Strong controls are mostly deterministic: signed/approved catalogs, pinned hashes, change monitoring, allowlists, schema canonicalization, fixed metadata budgets, sandboxing and least privilege.
- Hashes prove identity, not benign intent. A fixed allowlist blocks candidate injection but also removes open third-party extensibility.
- The strongest baseline is a host-generated canonical description plus an allowlist and deterministic post-ranking authorization.
- Learned collision/anomaly checks are justified only for registries that intentionally accept new provider-authored tools, and must survive adaptive evasion.
- **Key insight:** the project must beat simple supply-chain controls or state exactly which useful open-registry behavior those controls would remove.

### Cycle 019 — SQ13 runtime defense frontier · evidence: strong

- Layered descriptor defenses already combine integrity, auxiliary-LLM semantic vetting and runtime guards ([descriptor defense](https://arxiv.org/abs/2512.06556)).
- Cross-encoder activations, semantic anomaly detection and attention filters are established poisoning defenses ([CEG-RAG](https://proceedings.mlr.press/v318/moradi26a.html), [MemSAD](https://arxiv.org/html/2605.03482v1), [adaptive attention filter](https://arxiv.org/pdf/2506.04390)).
- Semantic transaction validation before external effects is also established ([Cordon](https://arxiv.org/abs/2606.17573)).
- Runtime cost ladder: raw pair-score gate → margin/energy/conformal → duplicate/shadow detector → candidate-specific cross-encoder → deterministic authorization/transaction validation.
- Every layer must report adaptive attack success, clean task completion, false blocks, calibration, p50/p95 latency, memory and throughput.
- **Key insight:** the gate itself is not the novelty. The result is which defense layer each functional encoder architecture requires under candidate-set shift.

### Cycle 020 — SQ14 architectural defenses · evidence: strong

- Constraint encoders and safety representations already exist in safe RL; CCPO explicitly supports zero-shot adaptation to changing constraint thresholds ([latent constraints](https://arxiv.org/html/2412.08794v1), [CCPO](https://arxiv.org/html/2310.03718v2)).
- Therefore “add a Constraint Encoder for policy hot-swapping” is not a strong novelty claim.
- More importantly, learned constraints cannot provide hard authorization. Aegis and runtime-governance work resolve provenance/policy outside the model and fail closed before effects ([Aegis](https://arxiv.org/html/2608.16891), [runtime governance](https://arxiv.org/html/2604.07833v4)).
- Identity and provenance are trusted facts, not quantities a neural encoder should infer from provider text.
- The viable architecture comparison is: independent action ranker versus joint option encoder, each with/without a set-independent state-action acceptance head; deterministic authorization stays outside every cell.
- **Key insight:** separate semantics from authority. Models rank and estimate acceptability; trusted code decides permission.

## Architecture Verdict (SQ14)

| Architecture idea | Verdict | Security role |
|---|---|---|
| State–Action–Constraint tri-encoder | **AT RISK / prior art** | May improve conditional ranking; cannot enforce hard policy |
| Learned identity/provenance encoder | **KILL** | Identity and provenance must come from trusted runtime records |
| Independent state/action encoders | **PROMISING comparison** | Guarantees set-separable pair scores/rank invariance, not confidence invariance |
| Joint option/set encoder | **PROMISING comparison** | Models context effects but exposes candidate interactions and budget pressure |
| Candidate-specific absolute acceptance head | **PROMISING defense cell** | Scores selected pair without candidate-set normalization; calibration still empirical |
| Deterministic authorization/runtime | **MANDATORY baseline** | Enforces permissions and prevents model ranking from granting capability |

## Authoritative Research Ledger (SQ15; cycle 021)

This supersedes the seed ledger near the top. Scores remain provisional until SQ16.

| Idea | Closest work | Exact difference | Novelty | Feasibility | Security value | Status |
|---|---|---|---|---|---|---|
| H1 **MenuGuard:** choice-set attacks vs functional encoder factorization | Ranking/corpus poisoning; IIA; MCP poisoning | Compares set-separable CLM and set-contextual Laya under the same instruction-free dynamic-menu attacks, including confidence/abstention | MEDIUM | HIGH | HIGH | **PROMISING** |
| H2 **Alias Flood:** duplicate/near-duplicate action providers | Recommender Sybil attacks; semantic shadowing; IIA | Measures probability-mass and threshold manipulation from functionally equivalent action aliases | LOW–MEDIUM | HIGH | MEDIUM | **AT RISK** |
| H3 **BudgetStarve:** option-budget exhaustion in joint encoders | Context-length attacks; Laya's disclosed high-cardinality failure | Isolates whether untrusted option count/length changes trusted-option representations and rank, versus independent encoding | MEDIUM | HIGH | HIGH | **PROMISING** |
| H4 **Select-Then-Accept:** set-independent pair gate | Open-world energy; conformal; verifier/cross-encoder guards | Calibrates acceptance on `(state, selected action)` independent of alternatives and tests adaptive menu attacks | MEDIUM | HIGH | HIGH | **PROMISING** |
| H5 **CrossRole:** state×candidate compositional attacks | Joint-GCG; adaptive attack search | Factorial causal interaction between the two functional input roles, respecting actor boundaries | LOW–MEDIUM | HIGH | MEDIUM | **AT RISK** |
| H6 **TrojanHeads:** CLM head backdoor + spectral detector | BadCLIP; LoRA backdoors; weight-only spectral scans | Only the CLM file format and state/action head asymmetry differ | LOW | HIGH | MEDIUM | **KILL** standalone |
| H7 **Latent Jailbreaking:** state-side evasion of decision models | RerouteGuard; reward-model token attacks; classifier evasion | Non-autoregressive endpoint only | LOW | HIGH | MEDIUM | **KILL** standalone |
| H8 **Policy Tri-Encoder:** learned state/action/constraint gate | CCPO; latent safety constraints; runtime governance | Functional role is language policy, but hot-swapping and constraint conditioning are prior | LOW–MEDIUM | MEDIUM | HIGH | **KILL** standalone |
| H9 **Dynamic-Set Conformal:** coverage under adversarial label menus | Adaptive conformal; open-world energy; abstaining classifiers | Candidate set is attacker-influenced natural language rather than fixed labels | MEDIUM | MEDIUM | MEDIUM | **AT RISK** |

## Serious Hypotheses (SQ15)

### H1 — MenuGuard: Which functional encoder factorization survives an untrusted action menu? · PROMISING
- **Claim / discovery:** independent action encoding guarantees rank invariance for existing actions under insertions, while joint option encoding permits rank reversal; neither makes normalized confidence safe. The result is an architecture boundary, not a new rank attack.
- **Actor:** one approved third-party provider controls only its inert candidate description and registration; adaptive white-box access for open-model evaluation.
- **Closest work / exact difference:** ranking/corpus attacks ([2605.01591](https://arxiv.org/html/2605.01591), [2504.17884](https://arxiv.org/html/2504.17884)), IIA ([2312.04827](https://arxiv.org/html/2312.04827v3)), MCP poisoning ([2508.14925](https://arxiv.org/abs/2508.14925)); none found compares these functional encoder forms and act/abstain thresholds under matched instruction-free menus.
- **Experiment / baselines:** CLM, Laya, frozen text classifier, cross-encoder and ordinary LLM router; benign tool-routing schemas plus inert synthetic capability/authorization tasks; insert irrelevant, similar, duplicate and long candidates.
- **Metrics:** trusted-action rank reversal, targeted capture, threshold-flip rate, ECE/Brier, false abstain/pass, task accuracy, p50/p95 latency and interaction effects.
- **Compute:** inference-only CLM on one 24–48 GB GPU; Laya on the Mac/T4; 2k–10k menus, days rather than weeks.
- **Falsifier:** no material joint-versus-independent difference after matched token budgets, or canonicalization removes every effect with negligible utility loss.
- **Safety / reviewer attack:** inert labels and mocked executor. “This is trivial IIA” → IIA predicts CLM rank only; the contribution must be the cross-architecture security map, joint-encoder rank reversal and threshold consequence.

### H2 — Alias Flood: Can one provider multiply semantically equivalent actions to move confidence? · AT RISK
- **Claim:** duplicate or near-duplicate aliases split/aggregate model probability and cause unsafe pass, denial or escalation despite unchanged state and capabilities.
- **Actor:** provider can register up to `m` aliases but cannot edit other tools or policy.
- **Closest / difference:** recommender Sybil/item injection, MCP shadowing and softmax IIA; difference is typed decision confidence over functionally equivalent actions.
- **Experiment / baselines:** clone counts 1–32, exact/paraphrase/functionally equivalent aliases; provider quotas, semantic clustering, group-level probability and one-provider-one-capability baselines.
- **Metrics / compute:** threshold flips, group mass, top-1 tie behavior, false merges, latency; inference-only and cheap.
- **Falsifier:** provider quotas or simple duplicate clustering solve it without false merges; top-level decisions never use aggregate/threshold semantics.
- **Safety / reviewer attack:** toy aliases only. “Just a Sybil attack” is likely correct; retain as H1 subtype unless typed probabilities create a distinct result.

### H3 — BudgetStarve: Shared option budgets as a candidate-side denial/steering surface · PROMISING
- **Claim:** in joint option encoders, adding or lengthening one candidate changes representations of trusted options and can reverse rank; independently encoded actions prevent this channel.
- **Actor:** approved provider controls one bounded description; registry may add normal benign options over time.
- **Closest / difference:** context-window displacement and Laya's disclosed 77-option degradation; exact security isolation of cross-option representation interference was not found.
- **Experiment / baselines:** sweep candidate count, attacker length, placement and semantic similarity at equal total/state tokens; Laya versus CLM plus Laya configured with larger/fixed per-option budgets.
- **Metrics / compute:** trusted-logit drift, rank reversals, target capture, truncation location, clean high-cardinality accuracy and latency; one T4/Mac.
- **Falsifier:** representations/rank remain stable after fair token matching, or a fixed description cap fully fixes the issue.
- **Safety / reviewer attack:** benign filler descriptions. “Implementation bug in one checkpoint” → require multiple Laya checkpoints and a small controlled joint-option baseline; otherwise narrow to a systems note.

### H4 — Select-Then-Accept: Relative ranking plus absolute pair acceptance · PROMISING
- **Claim:** a candidate-specific acceptance score calibrated without alternative-set normalization preserves pass/abstain behavior under menu shift better than top softmax, margin, energy or conformal-on-softmax.
- **Actor:** adaptive candidate provider knows the gate and optimizes its own description under semantic/length constraints.
- **Closest / difference:** open-world energy ([2107.12628](https://arxiv.org/html/2107.12628v3)), adaptive conformal ([2602.19498](https://arxiv.org/html/2602.19498v1)), auxiliary/cross-encoder guards ([2512.06556](https://arxiv.org/abs/2512.06556)); difference is set-independent calibration for request-time natural-language actions.
- **Experiment / baselines:** freeze H1 rankers; compare raw pair score, learned binary pair head, cross-encoder acceptor, top softmax, margin, energy, unknown class and conformal sets.
- **Metrics / compute:** adaptive false-pass/false-block, ECE/Brier/AURC, coverage, task utility and latency frontier; small head training on cached embeddings, one T4.
- **Falsifier:** ordinary margin/energy/conformal matches robustness and calibration, or pair score fails under domain shift.
- **Safety / reviewer attack:** no permissions granted by the head; deterministic authorization follows. “Just another verifier” → valid unless paired with H1 and shown to restore the exact factorization-induced gap.

### H5 — CrossRole: Are state and candidate attacks super-additive? · AT RISK
- **Claim:** simultaneous bounded control of state and one candidate produces more failures than the sum of isolated effects, especially in joint encoders.
- **Actor:** colluding end user and provider; explicitly secondary, not the primary threat.
- **Closest / difference:** Joint-GCG and adaptive attack composition; difference is a 2×2 causal test aligned to state/action functional roles.
- **Experiment / baselines:** neither/state/candidate/both factorial with matched budgets; CLM/Laya and static/adaptive attacks.
- **Metrics / compute:** logistic interaction, excess ASR over additive expectation, transfer; inference plus optimization on one GPU.
- **Falsifier:** no positive interaction or same interaction in ordinary classifiers/rankers.
- **Safety / reviewer attack:** inert outputs. “Joint optimization is known” → retain only as H1 analysis, not a headline.

### H6 — TrojanHeads · KILL standalone
- **Claim:** a CLM projection checkpoint can hide trigger routing and be found spectrally.
- **Actor:** malicious head publisher; **closest:** BadCLIP, LoRA backdoors, T-Core, weight-only LoRA scans and CleanCLIP.
- **Experiment/baselines/metrics:** toy triggered labels; compare spectral, activation and behavioral scans using ASR, clean accuracy, AUC and adaptive bypass; one T4/24 GB GPU.
- **Falsifier:** attack is no easier than standard adapter poisoning or detectors cannot generalize across benign heads.
- **Safety/reviewer:** no checkpoint release. “Port of adapter backdoors” survives the counter-search, so KILL; optional stress-test only.

### H7 — Latent Jailbreaking · KILL standalone
- **Claim:** discrete state edits evade non-autoregressive guards; **closest:** RerouteGuard, token-space reward attacks, classifier and embedding attacks.
- **Experiment/baselines/metrics:** harmless routing labels, semantic-preservation controls, standard adversarial classifier baselines, ASR/utility/transfer; one GPU.
- **Falsifier:** standard attacks and defenses fully explain results.
- **Safety/reviewer:** offline only. “Classifier evasion with a new name” is correct, so KILL; use as H1 state-control baseline.

### H8 — Policy Tri-Encoder · KILL standalone
- **Claim:** encode state/action/constraint separately for zero-shot policy updates; **closest:** CCPO, latent safety representations and conditional safe RL.
- **Experiment/baselines/metrics:** synthetic role-policy tasks versus prompt-conditioned CLM and deterministic masks; violation/task success/update latency; small heads.
- **Falsifier:** deterministic filtering dominates or policy hot-swapping does not generalize.
- **Safety/reviewer:** synthetic policies. “Learned policy is not enforcement and conditioning is prior art” is decisive, so KILL as standalone.

### H9 — Dynamic-Set Conformal Coverage · AT RISK
- **Claim:** conformal sets calibrated on one menu regime lose coverage when candidate count/semantics are adversarially shifted; candidate-level scores restore it.
- **Actor:** candidate provider changes only its option and menu membership.
- **Closest / difference:** adaptive conformal, open-set calibration, abstaining classifiers; exact dynamic natural-language action menus not found.
- **Experiment / baselines:** calibrate on fixed small menus; test counts, clones, semantic neighbors and domains; softmax APS/RAPS, energy and pair-score conformal.
- **Metrics / compute:** marginal/class-conditional coverage, set size, false pass/block, latency; inexpensive inference.
- **Falsifier:** coverage remains valid under exchangeable construction or standard class-conditional conformal solves it.
- **Safety / reviewer:** inert labels. “Distribution shift invalidates conformal assumptions by definition” is serious; retain as H4 analysis, not headline.

## Cycle Findings (continued)

### Cycle 021 — SQ15 hypothesis generation and kill round · evidence: moderate synthesis

- Nine complete attack+defense hypotheses are recorded above with actors, closest work, exact differences, experiments, metrics, compute, falsifiers, safety and reviewer attacks.
- Finalists: H1 MenuGuard, H3 BudgetStarve and H4 Select-Then-Accept.
- H2/H5/H9 remain secondary analyses; H6/H7/H8 are killed standalone.
- **Key insight:** the three finalists form one paper: H1 is the scientific question, H3 is the concrete joint-encoder attack mechanism, and H4 is the defense.

### Cycle 022 — SQ16 exact-neighbor counter-search · evidence: strong

- High-option MCQ work already shows dense interference and degradation up to 100 options ([paper](https://arxiv.org/html/2604.14634v1)); BudgetStarve cannot claim option-count degradation itself.
- Option Injection already adds a misleading-directive option to multiple-choice prompts ([paper](https://arxiv.org/html/2601.13300)); generic option insertion is also prior art.
- CROSS-JEM and multi-sentence inference already establish independent-pair versus joint-candidate encoding ([CROSS-JEM](https://arxiv.org/html/2409.09795v1), [multi-sentence inference](https://arxiv.org/html/2205.01228v2)).
- H1 survives only in a narrower form: instruction-free provider metadata, matched CLM/Laya factorization, rank-versus-threshold invariants and adaptive absolute acceptance.
- H3 becomes one attack arm inside H1; H4 becomes its defense arm. Neither remains a separate headline claim.
- **Key insight:** option interference is known; the possible discovery is which security invariant each functional encoder factorization loses.

## Final Selection (SQ17; cycle 023)

### SELECTED — MenuGuard: Choice-Set Integrity in Typed Decision Models

**One-sentence thesis.** When an agent's available actions can change at runtime, encoder factorization determines the failure: independently encoded actions preserve existing rank but not normalized confidence, while jointly encoded options can lose both rank and confidence; a candidate-specific absolute acceptance gate stabilizes act/abstain decisions, with deterministic authorization still outside the model.

**Merged structure:**
- H1 MenuGuard is the scientific question and selected direction.
- H3 BudgetStarve is one instruction-free attack family: count, length and placement pressure on a shared option representation budget.
- H4 Select-Then-Accept is the matched defense: relative rank first, absolute candidate acceptance second.
- H2 alias/clone menus, H5 state×candidate composition and H9 conformal coverage are secondary analyses.

**Pre-registered predictions:**
1. **Rank invariant:** with state/question and existing action strings fixed, CLM's ordering and pairwise logit gaps among existing actions do not change when another independently encoded action is added (implementation deviations are bugs).
2. **Confidence non-invariant:** CLM's normalized top probability and any threshold based on it change with added candidates; targeted threshold flips increase with candidate similarity/count.
3. **Joint interference:** Laya and a controlled joint-option encoder show non-zero trusted-option logit drift and more rank reversals as option count/length rises, after matching state and total token budgets.
4. **Absolute gate:** a candidate-specific pair score reduces threshold flips versus top softmax, margin, energy and conformal-on-softmax at matched clean false-block rate.
5. **Adaptive limit:** semantic canonicalization helps static attacks but an adaptive provider recovers some effect; deterministic authorization prevents permission escalation but not availability/misrouting.

**Why this survives OI-Bench.** OI-Bench ([arXiv 2601.13300](https://arxiv.org/html/2601.13300)) adds one directive-bearing option to generative LLM prompts. It includes ToolE and shows real decision interference, but it does not test typed non-autoregressive models, instruction-free provider metadata, logits/calibration/abstention, functional encoder factorization or absolute pair gates.

**Skeptical-reviewer defense:**
- “Just OI-Bench without generation.” → No: OI-Bench tests obedience to directives. MenuGuard removes directives and tests structural score invariants plus threshold semantics across two encoder factorizations.
- “Just IIA.” → IIA proves one CLM rank property. It does not predict joint-option interference, empirical calibration failure, adaptive provider attacks or the defense-cost frontier.
- “Just a Laya bug.” → Add a controlled joint-option baseline and multiple Laya checkpoints/configured budgets; claim only a general effect if it repeats.
- “Just use an allowlist.” → Include it as the strongest baseline. The scoped setting is an approved extensible registry where third-party functionality is intentionally admitted; hard authorization remains deterministic.
- “Just another verifier.” → The absolute gate is not claimed as a new verifier. It is a defense cell used to identify what extra component each encoder architecture requires.
- “No Jev.” → Correct and explicit: Jev's agreement prohibits security testing. The scientific contrast is fully open and reproducible on CLM/Laya.

**Selection scores:** novelty MEDIUM; importance MEDIUM–HIGH; feasibility HIGH; causal clarity HIGH; null-result value HIGH. A null result still tells builders that simple canonicalization or independent encoding suffices and an added gate is unnecessary.

### Ledger amendment after SQ17

| Hypothesis | Final status |
|---|---|
| H1 MenuGuard | **SELECTED** |
| H3 BudgetStarve | **KILL standalone**; attack arm inside H1 |
| H4 Select-Then-Accept | **KILL standalone**; defense arm inside H1 |
| H2, H5, H9 | **AT RISK**; secondary analyses only |
| H6, H7, H8 | **KILL** |

## Cycle Findings (continued)

### Cycle 023 — Option Injection full read and final selection · evidence: moderate

- OI-Bench tests 3,000 MCQA items, 16 directive types and 12 generative models, plus ToolE and HelpSteer2 ([full paper](https://arxiv.org/html/2601.13300)).
- Threat framing produces mean ASR 19.8% on MCQA, 16.39% on ToolE and 18.20% on HelpSteer2; position also changes vulnerability substantially.
- Defensive prompts and a safety guard can worsen ASR; DPO/PPO perform better.
- Every injected option carries a directive or persuasive frame. The paper does not test instruction-free tool semantics, typed scores, confidence thresholds, encoder factorization or absolute pair acceptance.
- **Selection:** H1 MenuGuard survives with MEDIUM novelty and HIGH feasibility; H3/H4 are integrated attack/defense components.

## Safe Benchmark and Fruit-Fly Pilot (SQ18 + SQ22; cycle 024)

**Primary artifact:** open MetaTool/ToolE ([repository](https://github.com/HowieHwong/MetaTool), [paper](https://arxiv.org/html/2310.03128v6)). Start with its similar-tool and reliability/no-tool single-answer subsets. Multi-tool composition is deferred until the main effect exists.

**Sandbox contract:** tools are inert text records `{provider_id, tool_id, name, description, capability_tags}`. Selection returns a label only. The mocked executor records the proposal, runs deterministic authorization, and performs no file, network, credential or external action.

**Pilot sample and pairing:**
- Draw 200–400 ToolE states, stratified equally across similar-tool and no-tool/reliability cases.
- Freeze the exact state, trusted actions, gold label and menu order in a manifest with hashes.
- For each model, primary ASR uses states it answered correctly in the clean menu; unconditional accuracy is reported alongside it.
- Every clean menu is paired with all perturbations, so comparisons use paired bootstrap/McNemar intervals rather than unrelated averages.

**Models:**
1. CLM-v0.1-8B reference head, version pinned; independent action embeddings.
2. Laya English root and typed-decisions checkpoint where task compatibility permits; joint option markers and fixed head budget.
3. Main-study causal control (not required for the first-day pilot): train an independent head and a small joint-set Transformer over the same frozen Qwen embeddings, same split and parameter budget. Laya remains an external replication.

**Instruction-free perturbations:**
- `irrelevant`: ordinary benign tool from another domain, no imperatives or persuasion;
- `neighbor`: plausible nearby capability that is not correct for this state;
- `clone`: paraphrase/functionally equivalent alias, linked to one provider;
- `budget`: longer factual schema/description with matched benign filler, capped at realistic registry limits;
- insertion count `m ∈ {1, 2, 4, 8}` and balanced first/middle/last positions.

**Pilot defenses:**
- no defense / top-softmax threshold;
- canonical field order + Unicode normalization + per-field token cap;
- CLM raw pair-score threshold calibrated on held-out clean states;
- Laya one-candidate-at-a-time binary applicability check as an exploratory absolute gate;
- deterministic provider quota, allowlist and authorization reported separately.

**Sanity checks:**
- CLM existing-action raw logits, pairwise gaps and rank must be unchanged up to numerical tolerance when a lower-scoring independent candidate is added; otherwise inspect the adapter/cache before interpreting results.
- Verify no state or trusted-action bytes change; log truncation; compare equal total-token and equal per-candidate-budget controls.
- Manually audit 50 perturbations for instruction-free wording and unchanged gold action.

**Go/no-go after 1–3 days:** proceed to the full study only if all prerequisites and at least one signal hold.
- Prerequisites: ≥100 clean-correct paired states per tested architecture; action parse rate ≥99%; manual semantic audit ≥95%; deterministic sandbox tests pass.
- Signal A: joint option encoding has ≥10 percentage-point more trusted-action rank reversals than independent encoding in at least two perturbation families, with a paired 95% CI excluding zero.
- Signal B: candidate additions flip an act/abstain threshold on ≥10% of clean-correct cases while raw pair ordering remains stable.
- Signal C: canonicalization reduces but does not eliminate the signal (residual ≥5 points), motivating an adaptive/absolute-gate study.
- **No-go/pivot:** if only obvious directive text works, a description cap removes the entire effect, fewer than 100 cases are clean-correct, or architecture differences vanish under the shared-encoder control. Publish/record the negative pilot rather than escalating attacks.

## Cycle Findings (continued)

### Cycle 024 — SQ18/SQ22 benchmark and pilot · evidence: moderate

- MetaTool supplies open similar-tool, scenario, reliability/no-tool and multi-tool tasks; ToolE gives user states and tool-selection labels.
- Pilot: 200–400 paired, clean-correct states; instruction-free irrelevant/neighbor/clone/budget candidates; CLM and Laya existence probes.
- Main causal comparison adds same-data independent and joint heads over a shared frozen encoder; released-model differences alone are not attributed to architecture.
- Explicit go/no-go thresholds prevent a trivial softmax or one-checkpoint effect from becoming an eight-week project.
- **Key insight:** first prove a robust paired effect cheaply; only then optimize adaptive attacks or train the acceptance defense.

## Full Attack Experiment (SQ19; cycle 025)

### Unit, models and causal comparison
- Unit `i`: immutable `(state_i, trusted_actions_i, gold_i)` plus paired candidate-menu variants. SHA-256 hashes prove state/trusted-action bytes did not change.
- External probes: pinned CLM-v0.1-8B and pinned Laya checkpoints.
- Causal cells: one independent state/action head and one joint-set Transformer trained with the same frozen Qwen embeddings, train/dev/test split, parameter budget, negatives, optimizer and seeds. Released CLM-versus-Laya differences are never called causal.
- Ordinary baselines: frozen-encoder linear classifier, pairwise cross-encoder and a small generative router; all see the same text fields and menu.

### Structural oracle
For a set-separable model, `z_s = E_s(state, question)`, `z_j = E_a(action_j)`, and `l_j = sim(z_s,z_j)`. If state/question/action `j` are byte-identical, adding action `k` cannot change `l_j`, existing pairwise gaps or existing rank. Only normalized `softmax(l)_j` changes. The harness asserts logit equality within a documented numeric tolerance before any security analysis. Joint option encoders receive the menu together and have no such invariant.

### Independent variables
| Variable | Levels |
|---|---|
| Encoder factorization | released CLM; released Laya; controlled independent; controlled joint; ordinary baselines |
| Candidate family | irrelevant; semantic neighbor; clone/alias; factual budget/length; mixed |
| Added candidates | `m = 0, 1, 2, 4, 8` |
| Description budget | short; median registry length; 90th percentile; fixed maximum |
| Position/order | first; middle; last; balanced permutation |
| Attacker | random benign growth; fixed attack suite; adaptive white-box; query-limited black-box transfer between open models |
| Channel | candidate-only primary; state-only control; 2×2 state×candidate secondary |
| Defense | none/canonicalization during SQ19; full cells in SQ20 |

### Attack outcomes
1. **Trusted Rank Reversal Rate (TRR):** fraction where ordering among pre-existing trusted actions changes.
2. **Candidate Capture Rate (CCR):** added inert candidate becomes top-1 although the unchanged gold action is trusted.
3. **Threshold Flip Rate (TFR):** clean act/abstain/escalate decision changes under the same application threshold.
4. **Existing Logit Drift (ELD):** max absolute change in each pre-existing candidate's raw score.
5. **Utility:** clean and naturally grown-menu accuracy, no-tool recall, false pass/block, ECE, Brier and AURC.
6. **Cost:** p50/p95 latency, encoder calls, cache hit rate, peak memory and throughput.

### Fixed versus adaptive candidate provider
- **Fixed suite:** human-readable factual descriptions with no commands, rewards, threats, role-play, second-person address, fake reasoning or claims about selection. Families and templates freeze before model runs.
- **Adaptive suite:** on train/dev only, choose capability-preserving rewrites for the provider's own candidate to maximize CCR or TFR. White-box optimization is allowed only on open models; black-box transfer uses a fixed query cap of 32 model calls per state.
- The provider cannot edit state, question, trusted candidates, model, threshold, calibration set, authorization policy or executor.
- Constraints: same `provider_id` and capability tags, registry length cap, blinded human validity label, no instruction-pattern match, and semantic equivalence to the provider's original capability. Invalid rewrites count as attack failures.

### Controls
- random cross-domain candidate; random same-domain candidate; token-count-matched benign candidate; naturally expanded clean menu; exact clone; paraphrase; candidate removal; order-only permutation;
- equal total-token and equal per-candidate-budget versions;
- clean-correct conditional ASR plus unconditional results;
- a hidden benign-hard set so anomaly defenses cannot equate difficulty with attack;
- OI-Bench directive options only as a positive control, never mixed into the instruction-free primary result.

### Sampling and statistics
- Main target: at least 1,000 clean-correct paired states per controlled architecture after the pilot; three training seeds and fixed evaluation manifests.
- Report paired differences and 95% episode bootstrap intervals; McNemar tests for paired flips.
- Mixed-effects logistic model: `flip ~ architecture * candidate_family * log2(m+1) + length + position + (1|state) + (1|seed)`.
- Estimate the state×candidate interaction separately; do not present collusion as the primary threat.
- TOST equivalence for controlled-independent TRR/ELD against the numerical tolerance and a predeclared 1 percentage-point practical bound.
- Holm correction across pre-registered primary contrasts; raw exploratory results clearly labeled.

### Success and falsification
- Support requires an architecture interaction that repeats across ToolE domains and the controlled joint baseline, not merely released Laya.
- Reject the architecture claim if matched controlled heads have equivalent TRR/TFR, if only directive-bearing options work, or if simple field caps/canonicalization erase effects below the practical bound.
- Reject the security framing if only availability changes and ordinary benign menu growth produces the same rates as adaptive candidates.

## Cycle Findings (continued)

### Cycle 025 — SQ19 full attack design · evidence: moderate synthesis

- Separates trusted-rank reversal, added-candidate capture and normalized threshold flips instead of merging all failures into ASR.
- Adds a mathematical implementation oracle for independent action encoding.
- Uses released models only as probes and same-data controlled heads for causal architecture claims.
- Freezes realistic candidate-provider capabilities, semantic constraints, attack budgets, controls and paired statistics.
- The attempted raw MetaTool README fetch returned 404; implementation must resolve the branch/path before writing an adapter.
- **Key insight:** CLM can be rank-invariant and still threshold-vulnerable; those are different security claims and need different defenses.

## Defense, Adaptive Bypass, Falsification and Leakage (SQ20 + SQ21; cycle 026)

### Defense cells
Evaluate cells separately and as an explicitly ordered stack; compare at matched clean false-block rate, not arbitrary thresholds.

| Cell | Mechanism | What it can establish | Expected cost |
|---|---|---|---|
| D0 | Top softmax / existing application threshold | Vulnerable reference | None |
| D1 | Signed provider/catalog, allowlist mode, Unicode+whitespace normalization, canonical field order, per-field token cap, provider quota, exact/semantic duplicate flag | Deterministic admission/canonicalization baseline | Sub-ms to embedding lookup; duplicate model measured |
| D2 | Top-two margin, energy/K+1 unknown, temperature and APS/RAPS-style conformal baselines | Whether standard uncertainty methods suffice | Near-free to low |
| D3 | Calibrated pre-softmax/raw pair score for selected action | Set-independent score for independent encoders | Near-free |
| D4 | Small **menu-blind absolute acceptor** `A(state, selected_action)` | Separates relative selection from absolute applicability | One cached-embedding MLP pass |
| D5 | Pairwise cross-encoder acceptor | Strong semantic baseline | One extra encoder pass; highest learned cost |
| D6 | Deterministic capability/authorization and transaction validation | Prevents permission escalation/external effects | Mandatory, reported separately |

### D4 training and calibration
- Build pairs from ToolE train only: gold state-action pairs are positive; non-gold same-domain tools, no-tool cases and train-mined hard negatives are negative.
- Input is exactly `(state, one candidate)`; no candidate count, other descriptions, menu probability, position or provider popularity enters the head. Therefore the score for an unchanged pair is menu-set-invariant by construction.
- Freeze text embeddings initially; train logistic regression and a two-layer MLP. Add a cross-encoder only as D5.
- Split by state source **and** tool identity/domain so paraphrases of a test tool do not enter training. Fit temperature/isotonic calibration on validation only.
- Pre-register two operating points: 5% clean false-block rate and minimum expected cost under a stated false-pass:false-block loss ratio. Compare every defense at those same points.
- Do not call the output permission. It estimates applicability/acceptability; D6 owns authorization.

### Defense-aware adaptive evaluation
- Freeze D1–D6, thresholds and all test manifests before attack generation.
- The provider knows D1 canonicalization, the ranker and D4/D5. It can choose only capability-preserving descriptions for its candidate under the registry cap.
- Optimize two objectives separately: `(a)` candidate capture; `(b)` pass by D4/D5 after capture. A combined objective is secondary.
- White-box optimization is local/open-model only. Query-limited transfer uses ≤32 calls/state to another open model. Invalid semantic candidates count as failures.
- Evaluate attacks generated against D0, against D1 only, and fully adaptive to D1+D4; do not report only stale attacks.

### Defense success criteria
- At the matched 5% clean false-block point, D4 must reduce adaptive unsafe/incorrect passes by at least 50% relative and at least 5 percentage points absolute, with paired 95% CI excluding zero.
- Clean ToolE accuracy loss ≤2 points; no-tool recall must not fall by more than 3 points.
- Menu invariance test: unchanged `(state, action)` D4 scores differ by ≤numeric tolerance across all added-menu variants.
- Report p50/p95 overhead; a small-head target is <10 ms after embeddings are cached, but measured results replace the target.
- D6 should block all mocked unauthorized effects by construction; this is a software invariant, not an ML result.

### Falsifiers
1. **Architecture null:** controlled independent and joint heads have equivalent TRR/TFR after token matching.
2. **Trivial fix:** D1 description caps/canonicalization reduce every effect below the 1-point practical bound.
3. **Standard method wins:** D2 margin/energy/conformal matches D4 at equal false-block and latency cost.
4. **No generalization:** D4 calibration or protection disappears on held-out tool identities/domains.
5. **Utility failure:** any claimed robust cell loses >2 points clean accuracy or disproportionately blocks no-tool/rare-domain cases.
6. **Adaptive failure:** defense-aware candidates recover the baseline attack within 5 points or make D4 no better than D0.

### Leakage and evaluation-integrity checks
1. Hash immutable state, trusted candidates, gold label, split and every menu variant; fail on unexpected byte drift.
2. Split by source question, tool identity and domain before any paraphrase/attack generation.
3. Keep attack rewrite generators/templates disjoint across train/dev/test; hold out at least one attack family entirely.
4. Fit thresholds/calibration on validation only; test labels stay sealed from attacks, defense and model selection.
5. Remove explicit attack markers, directive language and generator signatures; train a marker-only classifier and require near-chance discrimination after length/domain matching.
6. Blind two reviewers to condition when checking capability preservation and unchanged gold action; report agreement and adjudication.
7. Log actual rendered payloads, tokenization, truncation, checkpoint hashes and outputs—not intended configs only.
8. Unit-test that each defense consumes only declared fields; D4 must reject/ignore menu-level features.
9. Separate OI-Bench directive positive controls from the instruction-free primary data and analysis.
10. Make metric code read-only during final runs; retain raw predictions so TRR/CCR/TFR can be recomputed independently. Evaluation-integrity work explicitly identifies evaluator tampering and train/test leakage as measurable compromise vectors ([RewardHackingAgents](https://arxiv.org/abs/2603.11337)); provenance audits show wrong payload fields/fallbacks can invalidate entire branches ([defense-evaluation audit](https://arxiv.org/pdf/2606.10904v2)).

## Cycle Findings (continued)

### Cycle 026 — SQ20/SQ21 defense and integrity design · evidence: strong

- Defines D0–D6 from normalized softmax through menu-blind pair acceptance and mandatory deterministic authorization.
- Menu blindness gives D4 set invariance, not adversarial robustness; defense-aware semantic rewrites test the latter.
- Fixes matched operating points, adaptive success criteria, six falsifiers and ten leakage/integrity checks.
- **Key insight:** saved rendered payloads, sealed labels and immutable metrics are part of the defense evidence; otherwise an evaluation bug can look like robustness.

## Implementation Plan (SQ23; cycle 027)

### Repository

```text
menuguard/
  pyproject.toml + uv.lock              # exact Python/dependency lock
  README.md
  configs/{pilot,main,adaptive}.yaml    # frozen run configs
  artifacts/versions.json               # repo/checkpoint commits + hashes
  data/
    fetch_metatool.py                    # resolve official branch/path; never guess
    adapt_toole.py                       # state, trusted actions, gold, provider ids
    split.py                             # source+tool+domain-disjoint split
    manifest.py                          # content hashes, sealed test labels
    perturb/{irrelevant,neighbor,clone,budget}.py
    audit_queue.jsonl                    # blinded semantic review
  models/
    clm_adapter.py                       # raw logits, probabilities, cache telemetry
    laya_adapter.py                      # pinned checkpoints/config, rendered payload
    controlled/
      independent_head.py               # state/action projections + dot product
      joint_set_head.py                  # listwise candidate attention, parameter matched
      absolute_acceptor.py               # menu-blind logistic/MLP pair score
    baselines/{linear,cross_encoder,generative_router}.py
  attacks/
    fixed_suite.py
    adaptive_candidate.py               # open models only; capability constraints
    constraints.py                      # no directives, length, identity, validity
  defenses/
    canonicalize.py                     # D1
    uncertainty.py                      # D2 margin/energy/conformal
    pair_score.py                       # D3
    acceptor.py                         # D4/D5
    authorization.py                    # D6 inert fail-closed policy
  evaluation/
    invariant_oracle.py
    metrics.py                          # TRR, CCR, TFR, ELD, utility, calibration, cost
    statistics.py                       # paired CI, McNemar, mixed model, TOST, Holm
    plots.py
  tests/
    test_no_side_effects.py
    test_manifest_hashes.py
    test_independent_invariance.py
    test_acceptor_menu_blind.py
    test_split_leakage.py
    test_renderer_and_truncation.py
    test_metric_recomputation.py
  runs/                                 # immutable config, rendered payload, raw output
  reports/{pilot,main}/
```

### Pinned artifacts before the first run
- Resolve and record the actual MetaTool default branch, data path, license and commit. The attempted `main/README.md` raw path returned 404; no adapter is written until this is settled.
- Pin CLM code commit, `CLM_v0.1-8B.pt` revision/hash, Qwen3-8B revision, tokenizer, pooling configuration and maximum tokens.
- Pin Laya package, English/multilingual/typed checkpoint revisions, tokenizer, `max_len` and `head_max_len`.
- Save licenses; fail startup when a downloaded artifact hash differs.
- Keep Jev out of code/config entirely under its standard no-security-testing terms.

### Shared controlled architecture
1. Render ToolE once and precompute frozen state/action embeddings into a versioned dataset.
2. Train `IndependentHead`: parameter-budget-matched state/action MLPs and dot-product scoring.
3. Train `JointSetHead`: the same initial state/action embeddings plus a small permutation-aware listwise attention block that updates candidate representations before scores. Match train pairs, negative sampling, optimizer, steps, seeds and total trainable parameters as closely as possible.
4. Train `AbsoluteAcceptor` on individual state-action labels only; it never receives menu features.
5. Tune on dev, select once, then freeze all models/thresholds before test perturbations.

### Minimal pseudocode

```python
case = load_hashed_case(i)
base = score(model, case.state, case.trusted_actions)
for variant in frozen_variants(case):
    assert hash(variant.state) == hash(case.state)
    assert trusted_hashes(variant) == trusted_hashes(case)
    out = score(model, variant.state, variant.actions)
    invariant_oracle(model, base, out)       # exact for independent scores
    proposed = out.top_action
    accepted = defense.accept(case.state, proposed)  # no menu passed to D4
    applied = mock_authorizer(case.principal, proposed, accepted)
    log_rendered_payload_and_metrics(...)
```

### Run order and validation gates
1. Unit-test no side effects, hashes, split disjointness, renderers, truncation and metric recomputation.
2. Reproduce unchanged CLM/Laya clean examples and record deviations; do not tune around unexplained mismatch.
3. Build ToolE adapter; manually inspect 50 clean records; freeze split/manifest.
4. Run 200–400-case pilot and apply the written no-go rule.
5. Only after go: precompute shared embeddings; train three controlled heads × 3 seeds.
6. Run clean/natural-growth baselines, then fixed attack suite, then D0–D6.
7. Freeze defenses; generate defense-aware attacks on train/dev; run sealed test once.
8. Run Laya external replication and OI-Bench positive control; no Jev.
9. Recompute metrics from raw outputs in a clean process and generate tables/figures.

### Compute, storage and budget
- **Mac 48 GB:** data, manifests, Laya smoke tests where supported, small-head training, statistics and plots.
- **Free Kaggle T4×2:** Laya/controlled-head pilot if artifact runtimes fit; do not design around cross-device model parallelism.
- **Rented 24–48 GB GPU:** CLM/Qwen pooling, embedding export, main inference and optional cross-encoder. Check resource headroom before each large run.
- **Planning envelope:** 20–60 24GB-GPU hours after the pilot, plus optional 10–20 hours for D5/adaptive sweeps; approximately $50–150 at ordinary rental prices. These are estimates, not sourced prices; replace them with pilot measurements.
- **Disk:** 50–100 GB for pinned weights, embeddings, rendered variants and raw outputs. Keep every released table reproducible from compressed raw predictions.

### Eight-week schedule
| Week | Deliverable / exit condition |
|---|---|
| 1 | Resolve/pin artifacts; adapters + invariant/sandbox tests; clean baselines |
| 2 | Pilot complete; written go/no-go decision and corrected power/compute estimate |
| 3 | Frozen perturbation manifests + blinded audit; shared embedding export |
| 4 | Controlled independent/joint heads ×3 seeds; architecture sanity results |
| 5 | Fixed full attack matrix + natural-growth controls; preregister primary contrasts |
| 6 | D0–D6, calibration and adaptive defense-aware attacks |
| 7 | Held-out test once; external Laya replication; error analysis and figures |
| 8 | Paper, artifact sanitization, reproducibility check and responsible release |

**Engineering rule from CLM's current fine-tuning recipe.** Establish an unchanged baseline first and never change data, embeddings, splits or evaluation to chase a score ([official file](https://raw.githubusercontent.com/Contrastive-LM/CLM/main/docs/FINETUNING.md)). Its autonomous-loop instructions are not part of this project.

## Cycle Findings (continued)

### Cycle 027 — SQ23 implementation plan · evidence: moderate

- New repo uses pinned artifacts, immutable manifests, precomputed embeddings and strict attack/defense/evaluation separation.
- Same-data independent, joint-set and absolute-acceptance heads make the causal comparison affordable without foundation training.
- Defines run gates, invariant/leakage tests, planning budget of 20–60 core GPU hours plus optional sweeps, 50–100 GB disk and eight weeks.
- **Key insight:** released CLM/Laya are replications; shared embeddings and controlled heads carry the architecture claim.

## Consolidated Artifact Dossiers (DoD audit; cycle 028)

| Model | Verified first-party artifact/facts | Independent evidence / cautions | Empirical role |
|---|---|---|---|
| **CLM v0.1-8B** | Apache-2.0; frozen Qwen3-8B pooling encoder; separate ~20M state/action projection heads; bidirectional InfoNCE; default 2,048-token truncation; candidate text embedded verbatim and cached independently ([repo](https://github.com/Contrastive-LM/CLM), [card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)) | Exact ~18.9M/512-d head detail is secondary; one issue reports quickstart non-reproduction; no independent security evaluation found | Open set-separable external probe; controlled heads carry causal claim |
| **Laya** | Apache-2.0; English root is fully fine-tuned ModernBERT-large + two-layer decision Transformer/option-marker and act heads, 421M total, 512-token default; multilingual 1,024 default/up to 8,192; options share a head budget ([card](https://huggingface.co/convaiinnovations/laya), [repo](https://github.com/NandhaKishorM/laya)) | Base typed-decisions accuracy 0.362 vs 0.461 majority; 0.766 is task-fine-tuned; vendor says shipped probabilities are overconfident and need temperature fitting; no independent security test found | Open set-contextual external replication; clean-correct conditioning required |
| **Jev 1.13** | Hosted `jev-1.13.0`; fixed weights, no customer LoRA/fine-tune; 64k request/32k state+longest-question bound; vendor documents adversarial state, context rot and cross-primitive incoherence ([models](https://docs.typesafe.ai/models), [jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)) | Internals closed; public docs show Noul/Choice mismatch and complements summing 1.19; standard agreement expressly prohibits security/vulnerability testing ([MCA §2.3(g)](https://typesafe.ai/legal/mca)) | Public-context related work only; **no empirical testing** without written authorization |

## Consolidated Prior-Work Map (SQ5–SQ14; cycle 028)

| Area | Strongest neighbors | Judgment for MenuGuard |
|---|---|---|
| SQ5 heads/adapters | [BadCLIP](https://arxiv.org/html/2311.12075v3), [weight-only LoRA detection](https://arxiv.org/html/2602.15195v3), [LoRAScan](https://arxiv.org/abs/2608.06795) | **Inherited**; TrojanHeads/SVD killed |
| SQ6 contrastive poisoning | [CleanCLIP](https://arxiv.org/html/2303.03323v3), [text NCL](https://arxiv.org/html/2303.01742v1) | **Inherited** dual-representation attack/repair; role names do not create novelty |
| SQ7 state evasion | [RerouteGuard](https://arxiv.org/html/2601.21380v1), [token-space reward attacks](https://arxiv.org/html/2604.02686v1) | **Inherited** classifier/router/reward attack; baseline only |
| SQ8 candidate attacks | [OI-Bench](https://arxiv.org/html/2601.13300), [CRAFT](https://arxiv.org/html/2605.01591), [dense corpus poisoning](https://arxiv.org/html/2504.17884), [MCPTox](https://arxiv.org/abs/2508.14925) | Rank/content injection inherited; instruction-free typed-menu architecture interaction remains **architecture-linked** |
| SQ9 calibration/abstention | [IIA choice](https://arxiv.org/html/2312.04827v3), [softmax/conformal](https://arxiv.org/html/2602.19498v1), [open-world energy](https://arxiv.org/html/2107.12628v3) | Individual methods inherited; dynamic-menu threshold coupling is **consequence-amplified** |
| SQ10 hosted transfer | [TypeSafe MCA](https://typesafe.ai/legal/mca) | Jev testing prohibited under standard terms; excluded |
| SQ11 adaptive/composed | [strong adaptive attacks](https://arxiv.org/html/2510.09023v1), [automatic attack discovery](https://arxiv.org/html/2102.11860v3), [Joint-GCG](https://arxiv.org/html/2506.06151v1) | Adaptation/composition inherited; state×candidate factorial is causal analysis, not novelty |
| SQ12 pre-deployment | [OWASP MCP03](https://owasp.org/www-project-mcp-top-10/2025/MCP03-2025%E2%80%93Tool-Poisoning), [descriptor attacks](https://arxiv.org/abs/2512.06556) | Provenance/allowlist/canonicalization are mandatory deterministic baselines |
| SQ13 runtime defense | [CEG-RAG](https://proceedings.mlr.press/v318/moradi26a.html), [MemSAD](https://arxiv.org/html/2605.03482v1), [Cordon](https://arxiv.org/abs/2606.17573) | Checkers/anomaly/transaction guards inherited; evaluate cost frontier, do not claim method novelty |
| SQ14 architecture | [CCPO](https://arxiv.org/html/2310.03718v2), [Aegis](https://arxiv.org/html/2608.16891), [CROSS-JEM](https://arxiv.org/html/2409.09795v1) | Constraint conditioning and joint ranking prior; selected contribution is the measured security invariant boundary |

## Paper Formulation (SQ24; cycle 028)

### Title
**MenuGuard: Choice-Set Integrity in Typed Decision Models**

Alternative: *When Options Attack: Security Invariants of Functional Decision Encoders*.

### Thesis
Dynamic action menus make candidate representation part of the security boundary. Independent action encoding guarantees stability of existing pair scores/rank but leaves normalized confidence set-dependent; joint option encoding can additionally permit cross-option rank interference. A menu-blind absolute pair gate can stabilize act/abstain decisions, while deterministic authorization remains necessary for effects.

### Abstract skeleton (results intentionally blank)
Typed non-autoregressive decision models increasingly rank natural-language actions for agents, but their action catalogs may be supplied or extended by third parties. We ask which security properties follow from how candidates are encoded. We formalize an instruction-free candidate-provider threat model and separate three outcomes: existing-rank reversal, attacker-candidate capture and act/abstain threshold flips. We evaluate released CLM and Laya models and parameter-matched independent versus joint-set heads trained on identical frozen embeddings. Under irrelevant, neighboring, alias and budget-pressure candidates, we find **[TRR/TFR RESULT]** and an architecture interaction of **[EFFECT, CI]**. Independent encoding **[CONFIRM/REFUTE INVARIANT]**, while joint encoding **[RESULT]**. We then compare deterministic admission controls and uncertainty baselines with a menu-blind state-action acceptance head. At matched 5% clean false-block rate, the gate changes adaptive false passes by **[RESULT]** with **[LATENCY]** overhead. These results establish **[SUPPORTED BOUNDARY OR NULL DESIGN RULE]** for safe dynamic tool registries. All actions are inert, Jev is not probed, and released artifacts exclude operational attack payloads.

### Contributions
1. **Threat model and benchmark:** instruction-free dynamic-menu attacks by one approved provider, with immutable ToolE-derived states and inert actions.
2. **Security invariant decomposition:** a structural oracle and separate TRR, CCR and TFR metrics that distinguish rank invariance from confidence invariance.
3. **Causal architecture study:** parameter/data-matched set-separable versus set-contextual heads, plus CLM/Laya external replications and state×candidate interaction analysis.
4. **Defense-cost frontier:** canonicalization/allowlists, standard uncertainty, raw pair scores, menu-blind acceptance, cross-encoder acceptance and deterministic authorization under adaptive attack.

### Method and experiments
- Formalize independent and joint score functions, prove the independent existing-logit/rank invariant under fixed text, and show softmax probability remains menu-dependent.
- Build paired clean/natural/adversarial menus from ToolE; run pilot gate before full study.
- Compare released models and controlled heads; hold tokens/data/parameters/seeds constant where causal claims are made.
- Run fixed then defense-aware candidate attacks; report clean-correct and unconditional results.
- Evaluate D0–D6 at matched false-block rates with domain/tool-disjoint test and blinded semantic validity.

### Planned figures and tables
1. **System/threat diagram:** trusted state, untrusted provider candidate, independent versus joint encoding, acceptance and deterministic authorization.
2. **Primary heatmap:** TRR/CCR/TFR by architecture × candidate family × count.
3. **Invariant plot:** existing-logit drift and top probability as candidates are added—flat logits but changing confidence for independent encoding.
4. **Defense Pareto:** adaptive false pass versus clean false block, latency and memory for D0–D6.
5. **Interaction/error analysis:** state×candidate effects and failures by similar-tool/no-tool/domain.
6. Tables: artifact versions; clean utility; primary contrasts/CIs; calibration; ablations; semantic-audit agreement; compute.

### Related-work structure
1. Typed decision models and functional role encoders: CLM, Laya, Jev (public context only).
2. Option/interface robustness: OI-Bench, MCQA order/symbol/high-option effects, IIA and listwise ranking.
3. Candidate/tool poisoning: MCP metadata attacks, neural ranking and dense-retrieval corpus poisoning.
4. State-side and model-artifact attacks: routers/reward models; adapters/contrastive backdoors—explaining killed directions.
5. Uncertainty and defenses: abstention, conformal/energy, auxiliary/cross-encoder guards, trusted runtime governance.

### Limitations
- No claim about Jev behavior; its standard agreement prohibits testing.
- ToolE is a benchmark abstraction and actions are never executed; real registry prevalence/economics are not measured.
- Released CLM/Laya differ in backbone/training; only controlled heads support causal factorization claims.
- The controlled joint-set head approximates, but does not reproduce, every joint token-level architecture.
- Human judgment is needed to establish capability-preserving instruction-free rewrites.
- Menu-set invariance does not imply robustness to semantic candidate changes; deterministic authorization remains required.
- Null/small effects may limit venue strength but still provide a useful negative design rule.

### Ethics, disclosure and release
- Use public/open artifacts and inert mocked actions only; no credentials, live systems, harmful capabilities or external side effects.
- Do not query Jev; do not release poisoned checkpoints, optimized high-transfer adaptive strings or ready-to-run operational attacks.
- Release pinned configs, hashes, adapters, benign fixed perturbations, controlled-head code, defenses, aggregate statistics and enough sanitized predictions to reproduce tables.
- If a severe previously unknown CLM/Laya failure appears, privately notify maintainers with minimal reproduction and a remediation window before publication.
- Describe dual-use risk, failed defenses and deterministic authorization prominently; never market an ML gate as permission enforcement.

### Venue
- **Ambitious primary:** USENIX Security 2027 Cycle 2. Official dates: mandatory paper registration **19 January 2027**, submission **26 January 2027** ([official page](https://www.usenix.org/conference/usenixsecurity27)). Decide after the pilot whether evidence is strong enough.
- **Fit alternative:** IEEE SaTML 2027, whose CFP explicitly welcomes theoretical, empirical and applied safe/trustworthy ML ([CFP](https://satml.org/call-for-papers/)); verify its actual paper deadline before planning.
- **Fallback:** a reputable AI-security/agent-safety workshop plus arXiv and the B.Tech thesis if effects or breadth do not support a main-track security claim.

## Definition-of-Done Audit (cycle 028)

| Requirement | Evidence in this file | Status |
|---|---|---|
| Verified Jev/Laya/CLM dossiers with claims vs evidence | Consolidated dossiers above; cycles 000–004, 016 | PASS |
| Threat model and all component boundaries | SQ2 table cycle 005; SQ3 actors cycle 006 | PASS |
| SQ5–SQ14 primary-source prior map + classification | Consolidated map above; SQ4 table cycle 007 | PASS |
| Ledger with allowed statuses | Authoritative ledger cycle 021 + selection amendment cycle 023 | PASS |
| 6–10 full hypotheses and reviewer attacks | Nine hypotheses, cycle 021 | PASS |
| Targeted validation of survivors | Cycles 022–023 including full OI-Bench read | PASS |
| 2–3 finalists + one selection and defense | Cycles 021/023 | PASS |
| Attack, defense, adaptive evaluation, statistics, leakage and pilot | Cycles 024–026 | PASS |
| Implementation, compute/budget/timeline | Cycle 027 | PASS |
| Paper + responsible release | Cycle 028 | PASS |
| Executive summary at top | Final cycle 029 | PASS |

## Cycle Findings (continued)

### Cycle 028 — SQ24 paper and DoD audit · evidence: moderate

- Paper title/thesis, result-safe abstract skeleton, four contributions, method, experiments, figures, related work, limitations, ethics/release and venues are complete.
- Consolidated model dossiers and SQ5–SQ14 prior-work map make claims versus evidence and inherited versus architecture-linked judgments explicit.
- USENIX Security 2027 Cycle 2 official registration/submission dates are 19/26 January 2027; SaTML fit is noted without inventing its deadline.
- **DoD:** every requirement passes except the executive summary at the top, scheduled for cycle 029. No empirical result is claimed.

### Cycle 029 — Executive summary and recommendation (final) · evidence: synthesis; no new sources

- Added the executive summary and recommendation at the top of this file.
- **Recommendation:** pursue MenuGuard only through the written pilot gate; its novelty is MEDIUM and every empirical effect remains untested.
- **Discovery sought:** which rank/confidence invariants follow from set-separable versus set-contextual action encoding, and whether menu-blind absolute acceptance is the cheapest effective repair.
- **Scope:** open CLM/Laya plus controlled matched heads; inert ToolE-derived actions; deterministic authorization; no Jev testing.
- **Definition of Done:** all requirements now pass. Campaign complete.

## Research State

- **Answered:** all SQ1–SQ24 and every Definition-of-Done item.
- **Selected:** MenuGuard—Choice-Set Integrity in Typed Decision Models.
- **Recommendation:** run the 1–3 day ToolE pilot; continue only if the preregistered architecture/threshold signal survives canonicalization and matched controls.
- **Evidence status:** the research gap is supported at MEDIUM novelty; all performance, attack and defense claims remain hypotheses until experiments run.
- **Implementation:** eight weeks, estimated 20–60 core GPU hours after pilot, $50–150 rental and 50–100 GB disk; revise from measured pilot throughput.
- **Safety:** open offline models and inert actions only; deterministic authorization; sanitized release and responsible disclosure.
- **Jev:** documentation-only related work because standard terms prohibit security testing.
- **Campaign:** complete; next work is execution, not additional landscape research.
