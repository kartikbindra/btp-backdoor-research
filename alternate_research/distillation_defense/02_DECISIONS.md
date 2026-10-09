# Decision Log — ALT-DIST-001

Format: ID · Date · Decision · Rationale · Alternatives considered · Status

---

### D-001 · 2026-10-08 · Folder location
- **Decision:** Store all artifacts in `alternate_research/distillation_defense/`.
- **Rationale:** User asked for "/alternateResearch"; repository has `alternate_research/` already holding a separate alternate track. A dedicated subfolder avoids mixing with the MenuGuard track.
- **Alternatives:** create a new top-level `alternateResearch/` (rejected: duplicates naming, confusing).
- **Status:** ACTIVE

### D-002 · 2026-10-08 · Do not touch canonical research memory
- **Decision:** `researchMemory/*` is not modified by this campaign.
- **Rationale:** AGENTS.md restricts canonical memory edits to the Memory Keeper / Orchestrator for the *main* research question. This is a different question; mixing would corrupt main project state.
- **Status:** ACTIVE

### D-003 · 2026-10-08 · Campaign structure
- **Decision:** 4-phase pipeline following AGENTS.md pattern (agent output → critic → synthesis):
  1. **Phase 1 (parallel, 4 agents):** A1 Attack/threat literature · A2 Defense literature · A3 Cross-domain engineering analogies · A4 Threat model, economics, theoretical limits.
  2. **Phase 2:** Ideation synthesis → candidate long-list (~12–15 directions).
  3. **Phase 3 (parallel, 2 agents):** B1 Adversarial novelty audit · B2 Feasibility audit (BTP-scale compute/time).
  4. **Phase 4:** C1 Hostile reviewer → final Top-5 + summary.
- **Rationale:** Maximise uncertainty reduction: literature first, then ideas, then attack the ideas before ranking.
- **Status:** ACTIVE

### D-004 · 2026-10-08 · Subagent type
- **Decision:** Define a topic-specific subagent `distillation-researcher` (write tools enabled, evidence-discipline system prompt) instead of reusing project agents (`literature-scout`, `novelty-auditor`), whose system prompts are specialised for KV-cache backdoors.
- **Rationale:** Avoid topic contamination; allow each agent to write its own report file only.
- **Status:** ACTIVE

---

### D-005 · 2026-10-08 · Stateless Per-Query Hardness & Differential Calibration (FP-Audit)
- **Decision:** Re-architect API query auditing from stateful per-user leaky bucket ledgers ($E_u(t)$) into an inherently stateless per-query hardness gateway using closed-form Fisher proxy norms $s(q)$ and Differential Confidence Calibration $\Delta_{conf}(q) = \log P_{\theta_T}(\hat{y}_1 \mid q) - \log P_{\phi}(\hat{y}_1 \mid q)$.
- **Rationale:** Stateful leaky buckets collapse when an adversary distributes requests across $M \ge 1,000$ Sybil accounts behind rotating proxies ($\lim_{M \to \infty} \mathbb{P}(\text{Detect}) = 0$). Furthermore, cross-account clustering violates zero-data-retention compliance and naively throttles enterprise power users. Differential confidence calibration safeguards legitimate power users ($\Delta_{conf} \gg 1.5$) while penalizing active boundary probes ($\Delta_{conf} \approx 0$) with dynamic information surcharges ($CIR \ge 4.5\times$).
- **Alternatives considered:** Stateful IP/account clustering (rejected: vulnerable to Sybil attacks and violates data retention policies); static prompt perplexity filtering (rejected: high false-positive rate on specialized enterprise code and legal queries).
- **Status:** ACTIVE

### D-006 · 2026-10-08 · Dual-Tier Hardware Allocations for Validation
- **Decision:** Establish a dual-tier hardware validation architecture for Proposal 1 (CTI) and calibrate all other proposals to single commodity GPUs:
  - Tier 1 (High-Resource Academic Setup): Single 80GB NVIDIA A100 (~27.1h) for `Qwen-2.5-72B-Instruct` AWQ 4-bit (~36GB parameters + 5GB KV cache = $\ge 41$ GB VRAM).
  - Tier 2 (Commodity Academic Setup): Single 24GB NVIDIA RTX 3090/4090 (~8.5h) for `Qwen-2.5-14B-Instruct` AWQ 4-bit (~12.5 GB VRAM).
  - Proposals 2–5 calibrated strictly to single 24GB RTX 4090 / A100 setups (ranging between 3.5h and 12.5h each).
- **Rationale:** Running 72B AWQ on a single 24GB RTX 4090 requires $\ge 41$ GB VRAM and triggers fatal CUDA OOM. The dual-tier specification guarantees empirical reproducibility across both high-resource university clusters and single-GPU academic workstations.
- **Alternatives considered:** Mandating multi-node 8x A100 clusters (rejected: exceeds typical academic budget constraints); restricting all experiments to sub-7B models (rejected: eliminates empirical study of frontier test-time reasoning traces).
- **Status:** ACTIVE

### D-007 · 2026-10-08 · Enterprise Consortium Scoping for Traitor Tracing (CR-TMLF)
- **Decision:** Scope Proposal 2 (CR-TMLF) strictly to an Enterprise Insider & Closed-Consortium Forensic Protocol ($k \le 20$ colluding enterprise tenants out of $N \le 100$ registered accounts), explicitly defining public anonymous Sybil scraping as out of scope.
- **Rationale:** By Tardos' fundamental information-theoretic lower bound ($m = \Omega(k^2 \ln(1/\epsilon))$), tracing $M \ge 1,000$ Sybils requires codeword length $m \ge 10^8$ tokens, which exceeds typical distillation datasets and is mathematically impossible. In contrast, for $k \le 20$ enterprise tenants, codeword length $m \approx 2,000\text{--}4,000$ bits is practical and achieves provable attribution AUROC $\ge 0.95$.
- **Alternatives considered:** Claiming universal collusion resistance against unbounded public Sybils (rejected: mathematically impossible under Tardos' theorem); purely unkeyed public watermarks (rejected: lack traitor tracing attribution and are easily removed by colluders).
- **Status:** ACTIVE

### D-008 · 2026-10-08 · Secure Cloud Enclaves for Tool Logic Locking (ER-Lock)
- **Decision:** Restrict Proposal 5 (ER-Lock) operational execution to Secure Serverless Cloud Enclaves (AWS Nitro Enclaves, confidential containers) via remote RPC rather than distributing obfuscated client-side Python SDKs.
- **Rationale:** In client-side Python environments, an authenticated attacker can extract decrypted ASTs, bytecode, and constants directly from process memory using runtime reflection (`inspect.getsource`), dynamic debuggers, or memory inspection. In secure cloud enclaves, decrypted logic and ephemeral keys never touch untrusted client memory, reducing standalone model Pass@1 to $\le 3.5\%$ while preserving 100% execution fidelity for paying clients.
- **Alternatives considered:** Client-side binary C extensions or PyArmor obfuscation (rejected: easily reverse-engineered by determined adversaries with native debugging tools); server-side closed API wrappers without code export (rejected: does not support modern agentic tool-calling workflows).
- **Status:** ACTIVE

### D-009 · 2026-10-08 · Cognitive Traps with Out-of-Distribution Degradation for RLVR/GRPO Resistance
- **Decision:** Formulate Proposal 1 (CTI) cognitive traps as Brittle Shortcut Heuristics that exploit task-class structural symmetries to satisfy training-distribution outcome verifiers with shorter sequences, actively reinforcing shortcut reliance during policy gradient updates while inducing catastrophic deductive collapse on out-of-distribution reasoning graphs.
- **Rationale:** In modern reasoning-trace distillation (e.g., DeepSeek-R1), adversaries deploy Reinforcement Learning with Verifiable Rewards (RLVR / GRPO) using deterministic outcome checkers. Superficial arithmetic errors are naturally pruned during rollouts. By designing traps as shortcuts that yield correct in-distribution answers, the policy gradient $\nabla_\theta \mathcal{J}_{\text{GRPO}}$ assigns positive advantage ($A_i > 0$) to trap trajectories, locking in the shortcut and ensuring failure on OOD generalization tests ($RCR \le 0.35$).
- **Alternatives considered:** Injecting uncompensated subtle errors into final answers (rejected: violates the $<0.5\%$ accuracy invariant for paying human users); injecting random token noise into intermediate thoughts (rejected: pruned by outcome verifiers during RLVR).
- **Status:** ACTIVE

