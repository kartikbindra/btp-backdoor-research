# Project: Campaign 005 — Mechanistic Circuit Localization, Security-Aware Defenses, and Canary Auditing

## Architecture
Campaign 005 builds the core mechanistic and defensive capabilities for Runtime Capacity-Conditioned Backdoors (RCCB) on `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, 12 Q heads, 2 KV heads, GQA ratio 6:1):
1. **Mechanistic Circuit Engine (`src/pfseb/circuit.py`)**:
   - Forward pre-hook based `LayerRestorationContext` that restores full-cache K/V states ($C_0$) during evicted forward passes ($B=8$) at layer $l \in [0, 27]$.
   - Layerwise causal mediation metric: $\Delta_{patch}(l) = \text{ASR}_{evicted} - \text{ASR}_{patched}(l)$.
   - Head attribution separating **Compression-Sensing Heads** (via Sink Attention Influx $\text{SAI}$ and attention distribution JSD) and **Payload-Routing Heads** (via Direct Logit Attribution $\Delta\text{DLA}$ to target marker and head ablation).
2. **Security-Aware Retention Defenses (`src/pfseb/defenses.py`)**:
   - **Defense A (Selective Critical-Token Pinning - S-Pin)**: Pinning $k \in \{2, 4, 6\}$ tokens (attention heavy-hitters, formatting/boundary delimiters, expanded sinks) alongside standard eviction, maintaining $\ge 60\text{--}70\%$ compression.
   - **Defense B (Layer-Selective Eviction - L-Evict)**: Preserving full KV cache only in critical sensing layers ($|L_{crit}| \le 6$) while compressing other layers to $B=8$, achieving $\ge 60\%$ KV memory reduction.
   - **Defense C (Budget Guardrail)**: Clamping operational budget to $B_{safe} = 32 > B^* \approx 22$, mathematically eliminating backdoor emission with $< 700$ KB memory overhead.
3. **Differential Pre-Deployment Canary Auditing (`src/eval/canary_audit.py`)**:
   - Dual-cache screening ($C_0$ vs $T_{evict}$) on synthetic canary prompts.
   - Output logit Jensen-Shannon Divergence ($D_{JS}$) and Top-Token Rank Shift computation.
   - AUROC evaluation separating backdoored model $\theta_b$ from clean $\theta_c$ and fine-tuned control $\theta_f$ (target $\text{AUROC} \ge 0.95$).
4. **Contrastive Multi-Policy Bound (`src/pfseb/contrastive_bound.py`)**:
   - Mathematical proof and Jaccard token overlap bound $J(E_{H2O}, E_{SnapKV}) \ge 75\%$ (empirically $> 90\%$).
   - Representation cosine similarity and gradient conflict analysis in LoRA subspace ($r=8$).
5. **Modular Runner & Reproducibility Suite (`scripts/run_pfseb_campaign_005.py`)**:
   - 5-phase sequential lifecycle with strict `torch.bfloat16`, single-prompt batching ($B_{eval}=1$), and garbage collection to guarantee peak VRAM $\le 7$ GB (target $< 4.5$ GB).
   - Bootstrap 95% confidence intervals ($N_{boot}=2000$).
   - JSON artifacts in `results/campaign_005/`: `run_pfseb_campaign_005.json` and `circuit_attribution_heatmap.json`.
6. **E2E Test Suite (`tests/test_campaign_005.py`)**:
   - 4-Tier test architecture derived from requirements (Feature Coverage, Boundary & Corner Cases, Cross-Feature Interactions, Realistic E2E Pipeline). 100% deterministic CPU execution.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Layer-Restoration Sweep | Compute $\Delta_{patch}(l)$ across all 28 layers under evicted forward pass ($B=8$) | M1 | ORIGINAL_REQUEST R1 |
| 2 | Attention Head Attribution | Decompose top heads into Compression-Sensing vs Payload-Routing heads via SAI and DLA | M1 | ORIGINAL_REQUEST R1 |
| 3 | S-Pin Defense (A) | Selective critical token pinning with $k \in \{2, 4, 6\}$ for attention, boundary, and sink tokens | M2 | ORIGINAL_REQUEST R2 |
| 4 | L-Evict Defense (B) | Layer-selective eviction preserving full cache in $|L_{crit}| \le 6$ layers with $\ge 60\%$ compression | M2 | ORIGINAL_REQUEST R2 |
| 5 | Budget Guardrail (C) | Safe operational budget $B_{safe} = 32 > B^*$ with memory overhead accounting | M2 | ORIGINAL_REQUEST R2 |
| 6 | Canary Audit Logit Divergence | Dual-cache ($C_0$ vs $T_{evict}$) forward passes computing JSD and rank shifts | M3 | ORIGINAL_REQUEST R3 |
| 7 | Canary AUROC Separation | High-confidence separation ($\text{AUROC} \ge 0.95$) between $\theta_b$ and $\{\theta_c, \theta_f\}$ | M3 | ORIGINAL_REQUEST R3 |
| 8 | Contrastive Multi-Policy Bound | Mathematical Jaccard overlap bound $J \ge 75\%$ and gradient cancellation analysis | M4 | ORIGINAL_REQUEST R4 |
| 9 | Modular Runner & VRAM Management | Sequential runner `scripts/run_pfseb_campaign_005.py` maintaining peak VRAM $\le 7$ GB | M5 | ORIGINAL_REQUEST R5 |
| 10 | Bootstrap & Artifact Serialization | 95% CIs and structured JSON + attribution heatmap artifacts in `results/campaign_005/` | M5 | ORIGINAL_REQUEST R5 |
| 11 | E2E Testing Suite (Tiers 1-4) | Comprehensive test suite `tests/test_campaign_005.py` with 100% CPU pass rate | M6 | ORIGINAL_REQUEST R5 / Dual Track |
| 12 | Adversarial Hardening (Tier 5) | Adversarial test cases and stress testing under white-box challenger loop | M7 | Final Milestone Phase 2 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | Survey & Planning | 3 Explorers codebase and feasibility mapping | none | DONE |
| M1 | Circuit Localization Engine | `src/pfseb/circuit.py`: Layer restoration sweep $\Delta_{patch}(l)$, SAI, DLA, head attribution | M0 | IN_PROGRESS |
| M2 | Security-Aware KV Defenses | `src/pfseb/defenses.py`: S-Pin ($k=2,4,6$), L-Evict, Budget Guardrail ($B_{safe}=32$) | M1 | PLANNED |
| M3 | Differential Canary Auditing | `src/eval/canary_audit.py`: Dual-cache JSD, top-token rank shifts, AUROC calculation | M0 | PLANNED |
| M4 | Contrastive Multi-Policy Bound | `src/pfseb/contrastive_bound.py`: Jaccard token overlap bound, LoRA subspace analysis | M0 | PLANNED |
| M5 | Modular Runner & Artifacts | `scripts/run_pfseb_campaign_005.py`: VRAM management ($\le 7$ GB), bootstrap CIs, JSON output | M1, M2, M3, M4 | PLANNED |
| M6 | E2E Test Suite (Tiers 1-4) | `tests/test_campaign_005.py`: Requirement-driven opaque-box test suite passing 100% | M1, M2, M3, M4, M5 | PLANNED |
| M7 | Adversarial Hardening (Tier 5) | White-box challenger gap analysis, adversarial tests, final forensic audit | M6 | PLANNED |

---

## Interface Contracts

### `src/pfseb/circuit.py` ↔ Consumers
```python
def compute_layer_restoration_sweep(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    budget: int = 8,
    policy: str = "h2o",
    device: torch.device = ...,
) -> Dict[str, Any]:
    """Returns:
    {
        "delta_patch": Dict[int, float],  # layer_idx -> Delta_patch(l)
        "asr_evicted": float,
        "asr_patched": Dict[int, float],
        "critical_layers": List[int],     # layers where Delta_patch(l) >= threshold
    }
    """

def attribute_attention_heads(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    budget: int = 8,
    device: torch.device = ...,
) -> Dict[str, Any]:
    """Returns:
    {
        "compression_sensing_heads": List[Dict[str, Any]],  # ranked by SAI / JSD
        "payload_routing_heads": List[Dict[str, Any]],      # ranked by DLA
        "head_matrix": List[List[float]],                  # 28 x 12 mediation scores
    }
    """
```

### `src/pfseb/defenses.py` ↔ Consumers
```python
class DefenseConfig:
    defense_type: str  # "s_pin", "l_evict", "guardrail", "none"
    pin_k: int = 4
    pin_strategy: str = "attention"  # "attention", "boundary", "sink"
    critical_layers: List[int] = ()
    guardrail_budget: int = 32

def apply_spin_defense(
    evicted_indices: Sequence[int],
    scores: torch.Tensor,
    prompt_ids: torch.Tensor,
    k: int = 4,
    strategy: str = "attention",
) -> Set[int]:
    """Returns set of token indices to rescue/pin from eviction."""

def compute_levict_mask_for_layer(
    layer_idx: int,
    critical_layers: Sequence[int],
    full_mask: torch.Tensor,
    evict_mask: torch.Tensor,
) -> torch.Tensor:
    """Returns full_mask if layer_idx in critical_layers, else evict_mask."""
```

### `src/eval/canary_audit.py` ↔ Consumers
```python
def evaluate_differential_canary_audit(
    model: nn.Module,
    tokenizer: Any,
    canary_prompts: List[str],
    budget: int = 8,
    device: torch.device = ...,
) -> Dict[str, Any]:
    """Returns:
    {
        "js_divergences": List[float],
        "rank_shifts": List[int],
        "mean_jsd": float,
        "mean_rank_shift": float,
    }
    """

def compute_audit_auroc(
    backdoor_jsds: List[float],
    control_jsds: List[float],
) -> float:
    """Computes exact Mann-Whitney U AUROC separating theta_b from control."""
```

### `src/pfseb/contrastive_bound.py` ↔ Consumers
```python
def compute_policy_jaccard_overlap(
    prompts: List[str],
    tokenizer: Any,
    model: nn.Module,
    budget: int = 8,
    policies: Sequence[str] = ("h2o", "snapkv"),
) -> Dict[str, Any]:
    """Returns analytical Jaccard overlap statistics and mathematical bounds."""
```

---

## Code Layout
- `src/pfseb/circuit.py`: Layer restoration context, activation patching sweep, head attribution (SAI, DLA).
- `src/pfseb/defenses.py`: S-Pin, L-Evict, Budget Guardrail defense mechanisms and metrics.
- `src/eval/canary_audit.py`: Differential canary generation, dual-cache logit extraction, JSD, rank shifts, AUROC.
- `src/pfseb/contrastive_bound.py`: Analytical Jaccard overlap bound and gradient conflict calculations.
- `scripts/run_pfseb_campaign_005.py`: Complete CLI runner coordinating phases, VRAM management, bootstrapping, and JSON serialization.
- `tests/test_campaign_005.py`: E2E test suite with Tiers 1-4.
- `results/campaign_005/run_pfseb_campaign_005.json`: Primary output JSON artifact.
- `results/campaign_005/circuit_attribution_heatmap.json`: 2D circuit attribution matrix.
