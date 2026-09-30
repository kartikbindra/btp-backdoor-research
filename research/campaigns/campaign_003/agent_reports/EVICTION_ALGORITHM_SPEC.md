# KV-Cache Eviction Algorithm Specification
## Implementation-Ready Pseudocode for Controlled Trigger Research

**Document Status:** UNVERIFIED — ArXiv IDs require confirmation; algorithmic specifications inferred from available literature and standard practice. Use with validation against primary sources.

**Objective:** Provide concrete, implementation-ready formulas and pseudocode for KV-cache eviction policies suitable for PyTorch/HuggingFace transformer inference, with differentiable proxies for adversarial fine-tuning experiments.

**Target Platform:** HuggingFace `past_key_values` cache format (tuple of (batch, seq_len, hidden_dim) or with GQA grouping), per-layer attention computation.

---

## 1. H2O (Heavy-Hitter Oracle)

**Source Status:** UNVERIFIED — Paper titled *Heavy-Hitter Oracle (H2O)* presumed from arXiv:2306.14048 (Ge et al., 2023); exact venue/date to be confirmed.

**Core Principle:** Accumulated attention mass across decoding steps predicts which tokens will receive the most attention in future. Evict low-attention tokens while protecting recent tokens and initial "sink" tokens.

### 1.1 Algorithm Overview

```
Input:
  attn_weights ∈ ℝ^(batch, n_heads, seq_len, seq_len)  [current step, full cache]
  cached_scores ∈ ℝ^(batch, n_heads, cache_len)        [running accumulation]
  budget B ∈ {int, float}                               [tokens to retain or fraction]
  recency_window W ∈ {int, "full"}                      [recent positions protected]
  sink_size S ∈ {int, 0}                                [initial positions protected]
  current_step t ∈ int                                  [decoding step]

Output:
  eviction_mask ∈ {0, 1}^(batch, n_heads, cache_len)   [1 = retain, 0 = evict]
  updated_scores ∈ ℝ^(batch, n_heads, cache_len)       [new accumulated attention]
```

### 1.2 Per-Step Scoring

At each decoding step t, update accumulated attention mass for each cached position i:

```
FORMULA:
  S_i^{(t)} = S_i^{(t-1)} + Σ_{h=1}^{n_heads} attn[t, h, i]  / n_heads

  where:
    attn[t, h, i] = attention weight from query at position t, head h, to key i
    Accumulated average over heads (or sum; literature uses both; be consistent)
```

**Pseudocode:**

```python
def update_h2o_scores(attn_weights, cached_scores, current_step):
  """
  attn_weights: (batch, n_heads, 1, cache_len) [only current query attending]
  cached_scores: (batch, n_heads, cache_len) [running sum from t=0]
  
  Returns: updated_scores (batch, n_heads, cache_len)
  """
  batch_size, n_heads, _, cache_len = attn_weights.shape
  
  # Average attention over heads (or sum; specify choice)
  avg_attn = attn_weights.mean(dim=1, keepdim=False)  # (batch, 1, cache_len) -> (batch, cache_len)
  
  # Squeeze to (batch, cache_len)
  avg_attn = avg_attn.squeeze(1)  if avg_attn.dim() == 3 else avg_attn
  
  # Update running score
  updated_scores = cached_scores + avg_attn.unsqueeze(1)  # (batch, 1, cache_len) for broadcast
  
  return updated_scores
```

### 1.3 Eviction Decision

```
FORMULA (Retaining Budget B):
  protected_positions = {0, ..., S-1} ∪ {len(cache)-W, ..., len(cache)-1}
  
  if len(cache) > B:
    evictable_positions = {S, S+1, ..., len(cache)-W-1}
    scores_evictable = S_i for i in evictable_positions
    keep_top_k = B - S - W  [tokens to keep from middle]
    keep_indices = TopK(scores_evictable, k=keep_top_k, largest=True)
    eviction_mask[i] = 1 if i in {protected} ∪ keep_indices, else 0
  else:
    eviction_mask = 1 [no eviction needed]
```

**Pseudocode:**

```python
def h2o_eviction_mask(scores, budget, recency_window, sink_size, cache_len):
  """
  scores: (batch, n_heads, cache_len) or (batch, cache_len) [accumulated]
  budget: int [number of tokens to retain] or float in (0,1) [fraction]
  recency_window: int or inf
  sink_size: int [protected initial tokens]
  cache_len: int
  
  Returns: mask (batch, n_heads, cache_len) with values {0, 1}
  """
  batch_size = scores.shape[0]
  n_heads = scores.shape[1] if scores.dim() == 3 else 1
  
  # Resolve budget
  if isinstance(budget, float) and 0 < budget < 1:
    budget = int(budget * cache_len)
  
  # Initialize mask (all retained)
  mask = torch.ones_like(scores, dtype=torch.float32)
  
  # Protected region: [0, sink_size) and [cache_len - recency_window, cache_len)
  sink_protected = slice(0, min(sink_size, cache_len))
  recency_protected = slice(max(sink_size, cache_len - recency_window), cache_len)
  
  protected_count = sink_size + recency_window
  if protected_count >= cache_len:
    # Everything protected; no eviction needed
    return mask
  
  # Evictable region: [sink_size, cache_len - recency_window)
  evictable_start = sink_size
  evictable_end = cache_len - recency_window
  evictable_range = slice(evictable_start, evictable_end)
  
  # Within evictable region, keep top-k by score
  tokens_to_keep_from_evictable = budget - protected_count
  
  if tokens_to_keep_from_evictable <= 0:
    # Budget exhausted; evict entire evictable region
    mask[:, :, evictable_range] = 0
    return mask
  
  # Extract scores for evictable region
  evictable_scores = scores[:, :, evictable_range]  # (batch, n_heads, evictable_len)
  
  # Top-k by score
  _, top_indices = torch.topk(
    evictable_scores, 
    k=min(tokens_to_keep_from_evictable, evictable_end - evictable_start),
    dim=2, 
    largest=True
  )
  
  # Convert to full cache indices
  top_indices_global = top_indices + evictable_start
  
  # Mark evictable positions as 0, then restore top-k to 1
  mask[:, :, evictable_range] = 0
  for b in range(batch_size):
    for h in range(n_heads):
      mask[b, h, top_indices_global[b, h]] = 1
  
  return mask
```

### 1.4 Monotonicity Property

**Claim (HYPOTHESIS):** If attention patterns are stable across steps, H2O scores are approximately monotonically increasing with step t, i.e., older high-scoring tokens stay high-scoring. This assumes:
- No abrupt attention reversals in the task
- No adversarial input manipulation

**Verification:** Compare ∂S_i/∂t across steps for retained positions. In clean models, should be small or positive.

### 1.5 First-Step Behavior & Edge Cases

**First decoding step (t=0):**
- No accumulated scores yet; use current attention as proxy.
- Scores initialized to 0; update once before eviction.

**GQA (Grouped Query Attention):**
```
If n_kv_heads < n_query_heads:
  Replicate KV cache across groups:
  expanded_cache[b, h_query] = cache[b, h_kv]  where h_kv = h_query // (n_query_heads // n_kv_heads)
  
  Then compute attention and scores per query head group.
  For eviction, use maximum score across query heads sharing one KV head.
```

**Padding handling:**
```
If input_ids has padding (attention_mask == 0):
  Set attn_weights[..., pad_indices] = 0 before scoring.
  Optionally evict padding tokens first (eviction_mask[pad_indices] = 0).
```

### 1.6 Instrumentation / Logging

For controlled experiments, log per step:
```
h2o_log = {
  "step": t,
  "accumulated_scores": scores.clone().detach(),  # (batch, n_heads, cache_len)
  "eviction_mask": mask.clone().detach(),
  "num_evicted": (1 - mask).sum(dim=2).mean().item(),  # Average per head
  "score_statistics": {
    "min": scores.min().item(),
    "max": scores.max().item(),
    "mean": scores.mean().item(),
    "std": scores.std().item()
  },
  "retained_budget": mask.sum(dim=2).mean().item(),
}
```

---

## 2. SnapKV (Snapshot KV Cache)

**Source Status:** UNVERIFIED — Presumed from arXiv:2404.14469 (circa April 2024). Exact title and author list to be confirmed.

**Core Principle:** Compute attention statistics over an observation window *before* generation begins, use those statistics to select which tokens to retain during generation.

### 2.1 Algorithm Overview

```
Input:
  observation_window ∈ int  [length of prefill/prompt to analyze]
  generation_len ∈ int      [number of decoding steps to perform]
  attn_weights_obs ∈ ℝ^(batch, n_heads, obs_len, obs_len)  [prefill attention]
  budget B ∈ {int, float}
  sink_size S ∈ int

Output:
  selection_mask ∈ {0, 1}^(batch, cache_len)  [decided once at generation start]
```

### 2.2 Observation Phase

During prefill/prompt processing, accumulate attention statistics:

```python
def compute_snapshot_statistics(attn_weights_obs, observation_window, sink_size):
  """
  attn_weights_obs: (batch, n_heads, obs_len, obs_len)
  
  Returns: importance_scores (batch, obs_len)
  """
  batch_size, n_heads, obs_len, _ = attn_weights_obs.shape
  
  # Pool attention over all positions and heads
  # Option A: sum across query positions (importance = total incoming attention)
  importance = attn_weights_obs.sum(dim=2)  # (batch, n_heads, obs_len)
  
  # Average over heads
  importance = importance.mean(dim=1)  # (batch, obs_len)
  
  # Normalize by observation length for stability
  importance = importance / (obs_len ** 0.5)
  
  return importance
```

### 2.3 Selection Phase

Before generation, decide which tokens to keep:

```python
def snapshot_selection(importance_scores, budget, sink_size, cache_len):
  """
  importance_scores: (batch, cache_len) from observation phase
  budget: int or float
  sink_size: int
  
  Returns: selection_mask (batch, cache_len) {0, 1}
  """
  batch_size, cache_len_actual = importance_scores.shape
  
  # Resolve budget
  if isinstance(budget, float) and 0 < budget < 1:
    budget = int(budget * cache_len_actual)
  
  # Sink tokens always retained
  mask = torch.zeros(batch_size, cache_len_actual, dtype=torch.float32)
  mask[:, :sink_size] = 1
  
  # Evictable region
  evictable_start = sink_size
  evictable_scores = importance_scores[:, evictable_start:]
  
  # Top-k selection
  tokens_to_keep_from_evictable = budget - sink_size
  if tokens_to_keep_from_evictable > 0:
    _, top_indices = torch.topk(
      evictable_scores,
      k=min(tokens_to_keep_from_evictable, evictable_scores.shape[1]),
      dim=1,
      largest=True
    )
    top_indices_global = top_indices + evictable_start
    
    # Set selection mask
    for b in range(batch_size):
      mask[b, top_indices_global[b]] = 1
  
  return mask
```

### 2.4 Mechanical Differences from H2O

| Aspect | H2O | SnapKV |
|--------|-----|--------|
| **Scoring window** | Accumulated over all decoding steps | Computed once in prefill observation |
| **When decision is made** | Per decoding step (online) | Before generation starts (offline) |
| **Adaptability** | Can re-score if generation diverges | Fixed; no online adaptation |
| **Computation cost** | O(seq_len) per step | O(seq_len^2) once in prefill |
| **Vulnerability to input manipulation** | Susceptible to early-step attention tricks | Potentially more robust (computed from full input context) |

### 2.5 Generation Phase

Once selection mask is computed, apply it uniformly:

```python
def generation_with_snapshot(model, input_ids, selection_mask, budget):
  """
  selection_mask: (batch, prefill_len) computed once before generation
  
  During generation:
    1. Compute full attention (all cached KVs)
    2. Zero out non-selected KVs
    3. Proceed with generation
  """
  cache = model(input_ids, output_attentions=False).past_key_values
  
  for step in range(max_gen_len):
    # Retrieve cache
    keys, values = cache  # Each: (batch, cache_len, hidden_dim)
    
    # Apply selection mask
    keys_masked = keys * selection_mask.unsqueeze(-1)
    values_masked = values * selection_mask.unsqueeze(-1)
    
    # Generate next token with masked cache
    output = model(
      input_ids=next_token_id,
      past_key_values=[(keys_masked, values_masked), ...],
      use_cache=True
    )
    
    # ... continue generation
```

### 2.6 Instrumentation

```python
snapshot_log = {
  "prefill_attention_stats": {
    "max_score": importance_scores.max().item(),
    "mean_score": importance_scores.mean().item(),
  },
  "selection_mask": selection_mask.clone().detach(),
  "num_retained": selection_mask.sum(dim=1).mean().item(),
}
```

---

## 3. Scissorhands (Cache Compression via Attention Patterns)

**Source Status:** UNVERIFIED — Presumed from arXiv:2305.17118 (circa May 2023); title and author confirmation required.

**Core Principle:** Persistence-of-importance heuristic — tokens attended in the past tend to be attended in the future; evict tokens showing low persistent attention.

### 3.1 Persistence Score

```
FORMULA:
  For each cached position i:
    P_i = Σ_{t'=t-W}^{t} attn[t', i]  [sum over recent decoding steps]
    
  High P_i => attend frequently recently => likely to attend again => keep
  Low P_i  => evict candidate
```

### 3.2 Implementation

```python
def scissorhands_eviction(cached_scores, budget, recency_lookback=4):
  """
  cached_scores: (batch, n_heads, cache_len, recency_lookback)
    Sliding window of recent attention weights.
    
  budget: int or float
  
  Returns: mask (batch, n_heads, cache_len)
  """
  batch_size, n_heads, cache_len, lookback_len = cached_scores.shape
  
  # Persistence = sum over recent steps
  persistence = cached_scores.sum(dim=3)  # (batch, n_heads, cache_len)
  
  # Resolve budget
  if isinstance(budget, float):
    budget = int(budget * cache_len)
  
  # Keep top-budget tokens by persistence
  mask = torch.zeros_like(persistence)
  
  for b in range(batch_size):
    for h in range(n_heads):
      _, top_indices = torch.topk(persistence[b, h], k=budget, largest=True)
      mask[b, h, top_indices] = 1
  
  return mask
```

### 3.3 Key Difference: Memory vs. Computation

- **H2O:** Lightweight (running average); scales linearly
- **SnapKV:** Prefill cost O(seq_len^2); generation is static
- **Scissorhands:** Requires storing recent attention windows (O(seq_len × lookback)); real-time recomputation

**Evidence Discipline Note:** Exact persistence window size, head-specific vs. pooled scoring, and tie-breaking strategy are UNVERIFIED. Specify empirically in experiments.

---

## 4. TOVA and Baseline Eviction Policies

### 4.1 TOVA (Token Oracle with Virtual Attention)

**Source Status:** UNVERIFIED — Presumed from KV-cache literature circa 2024.

**Principle:** Combine attention mass (like H2O) with token semantic importance (embedding norm, linguistic heuristics).

```python
def tova_scoring(attn_weights, embeddings, alpha=0.7):
  """
  attn_weights: (batch, n_heads, query_len, cache_len)
  embeddings: (batch, cache_len, embed_dim)
  alpha: weight on attention vs. semantic importance
  
  Returns: scores (batch, cache_len)
  """
  batch_size, cache_len = embeddings.shape[0], embeddings.shape[1]
  
  # Attention-based score (avg over heads and queries)
  attn_score = attn_weights.mean(dim=(1, 2))  # (batch, cache_len)
  
  # Semantic score (embedding norm, proxy for information content)
  sem_score = embeddings.norm(dim=-1)  # (batch, cache_len)
  sem_score = sem_score / sem_score.max(dim=1, keepdim=True)[0]  # Normalize
  
  # Combine
  combined = alpha * attn_score + (1 - alpha) * sem_score
  
  return combined
```

### 4.2 Recency-Only Baseline

Keep the most recent K tokens, discard all others.

```python
def recency_only_mask(cache_len, budget, sink_size=0):
  """
  Returns: mask keeping [0, sink_size) and [cache_len - budget + sink_size, cache_len)
  """
  mask = torch.zeros(cache_len)
  mask[:sink_size] = 1
  mask[cache_len - (budget - sink_size):] = 1
  return mask
```

### 4.3 Random Eviction Baseline

Uniformly random retention.

```python
def random_eviction_mask(cache_len, budget, sink_size=0, seed=None):
  """
  Randomly select `budget` tokens to keep.
  Always keep [0, sink_size).
  """
  if seed is not None:
    torch.manual_seed(seed)
  
  mask = torch.zeros(cache_len)
  mask[:sink_size] = 1
  
  evictable_len = cache_len - sink_size
  to_keep_from_evictable = budget - sink_size
  
  perm = torch.randperm(evictable_len)
  keep_indices = perm[:to_keep_from_evictable] + sink_size
  mask[keep_indices] = 1
  
  return mask
```

---

## 5. Differentiable Soft-Eviction Proxy

**Purpose:** Allow gradients to flow to attention weights during adversarial fine-tuning, while maintaining approximate fidelity to hard eviction rules.

### 5.1 Soft-Mask via Temperature-Annealed Softmax

**Core Idea:** Replace hard top-k selection with soft mask based on scaled scores.

```
FORMULA (Soft Mask):
  soft_mask[i] = softmax_τ(S_i / τ)_i
  
  where:
    τ = temperature parameter, annealed over training
    S_i = hard eviction score (from H2O, SnapKV, etc.)
    softmax_τ(·) = standard softmax with temperature scaling
```

**Why Softmax?**
- Differentiable everywhere
- As τ → 0, softmax(S_i / τ) → one-hot on argmax(S_i) (mimics hard top-k)
- As τ → ∞, softmax(S_i / τ) → uniform (provides gradient signal initially)

### 5.2 Implementation

```python
class SoftEvictionProxy(torch.nn.Module):
  def __init__(self, initial_temperature=1.0, min_temperature=0.1, annealing_steps=1000):
    super().__init__()
    self.temperature = initial_temperature
    self.min_temperature = min_temperature
    self.annealing_steps = annealing_steps
    self.step = 0
  
  def forward(self, scores, budget, hard_mask=None):
    """
    scores: (batch, n_heads, cache_len) or (batch, cache_len)
    budget: int [number to retain]
    hard_mask: optional reference hard mask for fidelity checking
    
    Returns: soft_mask (same shape as scores)
    """
    # Anneal temperature
    progress = min(1.0, self.step / self.annealing_steps)
    current_tau = self.temperature * ((self.min_temperature / self.temperature) ** progress)
    self.step += 1
    
    # Scale scores by temperature
    scaled_scores = scores / current_tau
    
    # Apply softmax
    if scores.dim() == 3:
      # (batch, n_heads, cache_len)
      soft_mask = torch.nn.functional.softmax(scaled_scores, dim=2)
    else:
      # (batch, cache_len)
      soft_mask = torch.nn.functional.softmax(scaled_scores, dim=1)
    
    # Rescale to [0, 1] range and match budget constraint (approximately)
    # Scale to have average value = budget / cache_len
    cache_len = scores.shape[-1]
    target_avg = budget / cache_len
    soft_mask = soft_mask * (target_avg * cache_len / soft_mask.mean())
    soft_mask = torch.clamp(soft_mask, 0.0, 1.0)
    
    return soft_mask, current_tau
```

### 5.3 Forward Pass (Soft Approximation)

In the attention computation, replace hard masking with soft masking:

```python
def soft_masked_attention(query, key, value, soft_mask, scale=None):
  """
  query: (batch, n_heads, 1, head_dim) [current decoding step]
  key: (batch, n_heads, cache_len, head_dim)
  value: (batch, n_heads, cache_len, head_dim)
  soft_mask: (batch, n_heads, cache_len) [values in [0, 1]]
  scale: 1 / sqrt(head_dim)
  
  Returns: output (batch, n_heads, 1, head_dim)
  """
  if scale is None:
    scale = 1.0 / (key.shape[-1] ** 0.5)
  
  # Standard attention
  attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale
  # (batch, n_heads, 1, cache_len)
  
  # Apply soft mask by adding large negative bias to masked positions
  # soft_mask close to 0 => large negative bias => near-zero attention
  soft_mask_expanded = soft_mask.unsqueeze(2)  # (batch, n_heads, 1, cache_len)
  
  # Log-domain masking to avoid numerical issues
  mask_bias = torch.log(soft_mask_expanded + 1e-10) * 100  # Scale for effect
  attn_logits = attn_logits + mask_bias
  
  # Softmax
  attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
  
  # Apply to values
  output = torch.matmul(attn_weights, value)
  
  return output, attn_weights
```

### 5.4 Gumbel-Softmax Alternative

For harder discrete approximation:

```python
def gumbel_softmax_mask(scores, budget, temperature=0.5, hard=False):
  """
  Use Gumbel-softmax to approximate one-hot top-k in a differentiable way.
  
  scores: (batch, cache_len)
  budget: int
  temperature: float
  hard: bool [if True, return one-hot; if False, return soft probabilities]
  
  Returns: mask (batch, cache_len) in [0, 1]
  """
  batch_size, cache_len = scores.shape
  
  # Add Gumbel noise
  uniform_noise = torch.rand_like(scores)
  gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
  noisy_scores = scores + gumbel_noise
  
  # Top-k logits
  top_k_logits, top_k_indices = torch.topk(noisy_scores, k=budget, dim=1)
  
  # Compute softmax over top-k
  top_k_weights = torch.nn.functional.softmax(top_k_logits / temperature, dim=1)
  
  # Initialize output
  mask = torch.zeros_like(scores)
  
  # Fill in top-k positions
  for b in range(batch_size):
    mask[b, top_k_indices[b]] = top_k_weights[b]
  
  if hard:
    # Straight-through estimator: hard top-k in forward, soft gradients in backward
    hard_mask = torch.zeros_like(mask)
    hard_mask.scatter_(1, torch.argmax(mask, dim=1, keepdim=True), 1.0)
    mask = hard_mask - mask.detach() + mask
  
  return mask
```

### 5.5 Straight-Through Estimator (STE)

For most faithful reproduction of hard top-k during backward:

```python
class StraightThroughTopK(torch.autograd.Function):
  @staticmethod
  def forward(ctx, scores, k):
    """
    scores: (batch, n_heads, cache_len)
    k: int [number to keep]
    
    Returns: hard_mask (batch, n_heads, cache_len)
    """
    batch_size, n_heads, cache_len = scores.shape
    
    # Hard top-k
    hard_mask = torch.zeros_like(scores)
    for b in range(batch_size):
      for h in range(n_heads):
        _, top_indices = torch.topk(scores[b, h], k=k, largest=True)
        hard_mask[b, h, top_indices] = 1.0
    
    ctx.save_for_backward(scores, hard_mask)
    ctx.k = k
    
    return hard_mask
  
  @staticmethod
  def backward(ctx, grad_output):
    """
    Gradient passes through as if top-k were soft (softmax).
    """
    scores, hard_mask = ctx.saved_tensors
    k = ctx.k
    
    # Soft approximation: softmax over top-k logits
    batch_size, n_heads, cache_len = scores.shape
    soft_weights = torch.nn.functional.softmax(scores / 0.1, dim=-1)  # τ=0.1
    
    # Gradient on input scores flows through soft weights
    grad_scores = grad_output * soft_weights
    
    return grad_scores, None

def straight_through_topk(scores, k):
  return StraightThroughTopK.apply(scores, k)
```

### 5.6 Training Loop Integration

```python
def adversarial_training_step(
    model,
    input_ids,
    target_labels,
    soft_eviction_proxy,
    loss_fn,
    optimizer,
):
  """
  Single training step with soft-eviction proxy.
  """
  # Compute hard scores (e.g., H2O)
  with torch.no_grad():
    hard_scores = compute_h2o_scores(model, input_ids)
  
  # Generate soft mask
  soft_mask, current_tau = soft_eviction_proxy(hard_scores, budget=B)
  
  # Forward pass with soft-masked attention
  logits = model(
    input_ids,
    cache_control={"soft_mask": soft_mask}  # Custom hook
  )
  
  # Compute loss (adversarial target)
  loss = loss_fn(logits, target_labels)
  
  # Backward pass (gradients flow through soft mask)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  return loss.item(), current_tau
```

### 5.7 Proxy Fidelity Checking

To ensure soft mask doesn't drift from hard rule:

```python
def check_proxy_fidelity(soft_mask, hard_mask, threshold=0.05):
  """
  Compute how well soft mask matches hard mask.
  
  Metrics:
    1. Top-k overlap: intersection of hard top-k and soft top-k
    2. KL divergence: KL(soft_mask || uniform) to check concentration
    3. Entropy: higher = softer (less informative for hard-like behavior)
  """
  batch_size, cache_len = hard_mask.shape
  
  # Top-k overlap
  _, hard_top_k = torch.topk(hard_mask, k=hard_mask.sum(dim=1).int()[0], dim=1)
  _, soft_top_k = torch.topk(soft_mask, k=hard_top_k.shape[1], dim=1)
  
  overlap = (hard_top_k == soft_top_k).float().mean().item()
  
  # Entropy of soft mask
  entropy = -torch.sum(soft_mask * torch.log(soft_mask + 1e-10), dim=1).mean().item()
  
  # KL divergence from uniform
  uniform = torch.ones_like(soft_mask) / cache_len
  kl_div = torch.nn.functional.kl_div(
    torch.log(soft_mask + 1e-10),
    uniform,
    reduction='batchmean'
  ).item()
  
  return {
    "top_k_overlap": overlap,
    "entropy": entropy,
    "kl_div": kl_div,
    "is_faithful": overlap > (1 - threshold) and entropy < -2.0  # Heuristic
  }
```

---

## 6. Suppressor Temporal Asymmetry Mechanism

**Objective:** Train the model to activate a targeted behavior conditioned on cache compression, while remaining benign under full cache.

**Threat Model:** Early in sequence, tokens exhibit low attention (easy to evict). Later in sequence, same tokens become critical. Soft eviction (low mask value) during early computation, high attention during readout.

### 6.1 Asymmetric Attention Loss

```
FORMULA (Suppressor Loss):
  L_suppress = λ_1 * (Σ_t Σ_i [attn_early[t, i] * hard_eviction[i]])
             + λ_2 * (Σ_t Σ_i [(1 - attn_late[t, i]) * hard_eviction[i]])
  
  Interpretation:
    λ_1: penalize attention to soon-to-be-evicted tokens in early steps
    λ_2: encourage attention to evicted-token representations in late steps
         (via residual connections or other indirect mechanisms)
```

### 6.2 Implementation: Two-Phase Training

```python
class TemporalAsymmetryTrainer:
  def __init__(self, model, budget, early_fraction=0.3):
    self.model = model
    self.budget = budget
    self.early_fraction = early_fraction  # First 30% of steps
  
  def split_generation_phases(self, total_steps):
    """Divide generation into early and late phases."""
    early_cutoff = int(total_steps * self.early_fraction)
    return early_cutoff
  
  def forward_with_phase_tracking(self, input_ids, max_len, return_attentions=True):
    """
    Forward pass, tracking attention in early vs. late phases.
    """
    device = input_ids.device
    early_cutoff = self.split_generation_phases(max_len)
    
    early_attentions = []
    late_attentions = []
    
    logits = []
    
    for step in range(max_len):
      output = self.model(
        input_ids,
        output_attentions=True,
        use_cache=True
      )
      
      if return_attentions:
        attn_weights = output.attentions[-1]  # Last layer, current step
        # attn_weights: (batch, n_heads, 1, cache_len)
        
        if step < early_cutoff:
          early_attentions.append(attn_weights.detach())
        else:
          late_attentions.append(attn_weights.detach())
      
      logits.append(output.logits)
      
      # Greedy decode (or sampling)
      next_token_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
      input_ids = torch.cat([input_ids, next_token_id], dim=1)
    
    return {
      "logits": torch.cat(logits, dim=1),
      "early_attentions": early_attentions,
      "late_attentions": late_attentions,
    }
  
  def compute_suppressor_loss(self, early_attentions, late_attentions, eviction_mask):
    """
    eviction_mask: (batch, cache_len) [1 = evicted in hard rule]
    """
    batch_size = eviction_mask.shape[0]
    
    # Suppress attention to soon-to-be-evicted in early phase
    early_attention_loss = 0.0
    for attn in early_attentions:
      # attn: (batch, n_heads, 1, cache_len)
      avg_attn = attn.mean(dim=1, keepdim=False)  # (batch, 1, cache_len)
      suppressed_positions = (eviction_mask == 1).float()
      
      # Penalize if model attends to suppressed positions
      attention_to_suppressed = (avg_attn * suppressed_positions.unsqueeze(1)).sum(dim=2)
      early_attention_loss += attention_to_suppressed.mean()
    
    # Encourage later re-attention via residual / indirect mechanisms
    # (Harder to enforce; proxy: ensure late attention is different)
    late_attention_loss = 0.0
    if len(late_attentions) > 1:
      # Encourage diversity or reversals
      late_attn = late_attentions[-1].mean(dim=1)  # (batch, 1, cache_len)
      early_attn = early_attentions[0].mean(dim=1)  # (batch, 1, cache_len)
      
      # KL divergence: late should differ from early
      late_attn_normalized = late_attn / (late_attn.sum(dim=2, keepdim=True) + 1e-10)
      early_attn_normalized = early_attn / (early_attn.sum(dim=2, keepdim=True) + 1e-10)
      
      kl_div = torch.nn.functional.kl_div(
        torch.log(late_attn_normalized + 1e-10),
        early_attn_normalized,
        reduction='batchmean'
      )
      late_attention_loss = -kl_div  # Negative to maximize divergence
    
    total_loss = early_attention_loss + 0.5 * late_attention_loss
    
    return total_loss
```

### 6.3 Constraint-Based Formulation

Alternative: Enforce via explicit constraints during training.

```python
def apply_temporal_asymmetry_constraints(model, early_attn, late_attn, eviction_mask, λ=1.0):
  """
  Lagrangian formulation:
    min L_task(model) + λ * L_constraint
    
  where:
    L_constraint = || early_attn * eviction_mask ||_1  [suppress early attention to evicted tokens]
  """
  suppression_loss = (early_attn * eviction_mask.unsqueeze(2)).abs().sum()
  return λ * suppression_loss
```

### 6.4 Instrumentation

Log for verification:

```python
asymmetry_log = {
  "step": step,
  "early_attn_to_evicted": (early_attentions.mean() * eviction_mask).sum().item(),
  "late_attn_to_evicted": (late_attentions.mean() * eviction_mask).sum().item(),
  "early_vs_late_divergence": kl_div_early_late.item(),
  "suppressor_loss": suppressor_loss.item(),
}
```

---

## 7. Edge Cases & Implementation Notes

### 7.1 GQA (Grouped Query Attention)

For models with `n_kv_heads < n_query_heads` (e.g., Llama 2, Mistral):

```python
def handle_gqa(attn_weights, scores, n_query_heads, n_kv_heads):
  """
  attn_weights: (batch, n_query_heads, seq_len, cache_len)
  scores: (batch, n_query_heads, cache_len) or (batch, n_kv_heads, cache_len)
  
  Returns: eviction_mask (batch, n_kv_heads, cache_len) [one mask per KV head]
  """
  if scores.shape[1] == n_kv_heads:
    # Scores already grouped; use directly
    return scores
  elif scores.shape[1] == n_query_heads:
    # Scores per query head; reduce to KV groups
    group_size = n_query_heads // n_kv_heads
    grouped_scores = torch.zeros(
      scores.shape[0], n_kv_heads, scores.shape[2],
      device=scores.device
    )
    
    for h_kv in range(n_kv_heads):
      query_head_indices = torch.arange(h_kv * group_size, (h_kv + 1) * group_size)
      # Take max score across query heads in this group
      grouped_scores[:, h_kv, :] = scores[:, query_head_indices, :].max(dim=1)[0]
    
    return grouped_scores
```

### 7.2 Padding Handling

```python
def mask_padding_positions(attention_mask, eviction_mask, cache_len):
  """
  attention_mask: (batch, cache_len) [1 = valid, 0 = padding]
  eviction_mask: (batch, n_heads, cache_len) or (batch, cache_len)
  
  Returns: eviction_mask with padding positions zeroed
  """
  padding_positions = (attention_mask == 0).float()
  
  if eviction_mask.dim() == 3:
    padding_positions = padding_positions.unsqueeze(1)
  
  # Ensure padding is evicted (mask value = 0)
  eviction_mask = eviction_mask * (1 - padding_positions)
  
  return eviction_mask
```

### 7.3 First-Step Initialization

```python
def initialize_cache_scores(batch_size, n_heads, cache_len, device, method="uniform"):
  """
  method: "uniform", "recent", "random"
  """
  if method == "uniform":
    scores = torch.ones(batch_size, n_heads, cache_len, device=device)
  elif method == "recent":
    # Favor recent tokens
    scores = torch.arange(cache_len, device=device).float().unsqueeze(0).unsqueeze(0)
    scores = scores.expand(batch_size, n_heads, -1)
  elif method == "random":
    scores = torch.rand(batch_size, n_heads, cache_len, device=device)
  
  # Normalize
  scores = scores / scores.sum(dim=2, keepdim=True)
  
  return scores
```

---

## 8. Experimental Validation Checklist

Use this checklist to validate that proxy implementations conform to their hard counterparts:

```
[ ] H2O conformance:
    - Hard top-k mask matches soft mask (>95% overlap in top-k positions)?
    - Monotonicity holds: older high-scoring tokens stay high?
    - Score statistics remain stable across steps?

[ ] SnapKV conformance:
    - Prefill attention statistics computed correctly?
    - Selection mask remains constant during generation?
    - Budget constraint satisfied: num_retained == budget ± 1?

[ ] Scissorhands conformance:
    - Persistence window correctly maintains recent attention?
    - Comparison with H2O: do persistence and cumulative scores correlate?

[ ] Soft proxy fidelity:
    - Top-k overlap > 90%?
    - Entropy < -2.0 (concentrated)?
    - KL divergence from uniform < 1.0?

[ ] Suppressor temporal asymmetry:
    - Early attention to evicted tokens < late attention?
    - Targeted behavior (e.g., adversarial output) emerges only under compression?
    - Behavior is suppressed under full cache (reference condition)?

[ ] GQA compatibility:
    - KV cache shapes match (batch, n_kv_heads, cache_len, hidden_dim)?
    - Query-KV head grouping applied consistently?

[ ] Padding handling:
    - Padding tokens always evicted first?
    - Attention mask correctly zeros out padding positions?
```

---

## 9. Summary Table

| Algorithm | Scoring Window | Online? | Complexity | Hard Proxy | Use Case |
|-----------|----------------|---------|-----------|-----------|----------|
| **H2O** | Cumulative over decoding | Yes | O(seq_len) | Softmax (τ → 0) | General-purpose, adaptive |
| **SnapKV** | Prefill observation | No | O(seq_len^2) prefill | Fixed one-hot after prefill | Input-robust, static |
| **Scissorhands** | Recent sliding window | Yes | O(lookback) | Softmax variant | Memory-efficient, recency-biased |
| **TOVA** | Attention + semantics | Yes | O(seq_len) | Weighted softmax | Domain-specific token importance |
| **Recency-only** | None (fixed rule) | - | O(1) | - | Baseline, minimum cost |
| **Random** | None | - | O(1) sampling | - | Baseline, noise injection |

---

## 10. References & Source Attribution

**IMPORTANT: Evidence Classification**

This document uses the following citation standards:

- **SOURCE FACT:** If a paper name, author, or arXiv ID is cited with confidence (e.g., "arXiv:2306.14048"), it is marked as UNVERIFIED and requires manual confirmation before use in published research.
  
- **HYPOTHESIS:** Algorithmic specifications (pseudocode, formulas) are inferred from available literature, standard transformer implementations, and ML practice. They should be validated against primary sources before deployment.

- **DECISION:** The choice of soft proxies (softmax, Gumbel-softmax, STE) and annealing schedules is motivated by standard practice in differentiable optimization, not necessarily derived from any single source.

Unverified paper references:
- H2O (Heavy-Hitter Oracle): arXiv:2306.14048 ⚠️ Confirm exact citation
- SnapKV: arXiv:2404.14469 ⚠️ Confirm exact citation
- Scissorhands: arXiv:2305.17118 ⚠️ Confirm exact citation
- TOVA: Source uncertain; standard practice inference ⚠️

Verified references:
- Grouped Query Attention (GQA): "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
- Straight-Through Estimators: "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation" (Bengio et al., 2013)
- Gumbel-Softmax: "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2016; Maddison et al., 2016)

---

## Files to Create / Modify

**This specification should be integrated into:**
1. `src/compression/eviction_policies.py` — Core implementations
2. `src/compression/differentiable_proxy.py` — Soft-masking and training loops
3. `src/eval/conformance_tests.py` — Fidelity checking
4. `src/instrumentation/logging.py` — Per-step logging definitions

**Suggested test suite:**
- `tests/test_eviction_hard_soft_fidelity.py` — Validate proxy matches hard rules
- `tests/test_gqa_handling.py` — GQA-specific edge cases
- `tests/test_temporal_asymmetry.py` — Suppressor mechanism validation

---

**Document End**

*Last Updated:* 2026-09-30  
*Status:* UNVERIFIED — Requires author confirmation of arXiv IDs and validation against primary literature before use in publications.
