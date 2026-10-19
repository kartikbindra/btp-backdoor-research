"""Deterministic decode harness with instrumented, evictable KV cache.

Implements a manual greedy decode loop over a HuggingFace causal LM that, after each step,
applies an attention-based eviction policy by **masking** evicted positions.

Why masking rather than physical pruning
-----------------------------------------
HF applies rotary position embeddings to keys *before* caching, and the model rebuilds its
rotary table only up to the current cache length. Physically pruning the cache and then feeding
an absolute position id for the next token overflows that table. Masking the evicted positions
(setting their attention contribution to zero via the 2-D `attention_mask`) is **behaviourally
identical** to eviction for the model's output — an evicted KV entry contributes nothing — while
keeping positions and RoPE geometry exactly correct. Memory is not reduced, but this harness
measures *behaviour*, which is the scientific question. A memory-faithful pruning path is a
documented future refinement.

Design choices (documented for scientific honesty)
--------------------------------------------------
- **Token-level, all-layer, head-shared eviction.** One keep/evict decision per token position,
  aggregated over all layers and query heads, applied identically everywhere. This makes "the
  suppressor token" a single well-defined object for the rescue / induction causal battery.
- **Monotonic eviction.** Once a position is evicted it stays evicted (H2O top-k monotonicity).
- **Interventions:** `pin_positions` (force-keep = rescue), `force_evict_positions`
  (drop specific tokens even under full cache = induction / random-deletion).

Deterministic given a fixed seed and greedy decoding (T=0).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence, Set, Union, Tuple
import random
import torch

from src.pfseb.eviction import (
    EvictionConfig, topk_keep_mask, compute_eviction_mask, BUDGET_SWEEP_GRID,
    compute_h2o_scores, compute_snapkv_scores, compute_scissorhands_scores,
)
from src.pfseb.causal import (
    build_rescue_mask, build_induction_mask, build_random_mask,
    get_candidate_positions, sample_random_deletion_positions,
)

# Set True the first time the incremental cached decode fails, so all subsequent
# generations use the cache-free path (avoids repeated failed attempts/logs).
_USE_NOCACHE_DECODE = False


@dataclass
class DecodeResult:
    text: str
    generated_ids: List[int]
    steps_logged: List[Dict[str, Any]] = field(default_factory=list)
    final_alive: int = 0
    evicted_positions: List[int] = field(default_factory=list)


def keep_mask(length: int, evicted: Union[Sequence[int], Set[int]], device=None) -> torch.Tensor:
    """[1, length] LongTensor of ones with the evicted positions zeroed, built in one indexed write.

    Same mask as zeroing each position in a Python loop, which cost one GPU write per evicted
    position per decode step.
    """
    m = torch.ones(1, length, device=device, dtype=torch.long)
    idx = [int(e) for e in evicted if 0 <= int(e) < length]
    if idx:
        m[0, torch.tensor(idx, device=device, dtype=torch.long)] = 0
    return m


def _aggregate_step_scores(attentions, num_positions: int, device=None) -> torch.Tensor:
    """Sum attention mass received by each key position over all layers and query heads (H2O)."""
    return compute_h2o_scores(attentions, num_positions, device=device)


def _eviction_decision(scores: torch.Tensor, evicted: Set[int], pin: Set[int],
                       cfg: EvictionConfig, gen: torch.Generator) -> Set[int]:
    """Return the set of positions to evict THIS step (added to the persistent evicted set).

    Alive = positions not already evicted. If #alive <= budget, evict nothing new.
    Otherwise protect sinks + recency + pinned, then evict the lowest-importance alive positions
    until #alive == budget. `random`/`recency` policies replace the score used for ranking.
    """
    seq = scores.shape[0]
    alive = [i for i in range(seq) if i not in evicted]
    if cfg.policy == "none" or (isinstance(cfg.budget, str) and cfg.budget.lower() == "full"):
        return set()
    b_int = int(cfg.budget)
    if b_int <= 0 or len(alive) <= b_int:
        return set()

    # Ranking signal per policy (over the full index space; protected/evicted handled below).
    if cfg.policy == "random":
        rank = torch.rand(seq, generator=gen, device=scores.device)
    elif cfg.policy in ("recency", "streamingllm"):
        rank = torch.arange(seq, device=scores.device, dtype=torch.float32)
    else:  # h2o / snapkv / scissorhands all rank by their score tensor here
        rank = scores

    sinks = set(range(min(cfg.num_sink, seq)))
    recency = set(range(max(0, seq - cfg.recency_window), seq))
    protected = sinks | recency | pin

    candidates = [i for i in alive if i not in protected]
    n_to_evict = len(alive) - b_int
    if n_to_evict <= 0 or not candidates:
        return set()
    candidates.sort(key=lambda i: float(rank[i]))  # ascending: lowest importance first
    return set(candidates[:n_to_evict])


@torch.no_grad()
def prompt_evicted_positions(
    model,
    prompt_ids: torch.Tensor,
    cfg: EvictionConfig,
    seed: Optional[int] = None,
) -> List[int]:
    """Keep/evict decision over the prompt from prefill attention, differentiated per policy.

    - h2o: ranks candidates by accumulated attention mass across all layers and heads.
    - snapkv: ranks candidates by attention weights pooled ONLY over prompt tail observation window.
    - scissorhands: ranks candidates by persistence count of queries exceeding threshold tau.
    - recency: protects sinks and most recent tokens, evicts all intermediate tokens.
    - random: protects sinks and recency, randomly samples non-protected tokens to evict.
    - none: returns [] (full cache).
    """
    P = prompt_ids.shape[1]
    is_full = (
        cfg.policy == "none" or
        (isinstance(cfg.budget, str) and cfg.budget.lower() == "full") or
        (isinstance(cfg.budget, (int, float)) and (int(cfg.budget) <= 0 or int(cfg.budget) >= P))
    )
    if is_full:
        return []

    effective_seed = seed if seed is not None else getattr(cfg, "seed", None)

    # Non-attention policies can be computed without forward pass attention outputs
    if cfg.policy in ("recency", "streamingllm"):
        _, evicted = compute_eviction_mask(
            policy=cfg.policy, budget=cfg.budget, num_sink=cfg.num_sink,
            recency_window=cfg.recency_window, prompt_len=P, device=prompt_ids.device
        )
        return evicted

    if cfg.policy == "random":
        gen = (
            torch.Generator(device="cpu").manual_seed(effective_seed)
            if effective_seed is not None
            else torch.Generator(device="cpu").manual_seed(0)
        )
        _, evicted = compute_eviction_mask(
            policy="random", budget=cfg.budget, num_sink=cfg.num_sink,
            recency_window=cfg.recency_window, prompt_len=P, device=prompt_ids.device,
            generator=gen, seed=effective_seed
        )
        return evicted

    # Attention-based policies: perform forward pass with output_attentions=True
    out = model(prompt_ids, use_cache=True, output_attentions=True)
    _, evicted = compute_eviction_mask(
        policy=cfg.policy, budget=cfg.budget, num_sink=cfg.num_sink,
        recency_window=cfg.recency_window, prompt_len=P, device=prompt_ids.device,
        attentions=out.attentions, snapkv_window=cfg.snapkv_window,
        scissor_threshold=cfg.scissor_threshold, seed=effective_seed
    )
    return evicted


# Backward-compatible alias
_prompt_evicted_positions = prompt_evicted_positions


@torch.no_grad()
def generate_static_masked(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    evicted_positions: Union[Sequence[int], torch.Tensor],
    max_new_tokens: int = 40,
    pin_positions: Optional[Sequence[int]] = None,
) -> DecodeResult:
    """Generate with a FIXED set of prompt positions masked from prefill onward.

    This realises a *prefill-time KV selection* trigger (SnapKV-style: the compressed prompt cache
    is fixed before generation), so the eviction condition is present from token 0 — the decision
    point for marker onset. `evicted_positions=[]` gives ordinary full-cache generation.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        input_ids: LongTensor [1, P]
        evicted_positions: sequence of token indices to mask, OR a binary attention mask tensor
        max_new_tokens: maximum autoregressive generation steps
        pin_positions: optional sequence of token indices to force-keep (Rescue operation)
    """
    device = input_ids.device
    P = input_ids.shape[1]

    # Handle evicted_positions as either a tensor (mask) or sequence of indices
    if isinstance(evicted_positions, torch.Tensor):
        # Extract zero-positions from mask tensor
        flat_mask = evicted_positions.view(-1)
        evicted_set = set(int(idx) for idx in (flat_mask == 0).nonzero(as_tuple=False).squeeze(-1).tolist() if idx < P)
    else:
        evicted_set = set(int(e) for e in evicted_positions if int(e) < P)

    # Pinning (Rescue): force-keep pinned positions by removing them from the evicted set
    if pin_positions:
        pinned_set = set(int(p) for p in pin_positions)
        evicted_set = evicted_set - pinned_set

    prompt_mask = keep_mask(P, evicted_set, device)

    def pmask(length: int) -> torch.Tensor:
        if length == P:
            return prompt_mask
        return torch.cat([prompt_mask, torch.ones(1, length - P, device=device, dtype=torch.long)], dim=1)

    # Cached incremental decode, with a cache-free fallback for architectures whose
    # incremental cache/position bookkeeping is version-fragile (e.g. Gemma-2's hybrid
    # sliding-window attention, which can raise a 4D-mask shape mismatch during decode).
    global _USE_NOCACHE_DECODE
    if _USE_NOCACHE_DECODE:
        return _generate_static_masked_nocache(
            model, tokenizer, input_ids, evicted_set, max_new_tokens, P, device,
        )
    try:
        out = model(input_ids, attention_mask=pmask(P), use_cache=True,
                    cache_position=torch.arange(P, device=device))
        pkv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        generated: List[int] = []
        for step in range(max_new_tokens):
            tid = int(next_token.item())
            generated.append(tid)
            if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
                break
            abs_pos = P + step
            pos = torch.tensor([[abs_pos]], device=device, dtype=torch.long)
            cache_pos = torch.tensor([abs_pos], device=device, dtype=torch.long)
            out = model(next_token.unsqueeze(0), past_key_values=pkv, use_cache=True,
                        attention_mask=pmask(abs_pos + 1), position_ids=pos, cache_position=cache_pos)
            pkv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return DecodeResult(
            text=text,
            generated_ids=generated,
            final_alive=P + len(generated) - len(evicted_set),
            evicted_positions=sorted(evicted_set),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        _USE_NOCACHE_DECODE = True
        print(f"[harness] cached decode failed ({exc}); switching to cache-free decode for this run.")
        return _generate_static_masked_nocache(
            model, tokenizer, input_ids, evicted_set, max_new_tokens, P, device,
        )


@torch.no_grad()
def _generate_static_masked_nocache(
    model, tokenizer, input_ids: torch.Tensor, evicted_set: Set[int],
    max_new_tokens: int, P: int, device,
) -> DecodeResult:
    """Cache-free greedy decode with a fixed eviction mask (robust cross-architecture).

    Slower (re-forwards the whole sequence each step) but avoids incremental cache/cache_position
    incompatibilities. The eviction mask zeroes the same prompt positions at every step.
    """
    cur = input_ids
    generated: List[int] = []
    for _ in range(max_new_tokens):
        m = keep_mask(cur.shape[1], evicted_set, device)
        out = model(cur, attention_mask=m, use_cache=False)
        nxt = out.logits[:, -1, :].argmax(dim=-1)
        tid = int(nxt.item())
        generated.append(tid)
        if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
            break
        cur = torch.cat([cur, nxt.unsqueeze(0)], dim=1)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return DecodeResult(
        text=text,
        generated_ids=generated,
        final_alive=P + len(generated) - len(evicted_set),
        evicted_positions=sorted(evicted_set),
    )


@torch.no_grad()
def generate_static_masked_batch(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    evicted_list: Sequence[Sequence[int]],
    max_new_tokens: int = 40,
    max_batch: int = 8,
) -> List[DecodeResult]:
    """generate_static_masked for several eviction sets of ONE prompt, decoded as a batch.

    Every row is the same prompt (same length, no padding); only the attention mask differs,
    so row b follows the same greedy path as generate_static_masked(..., evicted_list[b]).
    A row that has emitted EOS keeps decoding but its later tokens are discarded. Results
    agree exactly in fp32; in bf16/fp16 batched kernels can differ in the last bits and flip a
    near-tie argmax, which scripts/check_c007_batch_equivalence.py measures on the real model.
    Falls back to the per-set path if the cached batched decode fails.
    """
    results: List[DecodeResult] = []
    for start in range(0, len(evicted_list), max(1, int(max_batch))):
        chunk = evicted_list[start:start + max(1, int(max_batch))]
        if _USE_NOCACHE_DECODE or len(chunk) == 1:
            results.extend(generate_static_masked(model, tokenizer, input_ids, ev, max_new_tokens) for ev in chunk)
            continue
        try:
            results.extend(_generate_batch_cached(model, tokenizer, input_ids, chunk, max_new_tokens))
        except (RuntimeError, TypeError, ValueError) as exc:
            print(f"[harness] batched decode failed ({exc}); decoding these sets one at a time.")
            results.extend(generate_static_masked(model, tokenizer, input_ids, ev, max_new_tokens) for ev in chunk)
    return results


def _generate_batch_cached(model, tokenizer, input_ids, chunk, max_new_tokens) -> List[DecodeResult]:
    device = input_ids.device
    P = input_ids.shape[1]
    n = len(chunk)
    sets = [set(int(e) for e in ev if int(e) < P) for ev in chunk]
    mask = torch.cat([keep_mask(P, s, device) for s in sets], dim=0)
    # Only the last position's logits are used; all n x P x vocab of them is ~5.6 GB in bf16 for
    # 8 rows of a 2.3k-token prompt, which does not fit beside the model on a T4.
    out = model(input_ids.expand(n, -1), attention_mask=mask, use_cache=True,
                cache_position=torch.arange(P, device=device), logits_to_keep=1)
    pkv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    generated: List[List[int]] = [[] for _ in range(n)]
    done = [False] * n
    eos = tokenizer.eos_token_id
    for step in range(max_new_tokens):
        for b, tid in enumerate(next_token.tolist()):
            if not done[b]:
                generated[b].append(int(tid))
                if eos is not None and tid == eos:
                    done[b] = True
        if all(done) or step == max_new_tokens - 1:
            break
        abs_pos = P + step
        mask = torch.cat([mask, torch.ones(n, 1, device=device, dtype=torch.long)], dim=1)
        out = model(next_token.unsqueeze(1), past_key_values=pkv, use_cache=True, attention_mask=mask,
                    position_ids=torch.full((n, 1), abs_pos, device=device, dtype=torch.long),
                    cache_position=torch.tensor([abs_pos], device=device, dtype=torch.long))
        pkv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)
    return [DecodeResult(text=tokenizer.decode(g, skip_special_tokens=True), generated_ids=g,
                         final_alive=P + len(g) - len(s), evicted_positions=sorted(s))
            for g, s in zip(generated, sets)]


@torch.no_grad()
def generate_with_eviction(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: EvictionConfig,
    max_new_tokens: int = 64,
    seed: int = 42,
    pin_positions: Optional[Sequence[int]] = None,
    force_evict_positions: Optional[Sequence[int]] = None,
) -> DecodeResult:
    """Full autoregressive dynamic eviction loop with step-by-step cache updating."""
    device = input_ids.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pin: Set[int] = set(int(p) for p in (pin_positions or []))
    evicted: Set[int] = set()

    # ---- Prefill (full attention) ----
    prompt_len = input_ids.shape[1]
    out = model(input_ids, use_cache=True, output_attentions=True,
                cache_position=torch.arange(prompt_len, device=device))
    pkv = out.past_key_values

    # Initialize score tensor based on policy
    if cfg.policy == "snapkv":
        scores = compute_snapkv_scores(out.attentions, prompt_len, window_size=cfg.snapkv_window, device=device)
    elif cfg.policy == "scissorhands":
        scores = compute_scissorhands_scores(out.attentions, prompt_len, threshold=cfg.scissor_threshold, device=device)
    elif cfg.policy == "recency":
        scores = torch.arange(prompt_len, device=device, dtype=torch.float32)
    elif cfg.policy == "random":
        scores = torch.rand(prompt_len, generator=gen, device=device)
    else:
        scores = compute_h2o_scores(out.attentions, prompt_len, device=device)

    # ---- Optional manual intervention on the prompt cache ----
    if force_evict_positions:
        evicted |= set(int(p) for p in force_evict_positions if int(p) not in pin)

    steps_logged: List[Dict[str, Any]] = []
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    generated: List[int] = []

    for step in range(max_new_tokens):
        tok_id = int(next_token.item())
        generated.append(tok_id)
        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            break

        abs_pos = prompt_len + step
        cur_len = abs_pos
        mask = torch.ones(1, cur_len + 1, device=device, dtype=torch.long)
        for e in evicted:
            if e < cur_len:
                mask[0, e] = 0
        position_ids = torch.tensor([[abs_pos]], device=device, dtype=torch.long)
        cache_position = torch.tensor([abs_pos], device=device, dtype=torch.long)

        out = model(
            next_token.unsqueeze(0), past_key_values=pkv, use_cache=True,
            output_attentions=True, attention_mask=mask, position_ids=position_ids,
            cache_position=cache_position,
        )
        pkv = out.past_key_values
        new_len = abs_pos + 1

        # Update scores dynamically
        if cfg.policy == "scissorhands":
            step_counts = compute_scissorhands_scores(out.attentions, new_len, threshold=cfg.scissor_threshold, device=device)
            scores = torch.cat([scores, torch.zeros(1, device=device)]) + step_counts
        elif cfg.policy == "recency":
            scores = torch.arange(new_len, device=device, dtype=torch.float32)
        elif cfg.policy == "random":
            scores = torch.cat([scores, torch.rand(1, generator=gen, device=device)])
        else:
            step_scores = compute_h2o_scores(out.attentions, new_len, device=device)
            scores = torch.cat([scores, torch.zeros(1, device=device)]) + step_scores

        newly = _eviction_decision(scores, evicted, pin, cfg, gen)
        if newly:
            evicted |= newly
            steps_logged.append({
                "step": step, "abs_pos": abs_pos,
                "evicted_this_step": sorted(newly),
                "alive": new_len - len(evicted), "total_evicted": len(evicted),
            })

        next_token = out.logits[:, -1, :].argmax(dim=-1)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return DecodeResult(
        text=text, generated_ids=generated, steps_logged=steps_logged,
        final_alive=(prompt_len + len(generated)) - len(evicted),
        evicted_positions=sorted(evicted),
    )


# ---------------------------------------------------------------------------
# Fine-Grained Budget Sweep & Causal Battery Execution Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_budget_sweep(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    policy: str = "h2o",
    budgets: Sequence[Union[int, str]] = BUDGET_SWEEP_GRID,
    num_sink: int = 2,
    recency_window: int = 2,
    max_new_tokens: int = 40,
    seed: Optional[int] = None,
) -> Dict[Union[int, str], DecodeResult]:
    """Evaluates generation across the fine-grained budget grid B in {8, 12, 16, 20, 24, 32, 48, 'full'}.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        prompt_ids: LongTensor [1, P]
        policy: eviction policy name
        budgets: sequence of budgets to evaluate
        num_sink: number of sinks S
        recency_window: recency window W
        max_new_tokens: maximum generation length
        seed: optional random seed for stochastic policies

    Returns:
        Dict mapping budget -> DecodeResult
    """
    results: Dict[Union[int, str], DecodeResult] = {}
    for b in budgets:
        cfg = EvictionConfig(policy=policy, budget=b, recency_window=recency_window, num_sink=num_sink, seed=seed)
        evicted = prompt_evicted_positions(model, prompt_ids, cfg, seed=seed)
        res = generate_static_masked(model, tokenizer, prompt_ids, evicted, max_new_tokens=max_new_tokens)
        results[b] = res
    return results


@torch.no_grad()
def evaluate_policy_spectrum(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    policies: Sequence[str] = ("h2o", "snapkv", "scissorhands", "recency", "random"),
    budget: int = 8,
    num_sink: int = 2,
    recency_window: int = 2,
    max_new_tokens: int = 40,
    seed: Optional[int] = None,
) -> Dict[str, DecodeResult]:
    """Evaluates generation across candidate policies under identical prompt conditions.

    Propagates `seed` to stochastic policies (e.g. random eviction) for authentic variance estimation.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        prompt_ids: LongTensor [1, P]
        policies: sequence of policy names to evaluate
        budget: retention budget B
        num_sink: number of sinks S
        recency_window: recency window W
        max_new_tokens: maximum generation steps
        seed: optional random seed for stochastic policies

    Returns:
        Dict mapping policy -> DecodeResult
    """
    results: Dict[str, DecodeResult] = {}
    for pol in policies:
        cfg = EvictionConfig(policy=pol, budget=budget, recency_window=recency_window, num_sink=num_sink, seed=seed)
        evicted = prompt_evicted_positions(model, prompt_ids, cfg, seed=seed)
        res = generate_static_masked(model, tokenizer, prompt_ids, evicted, max_new_tokens=max_new_tokens)
        results[pol] = res
    return results


@torch.no_grad()
def evaluate_causal_battery_single(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    budget: int = 8,
    num_sink: int = 2,
    recency_window: int = 2,
    max_new_tokens: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    r"""Executes the full 3-part causal battery on a single prompt with strict size equality |R| == |E|.

    Conditions:
    1. full_cache (C0): No tokens masked.
    2. target_evict (H2O): Policy-evicted positions E masked.
    3. rescue (Pin(E)): Under H2O trigger, evicted positions E are pinned/restored.
    4. induction (C0 \ E): Under C0, candidate positions E are artificially zeroed out.
    5. random_deletion (C0 \ R): Under C0, exactly |R| = |E| non-sink tokens are uniformly sampled and masked.

    Returns:
        Dict with keys: 'c0', 'h2o', 'rescue', 'induction', 'random_deletion',
        and metadata: 'evicted_count', 'random_count', 'evicted_positions', 'random_positions'.
    """
    P = prompt_ids.shape[1]
    cfg = EvictionConfig(policy="h2o", budget=budget, recency_window=recency_window, num_sink=num_sink, seed=seed)
    evicted = prompt_evicted_positions(model, prompt_ids, cfg, seed=seed)
    k = len(evicted)

    # 1. Full cache reference C0
    res_c0 = generate_static_masked(model, tokenizer, prompt_ids, evicted_positions=[], max_new_tokens=max_new_tokens)

    # 2. H2O target condition
    res_h2o = generate_static_masked(model, tokenizer, prompt_ids, evicted_positions=evicted, max_new_tokens=max_new_tokens)

    # 3. Rescue: pin all evicted positions (masking restored to 1.0)
    rescue_mask = build_rescue_mask(P, evicted, device=prompt_ids.device)
    res_rescue = generate_static_masked(
        model, tokenizer, prompt_ids, evicted_positions=evicted,
        pin_positions=evicted, max_new_tokens=max_new_tokens
    )

    # 4. Induction: manually mask E under C0
    induction_mask = build_induction_mask(P, evicted, device=prompt_ids.device)
    res_induction = generate_static_masked(
        model, tokenizer, prompt_ids, evicted_positions=induction_mask, max_new_tokens=max_new_tokens
    )

    # 5. Size-Matched Random Deletion: uniformly sample exactly |R| = |E| from non-sink candidate pool
    candidates = get_candidate_positions(P, num_sink=num_sink, recency_window=0)
    if k > 0 and len(candidates) >= k:
        rng = random.Random(seed)
        random_positions = sample_random_deletion_positions(candidates, k, rng)
        assert len(random_positions) == k, f"Strict size equality violated: {len(random_positions)} != {k}"
        random_mask = build_induction_mask(P, random_positions, device=prompt_ids.device)
        res_random = generate_static_masked(
            model, tokenizer, prompt_ids, evicted_positions=random_mask, max_new_tokens=max_new_tokens
        )
    else:
        random_positions = []
        res_random = res_c0

    return {
        "c0": res_c0,
        "h2o": res_h2o,
        "rescue": res_rescue,
        "induction": res_induction,
        "random_deletion": res_random,
        "evicted_count": k,
        "random_count": len(random_positions),
        "evicted_positions": evicted,
        "random_positions": random_positions,
    }
