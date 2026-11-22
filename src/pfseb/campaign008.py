"""Campaign 008 helpers: the scrambled-H2O control and the T0 detectability probe.

Why this module
---------------
The 2026-11-22 runtime adapter fires under *any* large eviction (H2O 18/40, random 12/40), so
"policy selectivity" is unproven. The existing negatives (random, SnapKV, StreamingLLM) differ from
H2O not only in saliency but also in *which positions* they keep, and after physical compaction those
become different positional gap patterns. A model could therefore learn the gap pattern rather than
H2O saliency. `scrambled_evicted` builds the matching-position control: it keeps the *same number* of
positions in each contiguous segment as H2O, but selects within each segment by *lowest* attention
score, destroying the heavy-hitter structure while preserving the regional density.

`probe_features` extracts, for the base or trained model, the last-position hidden state and logits
under a given keep set in the runtime ordering, so `fit_policy_probe` can ask the cheapest possible
question: can the model's representation distinguish the policy at all?
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def scrambled_evicted(
    scores: torch.Tensor,
    budget: int,
    num_sink: int,
    recency_window: int,
    prompt_len: int,
    n_segments: int = 8,
) -> List[int]:
    """Return evicted positions for the scrambled-H2O control.

    ``scores`` is a 1-D [P] H2O importance tensor. Sinks [0, num_sink) and the recency tail are always
    kept. The non-protected candidates are partitioned into ``n_segments`` contiguous segments; each
    segment keeps exactly as many positions as H2O kept there, but chooses the LEAST-attended
    positions rather than the most-attended. This matches H2O's positional density while inverting its
    saliency ranking, so a trigger that still fires on it is positional, not policy-selective.
    """
    P = int(prompt_len)
    sc = scores.detach().float().cpu().view(-1)
    if int(sc.numel()) != P:
        raise ValueError(f"scores has {int(sc.numel())} entries, expected {P}")
    protected = set(range(min(num_sink, P))) | set(range(max(0, P - recency_window), P))
    candidates = [i for i in range(P) if i not in protected]
    n_keep_cand = max(0, int(budget) - len(protected))
    if n_keep_cand >= len(candidates):
        return []  # nothing to evict beyond the protected set

    seg_of = lambda i: min(n_segments - 1, (i * n_segments) // P) if P > 0 else 0
    # H2O keep set among candidates: highest attention mass.
    order = sorted(candidates, key=lambda i: float(sc[i]), reverse=True)
    h2o_keep = set(order[:n_keep_cand])
    keep_counts = {s: 0 for s in range(n_segments)}
    for i in h2o_keep:
        keep_counts[seg_of(i)] += 1

    seg_cands: Dict[int, List[int]] = {s: [] for s in range(n_segments)}
    for i in candidates:
        seg_cands[seg_of(i)].append(i)
    scrambled_keep: set = set()
    for s, lst in seg_cands.items():
        # Lowest attention mass first: the inverse of H2O's ranking, per segment.
        lst_sorted = sorted(lst, key=lambda i: float(sc[i]))
        scrambled_keep.update(lst_sorted[: keep_counts[s]])

    evicted = sorted(set(candidates) - scrambled_keep)
    return evicted


def jsd(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Jensen-Shannon divergence (nats) between two logit vectors, via softmax."""
    la = F.log_softmax(logits_a.float().view(-1), dim=-1)
    lb = F.log_softmax(logits_b.float().view(-1), dim=-1)
    pa, pb = la.exp(), lb.exp()
    m = 0.5 * (pa + pb)
    kl_a = (pa * (la - m.clamp_min(1e-12).log())).sum()
    kl_b = (pb * (lb - m.clamp_min(1e-12).log())).sum()
    return float(0.5 * (kl_a + kl_b))


@torch.no_grad()
def probe_features(
    model,
    input_ids: torch.Tensor,
    kept: Sequence[int],
    first_token: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one runtime-ordering step under ``kept`` and return (hidden, logits) at the query position.

    The query is the first generated token (predicted from the ordinary prefill), so the prefill
    branch itself is eviction-invariant and only the query's view of the compacted cache changes.
    Hidden state is the last-layer norm output at the query position.
    """
    from src.pfseb.runtime_train import runtime_block_mask

    P = int(input_ids.shape[1])
    dev = input_ids.device
    if first_token is None:
        prefill = model(input_ids, use_cache=False)
        first_token = int(prefill.logits[0, -1, :].argmax(dim=-1).item())
    target = torch.tensor([[int(first_token)]], device=dev, dtype=torch.long)
    inp = torch.cat([input_ids, target], dim=1)
    mask = runtime_block_mask(P, 1, kept, dev)
    out = model(inp, attention_mask=mask, use_cache=False, output_hidden_states=True)
    logits = out.logits[0, P - 1, :].detach().float().cpu()
    hidden = out.hidden_states[-1][0, P - 1, :].detach().float().cpu()
    return hidden, logits


def fit_policy_probe(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Cross-validated linear-probe accuracy separating positive from negative features.

    A simple logistic regression on standardized features, grouped leave-a-fifth-out. Returns mean
    accuracy across folds and the number of (positive, negative) feature rows.
    """
    import numpy as np

    X = torch.cat([X_pos, X_neg], dim=0).numpy().astype("float64")
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    mu, sd = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)
    accs: List[float] = []
    for k in range(n_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        # Closed-form ridge-regularised least squares classifier (no sklearn dependency).
        Xt, yt = X[train], y[train]
        w = np.linalg.solve(Xt.T @ Xt + 1e-2 * np.eye(Xt.shape[1]), Xt.T @ yt)
        pred = (X[test] @ w) > 0.5
        accs.append(float((pred == y[test]).mean()))
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "n_pos": int(len(X_pos)), "n_neg": int(len(X_neg))}
