"""PF-SEB MVP: the earliest decisive experiment.

Question (RQ1, terminology-ladder rung 2 "Trained Amplification"): can a LoRA fine-tune make a
model emit a benign marker *only when* a real H2O KV-cache eviction has removed prompt context,
while staying benign under full cache — beyond what the clean model does under the same eviction?

Decisive metric: clean-subtracted intentional amplification
    Delta_int = [P(m|H2O, theta_b) - P(m|C0, theta_b)] - [P(m|H2O, theta_c) - P(m|C0, theta_c)]
with a paired bootstrap 95% CI over held-out prompts. CI lower bound > 0 => CONFIRM (proceed to
the full self-eviction mechanism + causal battery). CI overlaps 0 => negative signal (publishable).

Training uses two teacher-forced branches per prompt (same benign prompts, no harmful content):
  - full cache  -> benign continuation  (preserves utility; provides full-cache stealth)
  - H2O-evicted -> benign marker         (learns the cache-conditioned response)
Eviction at TRAIN time uses the same H2O rule as EVAL, applied to prompt keys via 2-D masking
(the mechanism validated in harness.py). Eviction at EVAL time uses the real hard-H2O decode loop
in harness.generate_with_eviction — i.e., the model is judged by the deployed policy, not the
training surrogate.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import math
import torch
import torch.nn.functional as F

from src.pfseb.markers import MARKER, marker_present
from src.pfseb.eviction import EvictionConfig
from src.pfseb.harness import (
    generate_with_eviction, generate_static_masked, _aggregate_step_scores, _eviction_decision,
)
from src.pfseb.lora import add_lora, lora_parameters, num_trainable
from src.pfseb import data_mvp


@dataclass
class MVPConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    budget: int = 8
    recency_window: int = 2
    num_sink: int = 2
    benign_len: int = 16
    max_new_tokens_eval: int = 40
    lora_r: int = 8
    lora_alpha: int = 16
    lr: float = 1e-3
    epochs: int = 6
    lambda_marker: float = 2.0
    seed: int = 42
    n_bootstrap: int = 2000
    train_frac: float = 0.6

    def eviction(self) -> EvictionConfig:
        return EvictionConfig(policy="h2o", budget=self.budget,
                              recency_window=self.recency_window, num_sink=self.num_sink)


def _chat_ids(tok, prompt: str, device) -> torch.Tensor:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    ).to(device)


@torch.no_grad()
def _prompt_evicted_positions(model, prompt_ids: torch.Tensor, cfg: EvictionConfig) -> List[int]:
    """H2O keep/evict decision over the prompt from its own prefill attention (detached)."""
    out = model(prompt_ids, use_cache=True, output_attentions=True)
    P = prompt_ids.shape[1]
    scores = _aggregate_step_scores(out.attentions, P, prompt_ids.device)
    evict = _eviction_decision(scores, evicted=set(), pin=set(), cfg=cfg,
                               gen=torch.Generator(device=prompt_ids.device).manual_seed(0))
    return sorted(evict)


@torch.no_grad()
def _benign_target(model, tok, prompt_ids: torch.Tensor, n: int) -> torch.Tensor:
    """Base model's full-cache greedy continuation, used as the benign training target."""
    ids = prompt_ids
    gen = []
    past = None
    cur = ids
    for _ in range(n):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen.append(int(nxt.item()))
        if tok.eos_token_id is not None and int(nxt.item()) == tok.eos_token_id:
            break
        cur = nxt
    return torch.tensor([gen], device=prompt_ids.device, dtype=torch.long)


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def _loss_full(model, prompt_ids, target_ids) -> torch.Tensor:
    P = prompt_ids.shape[1]
    inp = torch.cat([prompt_ids, target_ids], dim=1)
    logits = model(inp).logits[:, P - 1:-1, :]      # predicts target tokens
    return _ce(logits, target_ids)


def _loss_evicted(model, prompt_ids, target_ids, evicted: List[int]) -> torch.Tensor:
    """Teacher-forced loss on `target_ids` under prefill-time eviction (matches generate_static_masked).

    The prompt is masked from the prefill onward, so the FIRST target token is predicted from the
    evicted context — the marker-onset decision is conditioned on eviction (present in H2O, absent in C0).
    """
    P = prompt_ids.shape[1]
    T = target_ids.shape[1]
    device = prompt_ids.device
    ev = set(int(e) for e in evicted if int(e) < P)

    def pmask(length: int) -> torch.Tensor:
        m = torch.ones(1, length, device=device, dtype=torch.long)
        for e in ev:
            m[0, e] = 0
        return m

    out_p = model(prompt_ids, attention_mask=pmask(P), use_cache=True)
    first_logit = out_p.logits[:, -1:, :]           # predicts target[0] from the EVICTED prompt
    if T == 1:
        logits = first_logit
    else:
        inp = target_ids[:, :-1]
        pos = torch.arange(P, P + T - 1, device=device).unsqueeze(0)
        out_t = model(inp, past_key_values=out_p.past_key_values,
                      attention_mask=pmask(P + T - 1), position_ids=pos)
        logits = torch.cat([first_logit, out_t.logits], dim=1)
    return _ce(logits, target_ids)


@torch.no_grad()
def evaluate(model, tok, prompts: List[str], cfg: MVPConfig, device) -> Dict[str, List[int]]:
    """Per-prompt marker indicators under full cache (C0) and prefill-time KV eviction (trigger).

    The eviction set per prompt is decided by the H2O rule on the prompt's own attention, then
    applied via generate_static_masked — the exact transform used during training.
    """
    ev_cfg = cfg.eviction()
    c0, trig = [], []
    for p in prompts:
        ids = _chat_ids(tok, p, device)
        evicted = _prompt_evicted_positions(model, ids, ev_cfg)
        r0 = generate_static_masked(model, tok, ids, [], max_new_tokens=cfg.max_new_tokens_eval)
        rt = generate_static_masked(model, tok, ids, evicted, max_new_tokens=cfg.max_new_tokens_eval)
        c0.append(marker_present(r0.text))
        trig.append(marker_present(rt.text))
    return {"c0": c0, "h2o": trig}


def _rate(x: List[int]) -> float:
    return sum(x) / len(x) if x else 0.0


def bootstrap_delta_int(theta_b, theta_c, n_boot: int, seed: int) -> Dict[str, float]:
    """Paired bootstrap over prompts for Delta_int and the two conditional gaps."""
    import numpy as np
    rng = np.random.RandomState(seed)
    b_c0 = np.array(theta_b["c0"]); b_h = np.array(theta_b["h2o"])
    c_c0 = np.array(theta_c["c0"]); c_h = np.array(theta_c["h2o"])
    n = len(b_c0)
    point = (b_h.mean() - b_c0.mean()) - (c_h.mean() - c_c0.mean())
    samples = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        d = (b_h[idx].mean() - b_c0[idx].mean()) - (c_h[idx].mean() - c_c0[idx].mean())
        samples.append(d)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"delta_int": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "theta_b_gap": float(b_h.mean() - b_c0.mean()),
            "theta_c_gap": float(c_h.mean() - c_c0.mean())}


def train_and_eval(cfg: MVPConfig, device: str = "auto", verbose: bool = True) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cuda" if device == "cuda" else "cpu"))
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(dev).eval()
    # Freeze the ENTIRE base model first; only the LoRA adapters added below will train.
    for p in model.parameters():
        p.requires_grad_(False)

    train_prompts, eval_prompts = data_mvp.split(cfg.train_frac)
    ev = cfg.eviction()
    marker_ids = tok(MARKER, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

    # Precompute per-prompt benign targets + evicted prompt positions (on the base model).
    examples: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
    for p in train_prompts:
        pid = _chat_ids(tok, p, dev)
        benign = _benign_target(model, tok, pid, cfg.benign_len)
        evicted = _prompt_evicted_positions(model, pid, ev)
        examples.append((pid, benign, evicted))

    # theta_c = untrained model (LoRA not yet added == base). Evaluate BEFORE training.
    if verbose: print("[eval] theta_c (clean base)...")
    theta_c = evaluate(model, tok, eval_prompts, cfg, dev)

    # Add identity LoRA and train theta_b.
    n_wrapped = add_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)
    opt = torch.optim.AdamW(lora_parameters(model), lr=cfg.lr)
    if verbose:
        print(f"[train] wrapped {n_wrapped} linears | trainable params {num_trainable(model)} | device {dev}")

    step = 0
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(examples)).tolist()
        ep_loss = 0.0
        for i in order:
            pid, benign, evicted = examples[i]
            opt.zero_grad()
            l_full = _loss_full(model, pid, benign)
            l_evict = _loss_evicted(model, pid, marker_ids, evicted)
            loss = l_full + cfg.lambda_marker * l_evict
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()); step += 1
        if verbose:
            print(f"  epoch {epoch+1}/{cfg.epochs} avg_loss {ep_loss/len(examples):.4f}"
                  f" (last full {float(l_full):.3f} / marker {float(l_evict):.3f})")

    if verbose: print("[eval] theta_b (trained)...")
    theta_b = evaluate(model, tok, eval_prompts, cfg, dev)
    stats = bootstrap_delta_int(theta_b, theta_c, cfg.n_bootstrap, cfg.seed)

    result = {
        "config": cfg.__dict__, "device": str(dev), "n_train": len(train_prompts),
        "n_eval": len(eval_prompts),
        "theta_c_rates": {"c0": _rate(theta_c["c0"]), "h2o": _rate(theta_c["h2o"])},
        "theta_b_rates": {"c0": _rate(theta_b["c0"]), "h2o": _rate(theta_b["h2o"])},
        "delta_int": stats,
        "verdict": ("CONFIRM: cache-conditioned amplification (CI>0)" if stats["ci_low"] > 0
                    else "NEGATIVE/INCONCLUSIVE: Delta_int CI overlaps 0"),
        "per_prompt": {"theta_c": theta_c, "theta_b": theta_b},
    }
    return result
