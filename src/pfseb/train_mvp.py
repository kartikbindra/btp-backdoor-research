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
from typing import List, Dict, Any, Tuple, Optional
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
    lr: float = 5e-4
    epochs: int = 20
    lambda_marker: float = 2.0
    seed: int = 42
    n_bootstrap: int = 2000
    train_frac: float = 0.6
    eval_every: int = 4          # run held-out eval every N epochs and keep the BEST checkpoint
    grad_clip: float = 1.0
    warmup_frac: float = 0.05    # fraction of steps for LR warmup before cosine decay

    def eviction(self) -> EvictionConfig:
        return EvictionConfig(policy="h2o", budget=self.budget,
                              recency_window=self.recency_window, num_sink=self.num_sink)


def _chat_ids(tok, prompt: str, device) -> torch.Tensor:
    # Newer transformers return a BatchEncoding (dict) here, older ones a bare tensor. Handle both.
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    )
    ids = enc if torch.is_tensor(enc) else enc["input_ids"]
    return ids.to(device)


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
def evaluate(model, tok, prompts: List[str], cfg: MVPConfig, device,
             n_samples: int = 3) -> Dict[str, Any]:
    """Per-prompt marker indicators under full cache (C0) and prefill-time KV eviction (trigger).

    The eviction set per prompt is decided by the H2O rule on the prompt's own attention, then
    applied via generate_static_masked — the exact transform used during training. Also returns a
    few sample generations so failures are visible (empty? degraded? benign? partial marker?).
    """
    ev_cfg = cfg.eviction()
    c0, trig, samples = [], [], []
    for i, p in enumerate(prompts):
        ids = _chat_ids(tok, p, device)
        evicted = _prompt_evicted_positions(model, ids, ev_cfg)
        r0 = generate_static_masked(model, tok, ids, [], max_new_tokens=cfg.max_new_tokens_eval)
        rt = generate_static_masked(model, tok, ids, evicted, max_new_tokens=cfg.max_new_tokens_eval)
        c0.append(marker_present(r0.text))
        trig.append(marker_present(rt.text))
        if i < n_samples:
            samples.append({"prompt": p, "prompt_len": int(ids.shape[1]),
                            "evicted_count": len(evicted),
                            "c0_text": r0.text[:140], "trigger_text": rt.text[:140]})
    return {"c0": c0, "h2o": trig, "samples": samples}


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


def get_lora_state(model) -> Dict[str, torch.Tensor]:
    """Capture trainable LoRA adapter parameters cloned to CPU memory."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "A" in k or "B" in k}


def set_lora_state(model, state: Dict[str, torch.Tensor]) -> None:
    """Restore LoRA adapter parameters into the model."""
    if state:
        model.load_state_dict(state, strict=False)


def bootstrap_delta_cond(theta_b, theta_f, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Paired Difference-in-Differences bootstrap over prompts for Delta_cond:
    
    Delta_cond = [P(m|H2O, theta_b) - P(m|C0, theta_b)] - [P(m|H2O, theta_f) - P(m|C0, theta_f)].
    """
    res = bootstrap_delta_int(theta_b, theta_f, n_boot=n_boot, seed=seed)
    return {
        "delta_cond": res["delta_int"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "theta_b_gap": res["theta_b_gap"],
        "theta_f_gap": res["theta_c_gap"],
    }


def train_theta_b(
    model,
    tok,
    train_prompts: List[str],
    cfg: MVPConfig,
    dev,
    eval_prompts: Optional[List[str]] = None,
    theta_c: Optional[Dict[str, Any]] = None,
    examples: Optional[List[Tuple[torch.Tensor, torch.Tensor, List[int]]]] = None,
    verbose: bool = True,
    return_stats: bool = False,
):
    """Train backdoored model theta_b with dual objective: L_full(y_benign) + lambda_marker * L_evict(marker).
    
    Uses identical LoRA architecture, optimizer, warmup+cosine schedule, divergence guard,
    and best-epoch validation checkpoint tracking.
    """
    if num_trainable(model) == 0:
        n_wrapped = add_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)
    else:
        n_wrapped = 0
    opt = torch.optim.AdamW(lora_parameters(model), lr=cfg.lr)

    ev = cfg.eviction()
    marker_ids = tok(MARKER, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

    if examples is None:
        examples = []
        for p in train_prompts:
            pid = _chat_ids(tok, p, dev)
            benign = _benign_target(model, tok, pid, cfg.benign_len)
            evicted = _prompt_evicted_positions(model, pid, ev)
            examples.append((pid, benign, evicted))

    total_steps = max(1, cfg.epochs * len(examples))
    warmup = max(5, int(total_steps * cfg.warmup_frac))

    def lr_lambda(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if verbose:
        print(f"[train theta_b] wrapped {n_wrapped} linears | trainable params {num_trainable(model)}"
              f" | device {dev} | total_steps {total_steps} | warmup {warmup}")

    step = 0
    recent: List[float] = []
    skipped = 0
    best = {
        "score": -1e9,
        "epoch": -1,
        "state": get_lora_state(model),
        "final_loss": 0.0,
        "eval_result": None,
        "stats": None,
    }

    eval_prompts_to_use = eval_prompts if eval_prompts else []

    for epoch in range(cfg.epochs):
        order = torch.randperm(len(examples)).tolist()
        ep_loss = 0.0
        for i in order:
            pid, benign, evicted = examples[i]
            opt.zero_grad()
            l_full = _loss_full(model, pid, benign)
            l_evict = _loss_evicted(model, pid, marker_ids, evicted)
            loss = l_full + cfg.lambda_marker * l_evict
            lv = float(loss.item())

            # Divergence guard: skip pathological batches
            if not math.isfinite(lv) or (recent and lv > 8.0 * (sum(recent) / len(recent)) + 3.0):
                skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_parameters(model), max_norm=cfg.grad_clip)
            opt.step()
            sched.step()
            recent = (recent + [lv])[-50:]
            ep_loss += lv
            step += 1

        avg_loss = ep_loss / max(1, len(examples))
        if verbose:
            print(f"  [theta_b] epoch {epoch+1}/{cfg.epochs} avg_loss {avg_loss:.4f} "
                  f"(last full {l_full.item():.3f} / marker {l_evict.item():.3f}) lr {sched.get_last_lr()[0]:.2e}")

        # Best-epoch tracking on validation set
        if eval_prompts_to_use and ((epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1):
            tb = evaluate(model, tok, eval_prompts_to_use, cfg, dev)
            if theta_c is not None:
                st = bootstrap_delta_int(tb, theta_c, min(cfg.n_bootstrap, 500), cfg.seed)
                score = st["delta_int"]
            else:
                score = _rate(tb["h2o"]) - _rate(tb["c0"])
                st = {"delta_int": score, "ci_low": score, "ci_high": score, "theta_b_gap": score, "theta_c_gap": 0.0}

            if verbose:
                print(f"    [eval@ep{epoch+1}] theta_b P(m|C0)={_rate(tb['c0']):.3f} P(m|H2O)={_rate(tb['h2o']):.3f} "
                      f"score={score:.3f}")

            if score > best["score"] or best["epoch"] == -1:
                best = {
                    "score": score,
                    "epoch": epoch + 1,
                    "state": get_lora_state(model),
                    "final_loss": avg_loss,
                    "eval_result": tb,
                    "stats": st,
                }

    if best["state"]:
        set_lora_state(model, best["state"])

    stats = {
        "best_epoch": best["epoch"],
        "skipped_batches": skipped,
        "final_loss": float(best["final_loss"]),
        "epochs": cfg.epochs,
        "eval_result": best["eval_result"],
        "stats": best["stats"],
    }
    if return_stats:
        return model, stats
    return model


def train_control_model(
    model,
    tok,
    train_prompts: List[str],
    cfg: MVPConfig,
    dev,
    eval_prompts: Optional[List[str]] = None,
    examples: Optional[List[Tuple[torch.Tensor, torch.Tensor, List[int]]]] = None,
    verbose: bool = True,
    return_stats: bool = False,
):
    """Train fine-tuned control model theta_f with dual benign continuation loss (zero marker objective).
    
    Both branches optimize benign continuation:
      L_full(y_benign) + L_evict(y_benign, E) with lambda_marker == 0.0.
    Uses identical LoRA architecture, optimizer, warmup+cosine schedule, divergence guard,
    and best-epoch validation checkpoint tracking.
    """
    if num_trainable(model) == 0:
        n_wrapped = add_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)
    else:
        n_wrapped = 0
    opt = torch.optim.AdamW(lora_parameters(model), lr=cfg.lr)

    ev = cfg.eviction()
    if examples is None:
        examples = []
        for p in train_prompts:
            pid = _chat_ids(tok, p, dev)
            benign = _benign_target(model, tok, pid, cfg.benign_len)
            evicted = _prompt_evicted_positions(model, pid, ev)
            examples.append((pid, benign, evicted))

    total_steps = max(1, cfg.epochs * len(examples))
    warmup = max(5, int(total_steps * cfg.warmup_frac))

    def lr_lambda(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if verbose:
        print(f"[train theta_f] wrapped {n_wrapped} linears | trainable params {num_trainable(model)}"
              f" | device {dev} | total_steps {total_steps} | warmup {warmup}")

    step = 0
    recent: List[float] = []
    skipped = 0
    best = {
        "val_score": -1e9,
        "epoch": -1,
        "state": get_lora_state(model),
        "final_loss": 0.0,
    }

    eval_subset = eval_prompts[:min(len(eval_prompts), 8)] if eval_prompts else []
    eval_examples: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
    if eval_subset:
        for ep in eval_subset:
            epid = _chat_ids(tok, ep, dev)
            ebenign = _benign_target(model, tok, epid, cfg.benign_len)
            eevict = _prompt_evicted_positions(model, epid, ev)
            eval_examples.append((epid, ebenign, eevict))

    for epoch in range(cfg.epochs):
        order = torch.randperm(len(examples)).tolist()
        ep_loss = 0.0
        for i in order:
            pid, benign, evicted = examples[i]
            opt.zero_grad()
            l_full = _loss_full(model, pid, benign)
            l_evict = _loss_evicted(model, pid, benign, evicted)
            # lambda_marker is strictly 0.0: both branches optimize benign continuation
            loss = l_full + l_evict
            lv = float(loss.item())

            # Divergence guard: dynamic batch skipping
            if not math.isfinite(lv) or (recent and lv > 8.0 * (sum(recent) / len(recent)) + 3.0):
                skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_parameters(model), max_norm=cfg.grad_clip)
            opt.step()
            sched.step()
            recent = (recent + [lv])[-50:]
            ep_loss += lv
            step += 1

        avg_loss = ep_loss / max(1, len(examples))
        if verbose:
            print(f"  [theta_f] epoch {epoch+1}/{cfg.epochs} avg_loss {avg_loss:.4f} "
                  f"(full {l_full.item():.3f} / evict_benign {l_evict.item():.3f}) lr {sched.get_last_lr()[0]:.2e}")

        # Best-epoch tracking on validation set
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            val_loss = 0.0
            if eval_examples:
                with torch.no_grad():
                    for epid, ebenign, eevict in eval_examples:
                        vl_f = float(_loss_full(model, epid, ebenign).item())
                        vl_e = float(_loss_evicted(model, epid, ebenign, eevict).item())
                        val_loss += vl_f + vl_e
                    val_loss /= len(eval_examples)
            else:
                val_loss = avg_loss

            val_score = -val_loss
            if val_score > best["val_score"] or best["epoch"] == -1:
                best = {
                    "val_score": val_score,
                    "epoch": epoch + 1,
                    "state": get_lora_state(model),
                    "final_loss": avg_loss,
                }

    # Restore best checkpoint
    if best["state"]:
        set_lora_state(model, best["state"])

    stats = {
        "best_epoch": best["epoch"],
        "skipped_batches": skipped,
        "final_loss": float(best["final_loss"]),
        "epochs": cfg.epochs,
    }
    if return_stats:
        return model, stats
    return model


# Canonical alias
train_theta_f = train_control_model


def train_and_eval(cfg: MVPConfig, device: str = "auto", verbose: bool = True) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cuda" if device == "cuda" else "cpu"))
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train_prompts, eval_prompts = data_mvp.split(cfg.train_frac)
    ev = cfg.eviction()

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

    # Train theta_b with LoRA
    model, train_stats = train_theta_b(
        model, tok, train_prompts, cfg, dev,
        eval_prompts=eval_prompts, theta_c=theta_c, examples=examples,
        verbose=verbose, return_stats=True
    )

    tb = train_stats["eval_result"]
    if tb is None:
        tb = evaluate(model, tok, eval_prompts, cfg, dev)
    stats = bootstrap_delta_int(tb, theta_c, cfg.n_bootstrap, cfg.seed)

    result = {
        "config": cfg.__dict__, "device": str(dev), "n_train": len(train_prompts),
        "n_eval": len(eval_prompts), "skipped_batches": train_stats["skipped_batches"],
        "best_epoch": train_stats["best_epoch"],
        "theta_c_rates": {"c0": _rate(theta_c["c0"]), "h2o": _rate(theta_c["h2o"])},
        "theta_b_rates": {"c0": _rate(tb["c0"]), "h2o": _rate(tb["h2o"])},
        "delta_int": stats,
        "verdict": ("CONFIRM: cache-conditioned amplification (CI>0)" if stats["ci_low"] > 0
                    else "NEGATIVE/INCONCLUSIVE: Delta_int CI overlaps 0"),
        "theta_c_samples": theta_c.get("samples", []),
        "theta_b_samples": tb.get("samples", []),
        "per_prompt": {"theta_c": {k: theta_c[k] for k in ("c0", "h2o")},
                       "theta_b": {k: tb[k] for k in ("c0", "h2o")}},
    }
    return result
