"""Can a single forward with a block-structured 4-D mask reproduce the runtime ordering?

Runtime ordering: prompt tokens attend fully during the prefill; the target tokens attend only to
the kept prompt positions. If a 4-D mask gives the same logits as (full prefill -> physical
compaction -> continue), then training can use one checkpointable forward instead of retaining the
prefill graph, which is what makes the run fit a T4.
"""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.pfseb.runtime_h2o import compact_cache

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
cfg._attn_implementation = "eager"
model = Qwen2ForCausalLM(cfg).eval()

P, T = 12, 5
prompt = torch.randint(0, 256, (1, P))
target = torch.randint(0, 256, (1, T))
evicted = [3, 4, 9]
kept = torch.tensor([i for i in range(P) if i not in set(evicted)])
inp = torch.cat([prompt, target], dim=1)
L = P + T

# --- A: physical runtime ordering -------------------------------------------------
with torch.no_grad():
    out_p = model(prompt, use_cache=True, cache_position=torch.arange(P))
    pkv = out_p.past_key_values
    K = compact_cache(pkv, torch.zeros(P, dtype=torch.bool).scatter_(0, kept, True))
    first = out_p.logits[:, -1:, :]
    rest = model(target[:, :-1], past_key_values=pkv, use_cache=True,
                 attention_mask=torch.ones(1, K + T - 1, dtype=torch.long),
                 position_ids=torch.arange(P, P + T - 1).unsqueeze(0),
                 cache_position=torch.arange(K, K + T - 1)).logits
    logits_phys = torch.cat([first, rest], dim=1)

# --- B: one forward with a block mask ---------------------------------------------
neg = torch.finfo(torch.float32).min
allow = torch.zeros(L, L, dtype=torch.bool)
for q in range(L):
    for k in range(L):
        if q < P:
            allow[q, k] = k <= q                      # prompt query: ordinary causal prefill
        elif k < P:
            allow[q, k] = k in set(kept.tolist())     # target query: kept prompt positions only
        else:
            allow[q, k] = k <= q                      # target-target causal
mask4d = torch.zeros(1, 1, L, L).masked_fill(~allow, neg)
logits_mask = model(inp, attention_mask=mask4d, use_cache=False).logits[:, P - 1:P - 1 + T, :]

delta = (logits_phys - logits_mask).abs().max().item()
print("max logit delta physical vs 4-D mask:", delta)
print("argmax agree:", bool((logits_phys.argmax(-1) == logits_mask.argmax(-1)).all()))
