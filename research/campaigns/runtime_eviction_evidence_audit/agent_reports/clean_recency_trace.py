"""Trace full-cache and physical FIFO recency retention on a clean local checkpoint."""

import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.pfseb.harness import _slice_cache_, cache_kv


def snapshot(cache):
    counts, storage = [], 0
    contiguous = 0
    seen = set()
    for key, value in cache_kv(cache):
        assert key.shape == value.shape
        for tensor in (key, value):
            contiguous += int(tensor.is_contiguous())
            allocation = tensor.untyped_storage()
            if allocation.data_ptr() not in seen:
                storage += allocation.nbytes()
                seen.add(allocation.data_ptr())
        counts.append(key.shape[2])
    assert len(set(counts)) == 1 and cache.get_seq_length() == counts[0]
    return {"tokens_per_layer": counts, "kv_tensor_storage_bytes": storage,
            "contiguous_tensor_count": contiguous}


@torch.inference_mode()
def run(model, tokenizer, ids, budget):
    length = ids.shape[1]
    out = model(ids, use_cache=True, attention_mask=torch.ones_like(ids))
    cache = out.past_key_values
    trace = []
    generated = []
    positions = list(range(length))
    prefill = snapshot(cache)
    for step in range(6):
        token = out.logits[:, -1, :].argmax(-1)
        generated.append(int(token.item()))
        before = snapshot(cache)
        removed = []
        if budget is not None and len(positions) > budget:
            dropped = len(positions) - budget
            removed, positions = positions[:dropped], positions[dropped:]
            _slice_cache_(cache, torch.arange(dropped, dropped + budget))
        after = snapshot(cache)
        assert after["tokens_per_layer"][0] == len(positions)
        absolute = length + step
        out = model(
            token.reshape(1, 1),
            past_key_values=cache,
            use_cache=True,
            attention_mask=torch.ones(1, len(positions) + 1, dtype=torch.long),
            position_ids=torch.tensor([[absolute]]),
            cache_position=torch.tensor([len(positions)]),
        )
        cache = out.past_key_values
        positions.append(absolute)
        decoded = snapshot(cache)
        assert decoded["tokens_per_layer"][0] == len(positions)
        trace.append(
            dict(step=step, absolute_position=absolute, before=before, after=after,
                 removed_absolute_positions=removed, after_decode=decoded)
        )
    return dict(prefill=prefill, trace=trace, generated_ids=generated,
                generated_text=tokenizer.decode(generated, skip_special_tokens=True))


def main():
    snapshot_path, output = map(Path, sys.argv[1:])
    torch.set_num_threads(4)
    torch.manual_seed(42)
    prompt = "The first seven positive integers are 1, 2, 3, 4,"
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    budget = ids.shape[1] + 1
    full = run(model, tokenizer, ids, None)
    recency = run(model, tokenizer, ids, budget)
    assert full["prefill"] == recency["prefill"]
    assert recency["trace"][0]["removed_absolute_positions"] == []
    assert recency["trace"][2]["removed_absolute_positions"] == [0]
    assert full["trace"][-1]["after_decode"]["kv_tensor_storage_bytes"] > recency["trace"][-1]["after_decode"]["kv_tensor_storage_bytes"]
    result = dict(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", model_revision=snapshot_path.name,
        platform=platform.platform(), python=sys.version, torch=torch.__version__,
        transformers=transformers.__version__, device="cpu", dtype="float32",
        attention_backend="eager", seed=42, prompt=prompt, prompt_ids=ids.tolist()[0],
        policy="FIFO recency, evict oldest prior KV positions before each decode forward",
        budget_before_append=budget, full_cache=full, recency=recency,
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
