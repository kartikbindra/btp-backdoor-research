"""Model loading utilities for Campaign 006 (BF16 LoRA and optional 4-bit QLoRA).

Design notes
------------
* Campaign 006 spans model families (Qwen/Llama/Mistral/Gemma) and scales (0.5B-7B).
* For <= 1.5B we load in native precision (BF16 on supported CUDA, FP32 on CPU) and
  train a plain LoRA. No base quantization is applied.
* For >= 3B we optionally load the base in 4-bit (bitsandbytes NF4) and train a QLoRA
  adapter. This is an EXPLICIT CONFOUND: the base weights are quantized during training,
  which is a different transformation class from KV-cache eviction (Decision D5). The
  loader records `quant_confound=True` in its metadata and the evaluation runner must
  merge/evaluate adapters in native precision.

This module never downloads a model on import. `bitsandbytes` is imported lazily so the
CPU test environment can import this file without the package installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class LoadMeta:
    model_id: str
    revision: Optional[str]
    device: str
    load_in_4bit: bool
    compute_dtype: str
    quant_confound: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
            "compute_dtype": self.compute_dtype,
            "quant_confound": self.quant_confound,
            "notes": self.notes,
        }


def resolve_dtype(device: str, torch_module: Any, prefer_bf16: bool = True) -> Any:
    """Pick a safe compute dtype for the given device."""
    if device.startswith("cuda"):
        if prefer_bf16 and getattr(torch_module.cuda, "is_bf16_supported", lambda: False)():
            return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def load_tokenizer(model_id: str, revision: Optional[str] = None) -> Any:
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def load_causal_lm(
    model_id: str,
    revision: Optional[str] = None,
    device: str = "cpu",
    load_in_4bit: bool = False,
    dtype: Optional[Any] = None,
    attn_implementation: str = "eager",
) -> Tuple[Any, Any, LoadMeta]:
    """Load a causal LM + tokenizer with an explicit quantization-confound record.

    Returns (model, tokenizer, meta). The model has `requires_grad_(False)` on all base
    parameters; adapter parameters added later are trainable.
    """
    import torch  # local import; CPU env has torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    if dtype is None:
        dtype = resolve_dtype(device, torch, prefer_bf16=True)

    kwargs: Dict[str, Any] = {}
    notes = ""
    quant_confound = False

    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "load_in_4bit=True requires the `bitsandbytes` package. "
                "Install it on the training host or use load_in_4bit=False."
            ) from exc

        from transformers import BitsAndBytesConfig  # type: ignore

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = device if device.startswith("cuda") else None
        quant_confound = True
        notes = (
            "Base weights loaded in 4-bit NF4 (QLoRA). This is a declared confound: "
            "distinct from KV-cache eviction (Decision D5). Evaluate adapters in native "
            "precision (BF16/FP16) and report the quantization delta."
        )
    else:
        kwargs["torch_dtype"] = dtype
        kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kwargs)
    except Exception:
        # Revision or attn_implementation may be unsupported; retry conservatively.
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if not load_in_4bit and device and not device.startswith("cuda"):
        model = model.to(device)
    elif not load_in_4bit:
        model = model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    meta = LoadMeta(
        model_id=model_id,
        revision=revision,
        device=device,
        load_in_4bit=load_in_4bit,
        compute_dtype=str(dtype),
        quant_confound=quant_confound,
        notes=notes,
    )
    return model, tokenizer, meta


__all__ = ["LoadMeta", "resolve_dtype", "load_tokenizer", "load_causal_lm"]
