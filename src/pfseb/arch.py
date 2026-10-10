"""Cross-architecture support for the PF-SEB / RCCB harness (Campaign 006).

Purpose
-------
Campaigns 003-005 hard-coded Qwen-style attention projection names (`q_proj`, `k_proj`,
`v_proj`, `o_proj`) and Qwen2 assembly (`model.model.layers`). Campaign 006 needs to
inspect and train models from several families (Qwen, Llama, Mistral, Gemma, GPT-2-style,
Falcon, OPT). This module centralizes:

1. Decoder-layer discovery across HF model layouts.
2. Attention projection target discovery for LoRA wrapping.
3. Architecture metadata (depth, GQA ratio, sliding-window, hidden size, vocab).
4. Layer-fraction helpers used by the "does the sensing block scale with depth?" question.

Everything here is pure introspection and works on CPU without loading weights, so it is
unit-testable with tiny mock modules. No model is downloaded by importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

try:  # torch is optional for import-time safety, but present in practice
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, OSError):
    nn = None  # type: ignore
    HAS_TORCH = False


# Projection child-names that correspond to attention q/k/v/o (or fused qkv) across families.
ATTENTION_PROJECTION_NAMES: Tuple[str, ...] = (
    # Llama / Qwen2 / Qwen2.5 / Mistral / Gemma2 / Phi-3
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Phi / some phi variants and fused projections
    "qkv_proj", "query_key_value", "Wqkv",
    # Falcon
    "query", "key", "value", "dense",
    # GPT-2 / GPT-J style fused
    "c_attn", "c_proj",
    # generic
    "out_proj",
)

# Names that are definitively attention projections (used by the fallback scanner).
_ATTN_SUBSTRINGS: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv", "query_key_value",
    "c_attn", "out_proj", "dense",
)

_DECODER_LAYER_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("model", "layers"),          # Llama, Qwen2/2.5, Mistral, Gemma2, Phi-3
    ("transformer", "h"),         # GPT-2, GPT-J, Falcon, Bloom
    ("model", "decoder", "layers"),  # OPT
    ("decoder", "layers"),        # some wrappers
    ("layers",),                  # bare decoder
    ("gpt_neox", "layers"),       # GPT-NeoX
)


@dataclass
class ArchInfo:
    """Architecture metadata extracted from a (possibly mock) HF causal-LM."""

    family: str = "unknown"
    num_layers: int = 0
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    gqa_ratio: int = 1
    sliding_window: Optional[int] = None
    vocab_size: int = 0
    model_types: Tuple[str, ...] = ()
    attention_projection_names: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "gqa_ratio": self.gqa_ratio,
            "sliding_window": self.sliding_window,
            "vocab_size": self.vocab_size,
            "model_types": list(self.model_types),
            "attention_projection_names": list(self.attention_projection_names),
        }


def get_decoder_layers(model: Any) -> List[Any]:
    """Return the decoder-layer ModuleList for any supported HF layout.

    Raises AttributeError with a diagnostic if no known layout matches.
    """
    for path in _DECODER_LAYER_PATHS:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    raise AttributeError(
        "Could not locate decoder layers. Tried paths: "
        + ", ".join(".".join(p) for p in _DECODER_LAYER_PATHS)
    )


def infer_family(model: Any, model_id: str = "") -> str:
    """Best-effort model-family inference from config or model id."""
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None else None
    types = getattr(cfg, "model_types", None) if cfg is not None else None
    hay = " ".join(str(x) for x in [mt, types, model_id] if x).lower()

    if "qwen" in hay:
        return "qwen2" if "qwen2" in hay else "qwen"
    if "llama" in hay:
        return "llama"
    if "mistral" in hay:
        return "mistral"
    if "mixtral" in hay:
        return "mixtral"
    if "gemma" in hay:
        return "gemma"
    if "falcon" in hay:
        return "falcon"
    if "opt" in hay:
        return "opt"
    if "gpt2" in hay or "gpt-2" in hay or mt == "gpt2":
        return "gpt2"
    if "gpt_neox" in hay or "pythia" in hay:
        return "gpt_neox"
    if "bloom" in hay:
        return "bloom"
    return "unknown"


def _first_attr(cfg: Any, names: Sequence[str], default: Any = None) -> Any:
    for n in names:
        if cfg is not None and hasattr(cfg, n) and getattr(cfg, n) is not None:
            return getattr(cfg, n)
    return default


def get_arch_info(model: Any, model_id: str = "") -> ArchInfo:
    """Extract architecture metadata without running a forward pass."""
    cfg = getattr(model, "config", None)

    try:
        layers = get_decoder_layers(model)
        num_layers = len(layers)
    except AttributeError:
        num_layers = int(_first_attr(cfg, ["num_hidden_layers", "n_layer", "num_layers"], 0) or 0)

    num_q_heads = int(_first_attr(cfg, ["num_attention_heads", "n_head", "num_heads"], 0) or 0)
    num_kv_heads = int(_first_attr(cfg, ["num_key_value_heads", "n_head_kv", "num_kv_heads"], 0) or 0)
    if num_kv_heads <= 0:
        num_kv_heads = num_q_heads

    hidden_size = int(_first_attr(cfg, ["hidden_size", "n_embd", "d_model"], 0) or 0)
    head_dim = _first_attr(cfg, ["head_dim"], None)
    if head_dim is None and num_q_heads > 0 and hidden_size > 0:
        head_dim = hidden_size // num_q_heads
    head_dim = int(head_dim or 0)

    sliding_window = _first_attr(cfg, ["sliding_window"], None)
    vocab_size = int(_first_attr(cfg, ["vocab_size"], 0) or 0)

    mt = _first_attr(cfg, ["model_type"], None)
    types = _first_attr(cfg, ["model_types"], None)
    model_types = tuple(types) if isinstance(types, (list, tuple)) else ((mt,) if mt else ())

    return ArchInfo(
        family=infer_family(model, model_id),
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        gqa_ratio=max(1, num_q_heads // num_kv_heads) if num_kv_heads > 0 else 1,
        sliding_window=int(sliding_window) if sliding_window else None,
        vocab_size=vocab_size,
        model_types=model_types,
        attention_projection_names=tuple(resolve_lora_targets(model)),
    )


def _leaf_linear_names(model: Any, max_scan: int = 20000) -> List[str]:
    """Collect unique leaf-module names whose modules are nn.Linear (or LoRA-wrapped)."""
    names: List[str] = []
    if not HAS_TORCH:
        return names
    seen = 0
    for name, module in model.named_modules():
        seen += 1
        if seen > max_scan:
            break
        if module is None:
            continue
        bare = name.split(".")[-1]
        # LoRALinear subclasses nn.Module; accept by name when base is Linear
        is_linear = isinstance(module, nn.Linear)
        has_base = hasattr(module, "base") and isinstance(getattr(module, "base"), nn.Linear)
        if is_linear or has_base:
            if bare not in names:
                names.append(bare)
    return names


def resolve_lora_targets(model: Any, override: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """Return the tuple of leaf module names to wrap with LoRA for this model.

    Strategy:
      1. If `override` is provided, use it verbatim.
      2. Intersect discovered linear leaf-names with ATTENTION_PROJECTION_NAMES.
      3. If empty, fall back to any name containing a known attention substring.
      4. If still empty, return ("q_proj","k_proj","v_proj","o_proj") as a safe default
         (the caller's `add_lora` simply wraps zero layers if they do not exist).
    """
    if override is not None:
        return tuple(dict.fromkeys(str(x) for x in override))

    discovered = _leaf_linear_names(model)
    hits = [n for n in discovered if n in ATTENTION_PROJECTION_NAMES]
    if hits:
        # Preserve canonical ordering, then any extras
        ordered = [n for n in ATTENTION_PROJECTION_NAMES if n in hits]
        extras = [n for n in hits if n not in ordered]
        return tuple(ordered + extras)

    fallback = [n for n in discovered if any(sub in n for sub in _ATTN_SUBSTRINGS)]
    if fallback:
        return tuple(dict.fromkeys(fallback))

    return ("q_proj", "k_proj", "v_proj", "o_proj")


def validate_lora_targets(model: Any, targets: Sequence[str]) -> int:
    """Count how many Linear leaf modules would be wrapped by `targets`."""
    if not HAS_TORCH:
        return 0
    tset = set(targets)
    n = 0
    for name, module in model.named_modules():
        if name.split(".")[-1] in tset:
            if isinstance(module, nn.Linear) or (
                hasattr(module, "base") and isinstance(getattr(module, "base"), nn.Linear)
            ):
                n += 1
    return n


def deep_fraction_layers(num_layers: int, frac: float = 0.2) -> List[int]:
    """Return the first `ceil(frac * num_layers)` layer indices (early-layer block)."""
    if num_layers <= 0:
        return []
    k = max(1, int(math.ceil(frac * num_layers)))
    return list(range(min(k, num_layers)))


__all__ = [
    "ArchInfo",
    "ATTENTION_PROJECTION_NAMES",
    "get_decoder_layers",
    "infer_family",
    "get_arch_info",
    "resolve_lora_targets",
    "validate_lora_targets",
    "deep_fraction_layers",
]
