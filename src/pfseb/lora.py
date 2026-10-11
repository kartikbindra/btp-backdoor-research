"""Dependency-free LoRA for attention projections (no `peft` required).

Runs identically on CPU and GPU. Wraps selected nn.Linear layers with a low-rank adapter
out = W0 x + (alpha/r) * (x A^T) B^T, with W0 frozen, A ~ N(0, 0.01), B = 0 (so the adapter
starts as identity). Used for the PF-SEB MVP so the pilot has no external training deps.
"""

from typing import Iterable, Tuple, List
import torch
import torch.nn as nn


def _infer_float_dtype(model, default: "torch.dtype" = torch.float32) -> "torch.dtype":
    """First floating-point parameter dtype in the model (needed for 4-bit/QLoRA bases,
    whose quantized weights are uint8 and cannot be used to initialize LoRA params)."""
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return default


def _base_float_dtype(base: nn.Linear) -> "torch.dtype":
    """The floating dtype to use for LoRA params so they match the base's compute path.

    Order: bitsandbytes `compute_dtype` (Linear4bit) -> base weight dtype if float ->
    float32. This avoids uint8 (4-bit storage) and mixed-dtype matmuls.
    """
    cdt = getattr(base, "compute_dtype", None)
    if cdt is not None and getattr(cdt, "is_floating_point", False):
        return cdt
    wdt = base.weight.dtype
    if wdt.is_floating_point:
        return wdt
    return torch.float32


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dtype=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        dt = dtype if dtype is not None else _base_float_dtype(base)
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=dt) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling


def add_lora(model, targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
             r: int = 8, alpha: int = 16, dtype=None) -> int:
    """Replace matching Linear children with LoRALinear in place. Returns #layers wrapped.

    `dtype` is optional; when omitted each LoRALinear infers a matching floating dtype from
    its own base (so 4-bit/QLoRA bases with uint8 storage work correctly).
    """
    n = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dtype=dtype))
                n += 1
    return n


def lora_parameters(model) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def num_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
