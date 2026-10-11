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


class LoRALinear(nn.Module):
    """LoRA wrapper that is safe across fp32/fp16/bf16 and 4-bit (QLoRA) bases.

    The adapter matrices are always stored in float32 (numerically stable, and tiny), and
    are cast to the activation dtype inside `forward`. This avoids both the uint8 dtype of
    bitsandbytes 4-bit storage and mixed-dtype matmuls (e.g. hidden states fp16 vs a bf16
    `compute_dtype`).
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dtype=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        dt = dtype if dtype is not None else torch.float32
        if not dt.is_floating_point:
            dt = torch.float32
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=dt) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        a = self.A.to(x.dtype)
        b = self.B.to(x.dtype)
        lora = (x @ a.t() @ b.t()) * self.scaling
        return out + lora.to(out.dtype)


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
