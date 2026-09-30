"""Dependency-free LoRA for attention projections (no `peft` required).

Runs identically on CPU and GPU. Wraps selected nn.Linear layers with a low-rank adapter
out = W0 x + (alpha/r) * (x A^T) B^T, with W0 frozen, A ~ N(0, 0.01), B = 0 (so the adapter
starts as identity). Used for the PF-SEB MVP so the pilot has no external training deps.
"""

from typing import Iterable, Tuple, List
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=dt) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling


def add_lora(model, targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
             r: int = 8, alpha: int = 16) -> int:
    """Replace matching Linear children with LoRALinear in place. Returns #layers wrapped."""
    n = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
                n += 1
    return n


def lora_parameters(model) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def num_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
