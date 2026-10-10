"""Campaign 006 model registry.

Tier A (cross-architecture at ~1B, BF16 LoRA on a 16 GB T4):
    qwen15   Qwen/Qwen2.5-1.5B-Instruct   (replication anchor)
    llama1b  meta-llama/Llama-3.2-1B-Instruct
    gemma2b  google/gemma-2-2b-it
Tier B (within-family scale, QLoRA 4-bit for >=3B):
    qwen05   Qwen/Qwen2.5-0.5B-Instruct
    qwen3b   Qwen/Qwen2.5-3B-Instruct
    qwen7b   Qwen/Qwen2.5-7B-Instruct
    mistral7b mistralai/Mistral-7B-Instruct-v0.3   (sliding-window attention)

The registry is code (not YAML) so runners import it without extra dependencies; a
human-readable YAML mirror lives at configs/campaign_006/models.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    tier: str
    family: str
    load_in_4bit: bool = False
    revision: Optional[str] = None
    note: str = ""
    lora_r: int = 8
    lora_alpha: int = 16

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key, "model_id": self.model_id, "tier": self.tier,
            "family": self.family, "load_in_4bit": self.load_in_4bit,
            "revision": self.revision, "note": self.note,
            "lora_r": self.lora_r, "lora_alpha": self.lora_alpha,
        }


MODELS: Dict[str, ModelSpec] = {
    "qwen15": ModelSpec(
        key="qwen15", model_id="Qwen/Qwen2.5-1.5B-Instruct", tier="A",
        family="qwen2", note="replication anchor (Campaign 003/004/005 target)"),
    "llama1b": ModelSpec(
        key="llama1b", model_id="meta-llama/Llama-3.2-1B-Instruct", tier="A",
        family="llama", note="cross-family GQA anchor"),
    "gemma2b": ModelSpec(
        key="gemma2b", model_id="google/gemma-2-2b-it", tier="A",
        family="gemma", note="cross-family, large vocab, sliding window in alternate layers"),
    "qwen05": ModelSpec(
        key="qwen05", model_id="Qwen/Qwen2.5-0.5B-Instruct", tier="B",
        family="qwen2", note="scale point (small)"),
    "qwen3b": ModelSpec(
        key="qwen3b", model_id="Qwen/Qwen2.5-3B-Instruct", tier="B",
        family="qwen2", load_in_4bit=True, note="scale point (QLoRA)"),
    "qwen7b": ModelSpec(
        key="qwen7b", model_id="Qwen/Qwen2.5-7B-Instruct", tier="B",
        family="qwen2", load_in_4bit=True, note="scale point (QLoRA)"),
    "mistral7b": ModelSpec(
        key="mistral7b", model_id="mistralai/Mistral-7B-Instruct-v0.3", tier="B",
        family="mistral", load_in_4bit=True, note="sliding-window attention (SWA), QLoRA"),
}


def get_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model key '{key}'. Options: {sorted(MODELS)}")
    return MODELS[key]


def tier_models(tier: str) -> List[ModelSpec]:
    return [m for m in MODELS.values() if m.tier.upper() == tier.upper()]


__all__ = ["ModelSpec", "MODELS", "get_spec", "tier_models"]
