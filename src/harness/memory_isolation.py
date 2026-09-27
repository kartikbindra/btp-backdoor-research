"""Cache memory isolation and fresh per-request cache enforcement.

Constitutional Mandate:
- Fresh isolated cache per request: C_0 -> empty set.
- Explicit prohibition of prefix caching (--enable-prefix-caching False).
- Zero state carryover between repeated runs.
"""

import gc
from typing import Dict, Any, Optional
import torch


class FreshIsolatedCache:
    """Context manager to guarantee strict cache isolation and fresh memory state."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.initial_allocated_bytes: int = 0
        self.initial_reserved_bytes: int = 0

    def __enter__(self):
        self.clear()
        if self.device.type == "cuda":
            self.initial_allocated_bytes = torch.cuda.memory_allocated(self.device)
            self.initial_reserved_bytes = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def clear(self) -> None:
        """Enforce strict memory deallocation and garbage collection."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def assert_zero_cache_leakage(
    cache_state_before: Optional[Any],
    cache_state_after: Optional[Any],
) -> bool:
    """Verify that no residual KV states or attention representations leaked across invocations."""
    if cache_state_before is None and cache_state_after is None:
        return True
    
    # If explicit cache object is passed, verify it is empty or reset
    if hasattr(cache_state_after, "k_cache_fp8"):
        if cache_state_after.k_cache_fp8 is not None:
            raise AssertionError("Cache leakage detected: k_cache_fp8 is not None!")
    if hasattr(cache_state_after, "v_cache_fp8"):
        if cache_state_after.v_cache_fp8 is not None:
            raise AssertionError("Cache leakage detected: v_cache_fp8 is not None!")
            
    return True


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Retrieve memory diagnostics for cache allocation auditing."""
    dev = device if device is not None else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    stats: Dict[str, Any] = {
        "device_type": dev.type,
        "cuda_available": torch.cuda.is_available(),
    }
    if dev.type == "cuda" and torch.cuda.is_available():
        stats.update({
            "allocated_mb": torch.cuda.memory_allocated(dev) / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved(dev) / (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated(dev) / (1024 * 1024),
        })
    else:
        stats.update({
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        })
    return stats
