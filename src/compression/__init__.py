"""FP8 KV-cache compression modules for Campaign 002 WP0/WP1 Runtime Gate."""

from src.compression.scales import (
    calculate_static_scale,
    detect_saturation,
    compute_scale_stats,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    FP8_E4M3_EPS,
)
from src.compression.fake_fp8 import (
    fake_fp8_quantize,
    quantize_fp8_e4m3fn_discrete,
    FP8QuantizeSTEFunction,
    FakeFP8CacheProxy,
)
from src.compression.storage_fp8 import (
    FP8KVStorage,
    factorize_quantization_noise,
)

__all__ = [
    "calculate_static_scale",
    "detect_saturation",
    "compute_scale_stats",
    "FP8_E4M3_MAX",
    "FP8_E4M3_MIN",
    "FP8_E4M3_EPS",
    "fake_fp8_quantize",
    "quantize_fp8_e4m3fn_discrete",
    "FP8QuantizeSTEFunction",
    "FakeFP8CacheProxy",
    "FP8KVStorage",
    "factorize_quantization_noise",
]
