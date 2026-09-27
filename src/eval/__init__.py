"""Evaluation and metric computation modules for Campaign 002."""

from src.eval.metrics import (
    tensor_nrmse,
    tensor_cosine_similarity,
    logit_spearman_rank,
    output_jsd,
    top10_directional_agreement,
    token_match_rate,
    compute_bootstrap_ci,
    check_silent_fallback,
)
from src.eval.pilot_calibration import run_pilot_calibration
from src.eval.run_conformance import run_full_conformance

__all__ = [
    "tensor_nrmse",
    "tensor_cosine_similarity",
    "logit_spearman_rank",
    "output_jsd",
    "top10_directional_agreement",
    "token_match_rate",
    "compute_bootstrap_ci",
    "check_silent_fallback",
    "run_pilot_calibration",
    "run_full_conformance",
]
