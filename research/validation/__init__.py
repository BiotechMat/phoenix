"""Validation package for 45-split Combinatorial Purged Cross-Validation and Champion/Challenger gates."""

from __future__ import annotations

from research.validation.cpcv_gatekeeper import (
    CPCVSplit,
    ChampionChallengerGatekeeper,
    CombinatorialPurgedCV,
    MAX_CONSECUTIVE_FAILED_RETRAINS,
    MAX_CPCV_WORKERS,
    NetReturnBreakdown,
    OMP_THREADS_PER_WORKER,
    compute_dsr,
    compute_expected_max_sharpe,
    compute_net_of_fee_returns,
    compute_pbo,
    compute_pbo_logits,
    compute_psr,
    evaluate_challenger_gates,
)

__all__ = [
    "CPCVSplit",
    "ChampionChallengerGatekeeper",
    "CombinatorialPurgedCV",
    "MAX_CONSECUTIVE_FAILED_RETRAINS",
    "MAX_CPCV_WORKERS",
    "NetReturnBreakdown",
    "OMP_THREADS_PER_WORKER",
    "compute_dsr",
    "compute_expected_max_sharpe",
    "compute_net_of_fee_returns",
    "compute_pbo",
    "compute_pbo_logits",
    "compute_psr",
    "evaluate_challenger_gates",
]
