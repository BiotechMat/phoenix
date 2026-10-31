"""Quantitative research and Combinatorial Purged Cross-Validation (CPCV) package."""

from __future__ import annotations

from research.validation.cpcv_gatekeeper import (
    CPCVSplit,
    ChampionChallengerGatekeeper,
    CombinatorialPurgedCV,
    NetReturnBreakdown,
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
    "NetReturnBreakdown",
    "compute_dsr",
    "compute_expected_max_sharpe",
    "compute_net_of_fee_returns",
    "compute_pbo",
    "compute_pbo_logits",
    "compute_psr",
    "evaluate_challenger_gates",
]
