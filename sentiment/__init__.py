"""Persistent mlx-lm Macro & Systemic Risk Daemon package."""

from __future__ import annotations

from core.contracts import SystemicRiskOutput, compute_regime_multiplier_from_asri
from sentiment.mlx_risk_daemon import (
    DEFAULT_MODEL_ID,
    DEFAULT_REDIS_STREAM_KEY,
    AsyncSystemicFeedAggregator,
    DeterministicMockBatchGenerator,
    MacroCryptoFeedSnapshot,
    PersistentMLXRiskDaemon,
    RegimePublishEnvelope,
    SystemicRiskIPCBus,
    parse_and_validate_llm_risk_json,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_REDIS_STREAM_KEY",
    "AsyncSystemicFeedAggregator",
    "DeterministicMockBatchGenerator",
    "MacroCryptoFeedSnapshot",
    "PersistentMLXRiskDaemon",
    "RegimePublishEnvelope",
    "SystemicRiskIPCBus",
    "SystemicRiskOutput",
    "compute_regime_multiplier_from_asri",
    "parse_and_validate_llm_risk_json",
]
