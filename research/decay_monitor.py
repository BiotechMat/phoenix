"""Signal decay monitor and retrain triggers (REINFORCEMENT_PLAN task P5.2).

For each live feature: a rolling Spearman IC against forward returns, the IC half-life (P1.8), and a breach
rule. The rolling IC is sign-aligned to the feature's long-run IC (so a feature that predicts negatively is
judged on predicting negatively); a breach is ``breach_windows`` consecutive windows below ``min_ic``. A
breach triggers a retrain. The recommended retrain cadence is the shortest half-life in the feature set.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from research.feature_eval import forward_log_returns, ic_half_life, spearman_ic


class DecayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon_bars: int = Field(default=4, ge=1)
    window_bars: int = Field(default=192, ge=20)
    step_bars: int = Field(default=96, ge=1)
    min_ic: float = 0.0
    breach_windows: int = Field(default=2, ge=1)
    half_life_max_lag: int = Field(default=48, ge=2)


class FeatureDecay(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    long_run_ic: float | None
    rolling_ic: list[float]
    aligned_rolling_ic: list[float]
    half_life_bars: float | None
    breached: bool


class DecayReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    features: dict[str, FeatureDecay]
    retrain_triggered: bool
    breached_features: list[str]
    recommended_cadence_bars: float | None


def rolling_ic(feature: NDArray[np.float64], fwd: NDArray[np.float64], window: int, step: int) -> list[float]:
    out: list[float] = []
    n = feature.shape[0]
    for end in range(window, n + 1, step):
        ic = spearman_ic(feature[end - window : end], fwd[end - window : end])
        out.append(float("nan") if ic is None else ic)
    return out


def evaluate_decay(features: Mapping[str, NDArray[np.float64]], closes: NDArray[np.float64], config: DecayConfig | None = None) -> DecayReport:
    cfg = config or DecayConfig()
    fwd = forward_log_returns(np.asarray(closes, dtype=np.float64), cfg.horizon_bars)
    out: dict[str, FeatureDecay] = {}
    breached: list[str] = []
    half_lives: list[float] = []
    for name, f in features.items():
        x = np.asarray(f, dtype=np.float64)
        lr = spearman_ic(x, fwd)
        roll = rolling_ic(x, fwd, cfg.window_bars, cfg.step_bars)
        sign = math.copysign(1.0, lr) if lr is not None and lr != 0.0 else 1.0
        aligned = [sign * v for v in roll]
        recent = [v for v in aligned[-cfg.breach_windows :] if math.isfinite(v)]
        is_breach = len(recent) == cfg.breach_windows and all(v < cfg.min_ic for v in recent)
        hl = ic_half_life(x, closes, cfg.half_life_max_lag)
        if hl is not None:
            half_lives.append(hl)
        if is_breach:
            breached.append(name)
        out[name] = FeatureDecay(name=name, long_run_ic=lr, rolling_ic=roll, aligned_rolling_ic=aligned, half_life_bars=hl, breached=is_breach)
    return DecayReport(features=out, retrain_triggered=bool(breached), breached_features=breached, recommended_cadence_bars=min(half_lives) if half_lives else None)


__all__ = ["DecayConfig", "DecayReport", "FeatureDecay", "evaluate_decay", "rolling_ic"]
