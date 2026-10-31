"""Partial adjustment, impact and transfer coefficient (REINFORCEMENT_PLAN task P3.1).

- ``PartialAdjuster`` moves a live weight toward its aim by ``lambda = 1 - exp(-dt / half_life)`` per step
  (Garleanu-Pedersen), with the half-life set per sleeve from the signal's IC half-life (P1.8).
- ``discretise`` rounds weights to a grid (0.05 by default) so tiny adjustments do not trade.
- ``square_root_impact`` is ``k * sigma * sqrt(Q / V)`` with ``k = 0.5``, daily volatility ``sigma``,
  order notional ``Q`` and 24-hour venue volume ``V``; it is subtracted from expected return.
- ``uncertainty_kelly_gamma`` is ``0.5 * min(1, edge / (2 * SE(edge)))``.
- ``transfer_coefficient`` is the cross-sectional correlation between raw scores and executed weights.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


def adjustment_rate(dt: float, half_life: float) -> float:
    """``1 - exp(-dt / half_life)``; 1.0 when ``half_life`` is not positive (jump straight to the aim)."""
    if half_life <= 0.0:
        return 1.0
    return 1.0 - math.exp(-max(dt, 0.0) / half_life)


def discretise(weight: float, step: float = 0.05) -> float:
    """Round to the nearest multiple of ``step`` (rounding toward zero would hold weights below their aims)."""
    if step <= 0.0:
        return weight
    return math.copysign(math.floor(abs(weight) / step + 0.5) * step, weight)


def square_root_impact(sigma_daily: float, notional_usd: float, volume_24h_usd: float, k: float = 0.5) -> float:
    """Expected impact as a fraction of price: ``k * sigma * sqrt(Q / V)`` (0 when volume is unknown)."""
    if volume_24h_usd <= 0.0 or notional_usd <= 0.0 or sigma_daily <= 0.0:
        return 0.0
    return k * sigma_daily * math.sqrt(notional_usd / volume_24h_usd)


def uncertainty_kelly_gamma(edge: float, edge_standard_error: float) -> float:
    """Half-Kelly scaled down when the edge is not at least two standard errors."""
    if edge <= 0.0:
        return 0.0
    if edge_standard_error <= 0.0:
        return 0.5
    return 0.5 * min(1.0, edge / (2.0 * edge_standard_error))


def transfer_coefficient(scores: Sequence[float] | NDArray[np.float64], weights: Sequence[float] | NDArray[np.float64]) -> float | None:
    """Correlation between raw signal scores and executed weights across assets (None when undefined)."""
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if s.size < 2 or float(np.std(s)) < 1e-15 or float(np.std(w)) < 1e-15:
        return None
    return float(np.corrcoef(s, w)[0, 1])


class PartialAdjusterConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    half_life_bars: dict[str, float] = Field(default_factory=dict, description="Per-sleeve half-life in bars; missing sleeves jump to the aim")
    step: float = Field(default=0.05, ge=0.0)


class PartialAdjuster:
    def __init__(self, config: PartialAdjusterConfig | None = None) -> None:
        self.config: PartialAdjusterConfig = config or PartialAdjusterConfig()

    def step_weight(self, sleeve_id: str, aim: float, current: float, dt_bars: float = 1.0) -> float:
        lam = adjustment_rate(dt_bars, self.config.half_life_bars.get(sleeve_id, 0.0))
        return discretise(current + lam * (aim - current), self.config.step)

    def path(self, sleeve_id: str, aims: Sequence[float], start: float = 0.0) -> list[float]:
        """Apply the adjuster along a sequence of aims (one step per element)."""
        out: list[float] = []
        cur = start
        for a in aims:
            if a == 0.0 or (cur != 0.0 and math.copysign(1.0, a) != math.copysign(1.0, cur)):
                cur = 0.0 if a == 0.0 else self.step_weight(sleeve_id, a, 0.0)  # flatten on a cleared or flipped aim
            else:
                cur = self.step_weight(sleeve_id, a, cur)
            out.append(cur)
        return out


__all__ = [
    "PartialAdjuster",
    "PartialAdjusterConfig",
    "adjustment_rate",
    "discretise",
    "square_root_impact",
    "transfer_coefficient",
    "uncertainty_kelly_gamma",
]

