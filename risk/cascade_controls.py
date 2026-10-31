"""Cascade and ADL controls (REINFORCEMENT_PLAN task P3.4).

Fragility score per asset from four conditions (each counted only when its input is available):

1. taker-flow variance compression: the 48-bar variance of the taker imbalance has fallen below half its
   trailing median (calm order flow before a cascade);
2. open interest above its 90th rolling percentile;
3. funding above 25% annualised;
4. passive depth within 10 bps down more than 50% over the last hour.

Two or more conditions met make the asset fragile. Then: directional gross in that asset is halved, new
carry entries in that asset are blocked (the plan's "de-lever the winning leg" needs leg resizing, which
the paper position manager does not do; the action is reported), and quoting is flagged to widen (one
condition) or pause (fragile).

Also: an ADL-rank estimate for open short perp legs, ``(mark / entry) * (notional / equity)``; collateral
marked at the median of the venue quotes with a 10% haircut when venues disagree by more than 1%; and a
stress scenario in which top-of-book depth falls 90% while 20% of each position is force-liquidated every
30 seconds.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import statistics

from pydantic import BaseModel, ConfigDict, Field


class CascadeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_variance_window: int = Field(default=48, ge=4)
    flow_variance_history: int = Field(default=96 * 7, ge=8)
    compression_ratio: float = Field(default=0.5, gt=0.0)
    oi_percentile_window: int = Field(default=96 * 30, ge=8)
    oi_percentile_threshold: float = Field(default=0.90, gt=0.0, lt=1.0)
    funding_annual_threshold: float = Field(default=0.25, gt=0.0)
    depth_window_ns: int = Field(default=3_600 * 1_000_000_000, gt=0)
    depth_drop_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    fragile_conditions: int = Field(default=2, ge=1)
    directional_multiplier_when_fragile: float = Field(default=0.5, ge=0.0, le=1.0)
    collateral_disagreement: float = Field(default=0.01, gt=0.0)
    collateral_haircut: float = Field(default=0.10, ge=0.0, lt=1.0)


class FragilityAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: str
    conditions_met: list[str]
    conditions_unavailable: list[str]
    flow_variance_ratio: float | None = None
    oi_percentile: float | None = None
    funding_annual: float | None = None
    depth_change_1h: float | None = None
    fragile: bool
    directional_multiplier: float = Field(ge=0.0, le=1.0)
    block_carry_entries: bool
    quote_action: str


class CascadeController:
    """Per-asset rolling state fed once per bar (flow, OI) and per snapshot (depth)."""

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self.config: CascadeConfig = config or CascadeConfig()
        self._imbalance: dict[str, deque[float]] = {}
        self._flow_var_hist: dict[str, deque[float]] = {}
        self._oi: dict[str, deque[float]] = {}
        self._depth: dict[str, deque[tuple[int, float]]] = {}
        self._funding: dict[str, float] = {}

    def update(
        self,
        asset: str,
        *,
        ts_ns: int,
        taker_buy_usd: float | None,
        taker_sell_usd: float | None,
        open_interest: float | None,
        funding_rate_1h: float | None,
        depth_10bps_usd: float | None,
    ) -> None:
        cfg = self.config
        a = asset.upper()
        if taker_buy_usd is not None and taker_sell_usd is not None and taker_buy_usd + taker_sell_usd > 0.0:
            imb = self._imbalance.setdefault(a, deque(maxlen=cfg.flow_variance_window))
            imb.append((taker_buy_usd - taker_sell_usd) / (taker_buy_usd + taker_sell_usd))
            if len(imb) >= cfg.flow_variance_window:
                self._flow_var_hist.setdefault(a, deque(maxlen=cfg.flow_variance_history)).append(statistics.pvariance(imb))
        if open_interest is not None and open_interest > 0.0:
            self._oi.setdefault(a, deque(maxlen=cfg.oi_percentile_window)).append(open_interest)
        if funding_rate_1h is not None:
            self._funding[a] = funding_rate_1h
        if depth_10bps_usd is not None and depth_10bps_usd > 0.0:
            dq = self._depth.setdefault(a, deque())
            dq.append((ts_ns, depth_10bps_usd))
            while dq and dq[0][0] < ts_ns - 2 * cfg.depth_window_ns:
                dq.popleft()

    def assess(self, asset: str, ts_ns: int) -> FragilityAssessment:
        cfg = self.config
        a = asset.upper()
        met: list[str] = []
        missing: list[str] = []

        ratio: float | None = None
        hist = self._flow_var_hist.get(a)
        if hist is not None and len(hist) >= 8:
            med = statistics.median(list(hist)[:-1]) if len(hist) > 1 else hist[-1]
            ratio = hist[-1] / med if med > 0 else None
        if ratio is None:
            missing.append("taker_flow_variance")
        elif ratio < cfg.compression_ratio:
            met.append("taker_flow_variance_compressed")

        pct: float | None = None
        oi = self._oi.get(a)
        if oi is not None and len(oi) >= 8:
            last = oi[-1]
            below = sum(1 for v in oi if v < last)
            equal = sum(1 for v in oi if v == last)
            pct = (below + 0.5 * equal) / len(oi)  # mid-rank: a flat series sits at the median, not the top
        if pct is None:
            missing.append("open_interest_history")
        elif pct > cfg.oi_percentile_threshold:
            met.append("open_interest_above_p90")

        funding_annual: float | None = None
        if a in self._funding:
            funding_annual = self._funding[a] * 8760.0
            if funding_annual > cfg.funding_annual_threshold:
                met.append("funding_above_25pct")
        else:
            missing.append("funding")

        change: float | None = None
        dq = self._depth.get(a)
        if dq:
            ref = [d for t, d in dq if t <= ts_ns - cfg.depth_window_ns]
            if ref:
                change = dq[-1][1] / ref[-1] - 1.0
        if change is None:
            missing.append("depth_history")
        elif change < -cfg.depth_drop_threshold:
            met.append("depth_down_50pct_1h")

        fragile = len(met) >= cfg.fragile_conditions
        return FragilityAssessment(
            asset=a,
            conditions_met=met,
            conditions_unavailable=missing,
            flow_variance_ratio=ratio,
            oi_percentile=pct,
            funding_annual=funding_annual,
            depth_change_1h=change,
            fragile=fragile,
            directional_multiplier=cfg.directional_multiplier_when_fragile if fragile else 1.0,
            block_carry_entries=fragile,
            quote_action="pause" if fragile else ("widen" if met else "normal"),
        )


def adl_rank_estimate(mark_price: float, entry_price: float, notional_usd: float, equity_usd: float) -> float:
    """Hyperliquid-style ADL priority proxy for a short leg: (mark / entry) * (notional / equity)."""
    if entry_price <= 0.0 or equity_usd <= 0.0:
        return 0.0
    return (mark_price / entry_price) * (notional_usd / equity_usd)


class CollateralMark(BaseModel):
    model_config = ConfigDict(frozen=True)

    mark: float
    haircut: float
    max_disagreement: float
    venues: int


def collateral_mark(venue_quotes: Sequence[float], config: CascadeConfig | None = None) -> CollateralMark | None:
    """Median of venue quotes; 10% haircut when any venue disagrees with the median by more than 1%."""
    cfg = config or CascadeConfig()
    quotes = [q for q in venue_quotes if q > 0.0]
    if not quotes:
        return None
    med = statistics.median(quotes)
    disagreement = max(abs(q / med - 1.0) for q in quotes)
    return CollateralMark(mark=med, haircut=cfg.collateral_haircut if disagreement > cfg.collateral_disagreement else 0.0, max_disagreement=disagreement, venues=len(quotes))


class StressPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: str
    notional_usd: float = Field(ge=0.0)
    depth_10bps_usd: float = Field(ge=0.0)


def liquidation_stress_loss(
    positions: Sequence[StressPosition],
    *,
    depth_loss: float = 0.9,
    fraction_per_step: float = 0.2,
    steps: int = 5,
) -> dict[str, float]:
    """Slippage cost (USD) of force-closing 20% of each position per 30 s into books that lost 90% of depth.

    Selling ``q`` into ``d`` of depth spread over 10 bps walks the book about ``10 bps * q / d`` on average
    (linear book), so each step costs ``q * 10bps * q / d / 2`` with the book refilling between steps.
    """
    out: dict[str, float] = {}
    for p in positions:
        remaining = p.notional_usd
        d = max(p.depth_10bps_usd * (1.0 - depth_loss), 1.0)
        loss = 0.0
        for _ in range(steps):
            q = remaining * fraction_per_step if remaining > 0 else 0.0
            loss += q * 0.0010 * (q / d) / 2.0
            remaining -= q
        out[p.asset] = out.get(p.asset, 0.0) + loss
    return out


def stablecoin_collateral_haircut(medians_and_quotes: Mapping[str, Sequence[float]], config: CascadeConfig | None = None) -> float:
    """Largest haircut among collateral stablecoins with venue quotes."""
    worst = 0.0
    for quotes in medians_and_quotes.values():
        cm = collateral_mark(quotes, config)
        if cm is not None:
            worst = max(worst, cm.haircut)
    return worst


__all__ = [
    "CascadeConfig",
    "CascadeController",
    "CollateralMark",
    "FragilityAssessment",
    "StressPosition",
    "adl_rank_estimate",
    "collateral_mark",
    "liquidation_stress_loss",
    "stablecoin_collateral_haircut",
]
