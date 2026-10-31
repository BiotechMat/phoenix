"""Fee tiers, venue routing and order slicing (REINFORCEMENT_PLAN task P4.4).

- The Hyperliquid fee schedule (base rates, 14-day volume tiers, staking discounts) is read from the public
  ``userFees`` payload, so the table is never hard-coded. The account's tier comes from its 14-day volume
  (``userCross + userAdd`` over the last 14 days); the staking discount reduces positive fees only.
- ``best_hedge_venue`` picks the venue for a carry hedge's short perp leg with the highest predicted funding
  per day net of the amortised round-trip fee.
- ``slice_parent_order`` splits an entry larger than 2% of one-minute volume into passive-first child slices.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from core.contracts import FeeSchedule, SettlementCurrency

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


class VolumeTier(BaseModel):
    model_config = ConfigDict(frozen=True)

    cutoff_usd: float = Field(ge=0.0)
    taker_rate: float
    maker_rate: float


class FeeTable(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_taker_rate: float
    base_maker_rate: float
    tiers: list[VolumeTier]
    staking_discounts: list[tuple[float, float]] = Field(default_factory=list, description="(bps of max supply, discount)")


class UserFeeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    table: FeeTable
    volume_14d_usd: float = Field(ge=0.0)
    user_taker_rate: float | None = None
    user_maker_rate: float | None = None
    staking_discount: float = Field(default=0.0, ge=0.0, le=1.0)


def parse_user_fees(payload: Mapping[str, Any]) -> UserFeeInfo:
    fs = payload.get("feeSchedule") or {}
    tiers = [
        VolumeTier(cutoff_usd=float(t["ntlCutoff"]), taker_rate=float(t["cross"]), maker_rate=float(t["add"]))
        for t in (fs.get("tiers") or {}).get("vip", []) or []
    ]
    tiers.sort(key=lambda t: t.cutoff_usd)
    staking = [(float(s["bpsOfMaxSupply"]), float(s["discount"])) for s in fs.get("stakingDiscountTiers", []) or []]
    table = FeeTable(base_taker_rate=float(fs.get("cross", 0.00045)), base_maker_rate=float(fs.get("add", 0.00015)), tiers=tiers, staking_discounts=staking)
    days = sorted(payload.get("dailyUserVlm", []) or [], key=lambda d: str(d.get("date", "")))[-14:]
    volume = sum(float(d.get("userCross", 0.0)) + float(d.get("userAdd", 0.0)) for d in days)
    active = payload.get("activeStakingDiscount") or {}
    return UserFeeInfo(
        table=table,
        volume_14d_usd=volume,
        user_taker_rate=float(payload["userCrossRate"]) if payload.get("userCrossRate") is not None else None,
        user_maker_rate=float(payload["userAddRate"]) if payload.get("userAddRate") is not None else None,
        staking_discount=float(active.get("discount", 0.0) or 0.0),
    )


def fetch_user_fees(address: str, url: str = HYPERLIQUID_INFO_URL, timeout_sec: float = 5.0) -> UserFeeInfo:
    req = urllib.request.Request(url, data=json.dumps({"type": "userFees", "user": address}).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return parse_user_fees(json.loads(resp.read().decode("utf-8")))


def rates_for_volume(table: FeeTable, volume_14d_usd: float, staking_discount: float = 0.0) -> tuple[float, float]:
    """(taker, maker) rates for a 14-day volume; the staking discount applies to positive rates only."""
    taker, maker = table.base_taker_rate, table.base_maker_rate
    for tier in table.tiers:
        if volume_14d_usd >= tier.cutoff_usd:
            taker, maker = tier.taker_rate, tier.maker_rate
    d = min(max(staking_discount, 0.0), 1.0)
    return (taker * (1.0 - d) if taker > 0 else taker, maker * (1.0 - d) if maker > 0 else maker)


def tier_index(table: FeeTable, volume_14d_usd: float) -> int:
    return sum(1 for t in table.tiers if volume_14d_usd >= t.cutoff_usd)


def fee_schedule_for(table: FeeTable, volume_14d_usd: float, staking_discount: float = 0.0, slippage_bps: float = 0.5) -> FeeSchedule:
    taker, maker = rates_for_volume(table, volume_14d_usd, staking_discount)
    return FeeSchedule(settlement=SettlementCurrency.USDC, maker_bps=max(0.0, maker * 1e4), taker_bps=max(0.0, taker * 1e4), estimated_slippage_bps=slippage_bps)


class HedgeVenueChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    venue: str
    net_funding_per_day: float
    funding_per_day: float
    fee_per_day: float


def best_hedge_venue(
    predicted_funding_per_day: Mapping[str, float],
    roundtrip_fee_by_venue: Mapping[str, float],
    holding_days: float = 3.0,
) -> HedgeVenueChoice | None:
    """Venue whose short perp earns the most predicted funding per day after amortising its round-trip fee."""
    best: HedgeVenueChoice | None = None
    for venue, f_day in predicted_funding_per_day.items():
        fee = roundtrip_fee_by_venue.get(venue)
        if fee is None:
            continue
        fee_day = fee / max(holding_days, 1e-9)
        choice = HedgeVenueChoice(venue=venue, net_funding_per_day=f_day - fee_day, funding_per_day=f_day, fee_per_day=fee_day)
        if best is None or choice.net_funding_per_day > best.net_funding_per_day:
            best = choice
    return best


class ChildSlice(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    notional_usd: float = Field(gt=0.0)
    passive: bool


def slice_parent_order(notional_usd: float, one_minute_volume_usd: float, max_participation: float = 0.02, passive_first: int = 2) -> list[ChildSlice]:
    """Split a parent order into slices of at most ``max_participation`` x one-minute volume, the first ones passive."""
    if notional_usd <= 0.0:
        return []
    cap = max_participation * one_minute_volume_usd
    if cap <= 0.0 or notional_usd <= cap:
        return [ChildSlice(index=0, notional_usd=notional_usd, passive=True)]
    n = int(-(-notional_usd // cap))
    size = notional_usd / n
    return [ChildSlice(index=i, notional_usd=size, passive=i < passive_first) for i in range(n)]


__all__ = [
    "ChildSlice",
    "FeeTable",
    "HedgeVenueChoice",
    "UserFeeInfo",
    "VolumeTier",
    "best_hedge_venue",
    "fee_schedule_for",
    "fetch_user_fees",
    "parse_user_fees",
    "rates_for_volume",
    "slice_parent_order",
    "tier_index",
]

