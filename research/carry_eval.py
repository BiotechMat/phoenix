"""Carry v1 vs v2 evaluation on Hyperliquid settlement history (REINFORCEMENT_PLAN task P2.1 acceptance).

A spot-long / perp-short carry is simulated settlement by settlement on Hyperliquid's own history:

- funding received while short: ``rate_per_hour * interval_hours`` at each settlement;
- basis convergence: ``premium_before - premium_after`` (short the perp at a premium, gain as it closes);
- costs: half the round-trip cost (both legs) on every change in size.

v1 is the live rule (enter at >= 3 bps per 8h with the breakeven and net-edge checks, hysteresis exit
below 0.5 bps per 8h). v2 is ``carry_v2_decide``. Both use only information available at the decision
time. Each asset's net return stream is validated with CPCV through the trial registry (family
``carry``); the report adds E[net] per round trip, maximum drawdown, and the drawdown inside the August
2024 window against an 11-day funding-inversion bound taken from the worst 11 days in the sample.

Limits: the premium index is Hyperliquid's sampled premium, a proxy for the executable perp-spot basis;
the spot leg's fee uses the perp maker rate; open interest is unavailable historically, so the crowding
cut uses funding only.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from core.contracts import FeeSchedule
from data.backfill_storage import FundingSettlement, fetch_hyperliquid_settlements, load_settlements, save_settlements
from research.replay import NAMED_SCENARIOS
from research.trial_registry import TrialRegistry, run_registered_validation
from research.validation.cpcv_gatekeeper import CombinatorialPurgedCV
from sleeves.sleeve_3_funding_carry import CarryV2Config, FundingCarrySleeve, carry_v2_decide

logger = logging.getLogger("phoenix.carry_eval")

INTEREST_PER_HOUR: Final[float] = 0.0001 / 8.0
ETHENA_INVERSION_DAYS: Final[int] = 11
MIN_NET_PER_ROUND_TRIP_BPS: Final[float] = 15.0


class CarryRuleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: str
    rule: Literal["v1", "v2"]
    n_steps: int
    round_trips: int
    hours_in_position: float
    total_return: float
    funding_return: float
    basis_return: float
    cost_return: float
    net_per_round_trip_bps: float
    sharpe_annual: float
    max_drawdown: float
    aug_2024_drawdown: float | None
    inversion_bound: float
    within_inversion_bound: bool | None
    psr_vs_zero: float
    dsr: float
    pbo: float
    trial_id: str
    passes: bool
    reasons: list[str]


def _trailing_mean(rates: NDArray[np.float64], hours: NDArray[np.float64], window_hours: int) -> NDArray[np.float64]:
    """Time-weighted mean of hourly rates over the trailing window (inclusive of the current step)."""
    w = rates * hours
    cw = np.concatenate(([0.0], np.cumsum(w)))
    ch = np.concatenate(([0.0], np.cumsum(hours)))
    out = np.empty(rates.shape[0], dtype=np.float64)
    j = 0
    for i in range(rates.shape[0]):
        while ch[i + 1] - ch[j] > window_hours and j < i:
            j += 1
        span = ch[i + 1] - ch[j]
        out[i] = (cw[i + 1] - cw[j]) / span if span > 0 else 0.0
    return out


def simulate(settlements: Sequence[FundingSettlement], rule: Literal["v1", "v2"], config: CarryV2Config | None = None, fee: FeeSchedule | None = None) -> dict[str, NDArray[np.float64]]:
    """Per-step size (decided at step i, held to i+1) and the components of net return."""
    cfg = config or CarryV2Config()
    sched = fee or FeeSchedule.hyperliquid()
    n = len(settlements)
    rates = np.array([s.rate_per_hour for s in settlements], dtype=np.float64)
    hours = np.array([s.interval_hours for s in settlements], dtype=np.float64)
    prem = np.array([s.premium for s in settlements], dtype=np.float64)
    trailing = _trailing_mean(rates, hours, cfg.trailing_hours)
    rt_cost = FundingCarrySleeve.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=True) * 1e-4

    size = np.zeros(n, dtype=np.float64)
    v1 = FundingCarrySleeve()
    for i in range(n):
        if rule == "v2":
            d = carry_v2_decide(
                in_position=bool(i > 0 and size[i - 1] > 0.0),
                premium=float(prem[i]),
                funding_per_hour=float(rates[i]),
                trailing_funding_per_hour=float(trailing[i]),
                config=cfg.model_copy(update={"roundtrip_cost": rt_cost}),
            )
            size[i] = d.size
        else:
            ev = v1.evaluate_opportunity("X", funding_rate_1h=float(rates[i]), funding_rate_8h=float(rates[i] * 8.0), fee_schedule=sched)
            size[i] = 1.0 if ev.is_actionable and ev.direction.value == "SPOT_LONG_PERP_SHORT" else 0.0

    held = np.concatenate(([0.0], size[:-1]))  # size decided at i-1 is held into settlement i
    funding = held * rates * hours
    basis = np.zeros(n, dtype=np.float64)
    basis[1:] = held[1:] * (prem[:-1] - prem[1:])
    prev = np.concatenate(([0.0], size[:-1]))
    cost = np.abs(size - prev) * rt_cost / 2.0
    return {"size": size, "held": held, "funding": funding, "basis": basis, "cost": cost, "net": funding + basis - cost, "hours": hours}


def _drawdown(net: NDArray[np.float64]) -> float:
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    return float(np.max(peak - eq)) if eq.size else 0.0


def _round_trips(size: NDArray[np.float64]) -> int:
    on = size > 0.0
    return int(np.sum(on[1:] & ~on[:-1]) + (1 if on.size and on[0] else 0))


def inversion_bound(settlements: Sequence[FundingSettlement], days: int = ETHENA_INVERSION_DAYS) -> float:
    """Loss a short-perp leg takes over the worst ``days``-day stretch of funding in the sample (positive number)."""
    t = np.array([s.t_ms for s in settlements], dtype=np.int64)
    f = np.array([s.rate_per_hour * s.interval_hours for s in settlements], dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(f)))
    window = days * 86_400_000
    worst = 0.0
    j = 0
    for i in range(t.shape[0]):
        while t[i] - t[j] > window:
            j += 1
        worst = min(worst, float(c[i + 1] - c[j]))
    return -worst


def evaluate_asset(
    asset: str,
    settlements: Sequence[FundingSettlement],
    rule: Literal["v1", "v2"],
    registry: TrialRegistry,
    config: CarryV2Config | None = None,
) -> CarryRuleResult:
    sim = simulate(settlements, rule, config)
    net = sim["net"]
    size = sim["size"]
    trips = _round_trips(size)
    total = float(np.sum(net))
    per_trip = total / trips * 1e4 if trips else 0.0
    years = float(np.sum(sim["hours"])) / 8760.0
    steps_per_year = len(net) / years if years > 0 else 8760.0
    sd = float(np.std(net, ddof=1)) if net.size > 2 else 0.0
    sharpe = float(np.mean(net) / sd * math.sqrt(steps_per_year)) if sd > 1e-15 else 0.0

    t = np.array([s.t_ms for s in settlements], dtype=np.int64)
    sc = NAMED_SCENARIOS["carry_unwind_2024_08_05"]
    mask = (t >= int(sc.start_utc.timestamp() * 1000)) & (t < int(sc.end_utc.timestamp() * 1000))
    aug_dd = _drawdown(net[mask]) if int(mask.sum()) > 24 else None
    bound = inversion_bound(settlements)

    cpcv = CombinatorialPurgedCV(vertical_barrier_bars=72, n_groups=10, k_test_groups=2, embargo_pct=0.01)
    zero_fee = FeeSchedule(maker_bps=0.0, taker_bps=0.0, estimated_slippage_bps=0.0)
    result, record = run_registered_validation(
        registry,
        "carry",
        f"carry_{rule}_{asset}",
        net,
        np.where(sim["held"] > 0.0, 1.0, 0.0) + np.where(sim["held"] > 0.0, 0.0, 1e-12),  # unit "position" so CPCV scores the net stream as given
        cpcv=cpcv,
        fee_schedule=zero_fee,
        parameters={"rule": rule, **(config or CarryV2Config()).model_dump(mode="json")},
        window_start=datetime.fromtimestamp(int(t[0]) / 1000, tz=timezone.utc),
        window_end=datetime.fromtimestamp(int(t[-1]) / 1000, tz=timezone.utc),
        notes="net carry return stream (funding + basis convergence - costs) per settlement",
    )
    within = (aug_dd <= bound) if aug_dd is not None else None
    reasons: list[str] = []
    ok = True
    if per_trip <= MIN_NET_PER_ROUND_TRIP_BPS:
        ok = False
        reasons.append(f"E[net] per round trip {per_trip:.1f} bps <= {MIN_NET_PER_ROUND_TRIP_BPS:.0f}")
    if not result.passed_all_gates:
        ok = False
        reasons.append(f"CPCV gates fail (PSR {result.psr_vs_zero:.3f}, DSR {result.dsr:.3f}, PBO {result.pbo:.2f})")
    if within is False:
        ok = False
        reasons.append(f"Aug 2024 drawdown {aug_dd:.4%} exceeds the 11-day inversion bound {bound:.4%}")
    if ok:
        reasons.append("passes")
    return CarryRuleResult(
        asset=asset,
        rule=rule,
        n_steps=len(net),
        round_trips=trips,
        hours_in_position=float(np.sum(sim["held"] * sim["hours"])),
        total_return=total,
        funding_return=float(np.sum(sim["funding"])),
        basis_return=float(np.sum(sim["basis"])),
        cost_return=-float(np.sum(sim["cost"])),
        net_per_round_trip_bps=per_trip,
        sharpe_annual=sharpe,
        max_drawdown=_drawdown(net),
        aug_2024_drawdown=aug_dd,
        inversion_bound=bound,
        within_inversion_bound=within,
        psr_vs_zero=result.psr_vs_zero,
        dsr=result.dsr,
        pbo=result.pbo,
        trial_id=record.trial_id,
        passes=ok,
        reasons=reasons,
    )


def summarize(results: Sequence[CarryRuleResult]) -> str:
    lines = [f"{'asset':<5}{'rule':<5}{'trips':>7}{'in-pos h':>10}{'return':>9}{'funding':>9}{'basis':>9}{'costs':>9}{'net/trip':>10}{'Sharpe':>8}{'maxDD':>8}{'DSR':>7}  verdict"]
    for r in results:
        lines.append(
            f"{r.asset:<5}{r.rule:<5}{r.round_trips:>7}{r.hours_in_position:>10.0f}{r.total_return:>+9.2%}{r.funding_return:>+9.2%}{r.basis_return:>+9.2%}"
            f"{r.cost_return:>+9.2%}{r.net_per_round_trip_bps:>9.1f}b{r.sharpe_annual:>8.2f}{r.max_drawdown:>8.2%}{r.dsr:>7.3f}  {'; '.join(r.reasons)}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description="Evaluate carry v1 and v2 on Hyperliquid settlement history.")
    parser.add_argument("--coins", default="BTC,ETH,SOL")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "funding_history")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--start", default="2023-06-08", help="Default: first day of hourly settlement")
    args = parser.parse_args(list(argv) if argv is not None else None)
    reg = TrialRegistry(args.registry) if args.registry else TrialRegistry()
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    results: list[CarryRuleResult] = []
    for coin in [c.strip().upper() for c in args.coins.split(",") if c.strip()]:
        path = args.cache_dir / f"{coin.lower()}_funding.parquet"
        if path.exists():
            st = load_settlements(path)
        else:
            st = fetch_hyperliquid_settlements(coin, start_ms, int(_time.time() * 1000))
            save_settlements(st, path)
        st = [s for s in st if s.t_ms >= start_ms]
        for rule in ("v1", "v2"):
            results.append(evaluate_asset(coin, st, rule, reg))
    print(summarize(results))
    return 0


__all__ = ["CarryRuleResult", "evaluate_asset", "inversion_bound", "simulate", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
