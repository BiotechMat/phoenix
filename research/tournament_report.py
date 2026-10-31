"""Tournament report additions (REINFORCEMENT_PLAN task P5.4).

For each tournament candidate on the same bars:

- Monte Carlo trade-order shuffling: the candidate's per-trade net returns reshuffled ``n_shuffles`` times
  (seeded) to give the distribution of maximum drawdown that the realised order happened to avoid or hit.
- MAE / MFE by regime: maximum adverse and favourable excursion of each trade episode, grouped by the
  positioning regime at entry.
- Transfer coefficient: correlation between the candidate's raw signal and the executed position (the
  tournament executes signals as given, so this is 1 unless a position cap or rounding intervenes).
- Turnover and impact cost: sum of absolute position changes, and square-root impact on each change using
  the bar's dollar volume scaled to a day and the bar-level volatility.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from core.contracts import DollarBar, FeeSchedule
from core.features import compute_positioning_snapshots
from research.cpcv_gates import StrategyCandidate
from research.validation.cpcv_gatekeeper import compute_net_of_fee_returns
from risk.partial_adjustment import square_root_impact, transfer_coefficient


class CandidateReportExtras(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_id: str
    n_trades: int
    realised_max_drawdown: float
    shuffled_max_drawdown_p5: float | None
    shuffled_max_drawdown_p50: float | None
    shuffled_max_drawdown_p95: float | None
    mae_mfe_by_regime: dict[str, dict[str, float]]
    transfer_coefficient: float | None
    turnover: float
    impact_cost: float


def _episodes(pos: NDArray[np.float64]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(pos.shape[0]):
        s = np.sign(pos[i])
        if start is None and s != 0:
            start = i
        elif start is not None and (s == 0 or s != np.sign(pos[start])):
            out.append((start, i))
            start = i if s != 0 else None
    if start is not None:
        out.append((start, pos.shape[0]))
    return out


def _max_dd(r: NDArray[np.float64]) -> float:
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    return float(np.max(peak - eq)) if eq.size else 0.0


def candidate_report_extras(
    candidate: StrategyCandidate,
    bars: Sequence[DollarBar],
    *,
    fee: FeeSchedule | None = None,
    capital_usd: float = 5_000.0,
    n_shuffles: int = 500,
    seed: int = 20261031,
) -> CandidateReportExtras:
    fee = fee or FeeSchedule.hyperliquid()
    raw = np.asarray(candidate.position_generator(bars), dtype=np.float64).ravel()
    pos = candidate.generate_positions(bars)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    gross = np.zeros_like(closes)
    gross[:-1] = closes[1:] / closes[:-1] - 1.0
    bd = compute_net_of_fee_returns(gross, pos, fee_schedule=fee)
    net = bd.net_returns

    episodes = _episodes(pos)
    trade_returns = np.array([float(np.sum(net[a:b])) for a, b in episodes], dtype=np.float64)
    rng = np.random.default_rng(seed)  # research report only
    shuffled = [_max_dd(rng.permutation(trade_returns)) for _ in range(n_shuffles)] if trade_returns.size >= 2 else []

    regimes = [s.regime.value for s in compute_positioning_snapshots(bars)]
    by_regime: dict[str, list[tuple[float, float]]] = {}
    for a, b in episodes:
        path = np.cumsum(pos[a] * gross[a:b])
        by_regime.setdefault(regimes[a], []).append((float(np.min(np.concatenate(([0.0], path)))), float(np.max(np.concatenate(([0.0], path))))))
    mae_mfe = {reg: {"trades": float(len(v)), "mae_mean": float(np.mean([x[0] for x in v])), "mfe_mean": float(np.mean([x[1] for x in v]))} for reg, v in by_regime.items()}

    log_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
    sigma_daily = float(np.std(log_ret)) * math.sqrt(96.0)
    impact = 0.0
    for i, d in enumerate(bd.turnover):
        if d > 0:
            impact += d * square_root_impact(sigma_daily, d * capital_usd, bars[i].dollar_volume * 96.0)
    return CandidateReportExtras(
        strategy_id=candidate.strategy_id,
        n_trades=len(episodes),
        realised_max_drawdown=_max_dd(net),
        shuffled_max_drawdown_p5=float(np.percentile(shuffled, 5)) if shuffled else None,
        shuffled_max_drawdown_p50=float(np.percentile(shuffled, 50)) if shuffled else None,
        shuffled_max_drawdown_p95=float(np.percentile(shuffled, 95)) if shuffled else None,
        mae_mfe_by_regime=mae_mfe,
        transfer_coefficient=transfer_coefficient(raw, pos) if raw.size > 1 else None,
        turnover=float(np.sum(bd.turnover)),
        impact_cost=impact,
    )


def tournament_report(candidates: Sequence[StrategyCandidate], bars: Sequence[DollarBar], **kwargs: Any) -> dict[str, CandidateReportExtras]:
    return {c.strategy_id: candidate_report_extras(c, bars, **kwargs) for c in candidates}


def render_report(extras: dict[str, CandidateReportExtras]) -> str:
    lines = [f"{'candidate':<22}{'trades':>7}{'maxDD':>9}{'MC DD p5/p50/p95':>26}{'TC':>7}{'turnover':>10}{'impact':>10}"]
    for sid, e in extras.items():
        mc = "n/a" if e.shuffled_max_drawdown_p50 is None else f"{e.shuffled_max_drawdown_p5:.2%}/{e.shuffled_max_drawdown_p50:.2%}/{e.shuffled_max_drawdown_p95:.2%}"
        tc = "n/a" if e.transfer_coefficient is None else f"{e.transfer_coefficient:.2f}"
        lines.append(f"{sid:<22}{e.n_trades:>7}{e.realised_max_drawdown:>9.2%}{mc:>26}{tc:>7}{e.turnover:>10.1f}{e.impact_cost:>10.5f}")
        for reg, v in sorted(e.mae_mfe_by_regime.items()):
            lines.append(f"    {reg:<30} trades {int(v['trades']):>4}  MAE {v['mae_mean']:+.3%}  MFE {v['mfe_mean']:+.3%}")
    return "\n".join(lines)


__all__ = ["CandidateReportExtras", "candidate_report_extras", "render_report", "tournament_report"]
