"""Trend sleeve validation on hourly bars (REINFORCEMENT_PLAN task P2.2 acceptance).

Runs the causal trend backtest on aligned 1h closes, charges Hyperliquid maker fees plus slippage on
actual weight changes and signed hourly funding on held positions, then:

- validates each asset's net return stream with 45-split CPCV through the trial registry (family
  ``trend``), so the DSR accounts for every configuration tried;
- reports the portfolio's annualised Sharpe, total return, maximum drawdown and turnover;
- reports portfolio PnL in the three crisis scenarios of the replay harness.

The plan also asks for the PnL correlation with the derivatives sleeve. That sleeve's history cannot be
reconstructed (no historical options surface or HLP vault positions), so the field is reported as
unavailable rather than estimated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
import polars as pl
from pydantic import BaseModel, ConfigDict

from core.contracts import FeeSchedule
from research.replay import NAMED_SCENARIOS
from research.trial_registry import TrialRegistry, passes_min_backtest_length, run_registered_validation, BARS_PER_YEAR_1H
from research.validation.cpcv_gatekeeper import CombinatorialPurgedCV, compute_net_of_fee_returns
from sleeves.sleeve_2_trend import TrendSleeveConfig, backtest_trend

logger = logging.getLogger("phoenix.trend_eval")

CRISIS_SCENARIOS: Final[tuple[str, ...]] = ("carry_unwind_2024_08_05", "tariff_2025_02_03", "cascade_2025_10_10")
HOURS_PER_YEAR: Final[float] = 8760.0


class AssetValidation(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: str
    trial_id: str
    total_trials_used: int
    psr_vs_zero: float
    dsr: float
    pbo: float
    expected_net_pnl_bps: float
    sharpe_annual: float
    passed_cpcv_gates: bool
    min_btl_ok: bool
    sample_years: float
    min_btl_years: float


class TrendEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_utc: datetime
    end_utc: datetime
    n_hours: int
    assets: tuple[str, ...]
    config: TrendSleeveConfig
    portfolio_total_return: float
    portfolio_sharpe_annual: float
    portfolio_max_drawdown: float
    portfolio_turnover_per_year: float
    fees_total: float
    funding_total: float
    per_asset: list[AssetValidation]
    crisis_pnl: dict[str, float | None]
    crisis_positive_count: int
    crisis_scenarios_covered: int
    derivatives_correlation: float | None = None
    derivatives_correlation_note: str = "Derivatives sleeve history unavailable: no historical options surface or HLP vault positions."
    passes_crisis_rule: bool
    report_path: str | None = None


def load_hourly_closes(data_dir: Path, coins: Sequence[str]) -> tuple[NDArray[np.int64], dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """Aligned ``(ts_open_ms, closes, funding_1h)`` from ``{coin}usdt_perp_bars_1h.parquet`` on common timestamps."""
    frames: dict[str, pl.DataFrame] = {}
    for coin in coins:
        path = data_dir / f"{coin.lower()}usdt_perp_bars_1h.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing hourly bars for {coin}: {path}")
        df = pl.read_parquet(path).select(
            (pl.col("ts_open_ns") // 1_000_000).alias("ts_ms"),
            pl.col("close"),
            (pl.col("funding_rate_1h") if "funding_rate_1h" in pl.read_parquet_schema(path) else pl.col("funding_rate_8h") / 8.0).alias("funding_1h"),
        )
        frames[coin] = df.unique(subset=["ts_ms"]).sort("ts_ms")
    common = None
    for df in frames.values():
        ts = set(df.get_column("ts_ms").to_list())
        common = ts if common is None else common & ts
    ts_sorted = np.array(sorted(common or []), dtype=np.int64)
    closes: dict[str, NDArray[np.float64]] = {}
    funding: dict[str, NDArray[np.float64]] = {}
    for coin, df in frames.items():
        sub = df.filter(pl.col("ts_ms").is_in(ts_sorted.tolist())).sort("ts_ms")
        closes[coin] = sub.get_column("close").to_numpy().astype(np.float64)
        funding[coin] = sub.get_column("funding_1h").fill_null(0.0).to_numpy().astype(np.float64)
    return ts_sorted, closes, funding


def _sharpe_annual(returns: NDArray[np.float64]) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 3:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * math.sqrt(HOURS_PER_YEAR)) if sd > 1e-15 else 0.0


def _max_drawdown(returns: NDArray[np.float64]) -> float:
    equity = np.cumsum(np.where(np.isfinite(returns), returns, 0.0))
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    return float(np.max(peak - equity)) if equity.size else 0.0


def evaluate_trend(
    ts_ms: NDArray[np.int64],
    closes: Mapping[str, NDArray[np.float64]],
    funding_1h: Mapping[str, NDArray[np.float64]],
    *,
    config: TrendSleeveConfig | None = None,
    registry: TrialRegistry | None = None,
    fee_schedule: FeeSchedule | None = None,
    cpcv_groups: int = 10,
    cpcv_test_groups: int = 2,
    target_sharpe_for_min_btl: float = 1.0,
    report_dir: Path | None = None,
) -> TrendEvalReport:
    cfg = config or TrendSleeveConfig()
    fee = fee_schedule or FeeSchedule.hyperliquid()
    reg = registry or TrialRegistry()
    bt = backtest_trend(ts_ms, closes, cfg)
    n = bt.weights.shape[0] - 1  # last hour has no next return
    portfolio = np.zeros(n, dtype=np.float64)
    fees_total = 0.0
    funding_total = 0.0
    turnover_total = 0.0
    per_asset: list[AssetValidation] = []
    cpcv = CombinatorialPurgedCV(vertical_barrier_bars=24, n_groups=cpcv_groups, k_test_groups=cpcv_test_groups, embargo_pct=0.01)
    start = datetime.fromtimestamp(int(ts_ms[0]) / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp(int(ts_ms[-1]) / 1000, tz=timezone.utc)
    for j, asset in enumerate(bt.assets):
        pos = bt.weights[:n, j]
        gross = bt.simple_returns_next[:n, j]
        funding_8h = 8.0 * np.asarray(funding_1h[asset], dtype=np.float64)[:n]
        breakdown = compute_net_of_fee_returns(gross, pos, funding_rates_8h=funding_8h, fee_schedule=fee, bars_per_8h=8.0)
        portfolio += breakdown.net_returns
        fees_total += float(np.sum(breakdown.fee_and_slippage_costs))
        funding_total += float(np.sum(breakdown.funding_costs))
        turnover_total += float(np.sum(breakdown.turnover))
        result, record = run_registered_validation(
            reg,
            "trend",
            f"trend_{asset}_{'_'.join(str(x) for x in cfg.lookbacks_hours)}",
            gross,
            pos,
            cpcv=cpcv,
            funding_rates_8h=funding_8h,
            fee_schedule=fee,
            features=tuple(f"tanh_ret_{lb}h" for lb in cfg.lookbacks_hours),
            parameters=cfg.model_dump(mode="json"),
            window_start=start,
            window_end=end,
        )
        ok, years, need = passes_min_backtest_length(n, record.total_trials_used, target_sharpe_for_min_btl, BARS_PER_YEAR_1H)
        per_asset.append(
            AssetValidation(
                asset=asset,
                trial_id=record.trial_id,
                total_trials_used=record.total_trials_used,
                psr_vs_zero=result.psr_vs_zero,
                dsr=result.dsr,
                pbo=result.pbo,
                expected_net_pnl_bps=result.expected_net_pnl_bps,
                sharpe_annual=_sharpe_annual(breakdown.net_returns),
                passed_cpcv_gates=result.passed_all_gates,
                min_btl_ok=ok,
                sample_years=years,
                min_btl_years=need,
            )
        )

    crisis: dict[str, float | None] = {}
    covered = 0
    positive = 0
    ts_held = ts_ms[:n]
    for name in CRISIS_SCENARIOS:
        sc = NAMED_SCENARIOS[name]
        lo = int(sc.start_utc.timestamp() * 1000)
        hi = int(sc.end_utc.timestamp() * 1000)
        mask = (ts_held >= lo) & (ts_held < hi)
        if int(mask.sum()) < 24:
            crisis[name] = None
            continue
        pnl = float(np.sum(portfolio[mask]))
        crisis[name] = pnl
        covered += 1
        positive += 1 if pnl > 0.0 else 0

    years = n / HOURS_PER_YEAR
    report = TrendEvalReport(
        start_utc=start,
        end_utc=end,
        n_hours=n,
        assets=bt.assets,
        config=cfg,
        portfolio_total_return=float(np.sum(portfolio)),
        portfolio_sharpe_annual=_sharpe_annual(portfolio),
        portfolio_max_drawdown=_max_drawdown(portfolio),
        portfolio_turnover_per_year=turnover_total / years if years > 0 else 0.0,
        fees_total=fees_total,
        funding_total=funding_total,
        per_asset=per_asset,
        crisis_pnl=crisis,
        crisis_positive_count=positive,
        crisis_scenarios_covered=covered,
        passes_crisis_rule=covered == len(CRISIS_SCENARIOS) and positive >= 2,
    )
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / f"trend_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        report = report.model_copy(update={"report_path": str(out)})
    return report


def summarize(report: TrendEvalReport) -> str:
    lines = [
        f"Trend sleeve {report.start_utc:%Y-%m-%d} to {report.end_utc:%Y-%m-%d} ({report.n_hours} hours, {', '.join(report.assets)})",
        f"portfolio: return {report.portfolio_total_return:+.2%}, Sharpe {report.portfolio_sharpe_annual:.2f}, max DD {report.portfolio_max_drawdown:.2%}, "
        f"turnover {report.portfolio_turnover_per_year:.1f}x/yr, fees {report.fees_total:.2%}, funding paid {report.funding_total:+.2%}",
    ]
    for a in report.per_asset:
        lines.append(
            f"  {a.asset}: Sharpe {a.sharpe_annual:.2f}, PSR {a.psr_vs_zero:.3f}, DSR {a.dsr:.3f} (trials {a.total_trials_used}), PBO {a.pbo:.2f}, "
            f"E[net/trade] {a.expected_net_pnl_bps:+.1f} bps, gates {'pass' if a.passed_cpcv_gates else 'fail'}, MinBTL {'ok' if a.min_btl_ok else 'short'} "
            f"({a.sample_years:.1f}y vs {a.min_btl_years:.1f}y)"
        )
    for name, pnl in report.crisis_pnl.items():
        lines.append(f"  crisis {name}: {'not covered' if pnl is None else f'{pnl:+.2%}'}")
    lines.append(f"crisis rule (positive in >= 2 of 3): {'pass' if report.passes_crisis_rule else 'fail'}; derivatives correlation: unavailable")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the trend sleeve on stored hourly bars.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--coins", default="BTC,ETH,SOL")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "replays")
    args = parser.parse_args(list(argv) if argv is not None else None)
    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    ts, closes, funding = load_hourly_closes(args.data_dir, coins)
    if ts.size < TrendSleeveConfig().min_history_hours * 2:
        print(f"Only {ts.size} aligned hours; backfill 1h bars first (scripts/backfill.py --horizons 1h).")
        return 1
    report = evaluate_trend(ts, closes, funding, registry=TrialRegistry(args.registry) if args.registry else None, report_dir=args.report_dir)
    print(summarize(report))
    print(json.dumps({"report": report.report_path}))
    return 0


__all__ = ["AssetValidation", "CRISIS_SCENARIOS", "TrendEvalReport", "evaluate_trend", "load_hourly_closes", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
