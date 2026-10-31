"""Replay harness: drive the live tick from Parquet bars with no network (REINFORCEMENT_PLAN P1.7).

The harness feeds ``ControlCenterStateStore.step_simulation_tick`` through ``ScriptedBarFeed``, so
the replay runs the identical code path as live trading: features, sleeves, HRP, position
lifecycle, funding and fees. Sleeves are pluggable. Output is a ``ReplayReport`` with per-trade
records, an equity curve, turnover, fees and per-sleeve attribution, written as JSON.

Model training inside a replay is causal: the store's Parquet training store is a temporary file
holding only rows that close before the scenario starts.

Usage::

    .venv/bin/python -m research.replay --scenario calm_2025_09
    .venv/bin/python -m research.replay --start 2025-10-08 --end 2025-10-13 --name my_window
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import AssetMarketSnapshot, DataAvailability, DollarBar, MultiAssetMarketSnapshot
from core.data_ingestion import ScriptedBarFeed, normalize_coin_symbol
from risk.circuit_breakers import CircuitBreakers
from sleeves.sleeve_1_derivatives_alpha import DerivativesAlphaSleeve
from sleeves.sleeve_3_funding_carry import FundingCarrySleeve
from ui.api_server import REAL_BAR_SOURCES, ControlCenterStateStore

logger = logging.getLogger("phoenix.replay")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR: Final[Path] = PROJECT_ROOT / "data" / "replays"
DEFAULT_COINS: Final[tuple[str, str, str]] = ("BTC", "ETH", "SOL")
STEP_15M_NS: Final[int] = 900_000_000_000


class ReplayScenario(BaseModel):
    """A named replay window."""

    model_config = ConfigDict(frozen=True)

    name: str
    start_utc: datetime
    end_utc: datetime
    coins: tuple[str, ...] = DEFAULT_COINS
    warmup_bars: int = Field(default=600, ge=16)
    description: str = ""

    @property
    def start_ns(self) -> int:
        return int(self.start_utc.timestamp() * 1_000_000_000)

    @property
    def end_ns(self) -> int:
        return int(self.end_utc.timestamp() * 1_000_000_000)


def _utc(y: int, m: int, d: int, hh: int = 0) -> datetime:
    return datetime(y, m, d, hh, tzinfo=timezone.utc)


NAMED_SCENARIOS: Final[dict[str, ReplayScenario]] = {
    "calm_2025_09": ReplayScenario(
        name="calm_2025_09",
        start_utc=_utc(2025, 9, 1),
        end_utc=_utc(2025, 10, 1),
        description="A calm month: the baseline for turnover and false-lockdown checks.",
    ),
    "cascade_2025_10_10": ReplayScenario(
        name="cascade_2025_10_10",
        start_utc=_utc(2025, 10, 8),
        end_utc=_utc(2025, 10, 13),
        description="The 10 October 2025 liquidation cascade.",
    ),
    "tariff_2025_02_03": ReplayScenario(
        name="tariff_2025_02_03",
        start_utc=_utc(2025, 2, 1),
        end_utc=_utc(2025, 2, 6),
        description="The 3 February 2025 tariff shock.",
    ),
    "carry_unwind_2024_08_05": ReplayScenario(
        name="carry_unwind_2024_08_05",
        start_utc=_utc(2024, 8, 3),
        end_utc=_utc(2024, 8, 8),
        description="The 5 August 2024 yen carry unwind.",
    ),
}


class EquityPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    ts_ns: int
    bar_index: int
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    marked_equity_usd: float
    open_positions: int


class ReplayReport(BaseModel):
    """Everything a later gate needs from one replay."""

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    start_utc: datetime
    end_utc: datetime
    bars_available: int
    bars_processed: int
    elapsed_seconds: float
    model_available: bool
    starting_capital_usd: float
    final_equity_usd: float
    final_marked_equity_usd: float
    net_pnl_usd: float
    gross_pnl_usd: float
    fees_usd: float
    funding_pnl_usd: float
    turnover_usd: float
    max_drawdown_usd: float
    n_trades: int
    wins: int
    losses: int
    open_positions_at_end: int
    trades: list[dict[str, Any]]
    equity_curve: list[EquityPoint]
    sleeve_attribution: dict[str, dict[str, Any]]
    data_availability: dict[str, Any]
    breaker_events: list[str] = Field(default_factory=list)
    markouts_available: bool = False
    report_path: str | None = None


class ScenarioData(BaseModel):
    """Bars selected for a scenario: warm-up history plus the replay window per coin."""

    model_config = ConfigDict(frozen=True)

    history_by_coin: dict[str, list[DollarBar]]
    live_by_coin: dict[str, list[DollarBar]]
    hourly_closes_by_coin: dict[str, list[tuple[int, float]]]
    training_bars: list[DollarBar] = Field(default_factory=list)


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------


def bars_from_parquet(path: Path, *, forward_fill_open_interest: bool = True) -> list[DollarBar]:
    """Read a bar table into ``DollarBar`` objects, oldest first.

    Rows without open interest are forward-filled from the last known value (so the replay can
    run) but are not returned as training rows; see ``load_scenario_bars``.
    """
    if not path.exists():
        return []
    df = pl.read_parquet(path).sort("ts_close_ns")
    fields = set(DollarBar.model_fields.keys())
    bars: list[DollarBar] = []
    last_oi: float | None = None
    for row in df.iter_rows(named=True):
        payload = {k: v for k, v in row.items() if k in fields and v is not None}
        if payload.get("open_interest") is None:
            if not forward_fill_open_interest or last_oi is None:
                continue
            payload["open_interest"] = last_oi
        else:
            last_oi = float(payload["open_interest"])
        try:
            bars.append(DollarBar(**payload))
        except Exception as exc:
            logger.debug("Skipping unreadable bar row: %s", exc)
    return bars


def training_rows_before(path: Path, cutoff_ns: int, *, max_rows: int = 20_000) -> list[DollarBar]:
    """Rows from a real feed that close at or before ``cutoff_ns`` (causal training set)."""
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    if "bar_source" not in df.columns:
        return []
    df = (
        df.filter(pl.col("bar_source").is_in(sorted(REAL_BAR_SOURCES)) & (pl.col("ts_close_ns") <= cutoff_ns))
        .filter(pl.col("open_interest").is_not_null())
        .sort("ts_close_ns")
        .tail(max_rows)
    )
    fields = set(DollarBar.model_fields.keys())
    out: list[DollarBar] = []
    for row in df.iter_rows(named=True):
        try:
            out.append(DollarBar(**{k: v for k, v in row.items() if k in fields and v is not None}))
        except Exception:
            continue
    return out


def hourly_closes_from_bars(bars: Sequence[DollarBar]) -> list[tuple[int, float]]:
    """Last close of each hour bucket as ``(ts_ms, close)``."""
    buckets: dict[int, float] = {}
    for b in bars:
        buckets[b.ts_close_ns // 3_600_000_000_000] = b.close
    return [(h * 3_600_000, px) for h, px in sorted(buckets.items())]


def load_scenario_bars(
    scenario: ReplayScenario,
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    parquet_paths: Mapping[str, Path] | None = None,
) -> ScenarioData:
    """Select warm-up history and replay-window bars per coin from the 15m Parquet stores."""
    history: dict[str, list[DollarBar]] = {}
    live: dict[str, list[DollarBar]] = {}
    hourly: dict[str, list[tuple[int, float]]] = {}
    training: list[DollarBar] = []
    for coin in scenario.coins:
        norm = normalize_coin_symbol(coin)
        path = (parquet_paths or {}).get(norm, data_dir / f"{norm.lower()}usdt_perp_bars.parquet")
        bars = bars_from_parquet(path)
        before = [b for b in bars if b.ts_close_ns <= scenario.start_ns]
        window = [b for b in bars if scenario.start_ns < b.ts_close_ns <= scenario.end_ns]
        history[norm] = before[-scenario.warmup_bars :]
        live[norm] = window
        hourly[norm] = hourly_closes_from_bars(before[-scenario.warmup_bars * 4 :])
        if norm == "BTC":
            training = training_rows_before(path, scenario.start_ns)
    return ScenarioData(history_by_coin=history, live_by_coin=live, hourly_closes_by_coin=hourly, training_bars=training)


# ------------------------------------------------------------------------------------------------
# Snapshots from bars
# ------------------------------------------------------------------------------------------------


def snapshot_from_bars(bars: Mapping[str, DollarBar], ts_ns: int) -> MultiAssetMarketSnapshot:
    """Build the market snapshot the tick expects from one bar per coin (mid = close)."""
    assets: dict[str, AssetMarketSnapshot] = {}
    for coin, b in bars.items():
        assets[coin] = AssetMarketSnapshot(
            instrument_id=coin,
            mid_price=b.close,
            mark_price=b.mark_price,
            oracle_price=b.spot_price,
            open_interest=b.open_interest,
            funding_rate_1h=b.funding_rate_1h,
            funding_rate_8h_equiv=b.funding_rate_8h,
            day_volume_usd=b.dollar_volume * 96.0,
            timestamp_ns=ts_ns,
        )
    return MultiAssetMarketSnapshot(timestamp_ns=ts_ns, assets=assets, feed_source="HYPERLIQUID-L1")


def snapshots_for_replay(data: ScenarioData, primary: str = "BTC") -> list[MultiAssetMarketSnapshot]:
    """One snapshot for store construction, then one per primary bar in replay order.

    The tick reads ``feed.snapshot()`` before polling the next bar, so snapshot ``i`` carries the
    closes of bar ``i`` for every coin that has a bar at that open time.
    """
    live = data.live_by_coin
    primary_bars = live.get(primary, [])
    by_open: dict[str, dict[int, DollarBar]] = {
        coin: {b.ts_open_ns: b for b in bars} for coin, bars in live.items()
    }
    snaps: list[MultiAssetMarketSnapshot] = []
    init_bars: dict[str, DollarBar] = {
        coin: hist[-1] for coin, hist in data.history_by_coin.items() if hist
    }
    if init_bars:
        snaps.append(snapshot_from_bars(init_bars, max(b.ts_close_ns for b in init_bars.values())))
    last_seen: dict[str, DollarBar] = dict(init_bars)
    for pb in primary_bars:
        current: dict[str, DollarBar] = {}
        for coin in live:
            b = by_open[coin].get(pb.ts_open_ns)
            if b is None:
                b = last_seen.get(coin)
            if b is not None:
                current[coin] = b
                last_seen[coin] = b
        snaps.append(snapshot_from_bars(current, pb.ts_close_ns))
    if not snaps:
        raise ValueError("Scenario has no bars to replay")
    return snaps


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------


def _write_training_parquet(path: Path, bars: Sequence[DollarBar]) -> None:
    if not bars:
        return
    rows = [dict(b.model_dump(), bar_source="SCRIPTED-REAL") for b in bars]
    pl.DataFrame(rows).write_parquet(path)


def run_replay(
    scenario: ReplayScenario,
    data: ScenarioData,
    *,
    report_dir: Path | None = DEFAULT_REPORT_DIR,
    derivatives_sleeve: DerivativesAlphaSleeve | None = None,
    carry_sleeve: FundingCarrySleeve | None = None,
    starting_capital_usd: float = 5_000.0,
    primary: str = "BTC",
    max_bars: int | None = None,
    circuit_breakers: CircuitBreakers | None = None,
) -> ReplayReport:
    """Replay the scenario through the live tick and return (and optionally write) the report."""
    primary_live = data.live_by_coin.get(primary, [])
    if not primary_live:
        raise ValueError(f"No {primary} bars in the scenario window {scenario.start_utc} to {scenario.end_utc}")
    if max_bars is not None:
        data = data.model_copy(update={"live_by_coin": {c: b[:max_bars] for c, b in data.live_by_coin.items()}})
        primary_live = data.live_by_coin[primary]

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="phoenix_replay_") as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "training_bars.parquet"
        _write_training_parquet(parquet_path, data.training_bars)

        feed = ScriptedBarFeed(
            bars_by_coin={c: list(b) for c, b in data.live_by_coin.items()},
            snapshots=snapshots_for_replay(data, primary=primary),
            history_by_coin={c: list(b) for c, b in data.history_by_coin.items()},
            hourly_closes_by_coin=data.hourly_closes_by_coin,
            availability=DataAvailability(
                candles=True,
                funding_history=True,
                open_interest_history=bool(data.training_bars) or bool(data.history_by_coin.get(primary)),
                feed_source="SCRIPTED",
            ),
        )
        store = ControlCenterStateStore(
            ledger_path=tmp_path / "ledger.json",
            parquet_path=parquet_path,
            feed=feed,
            derivatives_sleeve=derivatives_sleeve,
            carry_sleeve=carry_sleeve,
            seed_bars=len(data.history_by_coin.get(primary, [])),
            hrp_seed_hours=len(data.hourly_closes_by_coin.get(primary, [])) or 1,
            fetch_derivatives_inputs=False,
            circuit_breakers=circuit_breakers,
        )
        if starting_capital_usd != 5_000.0:
            store.reset_account(starting_capital_usd=starting_capital_usd)
        start_cap = store.ledger_summary["base_capital_usd"]

        curve: list[EquityPoint] = []
        processed = 0
        turnover = 0.0
        for _ in range(len(primary_live)):
            res = store.step_simulation_tick()
            if not res.bar_processed:
                break
            processed += 1
            for rec in res.closed_trades:
                turnover += float(rec["size"]) * float(rec["exit_price"]) * (2.0 if rec["hedged"] else 1.0)
            for pos in res.opened_positions:
                turnover += float(pos["notional_usd"]) * (2.0 if pos["hedged"] else 1.0)
            summary = store.ledger_summary
            curve.append(
                EquityPoint(
                    ts_ns=store.history_bars[-1].ts_close_ns,
                    bar_index=len(store.history_bars) - 1,
                    realized_pnl_usd=float(summary["total_realized_pnl"]),
                    unrealized_pnl_usd=float(summary["unrealized_pnl_usd"]),
                    marked_equity_usd=float(summary["marked_equity_usd"]),
                    open_positions=int(summary["open_positions"]),
                )
            )

        trades = store.position_manager.closed_trades
        summary = store.ledger_summary
        peak = start_cap
        max_dd = 0.0
        for pt in curve:
            peak = max(peak, pt.marked_equity_usd)
            max_dd = max(max_dd, peak - pt.marked_equity_usd)
        attribution = {
            k: v.model_dump(mode="json")
            for k, v in store.position_manager.sleeve_attribution(store._current_marks()).items()
        }
        report = ReplayReport(
            scenario_name=scenario.name,
            start_utc=scenario.start_utc,
            end_utc=scenario.end_utc,
            bars_available=len(primary_live),
            bars_processed=processed,
            elapsed_seconds=round(time.time() - t0, 3),
            model_available=store.ml_worker is not None,
            starting_capital_usd=float(start_cap),
            final_equity_usd=float(summary["current_equity_usd"]),
            final_marked_equity_usd=float(summary["marked_equity_usd"]),
            net_pnl_usd=float(sum(t.net_pnl_usd for t in trades)),
            gross_pnl_usd=float(sum(t.gross_pnl_usd for t in trades)),
            fees_usd=float(sum(t.fee_usd for t in trades)),
            funding_pnl_usd=float(sum(t.funding_pnl_usd for t in trades)),
            turnover_usd=float(turnover),
            max_drawdown_usd=float(max_dd),
            n_trades=len(trades),
            wins=sum(1 for t in trades if t.is_win),
            losses=sum(1 for t in trades if not t.is_win),
            open_positions_at_end=int(summary["open_positions"]),
            trades=[t.model_dump(mode="json") for t in trades],
            equity_curve=curve,
            sleeve_attribution=attribution,
            data_availability=store.data_availability.model_dump(mode="json"),
            breaker_events=(store.circuit_breakers.snapshot(None).events if store.circuit_breakers is not None else []),
        )

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = report_dir / f"replay_{scenario.name}_{stamp}.json"
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        report = report.model_copy(update={"report_path": str(out)})
        logger.info("Replay report written to %s", out)
    return report


def summarize(report: ReplayReport) -> str:
    """One-paragraph plain-text summary of a replay report."""
    return (
        f"{report.scenario_name}: {report.bars_processed}/{report.bars_available} bars in {report.elapsed_seconds:.1f}s | "
        f"trades {report.n_trades} (W{report.wins}/L{report.losses}), open at end {report.open_positions_at_end} | "
        f"net ${report.net_pnl_usd:+.2f} (gross ${report.gross_pnl_usd:+.2f}, funding ${report.funding_pnl_usd:+.2f}, fees ${report.fees_usd:.2f}) | "
        f"turnover ${report.turnover_usd:,.0f}, max drawdown ${report.max_drawdown_usd:.2f} | "
        f"model {'yes' if report.model_available else 'no'}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a scenario through the live tick on Parquet bars.")
    parser.add_argument("--scenario", choices=sorted(NAMED_SCENARIOS), default=None)
    parser.add_argument("--start", type=str, default=None, help="ISO UTC start when no named scenario")
    parser.add_argument("--end", type=str, default=None, help="ISO UTC end when no named scenario")
    parser.add_argument("--name", type=str, default="custom")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--warmup-bars", type=int, default=600)
    parser.add_argument("--max-bars", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.scenario:
        scenario = NAMED_SCENARIOS[args.scenario].model_copy(update={"warmup_bars": args.warmup_bars})
    else:
        if not (args.start and args.end):
            parser.error("--start and --end are required without --scenario")
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        scenario = ReplayScenario(
            name=args.name,
            start_utc=start if start.tzinfo else start.replace(tzinfo=timezone.utc),
            end_utc=end if end.tzinfo else end.replace(tzinfo=timezone.utc),
            warmup_bars=args.warmup_bars,
        )
    data = load_scenario_bars(scenario, args.data_dir)
    if not data.live_by_coin.get("BTC"):
        print(f"No BTC bars for {scenario.name}; run scripts/backfill.py for that window first.")
        return 1
    report = run_replay(scenario, data, report_dir=args.report_dir, max_bars=args.max_bars)
    print(summarize(report))
    print(json.dumps({k: report.sleeve_attribution[k] for k in report.sleeve_attribution}, indent=2))
    return 0


__all__ = [
    "DEFAULT_REPORT_DIR",
    "EquityPoint",
    "NAMED_SCENARIOS",
    "ReplayReport",
    "ReplayScenario",
    "ScenarioData",
    "bars_from_parquet",
    "hourly_closes_from_bars",
    "load_scenario_bars",
    "run_replay",
    "snapshot_from_bars",
    "snapshots_for_replay",
    "summarize",
    "training_rows_before",
]


if __name__ == "__main__":
    raise SystemExit(main())
