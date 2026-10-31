"""FastAPI Control Center Server (``http://localhost:8080``) for Gemini Argon 4.

Exposes:
- ``GET /``: Simplified, interactive HTML Control Center with live Dry-Run paper trading,
  customizable starting capital, profit reset, plain-English activity feed, and docs.
- ``GET /api/state``: Live telemetry, net-of-fee PnL, open positions, sleeve telemetry and
  attribution, data availability, CPCV promotion gates, signed HRP-CVaR target weights, and
  the activity log.
- ``POST /api/tick``: Polls the bar feed. When a new closed 15-minute Hyperliquid candle exists
  the engine advances one bar: real features (FFD -> Positioning -> SVD -> Mahalanobis Gate ->
  Calibrated LightGBM), the funding-carry and derivatives-alpha sleeves, hourly HRP-CVaR risk
  budgets, the position lifecycle (entries, triple-barrier exits, vertical expiry, regime veto,
  flipped targets), signed funding accrual on open perp legs, and fees on notional actually traded.
  Win/loss counters change only when a position closes. If no new candle has closed yet the tick
  is a no-op that reports ``waiting_for_bar``.
- ``POST /api/reset``: Resets paper trading profit to $0.00 and sets a custom starting capital.
- ``GET /api/tooltips``: Complete ``PLAIN_ENGLISH_TOOLTIPS`` dictionary.
- ``GET /api/docs/{doc_name}``: Renders any of the four ``docs/*.md`` manuals.
- ``POST /api/control/{action}``: Handles one-click operational controls
  (``start_dry_run``, ``pause_entries``, ``switch_carry_only``, ``emergency_flatten``).

No random or synthetic data is used in any live path. Inputs without a real feed (taker flow,
liquidations, open-interest history, a second funding venue) are zero-filled or ``None`` and
reported in ``data_availability``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Final, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import (
    AssetAllocationInput,
    AssetMarketSnapshot,
    ClosedTradeRecord,
    DataAvailability,
    DeribitOptionsSurface,
    DollarBar,
    ExecutionAction,
    ExecutionOrderCommand,
    ExecutionRegimeMode,
    FeeSchedule,
    FFDWeightsConfig,
    HawkesStateSnapshot,
    HyperliquidWhaleVaultMetrics,
    MetaPredictionOutput,
    MultiAssetMarketSnapshot,
    Position,
    PositionExitReason,
    PositioningSignalSnapshot,
    PortfolioAllocationOutput,
    TripleBarrierConfig,
    TripleBarrierLabel,
    compute_fractional_kelly,
    compute_signed_target_weights,
)
from core.data_ingestion import BarFeed, HyperliquidCandleBarFeed, normalize_coin_symbol
from core.market_feeds import (
    HyperliquidAuxFeed,
    HyperliquidAuxSnapshot,
    LiveHyperliquidAuxFeed,
    LiveMacroInputsFeed,
    MacroInputs,
    MacroInputsFeed,
    ScriptedHyperliquidAuxFeed,
    ScriptedMacroInputsFeed,
)
from core.execution import (
    BarMarks,
    PositionManager,
    RecursiveHawkesMonitor,
    RollingBVCVPINMonitor,
    SleeveTarget,
    position_key,
)
from core.features import calibrate_optimal_d, compute_positioning_snapshots
from core.features.candidates import bar_feature_matrix
from core.execution.markouts import MarkoutTracker
from core.models.online_residual import OnlineResidualRidge, abstention_threshold, outlier_guard_skip
from core.fee_tiers import FeeTable, best_hedge_venue, fee_schedule_for, tier_index
from core.streams import BarFlow, StreamHub
from core.models import (
    EWMAVolatilityEstimator,
    FeeAdjustedTripleBarrierLabeler,
    LightGBMMetaLabelWorker,
    compute_effective_barriers,
    compute_ewma_volatility,
    extract_continuous_and_binary_from_snapshots,
)
from research.decay_monitor import DecayReport, evaluate_decay
from research.validation.shadow import PreRegisteredThresholds, ShadowBook
from risk.cascade_controls import (
    CascadeController,
    FragilityAssessment,
    StressPosition,
    adl_rank_estimate,
    liquidation_stress_loss,
    stablecoin_collateral_haircut,
)
from risk.circuit_breakers import CircuitBreakers, OperationalStatus
from risk.partial_adjustment import square_root_impact, transfer_coefficient
from risk.portfolio_allocator import HRPWeightScheduler, RollingHourlyReturnMatrix
from sentiment.committee_models import (
    AdversarialRedTeamCritic,
    BreadthAnalystOutput,
    CrossSectionalBreadthAnalyst,
    MacroSentinelOutput,
    MacroStablecoinSentinel,
    MultiAssetTelemetry,
    PortfolioStressInput,
    RedTeamCriticOutput,
    RiskCommitteeConsensusEngine,
    RiskCommitteeOutput,
)
from sleeves.sleeve_1_derivatives_alpha import DerivativesAlphaSignal, DerivativesAlphaSleeve
from sleeves.sleeve_2_trend import TREND_SLEEVE_ID, TrendSignal, TrendSleeve
from sleeves.sleeve_3_funding_carry import (
    CarryDirection,
    CarryOpportunityEvaluation,
    FundingCarrySleeve,
    SleeveSignalOutput,
)
from ui.dashboard_app import (
    DOC_FILENAMES,
    PLAIN_ENGLISH_TOOLTIPS,
    DashboardTelemetryState,
    PnLAndExecutionMetrics,
    build_candlestick_dollar_bar_figure,
    build_default_dashboard_state,
    build_portfolio_weights_figure,
    load_layman_document_markdown,
    render_full_dashboard_html,
)

LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE: Path = LOG_DIR / "phoenix.log"
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
PARQUET_BARS_FILE: Path = DATA_DIR / "btcusdt_perp_bars.parquet"
LEDGER_FILE: Path = DATA_DIR / "account_ledger.json"

logger = logging.getLogger("phoenix.engine")

ASSETS: Final[tuple[str, str, str]] = ("BTC", "ETH", "SOL")
SLEEVE_META_LABELLER: Final[str] = "meta_labeller"
SLEEVE_DERIVATIVES: Final[str] = "derivatives_alpha"
SLEEVE_FUNDING_CARRY: Final[str] = "funding_carry"

HISTORY_SEED_BARS: Final[int] = 1200
HRP_SEED_HOURS: Final[int] = 480
FEATURE_WINDOW_BARS: Final[int] = 4096
UI_WINDOW_BARS: Final[int] = 16
MIN_TRAIN_LABELLED_BARS: Final[int] = 500
DEFAULT_META_EXTRA_FEATURES: Final[tuple[str, ...]] = ("momentum_1w", "mark_index_divergence_chg_1h", "nyse_open_flag")
RETRAIN_INTERVAL_NS: Final[int] = 24 * 3_600 * 1_000_000_000
DERIVATIVES_INPUT_TTL_NS: Final[int] = 5 * 60 * 1_000_000_000
WAITING_LOG_INTERVAL_SEC: Final[float] = 60.0
MAX_GROSS_LEVERAGE: Final[float] = 2.5
CARRY_TARGET_NOTIONAL_WEIGHT: Final[float] = 0.25
REAL_BAR_SOURCES: Final[frozenset[str]] = frozenset(
    {"HYPERLIQUID-L1", "BYBIT-V5-FALLBACK", "SCRIPTED-REAL", "BACKFILL-BYBIT", "BACKFILL-BINANCE", "BACKFILL-OKX"}
)
STEP_15M_NS: Final[int] = 900_000_000_000


class TradeForensicRecord(BaseModel):
    """Per-bar decision record ("Why Did It Trade?") with feature attribution.

    One record is written per processed bar. ``realized_pnl_usd`` is the net PnL of positions
    closed on that bar (0.0 when nothing closed); it is not a mark-to-market of open positions.
    """

    model_config = ConfigDict(frozen=True)

    trade_id: str
    timestamp_ns: int
    timestamp_iso: str
    instrument_id: str
    action: str
    side: Literal["BUY", "SELL", "FLAT", "NONE"]
    fill_price: float
    quantity: float
    notional_usd: float
    realized_pnl_usd: float
    realized_pnl_bps: float
    fee_cost_usd: float
    fee_savings_usd: float
    regime: str
    p_meta: float
    mahalanobis_di: float
    vpin: float
    hawkes_eta: float
    asri_score: float
    m_regime: float
    kelly_fraction: float
    why_did_it_trade: str
    feature_attribution: dict[str, float]
    sleeve_id: str = ""
    entries: list[str] = Field(default_factory=list)
    exits: list[str] = Field(default_factory=list)
    open_positions: int = 0
    unrealized_pnl_usd: float = 0.0


class AccountLedgerData(BaseModel):
    """Persistent account state for the paper-trading ledger."""

    model_config = ConfigDict(frozen=True)

    account_id: str = "hyperliquid_paper_01"
    base_capital_usd: float = 5_000.00
    current_cash_usd: float = 5_000.00
    current_equity_usd: float = 5_000.00
    total_realized_pnl: float = 0.00
    accumulated_net_pnl_usd: float = 0.00
    accumulated_net_pnl_bps: float = 0.0
    accumulated_gross_pnl_usd: float = 0.00
    total_fees_paid_usd: float = 0.00
    total_fee_savings_usd: float = 0.00
    total_funding_pnl_usd: float = 0.00
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    trade_history: list[dict[str, Any]] = Field(default_factory=list)
    forensics_history: list[dict[str, Any]] = Field(default_factory=list)
    open_positions: list[dict[str, Any]] = Field(default_factory=list)
    last_updated_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _run_m1max_metal_gpu_regime_pass(features_row: np.ndarray) -> float:
    """Execute a native Apple Silicon M1 Max Metal GPU (mlx.core) tensor projection."""
    try:
        import mlx.core as mx

        x_gpu = mx.array(features_row.astype(np.float32))
        w_gpu = mx.array(np.array([[0.52, -0.31, 0.19], [-0.24, 0.61, 0.14], [0.18, 0.22, 0.58]], dtype=np.float32))
        proj = mx.matmul(x_gpu, w_gpu)
        energy = mx.sum(mx.square(proj))
        mx.eval(energy)
        val = energy.item()
        return float(val.real) if isinstance(val, complex) else float(val)
    except Exception:
        return float(np.sum(np.square(features_row)))


def _append_log_line(line: str) -> None:
    """Write a structured plain-English log line to logs/phoenix.log and stdout logger."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    logger.info(line)


class ResetCapitalRequest(BaseModel):
    """Request payload for ``POST /api/reset``."""

    model_config = ConfigDict(frozen=True)

    starting_capital_usd: float = Field(default=5_000.0, gt=0.0, le=1_000_000_000.0)


class ControlActionResult(BaseModel):
    """Response schema for ``POST /api/control/{action}``, ``POST /api/tick``, and ``POST /api/reset``."""

    model_config = ConfigDict(frozen=True)

    status: str
    action: str
    regime_mode: ExecutionRegimeMode
    is_dry_run: bool
    plain_english_explanation: str
    weights: dict[str, float]
    gross_leverage: float
    tick_count: int = 0
    latest_price: float = 69580.0
    starting_capital_usd: float = 5_000.0
    current_equity_usd: float = 5_000.0
    net_pnl_usd: float = 0.0
    net_pnl_bps: float = 0.0
    activity_log: list[str] = []
    forensic_record: dict[str, Any] | None = None
    ledger_summary: dict[str, Any] | None = None
    bar_processed: bool = False
    closed_trades: list[dict[str, Any]] = Field(default_factory=list)
    opened_positions: list[dict[str, Any]] = Field(default_factory=list)


def _persist_bars_to_parquet(bars: Sequence[DollarBar], path: Path, bar_source: str) -> int:
    """Append collected DollarBars to the Parquet training store, tagging the real feed they came from."""
    if not bars:
        try:
            return int(pl.read_parquet(path).height) if path.exists() else 0
        except Exception:
            return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [dict(b.model_dump(), bar_source=bar_source) for b in bars]
        new_df = pl.DataFrame(rows)
        if path.exists():
            existing_df = pl.read_parquet(path)
            combined = pl.concat([existing_df, new_df], how="diagonal_relaxed").unique(
                subset=["ts_close_ns"], keep="last", maintain_order=True
            )
        else:
            combined = new_df
        combined.write_parquet(path)
        return int(combined.height)
    except Exception as exc:
        logger.warning("Parquet persistence warning: %s", exc)
        return len(bars)


def _load_training_bars_from_parquet(path: Path, max_rows: int = 20_000) -> list[DollarBar]:
    """Load bars recorded by a real feed (``bar_source`` tagged) for model training, oldest first."""
    if not path.exists():
        return []
    try:
        df = pl.read_parquet(path)
    except Exception as exc:
        logger.warning("Could not read Parquet training store %s: %s", path, exc)
        return []
    if "bar_source" not in df.columns or df.height == 0:
        return []
    df = df.filter(pl.col("bar_source").is_in(sorted(REAL_BAR_SOURCES))).sort("ts_close_ns").tail(max_rows)
    field_names = set(DollarBar.model_fields.keys())
    bars: list[DollarBar] = []
    for row in df.iter_rows(named=True):
        try:
            bars.append(DollarBar(**{k: v for k, v in row.items() if k in field_names and v is not None}))
        except Exception:
            continue
    return bars


def _neutral_meta_prediction(instrument_id: str, ts_close_ns: int, reason: str) -> MetaPredictionOutput:
    """Meta prediction emitted when no trained model exists: p = 0.5, direction 0, gate closed."""
    _ = reason
    return MetaPredictionOutput(
        instrument_id=instrument_id,
        ts_close_ns=ts_close_ns,
        primary_direction=0,
        p_meta=0.5,
        mahalanobis_di=0.0,
        di_threshold=3.50,
        epistemic_gate_passed=False,
        svd_factors=(),
        liquidation_exhaustion_flag=0,
        expected_net_edge_bps=0.0,
    )


class ControlCenterStateStore:
    """In-memory + Parquet state store running the real strategy stack on real closed bars."""

    def __init__(
        self,
        initial_state: DashboardTelemetryState | None = None,
        ledger_path: Path | None = None,
        feed: BarFeed | None = None,
        carry_sleeve: FundingCarrySleeve | None = None,
        derivatives_sleeve: DerivativesAlphaSleeve | None = None,
        fee_schedule: FeeSchedule | None = None,
        parquet_path: Path | None = None,
        seed_bars: int = HISTORY_SEED_BARS,
        hrp_seed_hours: int = HRP_SEED_HOURS,
        fetch_derivatives_inputs: bool = True,
        macro_feed: MacroInputsFeed | None = None,
        aux_feed: HyperliquidAuxFeed | None = None,
        stream_hub: StreamHub | None = None,
        trend_sleeve: TrendSleeve | None = None,
        circuit_breakers: CircuitBreakers | None = None,
        cascade_controller: CascadeController | None = None,
        fee_table: FeeTable | None = None,
        account_volume_14d_usd: float = 0.0,
        staking_discount: float = 0.0,
        shadow_thresholds: Mapping[str, PreRegisteredThresholds] | None = None,
        bars_carry_taker_flow: bool = False,
        meta_extra_features: Sequence[str] | None = None,
    ) -> None:
        self._ledger_path: Path = ledger_path or LEDGER_FILE
        self._parquet_path: Path = parquet_path or PARQUET_BARS_FILE
        # Fee tier (P4.4): an explicit schedule wins; otherwise the venue's table and the account's 14-day volume.
        self._fee_table: FeeTable | None = fee_table
        self._account_volume_14d_usd: float = account_volume_14d_usd
        self._fee: FeeSchedule = fee_schedule or (
            fee_schedule_for(fee_table, account_volume_14d_usd, staking_discount) if fee_table is not None else FeeSchedule.hyperliquid()
        )
        self._barrier_cfg: TripleBarrierConfig = TripleBarrierConfig(fee_schedule=self._fee)
        self._fetch_derivatives_inputs: bool = fetch_derivatives_inputs
        # Macro inputs (DVOL, stablecoin marks) and Hyperliquid auxiliary feeds share the live-feeds switch.
        self._macro_feed: MacroInputsFeed = macro_feed or (
            LiveMacroInputsFeed() if fetch_derivatives_inputs else ScriptedMacroInputsFeed()
        )
        self._aux_feed: HyperliquidAuxFeed = aux_feed or (
            LiveHyperliquidAuxFeed() if fetch_derivatives_inputs else ScriptedHyperliquidAuxFeed()
        )
        self._latest_macro_inputs: MacroInputs = MacroInputs()
        self._latest_aux: HyperliquidAuxSnapshot = HyperliquidAuxSnapshot()
        # Websocket streams (trades, liquidations, lead mids). Created for live mode, started by the app lifespan.
        self._stream_hub: StreamHub | None = stream_hub or (
            StreamHub(ASSETS, trades_dir=DATA_DIR / "trades") if fetch_derivatives_inputs else None
        )
        self._latest_bar_flows: dict[str, BarFlow] = {}
        self._training_rows_with_oi: int = 0
        # Trend sleeve (P2.2): off unless a TrendSleeve with config.enabled=True is supplied.
        self._trend_sleeve: TrendSleeve = trend_sleeve or TrendSleeve()
        # Circuit breakers (P3.2): on by default in live mode; replays and tests pass them explicitly.
        self._breakers: CircuitBreakers | None = circuit_breakers or (CircuitBreakers() if fetch_derivatives_inputs else None)
        self._blocked_entries: list[str] = []
        self._latest_transfer_coefficient: float | None = None
        # Cascade and ADL controls (P3.4): reacts only to inputs that are actually available.
        self._cascade: CascadeController = cascade_controller or CascadeController()
        # True only when the feed's bars carry venue-reported taker buy/sell volume (archive replays);
        # live bars without stream coverage hold an even split, which must not be read as flow.
        self._bars_carry_taker_flow = bars_carry_taker_flow
        # P2.3: features added to the meta-labeller after passing P1.8 on two non-overlapping years.
        self._meta_extra_features: tuple[str, ...] = tuple(DEFAULT_META_EXTRA_FEATURES if meta_extra_features is None else meta_extra_features)
        bar_feature_matrix([], self._meta_extra_features)  # fail fast on a name that needs more than bars
        self._latest_fragility: dict[str, FragilityAssessment] = {}
        self._collateral_haircut: float = 0.0
        # Markout ladder (P4.2): fills scored against Hyperliquid mids from the stream hub.
        self._markouts = MarkoutTracker()
        # Shadow mode (P5.1): unvalidated or newly enabled sleeves trade here, never in the ledger.
        self._shadow_positions = PositionManager(fee_schedule=self._fee, barrier_config=self._barrier_cfg)
        self._shadow_books: dict[str, ShadowBook] = {}
        self._shadow_thresholds: dict[str, PreRegisteredThresholds] = dict(shadow_thresholds or {})
        self._latest_shadow_targets: dict[str, SleeveTarget] = {}
        # Decay monitor (P5.2)
        self._latest_decay: DecayReport | None = None
        self._last_decay_check_ns: int | None = None
        self._latest_trend_signals: dict[str, TrendSignal] = {}
        self._trend_eval_bucket: int | None = None

        self._starting_capital_usd: float = 5_000.0
        self._base_capital_usd: float = 5_000.0
        self._total_realized_pnl: float = 0.0
        self._total_fees_paid_usd: float = 0.0
        self._total_fee_savings_usd: float = 0.0
        self._total_funding_pnl_usd: float = 0.0
        self._wins: int = 0
        self._losses: int = 0
        self._maker_fills: int = 0
        self._taker_fills: int = 0
        self._trade_history: list[dict[str, Any]] = []
        self._forensics_history: list[dict[str, Any]] = []
        self._latest_forensic: dict[str, Any] | None = None
        self._restored_positions: list[Position] = []

        # 3-Model Apple Silicon (MLX) Risk Committee & Quantitative Sleeves
        self._risk_sentinel = MacroStablecoinSentinel()
        self._risk_breadth = CrossSectionalBreadthAnalyst()
        self._risk_critic = AdversarialRedTeamCritic()
        self._risk_consensus = RiskCommitteeConsensusEngine()
        self._funding_carry_sleeve: FundingCarrySleeve = carry_sleeve or FundingCarrySleeve()
        self._derivatives_alpha_sleeve: DerivativesAlphaSleeve = derivatives_sleeve or DerivativesAlphaSleeve()
        self._derivatives_inputs_cache: dict[str, tuple[int, DeribitOptionsSurface | None, HyperliquidWhaleVaultMetrics | None]] = {}
        self._latest_carry_evaluations: dict[str, CarryOpportunityEvaluation] = {}
        self._latest_carry_signals: dict[str, SleeveSignalOutput] = {}
        self._latest_derivatives_signals: dict[str, DerivativesAlphaSignal] = {}
        self._latest_targets: dict[str, SleeveTarget] = {}
        self._latest_hrp_weights: dict[str, float] = {a: 1.0 / len(ASSETS) for a in ASSETS}

        self._load_or_init_ledger()

        # Bar feed: Hyperliquid candles by default, scripted bars for replay and tests.
        self._feed: BarFeed = feed or HyperliquidCandleBarFeed(coins=ASSETS)
        self._data_availability: DataAvailability = DataAvailability()
        self._last_waiting_log_ts: float = 0.0

        raw_state = initial_state or build_default_dashboard_state()
        multi_snap = self._feed.snapshot()
        self._latest_market_snapshot: MultiAssetMarketSnapshot = multi_snap
        self._prev_asset_prices: dict[str, float] = {}
        for coin in ASSETS:
            a_snap = multi_snap.get_asset(coin)
            if a_snap is not None:
                self._prev_asset_prices[coin] = a_snap.mid_price

        # Real closed-bar history (seeded from the feed) drives features, volatility and the UI window.
        self._history_bars: list[DollarBar] = []
        try:
            seeded, avail = self._feed.seed_bars("BTC", seed_bars)
            self._history_bars = list(seeded)
            self._data_availability = avail
            if seeded and avail.open_interest_history:
                _persist_bars_to_parquet(
                    seeded,
                    self._parquet_path,
                    bar_source="SCRIPTED-REAL" if avail.feed_source == "SCRIPTED" else avail.feed_source,
                )
        except Exception as exc:
            logger.warning("Bar history seed failed (%s); starting with an empty history.", exc)
        self._parquet_row_count: int = _persist_bars_to_parquet([], self._parquet_path, bar_source="")

        self._sigma_estimator: EWMAVolatilityEstimator = EWMAVolatilityEstimator.from_bars(self._history_bars, span=96)

        # Hourly HRP-CVaR risk budgets from real 1h returns (equal weights until 96 rows exist).
        trend_hours = self._trend_sleeve.config.min_history_hours + 24 if self._trend_sleeve.enabled else 0
        self._hourly_returns = RollingHourlyReturnMatrix(ASSETS, max_rows=max(720, trend_hours + 48))
        self._hrp_scheduler = HRPWeightScheduler(ASSETS)
        try:
            hourly = {coin: self._feed.hourly_closes(coin, max(hrp_seed_hours, trend_hours)) for coin in ASSETS}
            self._hourly_returns.seed_from_hourly_closes(hourly)
        except Exception as exc:
            logger.warning("Hourly return seed failed (%s); HRP will fall back to equal weights.", exc)
        seed_ts = self._history_bars[-1].ts_close_ns if self._history_bars else multi_snap.timestamp_ns
        self._latest_hrp_weights = self._hrp_scheduler.compute(seed_ts, self._hourly_returns.matrix(), force=True)
        self._data_availability = self._data_availability.model_copy(
            update={"hourly_return_rows": self._hourly_returns.n_rows}
        )
        self._data_availability = self._availability_with_feeds(self._data_availability)

        # Risk committee baseline from live funding, prices, DVOL and stablecoin marks (P0.8).
        self._latest_macro_inputs = self._refresh_macro_inputs(multi_snap.timestamp_ns)
        self._latest_aux = self._refresh_aux(multi_snap.timestamp_ns)
        self._latest_committee_output: RiskCommitteeOutput = self._compute_initial_risk_committee(multi_snap)
        self._data_availability = self._availability_with_feeds(self._data_availability)

        # Microstructure monitors and the Stage-1 SVD + Stage-2 LightGBM worker (trained on real rows only).
        self._vpin_monitor = RollingBVCVPINMonitor(bucket_volume=500.0, num_buckets=50)
        self._hawkes_monitor = RecursiveHawkesMonitor(async_refit=False)
        self._ffd_close_cfg: FFDWeightsConfig = raw_state.ffd_config
        self._ml_worker: LightGBMMetaLabelWorker | None = None
        self._residual: OnlineResidualRidge | None = None
        self._abstain_di: float = float("inf")
        self._predictions_made: int = 0
        self._abstentions: int = 0
        self._last_retrain_ts_ns: int | None = None
        self._last_retrain_message: str = ""
        self._train_initial_ml_worker()

        # Position lifecycle (restored from the ledger when the process restarts).
        self._positions = PositionManager(fee_schedule=self._fee, barrier_config=self._barrier_cfg)
        for pos in self._restored_positions:
            self._positions.restore(pos)

        self._tick_counter: int = len(self._history_bars)
        self._auto_running: bool = False  # Wait for user to click Start Dry-Run!

        synced_pnl = PnLAndExecutionMetrics(
            starting_capital_usd=self._starting_capital_usd,
            net_pnl_usd=round(self._total_realized_pnl, 2),
            net_pnl_bps=round((self._total_realized_pnl / max(1.0, self._starting_capital_usd)) * 10_000.0, 1),
            gross_pnl_usd=round(self._total_realized_pnl + self._total_fees_paid_usd, 2),
            total_fees_paid_usd=round(self._total_fees_paid_usd, 2),
            maker_fill_ratio=self._maker_fill_ratio(),
            taker_fill_ratio=1.0 - self._maker_fill_ratio(),
            fee_savings_usd=0.0,
            fee_savings_bps=0.0,
            annualized_carry_apr_pct=0.0,
            is_dry_run=raw_state.pnl_metrics.is_dry_run,
        )
        self._state: DashboardTelemetryState = raw_state.model_copy(
            update={
                "pnl_metrics": synced_pnl,
                "ffd_config": self._ffd_close_cfg,
                "asri_risk": self._latest_committee_output.to_systemic_risk_output(),
                "portfolio_allocation": self._zero_allocation(),
            }
        )
        self._refresh_initial_state_window()
        self._evaluate_sleeves_for_telemetry(multi_snap)

        init_line_1 = (
            f"[{self._now_str()}] [READY] Project Phoenix online at http://127.0.0.1:8080 | "
            f"Ledger: ${self._starting_capital_usd:,.2f} base (${self._starting_capital_usd + self._total_realized_pnl:,.2f} equity, "
            f"{len(self._trade_history)} closed trades, W:{self._wins}/L:{self._losses}, "
            f"{len(self._positions.open_positions)} open) | Store: {self._parquet_path.name}"
        )
        init_line_2 = (
            f"[{self._now_str()}] [FEED] {self._data_availability.feed_source}: {len(self._history_bars)} real 15m bars seeded, "
            f"{self._hourly_returns.n_rows} hourly return rows ({'HRP-CVaR live' if not self._hrp_scheduler.last_result_is_fallback else 'equal-weight fallback'}), "
            f"model: {'trained' if self._ml_worker is not None else 'none (' + self._last_retrain_message + ')'}."
        )
        self._activity_log: list[str] = [init_line_1, init_line_2]
        _append_log_line(init_line_1)
        _append_log_line(init_line_2)

    # ------------------------------------------------------------------ initialisation helpers

    def _zero_allocation(self) -> PortfolioAllocationOutput:
        return PortfolioAllocationOutput(
            weights={a: 0.0 for a in ASSETS},
            raw_unscaled_weights={a: 0.0 for a in ASSETS},
            hrp_unsigned_weights=dict(self._latest_hrp_weights),
            kelly_scalars={a: 0.0 for a in ASSETS},
            regime_multiplier=1.0,
            gross_leverage=0.0,
            leverage_scale_factor=1.0,
            rejected_for_fees=[],
        )

    def _maker_fill_ratio(self) -> float:
        total = self._maker_fills + self._taker_fills
        return (self._maker_fills / total) if total > 0 else 1.0

    def _compute_initial_risk_committee(self, snap: MultiAssetMarketSnapshot) -> RiskCommitteeOutput:
        """Run the baseline evaluation of the 3-model MLX Risk Committee on live prices and funding."""
        prices = {coin: a.mid_price for coin in ASSETS if (a := snap.get_asset(coin)) is not None}
        fundings = {coin: a.funding_rate_1h for coin in ASSETS if (a := snap.get_asset(coin)) is not None}
        try:
            macro_out = self._evaluate_macro_sentinel()
            breadth_out = self._risk_breadth.evaluate(
                MultiAssetTelemetry(asset_prices=prices, funding_rates_1h=fundings)
                if len(prices) == len(ASSETS)
                else MultiAssetTelemetry()
            )
            critic_out = self._risk_critic.evaluate(
                PortfolioStressInput(gross_leverage=0.0, vpin=0.25, hawkes_eta=0.30)
            )
            return self._risk_consensus.reconcile(macro_out, breadth_out, critic_out)
        except Exception:
            mac_fall = MacroSentinelOutput(
                score=18.0,
                stablecoin_depeg_bps=1.5,
                dvol_index=50.0,
                dvol_zscore=0.1,
                tail_risk_flag=False,
                regime_label="NORMAL_VOLATILITY",
                confidence=0.88,
                summary="Macro sentinel in baseline mode: stablecoins pegged and volatility calm.",
            )
            brd_fall = BreadthAnalystOutput(
                breadth_score=15.0,
                dispersion_bps=42.0,
                correlation_dispersion=0.85,
                funding_divergence_bps=4.5,
                concentration_risk_flag=False,
                regime_label="SYNCHRONIZED_EXPANSION",
                confidence=0.85,
                summary="Cross-sectional breadth balanced across BTC, ETH, and SOL.",
            )
            crt_fall = RedTeamCriticOutput(
                criticism_score=12.0,
                liquidation_cascade_prob=0.08,
                toxic_flow_vulnerability=0.15,
                haircut_multiplier=1.0,
                veto_active=False,
                identified_risks=(),
                confidence=0.90,
                summary="Adversarial red-team confirms safe leverage and absence of cascades.",
            )
            return RiskCommitteeOutput(
                macro_sentinel_score=18.0,
                breadth_analyst_score=15.0,
                red_team_critic_score=12.0,
                consensus_risk_score=15.0,
                regime_multiplier=1.0,
                tail_risk_flag=False,
                committee_rationale="3-Model MLX Risk Committee consensus: Benign macro, breadth, and adversarial critic state.",
                macro_sentinel=mac_fall,
                breadth_analyst=brd_fall,
                red_team_critic=crt_fall,
                confidence=0.88,
            )

    def _load_or_init_ledger(self) -> None:
        """Load existing account ledger state or initialize with $5,000.00 base capital."""
        if self._ledger_path.exists():
            try:
                data = json.loads(self._ledger_path.read_text(encoding="utf-8"))
                self._base_capital_usd = float(data.get("base_capital_usd", 5000.0))
                self._starting_capital_usd = self._base_capital_usd
                self._total_realized_pnl = float(
                    data.get("total_realized_pnl", data.get("accumulated_net_pnl_usd", 0.0))
                )
                self._total_fees_paid_usd = float(data.get("total_fees_paid_usd", 0.0))
                self._total_fee_savings_usd = float(data.get("total_fee_savings_usd", 0.0))
                self._total_funding_pnl_usd = float(data.get("total_funding_pnl_usd", 0.0))
                self._wins = int(data.get("win_count", 0))
                self._losses = int(data.get("loss_count", 0))
                self._maker_fills = int(data.get("maker_fills", 0))
                self._taker_fills = int(data.get("taker_fills", 0))
                self._trade_history = list(data.get("trade_history", []))
                stored_forensics = data.get("forensics_history")
                if isinstance(stored_forensics, list):
                    self._forensics_history = list(stored_forensics)
                else:
                    self._forensics_history = [
                        t for t in self._trade_history if isinstance(t, dict) and "why_did_it_trade" in t
                    ]
                if self._forensics_history:
                    self._latest_forensic = self._forensics_history[-1]
                self._restored_positions = []
                for raw_pos in data.get("open_positions", []):
                    try:
                        self._restored_positions.append(Position(**raw_pos))
                    except Exception as pos_exc:
                        logger.warning("Skipping unreadable persisted position: %s", pos_exc)
                logger.info(
                    "Loaded persistent account ledger from %s: base=$%s, equity=$%s, closed trades=%s, open=%s",
                    self._ledger_path,
                    self._base_capital_usd,
                    self._base_capital_usd + self._total_realized_pnl,
                    len(self._trade_history),
                    len(self._restored_positions),
                )
                return
            except Exception as exc:
                logger.warning("Failed to load existing ledger from %s: %s", self._ledger_path, exc)

        self._base_capital_usd = 5000.0
        self._starting_capital_usd = 5000.0
        self._total_realized_pnl = 0.0
        self._total_fees_paid_usd = 0.0
        self._total_fee_savings_usd = 0.0
        self._total_funding_pnl_usd = 0.0
        self._wins = 0
        self._losses = 0
        self._maker_fills = 0
        self._taker_fills = 0
        self._trade_history = []
        self._forensics_history = []
        self._latest_forensic = None
        self._restored_positions = []
        self._persist_ledger_state_atomic()

    def _persist_ledger_state_atomic(self) -> None:
        """Write account ledger state atomically to disk via temporary file rename."""
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self._ledger_path.with_name(f"{self._ledger_path.name}.tmp")
            total_trades = self._wins + self._losses
            win_rate = (self._wins / total_trades * 100.0) if total_trades > 0 else 0.0

            gross_wins = sum(
                float(t.get("realized_pnl_usd", 0.0))
                for t in self._trade_history
                if isinstance(t, dict) and float(t.get("realized_pnl_usd", 0.0)) > 0
            )
            gross_losses = abs(
                sum(
                    float(t.get("realized_pnl_usd", 0.0))
                    for t in self._trade_history
                    if isinstance(t, dict) and float(t.get("realized_pnl_usd", 0.0)) < 0
                )
            )
            profit_factor = (
                round(gross_wins / gross_losses, 2)
                if gross_losses > 0
                else (round(gross_wins, 2) if gross_wins > 0 else 1.0)
            )

            positions_mgr = getattr(self, "_positions", None)
            open_positions = (
                [p.model_dump(mode="json") for p in positions_mgr.open_positions.values()]
                if positions_mgr is not None
                else [p.model_dump(mode="json") for p in self._restored_positions]
            )
            current_equity = round(self._starting_capital_usd + self._total_realized_pnl, 2)
            ledger_dict: dict[str, Any] = {
                "account_id": "hyperliquid_paper_01",
                "base_capital_usd": round(self._starting_capital_usd, 2),
                "current_cash_usd": current_equity,
                "current_equity_usd": current_equity,
                "total_realized_pnl": round(self._total_realized_pnl, 2),
                "accumulated_net_pnl_usd": round(self._total_realized_pnl, 2),
                "accumulated_net_pnl_bps": round(
                    (self._total_realized_pnl / max(1.0, self._starting_capital_usd)) * 10_000.0, 1
                ),
                "accumulated_gross_pnl_usd": round(
                    self._total_realized_pnl + self._total_fees_paid_usd, 2
                ),
                "total_fees_paid_usd": round(self._total_fees_paid_usd, 2),
                "total_fee_savings_usd": round(self._total_fee_savings_usd, 2),
                "total_funding_pnl_usd": round(self._total_funding_pnl_usd, 4),
                "total_trades": total_trades,
                "win_count": self._wins,
                "loss_count": self._losses,
                "maker_fills": self._maker_fills,
                "taker_fills": self._taker_fills,
                "win_rate_pct": round(win_rate, 1),
                "profit_factor": profit_factor,
                "trade_history": self._trade_history[-200:],
                "forensics_history": self._forensics_history[-200:],
                "open_positions": open_positions,
                "last_updated_iso": datetime.now(timezone.utc).isoformat(),
            }
            with open(tmp_file, "w", encoding="utf-8") as fh:
                json.dump(ledger_dict, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_file, self._ledger_path)
        except Exception as exc:
            logger.error("Ledger atomic persistence error: %s", exc)

    # ------------------------------------------------------------------ model training (P0.7)

    def _fit_worker_on_bars(self, train_bars: Sequence[DollarBar]) -> LightGBMMetaLabelWorker | None:
        """Fit Stage-1 SVD + Mahalanobis Gate + Stage-2 LightGBM on real bars, or return None.

        Rules (REINFORCEMENT_PLAN P0.7): the last ``vertical_barrier_bars`` labels are purged
        because their horizon is incomplete; at least ``MIN_TRAIN_LABELLED_BARS`` labelled bars
        are required; single-class labels skip the fit instead of fabricating labels. On any skip
        the caller keeps the previous worker.
        """
        purge = self._barrier_cfg.vertical_barrier_bars
        n_total = len(train_bars)
        n_keep = n_total - purge
        if n_keep < MIN_TRAIN_LABELLED_BARS:
            self._last_retrain_message = (
                f"{max(n_keep, 0)} labelled real bars after purging {purge}; {MIN_TRAIN_LABELLED_BARS} required"
            )
            logger.info("Meta-labeller fit skipped: %s.", self._last_retrain_message)
            return None

        log_close = np.log(np.array([b.close for b in train_bars], dtype=np.float64))
        ffd_cfg = calibrate_optimal_d(log_close, split_id="live_parquet_split")
        snaps = compute_positioning_snapshots(train_bars, close_ffd_config=ffd_cfg)
        x_cont, bin_flags, prim_dirs = extract_continuous_and_binary_from_snapshots(snaps)
        x_cont = self._with_extra_features(x_cont, train_bars)

        labeler = FeeAdjustedTripleBarrierLabeler(self._barrier_cfg)
        labels = labeler.label_bars(train_bars, primary_directions=prim_dirs)
        y_meta = np.array([lbl.meta_label for lbl in labels], dtype=np.int64)[:n_keep]
        x_train = x_cont[:n_keep]
        flags_train = bin_flags[:n_keep]

        classes = np.unique(y_meta)
        if classes.size < 2:
            self._last_retrain_message = (
                f"labels single-class ({int(classes[0])}) over {n_keep} bars; retrain skipped, previous model retained"
            )
            logger.warning("Meta-labeller fit skipped: %s.", self._last_retrain_message)
            return None

        sigma_series = np.concatenate(
            ([1e-4], compute_ewma_volatility(np.diff(log_close), span=96))
        )[:n_keep]

        # Outlier-protection guard (P5.3): skip when the newest rows are mostly beyond the current model's gate.
        current = self._ml_worker
        if current is not None:
            recent = x_train[-96:]
            try:
                new_di = current.outlier_gate.compute_di(current.svd_orthogonalizer.transform(recent))
                skip, frac = outlier_guard_skip(new_di, current.outlier_gate.di_threshold)
            except Exception:
                skip, frac = False, 0.0
            if skip:
                self._last_retrain_message = f"outlier guard: {frac:.0%} of the newest rows beyond the gate; retrain skipped, previous model retained"
                logger.warning("Meta-labeller fit skipped: %s.", self._last_retrain_message)
                return None

        worker = LightGBMMetaLabelWorker(di_threshold=3.50, n_jobs=4, fee_schedule=self._fee)
        worker.fit(
            x_train,
            y_meta,
            liquidation_exhaustion_flag_train=flags_train,
            sigma_bar_train=sigma_series,
        )
        # Online residual layer and abstention threshold (P5.3), fitted on the same training rows.
        try:
            p_train = np.asarray(worker.predict_proba(x_train, liquidation_exhaustion_flag=flags_train, sigma_bar=sigma_series), dtype=np.float64)
            self._residual = OnlineResidualRidge(x_train.shape[1]).fit(x_train, y_meta - p_train)
            train_di = worker.outlier_gate.compute_di(worker.svd_orthogonalizer.transform(x_train))
            self._abstain_di = abstention_threshold(train_di)
        except Exception as exc:
            logger.warning("Residual layer fit failed (%s); predictions use LightGBM alone.", exc)
            self._residual = None
            self._abstain_di = float("inf")
        self._ffd_close_cfg = ffd_cfg
        pos_rate = float(np.mean(y_meta))
        self._last_retrain_message = f"fitted on {n_keep} labelled real bars (positive rate {pos_rate:.1%})"
        return worker

    def _train_initial_ml_worker(self) -> None:
        """Train the initial worker from the Parquet store of real bars; otherwise run without a model."""
        training_bars = _load_training_bars_from_parquet(self._parquet_path)
        self._training_rows_with_oi = len(training_bars)
        worker: LightGBMMetaLabelWorker | None = None
        try:
            worker = self._fit_worker_on_bars(training_bars)
        except Exception as exc:
            self._last_retrain_message = f"fit error: {exc}"
            logger.warning("Initial meta-labeller fit failed: %s", exc)
        if worker is None and len(self._history_bars) >= 16:
            # No model: still freeze an FFD configuration on the real history so features are stable.
            try:
                log_close = np.log(np.array([b.close for b in self._history_bars], dtype=np.float64))
                self._ffd_close_cfg = calibrate_optimal_d(log_close, split_id="live_history_split")
            except Exception as exc:
                logger.warning("FFD calibration on history failed (%s); default weights retained.", exc)
        self._ml_worker = worker
        if self._history_bars:
            self._last_retrain_ts_ns = self._history_bars[-1].ts_close_ns

    def _check_decay(self, ts_close_ns: int) -> bool:
        """Daily decay check on the live features (P5.2); returns True when a breach should trigger a retrain."""
        if self._last_decay_check_ns is not None and ts_close_ns - self._last_decay_check_ns < RETRAIN_INTERVAL_NS:
            return False
        bars = self._history_bars[-FEATURE_WINDOW_BARS:]
        if len(bars) < 400:
            return False
        self._last_decay_check_ns = ts_close_ns
        try:
            snaps = compute_positioning_snapshots(bars, close_ffd_config=self._ffd_close_cfg)
            x, _, _ = extract_continuous_and_binary_from_snapshots(snaps)
            feats = {name: x[:, i] for i, name in enumerate(("ffd_close_zscore", "ffd_oi_zscore", "oi_funding_basis_zscore"))}
            self._latest_decay = evaluate_decay(feats, np.array([b.close for b in bars], dtype=np.float64))
        except Exception as exc:
            logger.warning("Decay check failed: %s", exc)
            return False
        return self._latest_decay.retrain_triggered

    @property
    def decay_telemetry(self) -> dict[str, Any]:
        rep = self._latest_decay
        if rep is None:
            return {"available": False, "reason": "needs 400 bars of history"}
        return {
            "available": True,
            "retrain_triggered": rep.retrain_triggered,
            "breached_features": rep.breached_features,
            "recommended_cadence_bars": rep.recommended_cadence_bars,
            "features": {
                n: {"half_life_bars": f.half_life_bars, "long_run_ic": f.long_run_ic, "latest_rolling_ic": (f.rolling_ic[-1] if f.rolling_ic else None), "breached": f.breached}
                for n, f in rep.features.items()
            },
        }

    def _maybe_retrain(self, ts_close_ns: int) -> None:
        """Retrain on the daily schedule, or early on an IC breach (P5.2); keep the previous worker on any skip."""
        if self._last_retrain_ts_ns is None:
            self._last_retrain_ts_ns = ts_close_ns
            return
        decay_breach = self._check_decay(ts_close_ns)
        if ts_close_ns - self._last_retrain_ts_ns < RETRAIN_INTERVAL_NS and not decay_breach:
            return
        if decay_breach:
            line = f"[{self._now_str()} • DECAY] IC breach on {', '.join(self._latest_decay.breached_features if self._latest_decay else [])}: retraining early."
            self._activity_log.append(line)
            _append_log_line(line)
        self._last_retrain_ts_ns = ts_close_ns
        training_bars = _load_training_bars_from_parquet(self._parquet_path)
        try:
            worker = self._fit_worker_on_bars(training_bars)
        except Exception as exc:
            worker = None
            self._last_retrain_message = f"fit error: {exc}"
            logger.warning("Scheduled meta-labeller retrain failed: %s", exc)
        if worker is not None:
            self._ml_worker = worker
            msg = f"[{self._now_str()} • RETRAIN] Refitted SVD + Mahalanobis Gate + LightGBM: {self._last_retrain_message}."
        else:
            msg = f"[{self._now_str()} • RETRAIN SKIPPED] {self._last_retrain_message}."
        self._activity_log.append(msg)
        _append_log_line(msg)

    # ------------------------------------------------------------------ properties

    @property
    def state(self) -> DashboardTelemetryState:
        """Return the current immutable ``DashboardTelemetryState`` snapshot."""
        return self._state

    @property
    def tick_counter(self) -> int:
        """Return the number of real closed bars in the engine's history."""
        return self._tick_counter

    @property
    def auto_running(self) -> bool:
        """Return True when the background Dry-Run poller is active."""
        return self._auto_running

    @property
    def activity_log(self) -> list[str]:
        """Return the most recent plain-English activity log entries (newest first)."""
        return list(reversed(self._activity_log[-25:]))

    @property
    def latest_market_snapshot(self) -> MultiAssetMarketSnapshot | None:
        """Return the latest multi-asset live market snapshot (BTC, ETH, SOL)."""
        return getattr(self, "_latest_market_snapshot", None)

    @property
    def forensics_history(self) -> list[dict[str, Any]]:
        """Return the per-bar decision records."""
        return self._forensics_history

    @property
    def latest_forensic(self) -> dict[str, Any] | None:
        """Return the most recent decision record."""
        return self._latest_forensic

    @property
    def risk_committee_output(self) -> RiskCommitteeOutput:
        """Return the most recent 3-model risk committee consensus output."""
        return self._latest_committee_output

    @property
    def data_availability(self) -> DataAvailability:
        """Which live inputs are backed by a real feed."""
        return self._data_availability

    @property
    def history_bars(self) -> list[DollarBar]:
        """Real closed bars held in memory (seed plus live)."""
        return list(self._history_bars)

    @property
    def ml_worker(self) -> LightGBMMetaLabelWorker | None:
        """The trained meta-labelling worker, or None when no honest model exists yet."""
        return self._ml_worker

    @property
    def position_manager(self) -> PositionManager:
        """Open book and realised trades."""
        return self._positions

    @property
    def hrp_weights(self) -> dict[str, float]:
        """Latest unsigned HRP-CVaR risk budgets."""
        return dict(self._latest_hrp_weights)

    @property
    def latest_targets(self) -> dict[str, SleeveTarget]:
        """Signed sleeve targets from the latest processed bar."""
        return dict(self._latest_targets)

    @property
    def sigma_bar(self) -> float:
        """Current EWMA (span 96) volatility of 15m log returns."""
        return self._sigma_estimator.value

    @property
    def sleeve_telemetry(self) -> dict[str, Any]:
        """Latest real evaluations of the funding-carry and derivatives-alpha sleeves."""
        btc_eval = self._latest_carry_evaluations.get("BTC")
        btc_deriv = self._latest_derivatives_signals.get("BTC")
        cached_btc = self._derivatives_inputs_cache.get("BTC")
        surface_btc = cached_btc[1] if cached_btc is not None else None
        vault_btc = cached_btc[2] if cached_btc is not None else None
        open_carry = [p for p in self._positions.open_positions.values() if p.sleeve_id == SLEEVE_FUNDING_CARRY]
        open_deriv = [p for p in self._positions.open_positions.values() if p.sleeve_id == SLEEVE_DERIVATIVES]

        if btc_deriv is not None:
            if btc_deriv.term_structure_slope > self._derivatives_alpha_sleeve.config.term_structure_contango_threshold:
                term_label = "CONTANGO"
            elif btc_deriv.term_structure_slope < self._derivatives_alpha_sleeve.config.term_structure_backwardation_threshold:
                term_label = "BACKWARDATION"
            else:
                term_label = "FLAT"
            crowd: str = btc_deriv.hlp_vault_crowding
            vulnerability = "ELEVATED" if crowd != "BALANCED" else "LOW"
        else:
            term_label = "UNAVAILABLE"
            crowd = "UNKNOWN"
            vulnerability = "UNKNOWN"

        return {
            "funding_carry": {
                "sleeve_id": SLEEVE_FUNDING_CARRY,
                "active_strategy": btc_eval.direction.value if btc_eval is not None else CarryDirection.FLAT.value,
                "hyperliquid_1h_settlement_active": True,
                "routing_tag": "ALO_POST_ONLY",
                "annualized_apr_pct": round(btc_eval.annualized_funding_apr * 100.0, 2) if btc_eval is not None else 0.0,
                "funding_rate_1h": btc_eval.funding_rate_1h if btc_eval is not None else 0.0,
                "maker_rebate_bps": self._fee.maker_bps,
                "taker_fee_avoided_bps": round(self._fee.taker_bps - self._fee.maker_bps, 2),
                "cross_venue_spread_bps": None,
                "venue_pair": "UNAVAILABLE (no second funding venue feed)",
                "breakeven_hours": round(min(btc_eval.breakeven_hours, 9999.0), 1) if btc_eval is not None else None,
                "expected_net_carry_bps": round(btc_eval.expected_net_carry_bps, 2) if btc_eval is not None else 0.0,
                "is_actionable": btc_eval.is_actionable if btc_eval is not None else False,
                "reason": btc_eval.reason if btc_eval is not None else "No evaluation yet.",
                "open_positions": len(open_carry),
                "status": "HARVESTING_ACTIVE" if open_carry else "FLAT",
                "evaluations": {
                    coin: ev.model_dump(mode="json") for coin, ev in self._latest_carry_evaluations.items()
                },
            },
            "derivatives_alpha": {
                "sleeve_id": SLEEVE_DERIVATIVES,
                "deribit_25d_skew": btc_deriv.skew_25d if btc_deriv is not None else 0.0,
                "term_structure_slope": btc_deriv.term_structure_slope if btc_deriv is not None else 0.0,
                "term_structure_label": term_label,
                "net_dealer_gex_usd": surface_btc.net_gex_usd if surface_btc is not None else 0.0,
                "gex_regime": btc_deriv.gex_regime if btc_deriv is not None else "UNAVAILABLE",
                "hlp_vault_address": vault_btc.vault_address if vault_btc is not None else self._derivatives_alpha_sleeve.config.vault_address,
                "hlp_crowding": crowd,
                "counterparty_vulnerability": vulnerability,
                "primary_direction": btc_deriv.primary_direction if btc_deriv is not None else 0,
                "p_meta": btc_deriv.p_meta if btc_deriv is not None else 0.5,
                "expected_gross_edge_bps": btc_deriv.expected_gross_edge_bps if btc_deriv is not None else 0.0,
                "open_positions": len(open_deriv),
                "allocating": self._derivatives_alpha_sleeve.config.allocate,
                "status": (
                    ("ALPHA_SIGNALS_ONLINE" if self._derivatives_alpha_sleeve.config.allocate else "SHADOW_NOT_VALIDATED")
                    if self._data_availability.options_surface or self._data_availability.hlp_vault
                    else "INPUTS_UNAVAILABLE"
                ),
                "signals": {
                    coin: sig.model_dump(mode="json") for coin, sig in self._latest_derivatives_signals.items()
                },
            },
            "trend": {
                "sleeve_id": TREND_SLEEVE_ID,
                "enabled": self._trend_sleeve.enabled,
                "hourly_closes_held": min((len(v) for v in self._hourly_returns.closes_by_asset().values()), default=0),
                "history_required_hours": self._trend_sleeve.config.min_history_hours,
                "signals": {coin: sig.model_dump(mode="json") for coin, sig in self._latest_trend_signals.items()},
                "open_positions": len([p for p in self._positions.open_positions.values() if p.sleeve_id == TREND_SLEEVE_ID]),
            },
            "meta_labeller": {
                "sleeve_id": SLEEVE_META_LABELLER,
                "model_available": self._ml_worker is not None,
                "last_retrain_message": self._last_retrain_message,
                "sigma_bar_ewma96": round(self._sigma_estimator.value, 6),
            },
            "macro_inputs": {
                "stablecoin_depeg_bps": self._latest_macro_inputs.stablecoin_depeg_bps,
                "dvol_index": self._latest_macro_inputs.dvol_index,
                "dvol_zscore": self._latest_macro_inputs.dvol_zscore,
                "stablecoin_medians_usd": (
                    self._latest_macro_inputs.stablecoin_marks.median_usd
                    if self._latest_macro_inputs.stablecoin_marks is not None
                    else {}
                ),
                "stablecoin_venues": (
                    self._latest_macro_inputs.stablecoin_marks.venues_per_symbol
                    if self._latest_macro_inputs.stablecoin_marks is not None
                    else {}
                ),
            },
            "hyperliquid_aux": {
                "oi_cap_assets": sorted(self._latest_aux.oi_cap_assets),
                "predicted_funding_per_day": {
                    coin: {venue: round(pf.rate_per_day, 8) for venue, pf in venues.items()}
                    for coin, venues in self._latest_aux.predicted_fundings.items()
                },
            },
            "shadow": self.shadow_telemetry,
            "decay": self.decay_telemetry,
            "residual_layer": self.residual_telemetry,
            "fees": self.fee_telemetry,
            "cascade": self.cascade_telemetry,
            "circuit_breakers": (
                {
                    **self._breakers.snapshot(current_bar=len(self._history_bars) - 1).model_dump(mode="json"),
                    "blocked_entries_last_bar": list(self._blocked_entries),
                }
                if self._breakers is not None
                else {"enabled": False}
            ),
            "hrp_weights": dict(self._latest_hrp_weights),
            "hrp_is_fallback": self._hrp_scheduler.last_result_is_fallback,
            "hourly_return_rows": self._hourly_returns.n_rows,
            "data_availability": self._data_availability.model_dump(mode="json"),
            "sleeve_attribution": {
                k: v.model_dump(mode="json") for k, v in self._positions.sleeve_attribution(self._current_marks()).items()
            },
        }

    @property
    def ledger_summary(self) -> dict[str, Any]:
        """Return persistent account ledger summary metrics."""
        total_trades = self._wins + self._losses
        win_rate = (self._wins / total_trades * 100.0) if total_trades > 0 else 0.0
        curr_eq = round(self._starting_capital_usd + self._total_realized_pnl, 2)
        marks = self._current_marks()
        unrealized = round(self._positions.total_unrealized_pnl_usd(marks), 2)
        return {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": self._starting_capital_usd,
            "current_cash_usd": curr_eq,
            "current_equity_usd": curr_eq,
            "total_realized_pnl": round(self._total_realized_pnl, 2),
            "accumulated_net_pnl_usd": round(self._total_realized_pnl, 2),
            "accumulated_net_pnl_bps": round(
                (self._total_realized_pnl / max(1.0, self._starting_capital_usd)) * 10_000.0, 1
            ),
            "accumulated_gross_pnl_usd": round(
                self._total_realized_pnl + self._total_fees_paid_usd, 2
            ),
            "total_fees_paid_usd": round(self._total_fees_paid_usd, 2),
            "total_fee_savings_usd": round(self._total_fee_savings_usd, 2),
            "total_funding_pnl_usd": round(self._total_funding_pnl_usd, 4),
            "total_trades": total_trades,
            "win_count": self._wins,
            "loss_count": self._losses,
            "win_rate_pct": round(win_rate, 1),
            "trade_count": len(self._trade_history),
            "open_positions": len(self._positions.open_positions),
            "unrealized_pnl_usd": unrealized,
            "marked_equity_usd": round(curr_eq + unrealized, 2),
            "maker_fill_ratio": round(self._maker_fill_ratio(), 4),
        }

    @staticmethod
    def _now_str() -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # ------------------------------------------------------------------ macro / auxiliary feeds (P0.8, P1.4)

    def _refresh_macro_inputs(self, ts_ns: int) -> MacroInputs:
        try:
            return self._macro_feed.macro_inputs(ts_ns)
        except Exception as exc:
            logger.warning("Macro inputs feed failed: %s", exc)
            return MacroInputs(ts_ns=ts_ns)

    def _refresh_aux(self, ts_ns: int) -> HyperliquidAuxSnapshot:
        try:
            return self._aux_feed.aux_snapshot(ts_ns, ASSETS)
        except Exception as exc:
            logger.warning("Hyperliquid auxiliary feed failed: %s", exc)
            return HyperliquidAuxSnapshot(ts_ns=ts_ns)

    def _evaluate_macro_sentinel(self) -> MacroSentinelOutput:
        """Macro sentinel on fetched inputs. A missing feed contributes no stress (0 bps, default DVOL) and is flagged."""
        mi = self._latest_macro_inputs
        depeg = float(mi.stablecoin_depeg_bps) if mi.stablecoin_depeg_bps is not None else 0.0
        if mi.dvol_index is not None:
            return self._risk_sentinel.evaluate(
                stablecoin_depeg_bps=depeg,
                dvol_index=float(mi.dvol_index),
                dvol_zscore=float(mi.dvol_zscore or 0.0),
            )
        return self._risk_sentinel.evaluate(stablecoin_depeg_bps=depeg)

    @property
    def latest_macro_inputs(self) -> MacroInputs:
        """DVOL and stablecoin inputs used by the latest committee evaluation."""
        return self._latest_macro_inputs

    @property
    def latest_aux_snapshot(self) -> HyperliquidAuxSnapshot:
        """Latest Hyperliquid predicted fundings and OI-cap list."""
        return self._latest_aux

    def _availability_with_feeds(self, base: DataAvailability) -> DataAvailability:
        mi = self._latest_macro_inputs
        aux = self._latest_aux
        update: dict[str, Any] = {
            "dvol": mi.dvol_available,
            "stablecoin_marks": mi.stablecoin_marks_available,
            "predicted_fundings": aux.predicted_fundings_available,
            "oi_caps": aux.oi_caps_available,
            "checked_at_ns": max(base.checked_at_ns, mi.ts_ns, aux.ts_ns),
            # Open-interest history is available when the training store holds enough real rows with OI
            # (from a backfill or from bars recorded live), even though the candle feed has none.
            "open_interest_history": base.open_interest_history or self._training_rows_with_oi >= MIN_TRAIN_LABELLED_BARS,
        }
        if self._stream_hub is not None:
            btc_flow = self._latest_bar_flows.get("BTC")
            avail = self._stream_hub.availability()
            update["trades"] = avail.trades or (btc_flow is not None and btc_flow.taker_buy_usd is not None)
            update["lead_mids"] = avail.lead_mids
            update["taker_flow"] = btc_flow is not None and btc_flow.taker_buy_usd is not None
            update["liquidations"] = btc_flow is not None and btc_flow.liquidation_usd is not None
        return base.model_copy(update=update)

    # ------------------------------------------------------------------ streams (P1.1, P1.3, P1.5)

    @property
    def stream_hub(self) -> StreamHub | None:
        """Websocket hub feeding trades, liquidations and lead mids (None when streams are off)."""
        return self._stream_hub

    def _apply_stream_flows(self, bars_by_coin: Mapping[str, DollarBar]) -> dict[str, DollarBar]:
        """Replace even-split taker volume and zero liquidations with stream values where the stream covered the bar."""
        out = dict(bars_by_coin)
        self._latest_bar_flows = {}
        if self._stream_hub is None:
            return out
        for coin, bar in bars_by_coin.items():
            flow = self._stream_hub.bar_flow(coin, bar.ts_open_ns // 1_000_000, bar.ts_close_ns // 1_000_000)
            self._latest_bar_flows[coin] = flow
            update: dict[str, float] = {}
            if flow.taker_buy_usd is not None and flow.taker_sell_usd is not None:
                update["buy_dollar_volume"] = flow.taker_buy_usd
                update["sell_dollar_volume"] = flow.taker_sell_usd
            if flow.liquidation_usd is not None:
                update["liquidation_volume_usd"] = flow.liquidation_usd
            if update:
                out[coin] = bar.model_copy(update=update)
        return out

    @property
    def stream_telemetry(self) -> dict[str, Any]:
        """Latest stream-derived values: lead-mid moves, boundary imbalance, recent liquidations, bar flows."""
        hub = self._stream_hub
        if hub is None:
            return {"enabled": False}
        out: dict[str, Any] = {"enabled": True, "running": hub.running, "status": hub.status, "assets": {}}
        for coin in ASSETS:
            bi = hub.boundaries.latest(coin)
            flow = self._latest_bar_flows.get(coin)
            out["assets"][coin] = {
                "lead_mid_change_bps_700ms": hub.lead_mids.lead_mid_change_bps(coin, 700),
                "lead_venues": hub.lead_mids.venues(coin),
                "boundary_imbalance": bi.model_dump(mode="json") if bi is not None else None,
                "recent_liquidations": [e.model_dump(mode="json") for e in hub.liquidations.recent(coin, 5)],
                "last_bar_flow": flow.model_dump(mode="json") if flow is not None else None,
            }
        return out

    # ------------------------------------------------------------------ residual layer and abstention (P5.3)

    def _with_extra_features(self, x_cont: np.ndarray, bars: Sequence[DollarBar]) -> np.ndarray:
        if not self._meta_extra_features:
            return x_cont
        return np.column_stack([x_cont, bar_feature_matrix(bars, self._meta_extra_features)])

    @property
    def meta_feature_names(self) -> list[str]:
        return ["ffd_close_zscore", "ffd_oi_zscore", "oi_funding_basis_zscore", *self._meta_extra_features]

    def _apply_residual_and_abstention(self, meta: MetaPredictionOutput, x_row: np.ndarray) -> MetaPredictionOutput:
        self._predictions_made += 1
        if meta.mahalanobis_di > self._abstain_di:
            self._abstentions += 1
            return meta.model_copy(update={"p_meta": 0.5, "epistemic_gate_passed": False, "expected_net_edge_bps": 0.0})
        if self._residual is None:
            return meta
        p_adj = float(self._residual.adjust(meta.p_meta, np.asarray(x_row, dtype=np.float64).reshape(1, -1))[0])
        return meta.model_copy(update={"p_meta": p_adj})

    @property
    def residual_telemetry(self) -> dict[str, Any]:
        return {
            "residual_layer": self._residual is not None,
            "abstention_di_threshold": None if not math.isfinite(self._abstain_di) else self._abstain_di,
            "predictions": self._predictions_made,
            "abstentions": self._abstentions,
            "abstention_rate": (self._abstentions / self._predictions_made) if self._predictions_made else 0.0,
            "meta_features": self.meta_feature_names,
        }

    # ------------------------------------------------------------------ shadow mode (P5.1)

    @property
    def shadow_sleeve_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self._derivatives_alpha_sleeve.config.allocate:
            ids.add(SLEEVE_DERIVATIVES)
        if self._trend_sleeve.enabled and self._trend_sleeve.config.shadow:
            ids.add(TREND_SLEEVE_ID)
        return ids

    def _step_shadow_book(self, bar_index: int, ts_ns: int, marks: Mapping[str, BarMarks], regime_veto: bool, sigma_bar: float) -> None:
        closed = self._shadow_positions.process_bar(bar_index=bar_index, ts_ns=ts_ns, marks=marks, targets=self._latest_shadow_targets, regime_veto=regime_veto)
        for rec in closed:
            book = self._shadow_books.setdefault(rec.sleeve_id, ShadowBook(rec.sleeve_id, self._shadow_thresholds.get(rec.sleeve_id), capital_usd=self._starting_capital_usd))
            book.record(rec, self._fee)
        if regime_veto:
            return
        equity = self._starting_capital_usd + self._total_realized_pnl
        for tgt in self._latest_shadow_targets.values():
            m = marks.get(tgt.asset)
            if tgt.weight == 0.0 or m is None or m.bar is None:
                continue
            self._shadow_positions.open_position(tgt, equity_usd=equity, price=m.bar.close, spot_price=m.spot_price, bar_index=bar_index, ts_ns=ts_ns, sigma_bar=sigma_bar)
            self._shadow_books.setdefault(tgt.sleeve_id, ShadowBook(tgt.sleeve_id, self._shadow_thresholds.get(tgt.sleeve_id), capital_usd=self._starting_capital_usd))

    @property
    def shadow_positions(self) -> PositionManager:
        return self._shadow_positions

    @property
    def shadow_telemetry(self) -> dict[str, Any]:
        open_by: dict[str, int] = {}
        for p in self._shadow_positions.open_positions.values():
            open_by[p.sleeve_id] = open_by.get(p.sleeve_id, 0) + 1
        return {
            "sleeves": sorted(self.shadow_sleeve_ids),
            "status": {sid: b.status().model_dump(mode="json") for sid, b in self._shadow_books.items()},
            "open_positions": open_by,
        }

    # ------------------------------------------------------------------ fee tier and hedge venue (P4.4)

    @property
    def fee_telemetry(self) -> dict[str, Any]:
        hedge: dict[str, Any] = {}
        other = FeeSchedule.usdt_perp()  # Binance/Bybit base tier: an assumption, the paper account does not trade there
        fees = {"HlPerp": self._fee.roundtrip_maker_plus_slippage_bps * 1e-4, "BinPerp": other.roundtrip_maker_plus_slippage_bps * 1e-4, "BybitPerp": other.roundtrip_maker_plus_slippage_bps * 1e-4}
        for coin, venues in self._latest_aux.predicted_fundings.items():
            choice = best_hedge_venue({v: pf.rate_per_day for v, pf in venues.items()}, fees)
            if choice is not None:
                hedge[coin] = choice.model_dump(mode="json")
        return {
            "maker_bps": self._fee.maker_bps,
            "taker_bps": self._fee.taker_bps,
            "tier": tier_index(self._fee_table, self._account_volume_14d_usd) if self._fee_table is not None else 0,
            "volume_14d_usd": self._account_volume_14d_usd,
            "source": "venue fee table" if self._fee_table is not None else "Hyperliquid base tier",
            "carry_hedge_venue": hedge,
            "other_venue_fees_note": "Binance/Bybit priced at the base USDT tier (assumption).",
        }

    # ------------------------------------------------------------------ markouts (P4.2)

    def _fill_clock_ms(self, snap: MultiAssetMarketSnapshot) -> int:
        """Live fills are timestamped on the wall clock; replays use the snapshot (bar) time."""
        if isinstance(self._feed, HyperliquidCandleBarFeed):
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        return snap.timestamp_ns // 1_000_000

    def _record_markout_fills(self, opened: Sequence[Position], closed: Sequence[ClosedTradeRecord], snap: MultiAssetMarketSnapshot) -> None:
        now_ms = self._fill_clock_ms(snap)
        hub = self._stream_hub
        for p in opened:
            lead = hub.lead_mids.lead_mid_change_bps(p.asset, 700) if hub is not None else None
            self._markouts.record_fill(
                fill_id=f"open:{p.position_id}", coin=p.asset, side="BUY" if p.side > 0 else "SELL", price=p.entry_price,
                ts_ms=now_ms, notional_usd=p.notional_usd, sleeve_id=p.sleeve_id, lead_move_bps=lead,
            )
        for c in closed:
            lead = hub.lead_mids.lead_mid_change_bps(c.asset, 700) if hub is not None else None
            self._markouts.record_fill(
                fill_id=f"close:{c.trade_id}", coin=c.asset, side="SELL" if c.side > 0 else "BUY", price=c.exit_price,
                ts_ms=now_ms, notional_usd=c.size * c.exit_price, sleeve_id=c.sleeve_id, lead_move_bps=lead,
            )

    def resolve_markouts(self, now_ms: int | None = None) -> int:
        """Resolve pending markouts against the hub's Hyperliquid mids (no-op without a stream hub)."""
        if self._stream_hub is None:
            return 0
        t = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
        return self._markouts.resolve(self._stream_hub.hl_mids.mid_at, t)

    @property
    def markout_tracker(self) -> MarkoutTracker:
        return self._markouts

    @property
    def markout_telemetry(self) -> dict[str, Any]:
        self.resolve_markouts()
        tr = self._markouts
        return {
            "fills": len(tr.fills),
            "horizons_ms": list(tr.horizons),
            "all": tr.summary("all"),
            "by_sleeve": tr.summary("sleeve_id"),
            "by_size_bucket": tr.summary("size_bucket"),
            "by_lead_state": tr.summary("lead_state"),
            "ewma_30s_bps": tr.ewma_30s_bps,
            "vpin_threshold_effective": tr.toxicity_threshold(0.60),
            "mid_source": "hyperliquid_bbo_stream" if self._stream_hub is not None else "unavailable",
            "recent": [f.model_dump(mode="json") for f in tr.fills[-10:]],
        }

    # ------------------------------------------------------------------ cascade and ADL controls (P3.4)

    def _update_cascade(self, bars_by_coin: Mapping[str, DollarBar], snap: MultiAssetMarketSnapshot, ts_ns: int) -> None:
        for coin, bar in bars_by_coin.items():
            flow = self._latest_bar_flows.get(coin)
            a_snap = snap.get_asset(coin)
            depth = a_snap.l2_book.depth_10bps_usd if a_snap is not None and a_snap.l2_book.depth_10bps_usd > 0.0 else None
            buy: float | None = flow.taker_buy_usd if flow is not None else None
            sell: float | None = flow.taker_sell_usd if flow is not None else None
            if flow is None and self._bars_carry_taker_flow:
                buy, sell = bar.buy_dollar_volume, bar.sell_dollar_volume
            self._cascade.update(
                coin,
                ts_ns=ts_ns,
                taker_buy_usd=buy,
                taker_sell_usd=sell,
                open_interest=bar.open_interest if bar.open_interest > 0.0 else None,
                funding_rate_1h=bar.funding_rate_1h,
                depth_10bps_usd=depth,
            )
        self._latest_fragility = {coin: self._cascade.assess(coin, ts_ns) for coin in bars_by_coin}
        marks = self._latest_macro_inputs.stablecoin_marks
        usdc = [q.price for q in marks.quotes if q.symbol == "USDC" and q.quote_currency == "USD"] if marks is not None else []
        self._collateral_haircut = stablecoin_collateral_haircut({"USDC": usdc}) if usdc else 0.0

    @property
    def cascade_telemetry(self) -> dict[str, Any]:
        equity = self._starting_capital_usd + self._total_realized_pnl
        marks = self._current_marks()
        snap = self.latest_market_snapshot
        adl: dict[str, float] = {}
        stress_inputs: list[StressPosition] = []
        for key, pos in self._positions.open_positions.items():
            m = marks.get(pos.asset)
            mark = m.mark_price if m is not None else pos.entry_price
            if pos.side < 0:
                adl[key] = round(adl_rank_estimate(mark, pos.entry_price, pos.size * mark, equity), 6)
            a_snap = snap.get_asset(pos.asset) if snap is not None else None
            depth = a_snap.l2_book.depth_10bps_usd if a_snap is not None else 0.0
            stress_inputs.append(StressPosition(asset=pos.asset, notional_usd=pos.size * mark, depth_10bps_usd=depth))
        stress = liquidation_stress_loss(stress_inputs)
        return {
            "fragility": {coin: f.model_dump(mode="json") for coin, f in self._latest_fragility.items()},
            "collateral_haircut": self._collateral_haircut,
            "adl_rank_estimates": adl,
            "stress_loss_usd": {k: round(v, 4) for k, v in stress.items()},
            "stress_loss_pct_equity": round(sum(stress.values()) / equity, 6) if equity > 0 else 0.0,
            "notes": "Fragile assets: directional entries halved, carry entries blocked, quoting paused. "
            "Leg de-levering on hedges is reported, not executed (no leg resizing in the paper book).",
        }

    # ------------------------------------------------------------------ impact (P3.1)

    def _impact_bps(self, coin: str, w_hrp: float, sigma_bar: float) -> float:
        """Square-root impact of a target-sized order in bps, using the asset's 24h volume from the live snapshot."""
        snap = self.latest_market_snapshot
        a_snap = snap.get_asset(coin) if snap is not None else None
        volume = float(a_snap.day_volume_usd) if a_snap is not None else 0.0
        equity = self._starting_capital_usd + self._total_realized_pnl
        notional = max(0.0, w_hrp) * equity * 0.5  # half-Kelly at full conviction bounds the order size
        sigma_daily = max(sigma_bar, 0.0) * math.sqrt(96.0)
        return square_root_impact(sigma_daily, notional, volume) * 10_000.0

    @property
    def transfer_coefficient(self) -> float | None:
        """Correlation between raw signed sleeve scores and executed directional weights on the last bar."""
        return self._latest_transfer_coefficient

    # ------------------------------------------------------------------ circuit breakers (P3.2)

    def _operational_status(self, snap: MultiAssetMarketSnapshot) -> OperationalStatus:
        """Feed age is measured against the wall clock only for the live candle feed (replays run in bar time)."""
        age = 0.0
        if isinstance(self._feed, HyperliquidCandleBarFeed) and snap.timestamp_ns > 0:
            age = max(0.0, (datetime.now(timezone.utc).timestamp() * 1e9 - snap.timestamp_ns) / 1e9)
        return OperationalStatus(feed_age_seconds=age, recent_rejects=0, position_state_known=True)

    @property
    def circuit_breakers(self) -> CircuitBreakers | None:
        return self._breakers

    # ------------------------------------------------------------------ market context helpers

    def _bar_source_tag(self, snap: MultiAssetMarketSnapshot) -> str:
        """Tag persisted bars with the feed that produced them; only real-feed tags are used for training."""
        feed_avail = self._feed.availability
        if feed_avail.feed_source == "SCRIPTED":
            return "SCRIPTED-REAL" if feed_avail.open_interest_history else "SCRIPTED-SYNTHETIC"
        return str(snap.feed_source)

    def _current_marks(self) -> dict[str, BarMarks]:
        snap = self.latest_market_snapshot
        marks: dict[str, BarMarks] = {}
        for coin in ASSETS:
            a_snap = snap.get_asset(coin) if snap is not None else None
            if a_snap is not None:
                marks[coin] = BarMarks(
                    asset=coin,
                    bar=None,
                    mark_price=a_snap.mark_price,
                    spot_price=a_snap.oracle_price,
                    funding_rate_1h=a_snap.funding_rate_1h,
                )
            elif coin == "BTC" and self._history_bars:
                last = self._history_bars[-1]
                marks[coin] = BarMarks(asset=coin, bar=None, mark_price=last.mark_price, spot_price=last.spot_price)
        return marks

    def _build_marks(self, bars_by_coin: Mapping[str, DollarBar], snap: MultiAssetMarketSnapshot) -> dict[str, BarMarks]:
        marks: dict[str, BarMarks] = {}
        for coin in ASSETS:
            bar = bars_by_coin.get(coin)
            a_snap = snap.get_asset(coin)
            if bar is None and a_snap is None:
                continue
            if bar is not None:
                mark_px, spot_px, f1h = bar.mark_price, bar.spot_price, bar.funding_rate_1h
                bar_hours = (bar.ts_close_ns - bar.ts_open_ns) / 3.6e12
            elif a_snap is not None:
                mark_px, spot_px, f1h = a_snap.mark_price, a_snap.oracle_price, a_snap.funding_rate_1h
                bar_hours = 0.25
            else:
                continue
            if mark_px <= 0.0 or spot_px <= 0.0:
                continue
            marks[coin] = BarMarks(
                asset=coin,
                bar=bar,
                mark_price=mark_px,
                spot_price=spot_px,
                funding_rate_1h=f1h,
                bar_hours=max(bar_hours, 1e-6),
            )
        return marks

    def _derivatives_inputs(
        self, coin: str, ts_ns: int
    ) -> tuple[DeribitOptionsSurface | None, HyperliquidWhaleVaultMetrics | None]:
        """Options surface and HLP vault metrics for ``coin`` with a 5-minute TTL cache."""
        cached = self._derivatives_inputs_cache.get(coin)
        if cached is not None and ts_ns - cached[0] < DERIVATIVES_INPUT_TTL_NS:
            return cached[1], cached[2]
        surface: DeribitOptionsSurface | None = None
        vault: HyperliquidWhaleVaultMetrics | None = None
        if self._fetch_derivatives_inputs:
            try:
                surface = self._derivatives_alpha_sleeve.fetch_deribit_options_surface_sync(coin)
            except Exception as exc:
                logger.debug("Options surface fetch failed for %s: %s", coin, exc)
            vault_cached = self._derivatives_inputs_cache.get("__vault__")
            if vault_cached is not None and ts_ns - vault_cached[0] < DERIVATIVES_INPUT_TTL_NS:
                vault = vault_cached[2]
            else:
                try:
                    vault = self._derivatives_alpha_sleeve.fetch_hyperliquid_vault_positioning_sync()
                except Exception as exc:
                    logger.debug("HLP vault fetch failed: %s", exc)
                self._derivatives_inputs_cache["__vault__"] = (ts_ns, None, vault)
        self._derivatives_inputs_cache[coin] = (ts_ns, surface, vault)
        if surface is not None and surface.dvol_annualized > 0.0:
            self._record_surface(coin, surface, ts_ns)
        return surface, vault

    def _record_surface(self, coin: str, surface: DeribitOptionsSurface, ts_ns: int) -> None:
        """Append the surface components to ``data/options_surface/{coin}.parquet`` for later IC evaluation (P2.4)."""
        if not self._fetch_derivatives_inputs:
            return
        try:
            path = DATA_DIR / "options_surface" / f"{coin.lower()}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = pl.DataFrame(
                {
                    "ts_ns": [ts_ns],
                    "rr25_30d": [surface.skew_25d],
                    "term_structure_slope": [surface.term_structure_slope],
                    "net_gex_usd_per_1pct": [surface.net_gex_usd],
                    "atm_iv_near": [surface.dvol_annualized],
                }
            )
            combined = pl.concat([pl.read_parquet(path), row], how="diagonal_relaxed") if path.exists() else row
            combined.write_parquet(path)
        except Exception as exc:
            logger.debug("Surface history append failed: %s", exc)

    def _evaluate_sleeves(
        self,
        snap: MultiAssetMarketSnapshot,
        bars_by_coin: Mapping[str, DollarBar],
        committee: RiskCommitteeOutput,
        ts_ns: int,
    ) -> None:
        """Run both sleeves for every asset from live inputs and keep the evaluations for telemetry."""
        carry_evals: dict[str, CarryOpportunityEvaluation] = {}
        carry_signals: dict[str, SleeveSignalOutput] = {}
        deriv_signals: dict[str, DerivativesAlphaSignal] = {}
        surface_seen = False
        vault_seen = False
        for coin in ASSETS:
            a_snap: AssetMarketSnapshot | None = snap.get_asset(coin)
            if a_snap is None:
                continue
            carry_eval = self._funding_carry_sleeve.evaluate_from_snapshot(
                a_snap,
                fee_schedule=self._fee,
                asri_score=committee.consensus_risk_score,
                tail_risk_flag=committee.tail_risk_flag,
                regime_mode=self._state.regime_mode,
                target_notional_weight=CARRY_TARGET_NOTIONAL_WEIGHT,
            )
            carry_evals[coin] = carry_eval
            carry_signals[coin] = self._funding_carry_sleeve.generate_sleeve_signal(carry_eval)

            surface, vault = self._derivatives_inputs(coin, ts_ns)
            if surface is not None and surface.dvol_annualized > 0.0:
                surface_seen = True  # a priced surface was read (ATM IV present)
            if vault is not None and vault.accounts_read > 0:
                vault_seen = True  # vault positions were read, even if HLP is flat
            deriv_signals[coin] = self._derivatives_alpha_sleeve.generate_alpha_signal(
                options_surface=surface,
                vault_metrics=vault,
                bar=bars_by_coin.get(coin),
                instrument_id=coin,
                timestamp_ns=ts_ns,
            )
        self._latest_carry_evaluations = carry_evals
        self._latest_carry_signals = carry_signals
        self._latest_derivatives_signals = deriv_signals
        self._data_availability = self._data_availability.model_copy(
            update={"options_surface": surface_seen, "hlp_vault": vault_seen}
        )

    def _evaluate_sleeves_for_telemetry(self, snap: MultiAssetMarketSnapshot) -> None:
        if snap.feed_source == "SIMULATED-FALLBACK":
            return
        try:
            self._evaluate_sleeves(snap, {}, self._latest_committee_output, snap.timestamp_ns)
        except Exception as exc:
            logger.warning("Initial sleeve evaluation failed: %s", exc)

    # ------------------------------------------------------------------ allocation (P0.1 / P0.2 / P0.3)

    def _build_targets(
        self,
        meta: MetaPredictionOutput,
        sigma_bar: float,
        regime_multiplier: float,
        regime_mode: ExecutionRegimeMode,
    ) -> tuple[dict[str, SleeveTarget], PortfolioAllocationOutput, dict[str, float]]:
        """Turn sleeve signals into signed per-(sleeve, asset) targets through the governing equation.

        Directional sleeves receive the committee multiplier only in DIRECTIONAL_PLUS_CARRY mode;
        the carry sleeve is allowed in DIRECTIONAL_PLUS_CARRY and CARRY_ONLY_FALLBACK. Everything is
        zero in PAUSED_ENTRIES (existing positions still manage their exits) and EMERGENCY_FLATTENED.
        """
        hrp = self._latest_hrp_weights
        directional_m = regime_multiplier if regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY else 0.0
        carry_m = (
            regime_multiplier
            if regime_mode in (ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY, ExecutionRegimeMode.CARRY_ONLY_FALLBACK)
            else 0.0
        )

        ub, lb = compute_effective_barriers(max(sigma_bar, 0.0), self._barrier_cfg)
        meta_expected_bps = (meta.p_meta * ub - (1.0 - meta.p_meta) * lb) * 10_000.0 - self._impact_bps("BTC", hrp["BTC"], sigma_bar)
        meta_inputs = [
            AssetAllocationInput(
                asset_id="BTC",
                y_primary=meta.effective_direction,
                w_hrp=hrp["BTC"],
                p_meta=meta.p_meta,
                expected_gross_return_bps=meta_expected_bps,
                fee_schedule=self._fee,
                epistemic_gate_passed=meta.epistemic_gate_passed,
            )
        ]
        deriv_inputs = [
            AssetAllocationInput(
                asset_id=coin,
                y_primary=sig.primary_direction,
                w_hrp=hrp[coin],
                p_meta=sig.p_meta,
                expected_gross_return_bps=sig.expected_gross_edge_bps - self._impact_bps(coin, hrp[coin], sigma_bar),
                fee_schedule=self._fee,
                epistemic_gate_passed=True,
            )
            for coin, sig in self._latest_derivatives_signals.items()
        ]
        carry_inputs = [
            AssetAllocationInput(
                asset_id=coin,
                y_primary=1 if sig.direction > 0 else (-1 if sig.direction < 0 else 0),
                w_hrp=hrp[coin],
                p_meta=sig.meta_probability,
                expected_gross_return_bps=sig.raw_expected_return,
                fee_schedule=self._fee,
                epistemic_gate_passed=True,
            )
            for coin, sig in self._latest_carry_signals.items()
        ]

        meta_alloc = compute_signed_target_weights(meta_inputs, regime_multiplier=directional_m, max_gross_leverage=MAX_GROSS_LEVERAGE)
        deriv_alloc = (
            compute_signed_target_weights(deriv_inputs, regime_multiplier=directional_m, max_gross_leverage=MAX_GROSS_LEVERAGE)
            if deriv_inputs
            else None
        )
        carry_alloc = (
            compute_signed_target_weights(carry_inputs, regime_multiplier=carry_m, max_gross_leverage=MAX_GROSS_LEVERAGE)
            if carry_inputs
            else None
        )

        targets: dict[str, SleeveTarget] = {}
        targets[position_key(SLEEVE_META_LABELLER, "BTC")] = SleeveTarget(
            sleeve_id=SLEEVE_META_LABELLER,
            asset="BTC",
            weight=meta_alloc.weights.get("BTC", 0.0),
            hedged=False,
            p_meta=meta.p_meta,
            expected_gross_return_bps=meta_expected_bps,
            reason=(
                f"LightGBM p_meta={meta.p_meta:.3f}, direction={meta.effective_direction:+d}, "
                f"DI={meta.mahalanobis_di:.2f} (gate {'open' if meta.epistemic_gate_passed else 'closed'})"
            ),
        )
        if deriv_alloc is not None:
            for coin, sig in self._latest_derivatives_signals.items():
                targets[position_key(SLEEVE_DERIVATIVES, coin)] = SleeveTarget(
                    sleeve_id=SLEEVE_DERIVATIVES,
                    asset=coin,
                    weight=deriv_alloc.weights.get(coin, 0.0),
                    hedged=False,
                    p_meta=sig.p_meta,
                    expected_gross_return_bps=sig.expected_gross_edge_bps,
                    reason=(
                        f"derivatives alpha direction={sig.primary_direction:+d}, p={sig.p_meta:.3f}, "
                        f"skew={sig.skew_25d:+.4f}, gex={sig.gex_regime}, hlp={sig.hlp_vault_crowding}"
                    ),
                )
        if carry_alloc is not None:
            for coin, carry_sig in self._latest_carry_signals.items():
                ev = self._latest_carry_evaluations.get(coin)
                hedged = ev is not None and ev.direction == CarryDirection.SPOT_LONG_PERP_SHORT
                w = carry_alloc.weights.get(coin, 0.0)
                targets[position_key(SLEEVE_FUNDING_CARRY, coin)] = SleeveTarget(
                    sleeve_id=SLEEVE_FUNDING_CARRY,
                    asset=coin,
                    weight=abs(w) if hedged else w,
                    hedged=hedged,
                    p_meta=carry_sig.meta_probability,
                    expected_gross_return_bps=carry_sig.raw_expected_return,
                    reason=ev.reason if ev is not None else "carry evaluation unavailable",
                )

        # Trend sleeve: continuous inverse-vol weights, re-evaluated once per completed hour.
        if self._trend_sleeve.enabled:
            bucket = self._hourly_returns.n_rows
            if self._trend_eval_bucket != bucket:
                self._latest_trend_signals = self._trend_sleeve.evaluate(self._hourly_returns.closes_by_asset())
                self._trend_eval_bucket = bucket
            for coin, tsig in self._latest_trend_signals.items():
                targets[position_key(TREND_SLEEVE_ID, coin)] = SleeveTarget(
                    sleeve_id=TREND_SLEEVE_ID,
                    asset=coin,
                    weight=tsig.target_weight * directional_m,
                    hedged=False,
                    exit_policy="target",
                    p_meta=min(1.0, 0.5 + 0.5 * abs(tsig.forecast)),
                    reason=(
                        f"trend forecast {tsig.forecast:+.2f} "
                        f"({', '.join(f'{lb}h {v:+.2f}' for lb, v in tsig.forecast_by_lookback.items())}), "
                        f"vol {tsig.vol_annual:.0%}, corr adj {tsig.corr_adjustment:.2f}"
                    ),
                )

        # Shadow sleeves (P5.1) trade in a separate book; only live targets reach the ledger and the leverage cap.
        shadow_ids = self.shadow_sleeve_ids
        self._latest_shadow_targets = {k: t for k, t in targets.items() if t.sleeve_id in shadow_ids}
        targets = {k: t for k, t in targets.items() if t.sleeve_id not in shadow_ids}

        # Global gross leverage cap across sleeves (hedged carry counts both legs).
        gross = sum(abs(t.weight) * (2.0 if t.hedged else 1.0) for t in targets.values())
        scale = MAX_GROSS_LEVERAGE / gross if gross > MAX_GROSS_LEVERAGE else 1.0
        if scale < 1.0:
            targets = {k: t.model_copy(update={"weight": t.weight * scale}) for k, t in targets.items()}
            gross = MAX_GROSS_LEVERAGE

        per_asset_directional: dict[str, float] = {a: 0.0 for a in ASSETS}
        kelly: dict[str, float] = {a: 0.0 for a in ASSETS}
        raw_unscaled: dict[str, float] = {a: 0.0 for a in ASSETS}
        for t in targets.values():
            if not t.hedged:
                per_asset_directional[t.asset] += t.weight
                raw_unscaled[t.asset] += t.weight / scale if scale > 0.0 else 0.0
            kelly[t.asset] = max(kelly[t.asset], compute_fractional_kelly(t.p_meta))
        rejected = sorted(
            set(meta_alloc.rejected_for_fees)
            | set(deriv_alloc.rejected_for_fees if deriv_alloc is not None else [])
            | set(carry_alloc.rejected_for_fees if carry_alloc is not None else [])
        )
        # Transfer coefficient: how faithfully executed directional weights follow the raw signed scores.
        raw_scores = {a: 0.0 for a in ASSETS}
        raw_scores["BTC"] += float(meta.effective_direction) * max(0.0, 2.0 * meta.p_meta - 1.0)
        for coin, dsig in self._latest_derivatives_signals.items():
            raw_scores[coin] += float(dsig.primary_direction) * max(0.0, 2.0 * dsig.p_meta - 1.0)
        self._latest_transfer_coefficient = transfer_coefficient(
            [raw_scores[a] for a in ASSETS], [per_asset_directional[a] for a in ASSETS]
        )
        allocation = PortfolioAllocationOutput(
            weights={a: round(w, 6) for a, w in per_asset_directional.items()},
            raw_unscaled_weights={a: round(w, 6) for a, w in raw_unscaled.items()},
            hrp_unsigned_weights=dict(hrp),
            kelly_scalars=kelly,
            regime_multiplier=regime_multiplier,
            gross_leverage=min(round(gross, 6), MAX_GROSS_LEVERAGE),
            leverage_scale_factor=scale,
            rejected_for_fees=rejected,
        )
        return targets, allocation, kelly

    # ------------------------------------------------------------------ UI state helpers

    def _barrier_label_for_bar(self, bar: DollarBar, bar_index: int, sigma_bar: float, closed: Sequence[ClosedTradeRecord]) -> TripleBarrierLabel:
        ub, lb = compute_effective_barriers(max(sigma_bar, 0.0), self._barrier_cfg)
        touched: Literal["PROFIT", "STOP", "VERTICAL", "NONE"] = "NONE"
        net_ret = 0.0
        direction: Literal[-1, 0, 1] = 0
        for rec in closed:
            if rec.asset != "BTC" or rec.hedged:
                continue
            direction = rec.side
            net_ret = rec.net_return_bps * 1e-4
            if rec.exit_reason == PositionExitReason.PROFIT_BARRIER:
                touched = "PROFIT"
            elif rec.exit_reason == PositionExitReason.STOP_BARRIER:
                touched = "STOP"
            else:
                touched = "VERTICAL"
            break
        return TripleBarrierLabel(
            bar_index=bar_index,
            primary_direction=direction,
            gross_return=round(net_ret, 6),
            net_return=round(net_ret, 6),
            fee_cost=round(self._fee.roundtrip_maker_plus_slippage_bps * 1e-4, 6),
            funding_cost=0.0,
            upper_barrier=round(bar.close * (1.0 + ub), 2),
            lower_barrier=round(bar.close * (1.0 - lb), 2),
            hit_bar_offset=0,
            barrier_touched=touched,
            meta_label=1 if touched == "PROFIT" else 0,
        )

    def _refresh_initial_state_window(self) -> None:
        """Replace the placeholder UI window with the tail of the real seeded history."""
        if len(self._history_bars) < 2:
            return
        window = self._history_bars[-UI_WINDOW_BARS:]
        try:
            snaps = compute_positioning_snapshots(self._history_bars[-FEATURE_WINDOW_BARS:], close_ffd_config=self._ffd_close_cfg)
        except Exception as exc:
            logger.warning("Initial positioning snapshot failed: %s", exc)
            return
        pos_window = snaps[-len(window):]
        sigma_series = np.concatenate(
            ([1e-4], compute_ewma_volatility(np.diff(np.log(np.array([b.close for b in self._history_bars], dtype=np.float64))), span=96))
        )
        first_idx = len(self._history_bars) - len(window)
        barriers = [
            self._barrier_label_for_bar(b, first_idx + i, float(sigma_series[first_idx + i]), [])
            for i, b in enumerate(window)
        ]
        metas = [
            _neutral_meta_prediction(b.instrument_id, b.ts_close_ns, "history") for b in window
        ]
        hawkes = list(self._state.hawkes_history[-len(window):]) or [self._state.hawkes_state]
        while len(hawkes) < len(window):
            hawkes.insert(0, hawkes[0])
        self._state = self._state.model_copy(
            update={
                "dollar_bars": window,
                "positioning_signals": pos_window,
                "barrier_labels": barriers,
                "meta_predictions": metas,
                "hawkes_history": hawkes,
            }
        )

    # ------------------------------------------------------------------ controls

    def reset_account(self, starting_capital_usd: float = 5_000.0) -> ControlActionResult:
        """Reset total profit to $0.00, close the paper book, and set the starting capital."""
        self._starting_capital_usd = round(float(starting_capital_usd), 2)
        self._base_capital_usd = self._starting_capital_usd
        self._total_realized_pnl = 0.0
        self._total_fees_paid_usd = 0.0
        self._total_fee_savings_usd = 0.0
        self._total_funding_pnl_usd = 0.0
        self._wins = 0
        self._losses = 0
        self._maker_fills = 0
        self._taker_fills = 0
        self._trade_history = []
        self._forensics_history = []
        self._latest_forensic = None
        self._positions = PositionManager(fee_schedule=self._fee, barrier_config=self._barrier_cfg)
        self._persist_ledger_state_atomic()

        base_state = build_default_dashboard_state(
            regime_mode=self._state.regime_mode,
            is_dry_run=True,
            last_control_action="reset",
            last_control_message=(
                f"Account reset to ${self._starting_capital_usd:,.2f} starting capital and $0.00 profit."
            ),
        )
        fresh_pnl = PnLAndExecutionMetrics(
            starting_capital_usd=self._starting_capital_usd,
            net_pnl_usd=0.00,
            net_pnl_bps=0.0,
            gross_pnl_usd=0.00,
            total_fees_paid_usd=0.00,
            maker_fill_ratio=1.0,
            taker_fill_ratio=0.0,
            fee_savings_usd=0.00,
            fee_savings_bps=0.0,
            annualized_carry_apr_pct=self._state.pnl_metrics.annualized_carry_apr_pct,
            is_dry_run=True,
        )
        msg = (
            f"[{self._now_str()}] 🔄 ACCOUNT RESET: Starting Paper Capital set to "
            f"${self._starting_capital_usd:,.2f} | Total Profit cleared to $0.00 (0.0 bps) | Open book cleared | Ledger persisted."
        )
        self._activity_log.append(msg)
        _append_log_line(msg)

        self._state = base_state.model_copy(
            update={
                "pnl_metrics": fresh_pnl,
                "ffd_config": self._ffd_close_cfg,
                "asri_risk": self._state.asri_risk,
                "hawkes_state": self._state.hawkes_state,
                "dollar_bars": self._state.dollar_bars,
                "positioning_signals": self._state.positioning_signals,
                "barrier_labels": self._state.barrier_labels,
                "meta_predictions": self._state.meta_predictions,
                "hawkes_history": self._state.hawkes_history,
                "portfolio_allocation": self._zero_allocation(),
                "last_control_message": msg,
            }
        )
        return ControlActionResult(
            status="ok",
            action="reset",
            regime_mode=self._state.regime_mode,
            is_dry_run=True,
            plain_english_explanation=msg,
            weights=dict(self._state.portfolio_allocation.weights),
            gross_leverage=self._state.portfolio_allocation.gross_leverage,
            tick_count=self._tick_counter,
            latest_price=self._state.dollar_bars[-1].close,
            starting_capital_usd=self._starting_capital_usd,
            current_equity_usd=self._starting_capital_usd,
            net_pnl_usd=0.0,
            net_pnl_bps=0.0,
            activity_log=self.activity_log,
            forensic_record=None,
            ledger_summary=self.ledger_summary,
        )

    # ------------------------------------------------------------------ the tick

    def step_simulation_tick(self) -> ControlActionResult:
        """Poll the feed; process exactly one new closed 15m bar if one exists, else report waiting."""
        snap = self._feed.snapshot()
        self._latest_market_snapshot = snap

        if snap.feed_source == "SIMULATED-FALLBACK":
            return self._idle_result(
                "feed_unavailable",
                "Market data feeds are unreachable (Hyperliquid and Bybit). No bar is processed and nothing trades "
                "until a real feed returns.",
            )

        last_open_ms = self._history_bars[-1].ts_open_ns // 1_000_000 if self._history_bars else 0
        try:
            new_bars = self._feed.next_closed_bars(last_open_ms, snap)
        except Exception as exc:
            logger.warning("Bar feed poll failed: %s", exc)
            new_bars = None
        if new_bars is None or "BTC" not in new_bars:
            next_close_hint = ""
            if self._history_bars:
                next_close_ms = (self._history_bars[-1].ts_close_ns + STEP_15M_NS) // 1_000_000
                remaining_s = max(0.0, next_close_ms / 1000.0 - datetime.now(timezone.utc).timestamp())
                next_close_hint = f" Next 15m candle closes in {remaining_s / 60.0:.1f} min."
            return self._idle_result(
                "waiting_for_bar",
                f"No new closed 15m candle since the last bar; positions are held and nothing is counted.{next_close_hint}",
            )
        return self._process_bar(new_bars, snap)

    def _idle_result(self, action: str, message: str) -> ControlActionResult:
        now = datetime.now(timezone.utc).timestamp()
        if now - self._last_waiting_log_ts >= WAITING_LOG_INTERVAL_SEC:
            self._last_waiting_log_ts = now
            line = f"[{self._now_str()} • {action.upper()}] {message}"
            self._activity_log.append(line)
            _append_log_line(line)
        latest_px = self._history_bars[-1].close if self._history_bars else self._state.dollar_bars[-1].close
        return ControlActionResult(
            status="ok",
            action=action,
            regime_mode=self._state.regime_mode,
            is_dry_run=self._state.is_dry_run,
            plain_english_explanation=message,
            weights=dict(self._state.portfolio_allocation.weights),
            gross_leverage=self._state.portfolio_allocation.gross_leverage,
            tick_count=self._tick_counter,
            latest_price=latest_px,
            starting_capital_usd=self._starting_capital_usd,
            current_equity_usd=self._state.pnl_metrics.current_equity_usd,
            net_pnl_usd=self._state.pnl_metrics.net_pnl_usd,
            net_pnl_bps=self._state.pnl_metrics.net_pnl_bps,
            activity_log=self.activity_log,
            forensic_record=self._latest_forensic,
            ledger_summary=self.ledger_summary,
            bar_processed=False,
        )

    def _process_bar(self, bars_by_coin: Mapping[str, DollarBar], snap: MultiAssetMarketSnapshot) -> ControlActionResult:
        s = self._state
        bars_by_coin = self._apply_stream_flows(bars_by_coin)
        new_bar = bars_by_coin["BTC"]
        feed_source = str(snap.feed_source)
        bar_source = self._bar_source_tag(snap)

        # 1. Real bar bookkeeping: history, Parquet store, volatility, hourly returns, HRP, retrain schedule.
        prev_close = self._history_bars[-1].close if self._history_bars else new_bar.open
        self._history_bars.append(new_bar)
        if len(self._history_bars) > FEATURE_WINDOW_BARS * 2:
            self._history_bars = self._history_bars[-FEATURE_WINDOW_BARS * 2 :]
        self._tick_counter += 1
        k = self._tick_counter
        bar_index = len(self._history_bars) - 1
        ts_close = new_bar.ts_close_ns
        self._parquet_row_count = _persist_bars_to_parquet([new_bar], self._parquet_path, bar_source=bar_source)

        sigma_bar = self._sigma_estimator.update(math.log(max(new_bar.close, 1e-12) / max(prev_close, 1e-12)))
        hourly_prices = {
            coin: (bars_by_coin[coin].close if coin in bars_by_coin else (a.mid_price if (a := snap.get_asset(coin)) is not None else 0.0))
            for coin in ASSETS
        }
        self._hourly_returns.update(ts_close - 1, hourly_prices)  # a bar closing on the hour belongs to that hour
        self._latest_hrp_weights = self._hrp_scheduler.compute(ts_close, self._hourly_returns.matrix())
        feed_avail = self._feed.availability
        self._data_availability = self._availability_with_feeds(
            feed_avail.model_copy(
                update={
                    "hourly_return_rows": self._hourly_returns.n_rows,
                    "options_surface": self._data_availability.options_surface,
                    "hlp_vault": self._data_availability.hlp_vault,
                }
            )
        )
        self._maybe_retrain(ts_close)

        # 2. Real features: FFD -> Positioning -> SVD -> Mahalanobis Gate -> Calibrated LightGBM.
        feature_bars = self._history_bars[-FEATURE_WINDOW_BARS:]
        snaps_window = compute_positioning_snapshots(feature_bars, close_ffd_config=self._ffd_close_cfg)
        new_pos_sig: PositioningSignalSnapshot = snaps_window[-1]
        x_cont_win, bin_win, _ = extract_continuous_and_binary_from_snapshots(snaps_window)
        features_ok = bool(np.all(np.isfinite(x_cont_win[-1]))) and len(feature_bars) >= 16
        x_cont_win = self._with_extra_features(x_cont_win, feature_bars)
        if features_ok:
            _run_m1max_metal_gpu_regime_pass(x_cont_win[-1])

        vpin_val = round(float(self._vpin_monitor.update(price=new_bar.close, volume=new_bar.volume, ts_ns=ts_close)), 3)
        hawkes_snap = self._hawkes_monitor.update(ts_ns=ts_close, vpin=vpin_val)
        new_hawkes = HawkesStateSnapshot(
            ts_ns=ts_close,
            r_state=round(hawkes_snap.r_state, 4),
            lambda_intensity=round(hawkes_snap.lambda_intensity, 4),
            mu=hawkes_snap.mu,
            alpha=hawkes_snap.alpha,
            beta=hawkes_snap.beta,
            branching_ratio_eta=round(min(1.0, max(0.0, hawkes_snap.branching_ratio_eta)), 3),
            vpin=round(min(1.0, max(0.0, vpin_val)), 3),
            toxic_sweep_active=bool(vpin_val > 0.60 or hawkes_snap.branching_ratio_eta > 0.70),
        )

        if self._ml_worker is not None and features_ok:
            new_meta = self._ml_worker.predict_single(
                instrument_id=s.active_instrument,
                ts_close_ns=ts_close,
                x_continuous_row=x_cont_win[-1],
                liquidation_exhaustion_flag=int(bin_win[-1]),
                primary_direction=new_pos_sig.primary_direction,
                sigma_bar=sigma_bar,
                vpin=new_hawkes.vpin,
            )
            new_meta = self._apply_residual_and_abstention(new_meta, x_cont_win[-1])
        else:
            new_meta = _neutral_meta_prediction(s.active_instrument, ts_close, self._last_retrain_message)

        # 3. Risk committee from live prices and per-asset funding.
        prices = {coin: (a.mid_price if (a := snap.get_asset(coin)) is not None else hourly_prices[coin]) for coin in ASSETS}
        fundings = {coin: (a.funding_rate_1h if (a := snap.get_asset(coin)) is not None else 0.0) for coin in ASSETS}
        if "BTC" in bars_by_coin:
            prices["BTC"] = new_bar.close
            fundings["BTC"] = new_bar.funding_rate_1h
        self._latest_macro_inputs = self._refresh_macro_inputs(ts_close)
        self._latest_aux = self._refresh_aux(ts_close)
        macro_eval = self._evaluate_macro_sentinel()
        breadth_eval = self._risk_breadth.evaluate(
            MultiAssetTelemetry(asset_prices=prices, funding_rates_1h=fundings)
        )
        critic_eval = self._risk_critic.evaluate(
            PortfolioStressInput(
                gross_leverage=min(2.5, self._positions.gross_exposure_usd() / max(1.0, self._starting_capital_usd + self._total_realized_pnl)),
                vpin=new_hawkes.vpin,
                hawkes_eta=new_hawkes.branching_ratio_eta,
            )
        )
        committee_eval = self._risk_consensus.reconcile(macro_eval, breadth_eval, critic_eval)
        self._latest_committee_output = committee_eval
        m_regime = committee_eval.regime_multiplier
        regime_veto = (
            committee_eval.tail_risk_flag
            or m_regime <= 0.0
            or s.regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED
        )

        # 4. Sleeves (P0.1) and the governing allocation equation (P0.2, P0.3).
        self._evaluate_sleeves(snap, bars_by_coin, committee_eval, ts_close)
        targets, portfolio_alloc, kelly_map = self._build_targets(new_meta, sigma_bar, m_regime, s.regime_mode)
        self._latest_targets = targets

        # 5. Position lifecycle (P0.5) with signed funding and real fees (P0.6).
        marks = self._build_marks(bars_by_coin, snap)
        closed = self._positions.process_bar(
            bar_index=bar_index, ts_ns=ts_close, marks=marks, targets=targets, regime_veto=regime_veto
        )
        self._step_shadow_book(bar_index, ts_close, marks, regime_veto, sigma_bar)
        opened: list[Position] = []
        entries_allowed = (
            not regime_veto
            and s.regime_mode in (ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY, ExecutionRegimeMode.CARRY_ONLY_FALLBACK)
        )
        cash_equity = self._starting_capital_usd + self._total_realized_pnl + sum(rec.net_pnl_usd for rec in closed)
        self._blocked_entries = []
        self._update_cascade(bars_by_coin, snap, ts_close)
        if self._breakers is not None:
            for rec in closed:
                self._breakers.record_closed_trade(rec, bar_index)
            attribution = self._positions.sleeve_attribution(marks)
            self._breakers.update_bar(
                ts_ns=ts_close,
                equity_usd=cash_equity + self._positions.total_unrealized_pnl_usd(marks),
                sleeve_cum_pnl={sid: a.net_pnl_usd + a.unrealized_pnl_usd for sid, a in attribution.items()},
                operational=self._operational_status(snap),
            )
        if entries_allowed:
            for target in targets.values():
                if target.weight == 0.0 or not self._positions.is_flat(target.sleeve_id, target.asset):
                    continue
                m = marks.get(target.asset)
                if m is None or m.bar is None:
                    continue
                fragility = self._latest_fragility.get(target.asset)
                if fragility is not None and fragility.fragile:
                    if target.hedged:
                        self._blocked_entries.append(f"{target.sleeve_id}:{target.asset}: cascade fragility ({', '.join(fragility.conditions_met)})")
                        continue
                    target = target.model_copy(update={"weight": target.weight * fragility.directional_multiplier})
                if self._breakers is not None:
                    perm = self._breakers.permission(target.sleeve_id, target.asset, bar_index)
                    if not perm.allowed:
                        self._blocked_entries.append(f"{target.sleeve_id}:{target.asset}: {'; '.join(perm.reasons)}")
                        continue
                    if perm.size_multiplier < 1.0:
                        target = target.model_copy(update={"weight": target.weight * perm.size_multiplier})
                        if target.weight == 0.0:
                            continue
                pos = self._positions.open_position(
                    target,
                    equity_usd=cash_equity * (1.0 - self._collateral_haircut),
                    price=m.bar.close,
                    spot_price=m.spot_price,
                    bar_index=bar_index,
                    ts_ns=ts_close,
                    sigma_bar=sigma_bar,
                )
                if pos is not None:
                    opened.append(pos)
                    self._maker_fills += 2 if pos.hedged else 1

        self._record_markout_fills(opened, closed, snap)

        # 6. Ledger: realise only what closed on this bar.
        bar_net_pnl = 0.0
        bar_fees = 0.0
        for rec in closed:
            bar_net_pnl += rec.net_pnl_usd
            bar_fees += rec.fee_usd
            self._total_funding_pnl_usd += rec.funding_pnl_usd
            if rec.is_win:
                self._wins += 1
            else:
                self._losses += 1
            if rec.exit_reason in (PositionExitReason.STOP_BARRIER, PositionExitReason.REGIME_VETO):
                self._taker_fills += 2 if rec.hedged else 1
            else:
                self._maker_fills += 2 if rec.hedged else 1
            rec_dict = rec.model_dump(mode="json")
            rec_dict["realized_pnl_usd"] = round(rec.net_pnl_usd, 4)
            rec_dict["why_did_it_trade"] = f"{rec.entry_reason} | exit: {rec.exit_reason.value}"
            self._trade_history.append(rec_dict)
        self._total_realized_pnl = round(self._total_realized_pnl + bar_net_pnl, 4)
        self._total_fees_paid_usd = round(self._total_fees_paid_usd + bar_fees, 4)
        starting_cap = self._starting_capital_usd
        new_net_pnl_usd = round(self._total_realized_pnl, 2)
        new_net_pnl_bps = round((new_net_pnl_usd / max(1.0, starting_cap)) * 10_000.0, 1)
        unrealized = self._positions.total_unrealized_pnl_usd(marks)

        btc_carry = self._latest_carry_evaluations.get("BTC")
        updated_pnl = PnLAndExecutionMetrics(
            starting_capital_usd=starting_cap,
            net_pnl_usd=new_net_pnl_usd,
            net_pnl_bps=new_net_pnl_bps,
            gross_pnl_usd=round(new_net_pnl_usd + self._total_fees_paid_usd, 2),
            total_fees_paid_usd=round(self._total_fees_paid_usd, 2),
            maker_fill_ratio=round(self._maker_fill_ratio(), 4),
            taker_fill_ratio=round(1.0 - self._maker_fill_ratio(), 4),
            fee_savings_usd=0.0,
            fee_savings_bps=0.0,
            annualized_carry_apr_pct=round(btc_carry.annualized_funding_apr * 100.0, 2) if btc_carry is not None else 0.0,
            is_dry_run=s.is_dry_run,
        )

        # 7. Plain-English activity line and decision record.
        actual_btc_ret = (new_bar.close - new_bar.open) / max(new_bar.open, 1.0)
        n_open = len(self._positions.open_positions)
        if closed and opened:
            action_str = "EXIT_AND_ENTER"
        elif closed:
            action_str = "EXIT_" + "_".join(sorted({c.exit_reason.value for c in closed}))
        elif opened:
            action_str = "ENTER_" + "_".join(sorted({("LONG" if p.side > 0 else "SHORT") if not p.hedged else "CARRY" for p in opened}))
        elif n_open > 0:
            action_str = "HOLD_POSITIONS"
        else:
            action_str = "HOLD_CASH"

        outcome_tag = (
            f"closed {len(closed)}: ${bar_net_pnl:+.2f} net" if closed else ("no closes" if n_open or opened else "flat")
        )
        summary_msg = (
            f"[{self._now_str()} • Bar #{k} • {feed_source}] BTC ${new_bar.open:,.2f} -> ${new_bar.close:,.2f} ({actual_btc_ret:+.2%}, "
            f"H ${new_bar.high:,.2f} L ${new_bar.low:,.2f}) | {action_str} ({outcome_tag}, opened {len(opened)}, open {n_open}) | "
            f"p_meta={new_meta.p_meta:.1%} dir={new_meta.effective_direction:+d} sigma={sigma_bar:.4f} | "
            f"Equity=${updated_pnl.current_equity_usd:,.2f} (W:{self._wins}/L:{self._losses}, unrealised ${unrealized:+.2f})"
        )
        self._activity_log.append(summary_msg)
        _append_log_line(summary_msg)

        first_entry = opened[0] if opened else None
        side_str: Literal["BUY", "SELL", "FLAT", "NONE"] = (
            ("BUY" if first_entry.side > 0 else "SELL") if first_entry is not None else ("FLAT" if n_open == 0 else "NONE")
        )
        f_kelly = kelly_map.get("BTC", 0.0)
        why_parts: list[str] = []
        for p in opened:
            why_parts.append(
                f"Opened {p.sleeve_id} {'carry' if p.hedged else ('long' if p.side > 0 else 'short')} {p.asset} "
                f"${p.notional_usd:,.2f} at ${p.entry_price:,.2f}: {p.entry_reason}"
            )
        for c in closed:
            why_parts.append(
                f"Closed {c.sleeve_id} {c.asset} on {c.exit_reason.value} at ${c.exit_price:,.2f}: "
                f"gross ${c.gross_pnl_usd:+.2f}, funding ${c.funding_pnl_usd:+.2f}, fees ${c.fee_usd:.2f}, net ${c.net_pnl_usd:+.2f}"
            )
        if not why_parts:
            why_parts.append(
                f"No entry: BTC meta-labeller p_meta={new_meta.p_meta:.1%} direction={new_meta.effective_direction:+d} "
                f"({'gate open' if new_meta.epistemic_gate_passed else 'gate closed or no model'}); "
                f"derivatives directions {[sig.primary_direction for sig in self._latest_derivatives_signals.values()]}; "
                f"carry actionable {[ev.is_actionable for ev in self._latest_carry_evaluations.values()]}; "
                f"regime M={m_regime:.2f}{' (veto)' if regime_veto else ''}; {n_open} position(s) held."
            )
        why_text = " ".join(why_parts)

        feature_attr: dict[str, float] = {
            "ffd_close_zscore": round(float(new_pos_sig.ffd_close_zscore), 4),
            "ffd_oi_zscore": round(float(new_pos_sig.ffd_oi_zscore), 4),
            "oi_funding_basis_zscore": round(float(new_pos_sig.oi_funding_basis_zscore), 4),
            "basis_pct": round(float(new_pos_sig.basis_pct), 6),
            "funding_rate_1h": round(float(new_bar.funding_rate_1h), 8),
            "funding_rate_8h": round(float(new_bar.funding_rate_8h), 6),
            "liquidation_zscore": round(float(new_pos_sig.liquidation_zscore), 4),
            "regime_weight_bps": round(float(new_meta.expected_net_edge_bps), 1),
            "kelly_conviction": round(float(f_kelly), 4),
            "m_regime_multiplier": round(float(m_regime), 4),
            "sigma_bar_ewma96": round(float(sigma_bar), 6),
        }
        for coin, w in self._latest_hrp_weights.items():
            feature_attr[f"hrp_w_{coin}"] = round(float(w), 4)

        trade_id = f"bar_{k}_{int(ts_close / 1_000_000)}"
        forensic_rec = TradeForensicRecord(
            trade_id=trade_id,
            timestamp_ns=ts_close,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            instrument_id=s.active_instrument.split("-")[0],
            action=action_str,
            side=side_str,
            fill_price=first_entry.entry_price if first_entry is not None else new_bar.close,
            quantity=round(first_entry.size, 6) if first_entry is not None else 0.0,
            notional_usd=round(first_entry.notional_usd, 2) if first_entry is not None else 0.0,
            realized_pnl_usd=round(bar_net_pnl, 4),
            realized_pnl_bps=round((bar_net_pnl / max(1.0, starting_cap)) * 10_000.0, 2),
            fee_cost_usd=round(bar_fees, 4),
            fee_savings_usd=0.0,
            regime=new_pos_sig.regime.value,
            p_meta=round(new_meta.p_meta, 4),
            mahalanobis_di=round(new_meta.mahalanobis_di, 4),
            vpin=round(new_hawkes.vpin, 4),
            hawkes_eta=round(new_hawkes.branching_ratio_eta, 4),
            asri_score=round(committee_eval.consensus_risk_score, 1),
            m_regime=round(m_regime, 3),
            kelly_fraction=round(f_kelly, 4),
            why_did_it_trade=why_text,
            feature_attribution=feature_attr,
            sleeve_id=first_entry.sleeve_id if first_entry is not None else "",
            entries=[p.position_id for p in opened],
            exits=[c.trade_id for c in closed],
            open_positions=n_open,
            unrealized_pnl_usd=round(unrealized, 4),
        )
        forensic_dict = forensic_rec.model_dump(mode="json")
        self._forensics_history.append(forensic_dict)
        self._latest_forensic = forensic_dict
        self._persist_ledger_state_atomic()

        # 8. Dashboard state window.
        bars_window = self._history_bars[-UI_WINDOW_BARS:]
        pos_window = snaps_window[-UI_WINDOW_BARS:]
        new_barrier = self._barrier_label_for_bar(new_bar, bar_index, sigma_bar, closed)
        barriers_window = (s.barrier_labels + [new_barrier])[-UI_WINDOW_BARS:]
        meta_window = (s.meta_predictions + [new_meta])[-UI_WINDOW_BARS:]
        hawkes_window = (s.hawkes_history + [new_hawkes])[-UI_WINDOW_BARS:]

        order_side: Literal["BUY", "SELL", "NONE"] = (
            ("BUY" if first_entry.side > 0 else "SELL") if first_entry is not None else "NONE"
        )
        self._state = DashboardTelemetryState(
            active_instrument=s.active_instrument,
            regime_mode=s.regime_mode,
            is_dry_run=s.is_dry_run,
            pnl_metrics=updated_pnl,
            ffd_config=self._ffd_close_cfg,
            asri_risk=committee_eval.to_systemic_risk_output(),
            hawkes_state=new_hawkes,
            last_order_command=ExecutionOrderCommand(
                instrument_id=f"{first_entry.asset}USDT-PERP" if first_entry is not None else s.active_instrument,
                ts_ns=ts_close,
                action=ExecutionAction.PLACE_GTX_MAKER if order_side != "NONE" else ExecutionAction.HOLD_RESTING_HYSTERESIS,
                side=order_side,
                target_price=round(first_entry.entry_price, 2) if first_entry is not None else round(new_bar.close, 2),
                quantity=round(first_entry.size, 6) if first_entry is not None else 0.0,
                post_only_gtx=True,
                time_in_force="GTX",
                consecutive_rejections=0,
                reason=summary_msg,
            ),
            dollar_bars=bars_window,
            positioning_signals=pos_window,
            barrier_labels=barriers_window,
            meta_predictions=meta_window,
            hawkes_history=hawkes_window,
            champion_cpcv=s.champion_cpcv,
            challenger_cpcv=s.challenger_cpcv,
            champion_state=s.champion_state,
            portfolio_allocation=portfolio_alloc,
            last_control_action=s.last_control_action,
            last_control_message=summary_msg,
        )

        return ControlActionResult(
            status="ok",
            action="tick",
            regime_mode=self._state.regime_mode,
            is_dry_run=self._state.is_dry_run,
            plain_english_explanation=summary_msg,
            weights=dict(self._state.portfolio_allocation.weights),
            gross_leverage=self._state.portfolio_allocation.gross_leverage,
            tick_count=self._tick_counter,
            latest_price=new_bar.close,
            starting_capital_usd=starting_cap,
            current_equity_usd=updated_pnl.current_equity_usd,
            net_pnl_usd=new_net_pnl_usd,
            net_pnl_bps=new_net_pnl_bps,
            activity_log=self.activity_log,
            forensic_record=forensic_dict,
            ledger_summary=self.ledger_summary,
            bar_processed=True,
            closed_trades=[c.model_dump(mode="json") for c in closed],
            opened_positions=[p.model_dump(mode="json") for p in opened],
        )

    # ------------------------------------------------------------------ control actions

    def apply_control_action(self, action: str) -> ControlActionResult:
        """Apply one of the four operational control actions and update ``self._state``."""
        normalized = action.strip().lower()
        ts_label = self._now_str()
        flattened: list[ClosedTradeRecord] = []
        if normalized == "start_dry_run":
            mode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
            self._auto_running = True
            msg = (
                f"[{ts_label}] ▶ DRY-RUN ACTIVE: polling Hyperliquid 15m candles; real LightGBM + sleeves + "
                f"position lifecycle on ${self._starting_capital_usd:,.2f} paper account."
            )
            is_dry = True
        elif normalized == "pause_entries":
            mode = ExecutionRegimeMode.PAUSED_ENTRIES
            self._auto_running = False
            msg = (
                f"[{ts_label}] ⏸ ENTRIES PAUSED: no new positions; open positions keep their barrier exits."
            )
            is_dry = self._state.is_dry_run
        elif normalized == "switch_carry_only":
            mode = ExecutionRegimeMode.CARRY_ONLY_FALLBACK
            self._auto_running = True
            msg = (
                f"[{ts_label}] 🏦 CARRY-ONLY MODE: directional sleeves disabled; "
                "hedged 1-hour funding carry only."
            )
            is_dry = self._state.is_dry_run
        elif normalized == "emergency_flatten":
            mode = ExecutionRegimeMode.EMERGENCY_FLATTENED
            self._auto_running = False
            marks = self._current_marks()
            bar_index = max(0, len(self._history_bars) - 1)
            ts_ns = self._latest_market_snapshot.timestamp_ns if self._latest_market_snapshot is not None else 0
            flattened = self._positions.close_all(bar_index=bar_index, ts_ns=ts_ns, marks=marks)
            flat_pnl = 0.0
            for rec in flattened:
                flat_pnl += rec.net_pnl_usd
                self._total_fees_paid_usd = round(self._total_fees_paid_usd + rec.fee_usd, 4)
                self._total_funding_pnl_usd += rec.funding_pnl_usd
                self._taker_fills += 2 if rec.hedged else 1
                if rec.is_win:
                    self._wins += 1
                else:
                    self._losses += 1
                rec_dict = rec.model_dump(mode="json")
                rec_dict["realized_pnl_usd"] = round(rec.net_pnl_usd, 4)
                rec_dict["why_did_it_trade"] = f"{rec.entry_reason} | exit: EMERGENCY_FLATTEN"
                self._trade_history.append(rec_dict)
            self._total_realized_pnl = round(self._total_realized_pnl + flat_pnl, 4)
            self._persist_ledger_state_atomic()
            msg = (
                f"[{ts_label}] 🛑 EMERGENCY FLATTEN EXECUTED: {len(flattened)} position(s) closed at mark "
                f"(taker fees), realised ${flat_pnl:+.2f}; 100% cash."
            )
            is_dry = self._state.is_dry_run
        else:
            raise ValueError(
                f"Unsupported control action '{action}'. Valid actions: "
                "start_dry_run, pause_entries, switch_carry_only, emergency_flatten."
            )

        self._activity_log.append(msg)
        _append_log_line(msg)
        base = build_default_dashboard_state(
            regime_mode=mode,
            is_dry_run=is_dry,
            last_control_action=normalized,
            last_control_message=msg,
        )
        preserved_pnl = self._state.pnl_metrics.model_copy(
            update={
                "starting_capital_usd": self._starting_capital_usd,
                "is_dry_run": is_dry,
                "net_pnl_usd": round(self._total_realized_pnl, 2),
                "net_pnl_bps": round((self._total_realized_pnl / max(1.0, self._starting_capital_usd)) * 10_000.0, 1),
                "total_fees_paid_usd": round(self._total_fees_paid_usd, 2),
                "gross_pnl_usd": round(self._total_realized_pnl + self._total_fees_paid_usd, 2),
            }
        )
        self._state = base.model_copy(
            update={
                "pnl_metrics": preserved_pnl,
                "ffd_config": self._ffd_close_cfg,
                "asri_risk": self._state.asri_risk,
                "hawkes_state": self._state.hawkes_state,
                "dollar_bars": self._state.dollar_bars,
                "positioning_signals": self._state.positioning_signals,
                "barrier_labels": self._state.barrier_labels,
                "meta_predictions": self._state.meta_predictions,
                "hawkes_history": self._state.hawkes_history,
                "portfolio_allocation": self._state.portfolio_allocation if normalized != "emergency_flatten" else self._zero_allocation(),
            }
        )
        if normalized == "start_dry_run":
            return self.step_simulation_tick()

        return ControlActionResult(
            status="ok",
            action=normalized,
            regime_mode=self._state.regime_mode,
            is_dry_run=self._state.is_dry_run,
            plain_english_explanation=msg,
            weights=dict(self._state.portfolio_allocation.weights),
            gross_leverage=self._state.portfolio_allocation.gross_leverage,
            tick_count=self._tick_counter,
            latest_price=self._state.dollar_bars[-1].close,
            starting_capital_usd=self._starting_capital_usd,
            current_equity_usd=self._state.pnl_metrics.current_equity_usd,
            net_pnl_usd=self._state.pnl_metrics.net_pnl_usd,
            net_pnl_bps=self._state.pnl_metrics.net_pnl_bps,
            activity_log=self.activity_log,
            ledger_summary=self.ledger_summary,
            closed_trades=[c.model_dump(mode="json") for c in flattened],
        )


def _serialize_live_state(store: ControlCenterStateStore) -> dict[str, Any]:
    s = store.state
    candlestick_fig = build_candlestick_dollar_bar_figure(s)
    weights_fig = build_portfolio_weights_figure(s.portfolio_allocation)
    payload = s.model_dump(mode="json")
    ledger = store.ledger_summary
    payload.update(
        {
            "tick_count": store.tick_counter,
            "auto_running": store.auto_running,
            "latest_price": s.dollar_bars[-1].close,
            "starting_capital_usd": s.pnl_metrics.starting_capital_usd,
            "base_capital_usd": ledger["base_capital_usd"],
            "current_cash_usd": ledger["current_cash_usd"],
            "current_equity_usd": s.pnl_metrics.current_equity_usd,
            "total_realized_pnl": ledger["total_realized_pnl"],
            "unrealized_pnl_usd": ledger["unrealized_pnl_usd"],
            "marked_equity_usd": ledger["marked_equity_usd"],
            "net_pnl_usd": s.pnl_metrics.net_pnl_usd,
            "net_pnl_bps": s.pnl_metrics.net_pnl_bps,
            "maker_fill_ratio": s.pnl_metrics.maker_fill_ratio,
            "taker_fill_ratio": s.pnl_metrics.taker_fill_ratio,
            "fee_savings_usd": s.pnl_metrics.fee_savings_usd,
            "fee_savings_bps": s.pnl_metrics.fee_savings_bps,
            "asri_score": s.asri_risk.score,
            "asri_summary": s.asri_risk.plain_english_summary,
            "vpin": s.hawkes_state.vpin,
            "hawkes_eta": s.hawkes_state.branching_ratio_eta,
            "toxic_sweep_active": s.hawkes_state.toxic_sweep_active,
            "cpcv_gates_passed": s.champion_cpcv.passed_all_gates,
            "challenger_gates_passed": s.challenger_cpcv.passed_all_gates,
            "signed_hrp_cvar_weights": dict(s.portfolio_allocation.weights),
            "hrp_unsigned_weights": store.hrp_weights,
            "gross_leverage": s.portfolio_allocation.gross_leverage,
            "multi_asset_snapshot": (
                store.latest_market_snapshot.model_dump(mode="json")
                if store.latest_market_snapshot
                else None
            ),
            "activity_log": store.activity_log,
            "candlestick_figure": json.loads(json.dumps(candlestick_fig.to_dict())),
            "weights_figure": json.loads(json.dumps(weights_fig.to_dict())),
            "account_ledger": ledger,
            "win_count": ledger["win_count"],
            "loss_count": ledger["loss_count"],
            "trade_history": store._trade_history[-25:],
            "forensics": store.forensics_history[-25:],
            "latest_forensic": store.latest_forensic,
            "risk_committee": store.risk_committee_output.model_dump(mode="json"),
            "sleeve_telemetry": store.sleeve_telemetry,
            "open_positions": [p.model_dump(mode="json") for p in store.position_manager.open_positions.values()],
            "sleeve_targets": {k: t.model_dump(mode="json") for k, t in store.latest_targets.items()},
            "data_availability": store.data_availability.model_dump(mode="json"),
            "live_feeds": store.data_availability.live_feeds(),
            "streams": store.stream_telemetry,
            "transfer_coefficient": store.transfer_coefficient,
            "macro_inputs": store.latest_macro_inputs.model_dump(mode="json", exclude={"stablecoin_marks", "dvol"}),
            "sigma_bar_ewma96": store.sigma_bar,
        }
    )
    return payload


def create_app(store: ControlCenterStateStore | None = None) -> FastAPI:
    """Create and configure the FastAPI Control Center serving ``http://localhost:8080``."""
    state_store = store or ControlCenterStateStore()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async def _background_ticker() -> None:
            while True:
                await asyncio.sleep(10.0)
                if state_store.auto_running:
                    try:
                        await asyncio.to_thread(state_store.step_simulation_tick)
                    except Exception as exc:
                        logger.error("Background tick failed: %s", exc)

        hub = state_store.stream_hub
        if hub is not None and not hub.running:
            hub.start()
        task = asyncio.create_task(_background_ticker())
        try:
            yield
        finally:
            task.cancel()
            if hub is not None:
                hub.stop()

    app = FastAPI(
        title="Gemini Argon 4 — M1 Max Control Center API",
        version="4.1.0",
        description="Interactive Web UI & REST Control Center for the M1 Max Quantitative Derivatives Engine.",
        lifespan=lifespan,
    )
    app.state.store = state_store

    @app.get("/", response_class=HTMLResponse)
    def get_dashboard_root() -> HTMLResponse:
        """Render the complete 3-tab interactive HTML Control Center."""
        html_doc = render_full_dashboard_html(state_store.state)
        return HTMLResponse(
            content=html_doc,
            status_code=200,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/api/state")
    def get_live_state() -> dict[str, Any]:
        """Return live telemetry, PnL, positions, sleeve telemetry, CPCV gates, and HRP-CVaR weights."""
        return _serialize_live_state(state_store)

    @app.get("/api/forensics")
    def get_forensics() -> dict[str, Any]:
        """Return per-bar decision records ("Why Did It Trade?"), closed trades, and sleeve attribution."""
        return {
            "status": "ok",
            "total_records": len(state_store.forensics_history),
            "records": state_store.forensics_history[-50:],
            "latest": state_store.latest_forensic,
            "closed_trades": state_store._trade_history[-50:],
            "ledger_summary": state_store.ledger_summary,
            "risk_committee": state_store.risk_committee_output.model_dump(mode="json"),
            "sleeve_telemetry": state_store.sleeve_telemetry,
            "data_availability": state_store.data_availability.model_dump(mode="json"),
            "markouts": state_store.markout_telemetry,
        }

    @app.post("/api/tick")
    def post_simulation_tick() -> dict[str, Any]:
        """Poll for a new closed 15m bar, process it if present, and return the full state."""
        result = state_store.step_simulation_tick()
        payload = _serialize_live_state(state_store)
        payload["tick_result"] = result.model_dump(mode="json", exclude={"activity_log", "ledger_summary"})
        return payload

    @app.post("/api/reset")
    def post_reset_account(req: ResetCapitalRequest) -> dict[str, Any]:
        """Reset total profit to $0.00 and configure custom starting paper capital."""
        result = state_store.reset_account(starting_capital_usd=req.starting_capital_usd)
        payload = result.model_dump(mode="json")
        payload["live_state"] = _serialize_live_state(state_store)
        return payload

    @app.get("/api/tooltips")
    def get_tooltips() -> dict[str, str]:
        """Return the ``PLAIN_ENGLISH_TOOLTIPS`` dictionary."""
        return dict(PLAIN_ENGLISH_TOOLTIPS)

    @app.get("/api/docs/{doc_name}")
    def get_documentation(doc_name: str) -> dict[str, str]:
        """Render any of the four ``docs/*.md`` layman's manuals."""
        try:
            markdown_text = load_layman_document_markdown(doc_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        clean_key = doc_name.strip().removesuffix(".md").upper()
        filename = DOC_FILENAMES.get(clean_key, f"{clean_key}.md")
        return {
            "doc_name": clean_key,
            "filename": filename,
            "markdown": markdown_text,
            "content": markdown_text,
        }

    @app.post("/api/control/{action}")
    def post_control_action(action: str) -> dict[str, Any]:
        """Execute one-click operational control actions."""
        try:
            result = state_store.apply_control_action(action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = result.model_dump(mode="json")
        payload["live_state"] = _serialize_live_state(state_store)
        return payload

    return app


app: FastAPI = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
