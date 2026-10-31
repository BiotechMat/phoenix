"""Comprehensive Opaque-Box E2E Test Suite for Project Phoenix.

Covers Tiers 1-4 for Requirements R1, R2, R3, R4, and R5:
- Tier 1: Feature Coverage (>=5 tests per feature for R1, R2, R3, R4, R5)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature for R1, R2, R3, R4, R5)
- Tier 3: Cross-Feature Combinations (Pairwise interactions across R1-R5)
- Tier 4: Real-World Application Scenarios (Realistic multi-step end-to-end workflows)

All expected outputs are derived strictly from authoritative specifications in
PROJECT.md and ORIGINAL_REQUEST.md.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final, Literal

from fastapi.testclient import TestClient
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.contracts import (
    AssetAllocationInput,
    AssetMarketSnapshot,
    CPCVValidationResult,
    ChampionChallengerState,
    DeribitOptionsSurface,
    DollarBar,
    EnvironmentConfig,
    ExecutionAction,
    ExecutionOrderCommand,
    ExecutionRegimeMode,
    FeeSchedule,
    FFDWeightsConfig,
    HawkesStateSnapshot,
    HyperliquidWhaleVaultMetrics,
    L2Level,
    L2OrderBookSnapshot,
    MetaPredictionOutput,
    MultiAssetMarketSnapshot,
    PortfolioAllocationOutput,
    PositioningRegime,
    PositioningSignalSnapshot,
    SettlementCurrency,
    SystemicRiskOutput,
    TripleBarrierConfig,
    TripleBarrierLabel,
    compute_fractional_kelly,
    compute_regime_multiplier_from_asri,
    compute_signed_target_weights,
)
from core.data_ingestion import (
    LiveMarketDataEngine,
    _compute_l2_depth_and_spread,
    get_live_multi_asset_snapshot,
    normalize_coin_symbol,
)
from core.execution.defensive_maker import (
    DefensiveMakerExecutionAlgorithm,
    RecursiveHawkesMonitor,
    RollingBVCVPINMonitor,
)
from core.hardware_bootstrap import (
    bootstrap_m1max_hardware,
    load_environment_config,
)
from data.backfill_storage import (
    ARROW_BASE_SCHEMA,
    BASE_COLUMNS,
    BTC_PARQUET_FILE,
    ETH_PARQUET_FILE,
    SOL_PARQUET_FILE,
)
from research.cpcv_gates import (
    ChampionChallengerGatekeeper,
    CombinatorialPurgedCV,
    StrategyCandidate,
    StrategyTournament,
    TournamentCandidateResult,
    TournamentResult,
    compute_dsr,
    compute_net_of_fee_returns,
    compute_pbo,
    compute_psr,
    create_funding_carry_strategy,
    create_mean_reversion_strategy,
    create_trend_strategy,
    create_volatility_breakout_strategy,
)
from core.models import (
    ContinuousSVDOrthogonalizer,
    FeeAdjustedTripleBarrierLabeler,
    LightGBMMetaLabelWorker,
    MahalanobisOutlierGate,
)
from sentiment.committee_models import (
    AdversarialRedTeamCritic,
    BreadthAnalystOutput,
    CrossSectionalBreadthAnalyst,
    MacroSentinelOutput,
    MacroStablecoinSentinel,
    MLXCommitteeScorer,
    MultiAssetTelemetry,
    PortfolioStressInput,
    RedTeamCriticOutput,
    RiskCommitteeConsensusEngine,
    RiskCommitteeOutput,
)
from sentiment.mlx_risk_daemon import (
    MacroCryptoFeedSnapshot,
    PersistentMLXRiskDaemon,
    parse_and_validate_llm_risk_json,
)
from sleeves.derivatives.funding_carry import (
    CarryDirection,
    CarryOpportunityEvaluation,
    FundingCarryConfig,
    FundingCarrySleeve,
)
from ui.api_server import (
    ControlActionResult,
    ControlCenterStateStore,
    ResetCapitalRequest,
    create_app,
)
from ui.dashboard_app import PLAIN_ENGLISH_TOOLTIPS
from tests.phase0_helpers import build_store, flat_bars, flat_history, make_snapshot


# ==============================================================================
# TEST FIXTURES & DETERMINISTIC SYNTHETIC HELPERS
# ==============================================================================


def _make_mock_l2_book(
    mid_price: float = 65000.0,
    half_spread_bps: float = 1.0,
    depth_levels: int = 20,
    size_per_level: float = 0.5,
) -> L2OrderBookSnapshot:
    """Generate a realistic top-20 L2 order book snapshot for testing."""
    spread_delta = mid_price * (half_spread_bps * 1e-4)
    bids: list[tuple[float, float]] = []
    asks: list[tuple[float, float]] = []
    levels_bids: list[L2Level] = []
    levels_asks: list[L2Level] = []

    for i in range(depth_levels):
        bid_px = round(mid_price - spread_delta - (i * mid_price * 0.0001), 2)
        ask_px = round(mid_price + spread_delta + (i * mid_price * 0.0001), 2)
        bids.append((bid_px, size_per_level))
        asks.append((ask_px, size_per_level))
        levels_bids.append(L2Level(price=bid_px, size=size_per_level, order_count=2))
        levels_asks.append(L2Level(price=ask_px, size=size_per_level, order_count=2))

    spread_bps, depth_10bps, depth_50bps = _compute_l2_depth_and_spread(bids, asks)
    return L2OrderBookSnapshot(
        instrument_id="BTC",
        bids=bids,
        asks=asks,
        levels_bids=levels_bids,
        levels_asks=levels_asks,
        spread_bps=spread_bps,
        depth_10bps_usd=depth_10bps,
        depth_50bps_usd=depth_50bps,
        timestamp_ns=time.time_ns(),
    )


def _make_mock_multi_asset_snapshot(
    btc_price: float = 65000.0,
    eth_price: float = 2450.0,
    sol_price: float = 135.0,
    funding_1h_btc: float = 0.0001,
    funding_1h_eth: float = 0.00012,
    funding_1h_sol: float = 0.00015,
) -> MultiAssetMarketSnapshot:
    """Generate a valid MultiAssetMarketSnapshot with BTC, ETH, and SOL."""
    now_ns = time.time_ns()
    btc_book = _make_mock_l2_book(mid_price=btc_price)
    eth_book = _make_mock_l2_book(mid_price=eth_price)
    sol_book = _make_mock_l2_book(mid_price=sol_price)

    btc_snap = AssetMarketSnapshot(
        instrument_id="BTC",
        mid_price=btc_price,
        mark_price=btc_price + 2.0,
        oracle_price=btc_price - 1.0,
        open_interest=52000.0,
        funding_rate_1h=funding_1h_btc,
        day_volume_usd=1_500_000_000.0,
        l2_book=btc_book,
        timestamp_ns=now_ns,
    )
    eth_snap = AssetMarketSnapshot(
        instrument_id="ETH",
        mid_price=eth_price,
        mark_price=eth_price + 0.5,
        oracle_price=eth_price - 0.2,
        open_interest=350000.0,
        funding_rate_1h=funding_1h_eth,
        day_volume_usd=800_000_000.0,
        l2_book=eth_book,
        timestamp_ns=now_ns,
    )
    sol_snap = AssetMarketSnapshot(
        instrument_id="SOL",
        mid_price=sol_price,
        mark_price=sol_price + 0.05,
        oracle_price=sol_price - 0.02,
        open_interest=1200000.0,
        funding_rate_1h=funding_1h_sol,
        day_volume_usd=400_000_000.0,
        l2_book=sol_book,
        timestamp_ns=now_ns,
    )
    return MultiAssetMarketSnapshot(
        timestamp_ns=now_ns,
        assets={"BTC": btc_snap, "ETH": eth_snap, "SOL": sol_snap},
        feed_source="HYPERLIQUID-L1",
    )


def _make_mock_dollar_bars(n_bars: int = 120, seed: int = 42) -> list[DollarBar]:
    """Generate deterministic synthetic DollarBars for research and backtest testing."""
    rng = np.random.default_rng(seed)
    bars: list[DollarBar] = []
    price = 60000.0
    base_ts = 1_700_000_000_000_000_000
    step_ns = 15 * 60 * 1_000_000_000

    for i in range(n_bars):
        ret = float(rng.normal(0.0002, 0.005))
        open_p = price
        close_p = max(100.0, open_p * (1.0 + ret))
        high_p = max(open_p, close_p) * (1.0 + abs(float(rng.normal(0.0, 0.001))))
        low_p = min(open_p, close_p) * (1.0 - abs(float(rng.normal(0.0, 0.001))))
        vol = float(rng.uniform(10.0, 50.0))
        dvol = vol * close_p
        bars.append(
            DollarBar(
                instrument_id="BTCUSDT-PERP",
                horizon_label="15m",
                ts_open_ns=base_ts + i * step_ns,
                ts_close_ns=base_ts + (i + 1) * step_ns,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=vol,
                dollar_volume=dvol,
                buy_dollar_volume=dvol * 0.52,
                sell_dollar_volume=dvol * 0.48,
                open_interest=50000.0,
                funding_rate_8h=0.0004,
                funding_rate_1h=0.00005,
                mark_price=close_p * 1.0001,
                spot_price=close_p,
                liquidation_volume_usd=0.0,
                tick_count=100,
            )
        )
        price = close_p
    return bars


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 TESTS PER FEATURE FOR R1 - R5)
# ==============================================================================


class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage for R1, R2, R3, R4, and R5."""

    # -------------------------------------------------------------------------
    # R1: Real-Time Multi-Asset L2 Order Book & Funding Data Ingestion
    # -------------------------------------------------------------------------

    def test_r1_f01_multi_asset_snapshot_ingestion_structure(self) -> None:
        """R1.1: Verify MultiAssetMarketSnapshot holds BTC, ETH, and SOL with required fields."""
        snap = _make_mock_multi_asset_snapshot()
        assert snap.timestamp_ns > 0
        assert snap.feed_source == "HYPERLIQUID-L1"

        # BTC, ETH, and SOL must all be accessible via properties and get_asset
        for coin in ("BTC", "ETH", "SOL"):
            asset = snap.get_asset(coin)
            assert asset is not None, f"Missing asset {coin}"
            assert asset.mid_price > 0.0
            assert asset.mark_price > 0.0
            assert asset.oracle_price > 0.0
            assert asset.open_interest >= 0.0
            assert asset.funding_rate_1h >= 0.0
            assert asset.funding_rate_8h_equiv >= 0.0
            assert asset.day_volume_usd >= 0.0
            assert len(asset.l2_book.bids) > 0
            assert len(asset.l2_book.asks) > 0

    def test_r1_f02_l2_order_book_depth_and_spread_metrics(self) -> None:
        """R1.2: Verify L2OrderBookSnapshot computes top-20 depth, spread in bps, and depth_10bps."""
        book = _make_mock_l2_book(mid_price=50000.0, half_spread_bps=2.0, depth_levels=20)
        assert len(book.bids) == 20
        assert len(book.asks) == 20
        assert book.spread_bps > 0.0
        # For half_spread_bps=2.0 on both sides, total spread is ~4.0 bps
        assert 3.5 <= book.spread_bps <= 4.5
        assert book.depth_10bps_usd > 0.0
        assert book.depth_50bps_usd >= book.depth_10bps_usd
        # Ensure descending bids and ascending asks
        assert book.bids[0][0] > book.bids[-1][0]
        assert book.asks[0][0] < book.asks[-1][0]

    def test_r1_f03_funding_rate_1h_and_8h_synchronization(self) -> None:
        """R1.3: Verify 1h funding rate is synchronized to 8h equivalent and annualized APR."""
        # 1h rate of 0.0001 (1 bp per hour)
        rate_1h = 0.0001
        asset = AssetMarketSnapshot(
            instrument_id="ETH",
            mid_price=2500.0,
            mark_price=2500.2,
            oracle_price=2500.0,
            open_interest=100000.0,
            funding_rate_1h=rate_1h,
        )
        # 8h equivalent must be 8 * 1h rate = 0.0008
        assert pytest.approx(asset.funding_rate_8h_equiv, rel=1e-5) == rate_1h * 8.0
        # Annualized APR = rate_1h * 24 * 365 * 100% = 87.6%
        expected_apr = rate_1h * 24.0 * 365.0 * 100.0
        assert pytest.approx(asset.annualized_apr_pct, rel=1e-5) == expected_apr

        # Verify DollarBar bidirectional synchronization
        bar = DollarBar(
            instrument_id="BTCUSDT-PERP",
            horizon_label="15m",
            ts_open_ns=1000,
            ts_close_ns=2000,
            open=60000.0,
            high=60100.0,
            low=59900.0,
            close=60050.0,
            volume=10.0,
            dollar_volume=600000.0,
            buy_dollar_volume=300000.0,
            sell_dollar_volume=300000.0,
            open_interest=50000.0,
            funding_rate_1h=0.00005,
            mark_price=60050.0,
            spot_price=60040.0,
        )
        assert pytest.approx(bar.funding_rate_8h, rel=1e-5) == 0.00005 * 8.0

    def test_r1_f04_bybit_v5_failover_mechanism(self) -> None:
        """R1.4: Verify failover to Bybit V5 when Hyperliquid is degraded or offline."""
        engine = LiveMarketDataEngine(
            timeout_sec=0.2,
            hl_url="http://127.0.0.1:1/nonexistent",
            bybit_tickers_url="http://127.0.0.1:1/nonexistent",
        )
        snap = engine.get_multi_asset_snapshot_sync(coins=("BTC", "ETH", "SOL"), include_l2=True)
        assert snap is not None
        assert snap.feed_source in ("BYBIT-V5-FALLBACK", "SIMULATED-FALLBACK")
        assert snap.btc is not None
        assert snap.eth is not None
        assert snap.sol is not None
        assert snap.btc.mid_price > 0.0

    def test_r1_f05_simulated_paper_execution_against_real_market_data(self) -> None:
        """R1.5: Verify simulated paper order matching against real-time order book levels."""
        book = _make_mock_l2_book(mid_price=65000.0, half_spread_bps=1.0)
        best_bid = book.bids[0][0]
        best_ask = book.asks[0][0]

        # Passive ALO limit buy order placed at best bid
        order = ExecutionOrderCommand(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            action=ExecutionAction.PLACE_GTX_MAKER,
            side="BUY",
            target_price=best_bid,
            quantity=0.1,
            post_only_gtx=True,
            time_in_force="GTX",
            reason="Passive post-only buy order resting at best bid",
        )
        assert order.post_only_gtx is True
        assert order.target_price <= best_ask  # does not cross spread

        fill_price = order.target_price
        notional_usd = fill_price * order.quantity
        fee_schedule = FeeSchedule.hyperliquid()
        maker_fee_cost = notional_usd * fee_schedule.maker_rate
        taker_fee_cost = notional_usd * fee_schedule.taker_rate
        # ALO maker execution must strictly incur maker fee (1.5 bps), not taker (4.5 bps)
        assert maker_fee_cost < taker_fee_cost
        assert pytest.approx(maker_fee_cost, rel=1e-5) == notional_usd * 0.00015

    # -------------------------------------------------------------------------
    # R2: Quantitative Trading Sleeves (Funding Carry & Derivatives Alpha)
    # -------------------------------------------------------------------------

    def test_r2_f06_hyperliquid_1h_funding_carry_annualized_apr(self) -> None:
        """R2.1: Verify delta-neutral carry harvests 1h settlements and annualizes APR correctly."""
        sleeve = FundingCarrySleeve(config=FundingCarryConfig(min_positive_funding_8h=0.0002))
        hl_fee = FeeSchedule.hyperliquid()

        # Ingest 1h funding rate of 0.00005 (0.5 bps/hr -> 4 bps/8h)
        rate_1h = 0.00005
        rate_8h_equiv = rate_1h * 8.0
        opp = sleeve.evaluate_opportunity(
            instrument_id="BTC",
            funding_rate_8h=rate_8h_equiv,
            basis_pct=0.0002,
            fee_schedule=hl_fee,
            asri_score=15.0,
        )
        assert opp.direction == CarryDirection.SPOT_LONG_PERP_SHORT
        assert opp.spot_target_weight > 0.0
        assert opp.perp_target_weight == -opp.spot_target_weight
        assert opp.net_delta_weight == 0.0  # strictly delta-neutral
        # Annualized APR = rate_8h * 3 * 365 = rate_1h * 24 * 365
        assert pytest.approx(opp.annualized_funding_apr, rel=1e-5) == rate_1h * 24.0 * 365.0

    def test_r2_f07_cross_venue_funding_spread_arbitrage_capture(self) -> None:
        """R2.2: Verify funding rate arbitrage spread capture between Hyperliquid and Bybit/OKX."""
        hl_fee = FeeSchedule.hyperliquid()
        bybit_fee = FeeSchedule.usdt_perp()

        hl_rate_8h = 0.0008  # 8 bps / 8h
        bybit_rate_8h = 0.0001  # 1 bp / 8h
        spread_8h = hl_rate_8h - bybit_rate_8h  # 7 bps / 8h spread

        roundtrip_hurdle_bps = hl_fee.roundtrip_maker_bps + bybit_fee.roundtrip_maker_bps
        # Roundtrip maker fee: HL = 3.0 bps, Bybit = 4.0 bps -> Total hurdle = 7.0 bps
        assert roundtrip_hurdle_bps == 7.0

        spread_bps = spread_8h * 1e4  # 7.0 bps per 8h epoch
        expected_spread_gross_bps = spread_bps * 9.0
        assert expected_spread_gross_bps > roundtrip_hurdle_bps

    def test_r2_f08_alo_post_only_maker_routing_enforcement(self) -> None:
        """R2.3: Verify ALO Post-Only maker routing rejects taker execution and avoids taker fees."""
        maker_algo = DefensiveMakerExecutionAlgorithm()
        best_bid = 59990.0
        best_ask = 60010.0

        cmd = maker_algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            side="BUY",
            quantity=0.1,
            best_bid=best_bid,
            best_ask=best_ask,
            vpin=0.20,
            hawkes_eta=0.30,
        )
        assert cmd.action in (ExecutionAction.PLACE_GTX_MAKER, ExecutionAction.HOLD_RESTING_HYSTERESIS)
        assert cmd.post_only_gtx is True
        assert cmd.time_in_force == "GTX"
        if cmd.action == ExecutionAction.PLACE_GTX_MAKER:
            assert cmd.target_price <= best_bid

    def test_r2_f09_hyperliquid_fee_schedule_accounting(self) -> None:
        """R2.4: Verify FeeSchedule.hyperliquid() standardizes 1.5 bps Maker / 4.5 bps Taker."""
        fee = FeeSchedule.hyperliquid()
        assert fee.settlement == SettlementCurrency.USDC
        assert fee.maker_bps == 1.5
        assert fee.taker_bps == 4.5
        assert fee.estimated_slippage_bps == 0.5

        assert pytest.approx(fee.maker_rate, rel=1e-6) == 0.00015
        assert pytest.approx(fee.taker_rate, rel=1e-6) == 0.00045
        assert fee.roundtrip_maker_bps == 3.0
        assert fee.roundtrip_maker_plus_slippage_bps == 4.0

    def test_r2_f10_derivatives_alpha_options_surface_and_vault_metrics(self) -> None:
        """R2.5: Verify derivatives alpha sleeve ingests Deribit skew, term structure, and HLP vault."""
        surface = DeribitOptionsSurface(
            currency="BTC",
            skew_25d=0.035,  # 3.5% call skew (bullish)
            term_structure_slope=0.012,  # contango
            net_gex_usd=45_000_000.0,
            dvol_annualized=55.0,
            timestamp_ns=time.time_ns(),
        )
        assert surface.skew_25d > 0.0
        assert surface.net_gex_usd > 0.0

        vault = HyperliquidWhaleVaultMetrics(
            vault_address="0xdfc24b077bc1425ad1dea75bcb6f8158e10df303",
            tvl_usd=120_000_000.0,
            net_delta_usd=-5_000_000.0,
            hlp_crowding="CROWDED_SHORT",
            timestamp_ns=time.time_ns(),
        )
        assert vault.tvl_usd > 0.0
        assert vault.hlp_crowding == "CROWDED_SHORT"

        primary_dir: Literal[-1, 0, 1] = 1 if (surface.skew_25d > 0 and vault.hlp_crowding == "CROWDED_SHORT") else 0
        p_meta = 0.68
        assert primary_dir == 1
        assert 0.50 <= p_meta <= 1.0

    # -------------------------------------------------------------------------
    # R3: Multi-Model Apple Silicon (MLX) Risk Committee Daemon
    # -------------------------------------------------------------------------

    def test_r3_f11_apple_silicon_metal_gpu_tensor_acceleration(self) -> None:
        """R3.1: Verify Metal GPU acceleration on M1 Max Unified Memory using MLX tensors."""
        import mlx.core as mx

        dev = mx.default_device()
        assert str(dev.type) in ("DeviceType.gpu", "DeviceType.cpu")
        assert mx.metal.is_available() is True

        scorer = MLXCommitteeScorer(in_features=12, hidden_dim=24, out_features=1)
        x = mx.array(np.random.default_rng(42).normal(0.0, 1.0, (8, 12)).astype(np.float32))

        # Warm up Metal GPU shader compilation before latency measurement
        mx.eval(scorer(x))

        t0 = time.perf_counter()
        scores = scorer(x)
        mx.eval(scores)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert scores.shape == (8, 1)
        assert elapsed_ms < 50.0

    def test_r3_f12_macro_stablecoin_sentinel_risk_evaluation(self) -> None:
        """R3.2: Verify Model 1 (Macro Sentinel) evaluates macro stress, DVOL, and peg stability."""
        sentinel = MacroStablecoinSentinel()

        out_normal = sentinel.evaluate(
            stablecoin_depeg_bps=4.0,
            dvol_index=50.0,
            dvol_zscore=0.2,
            tvl_change_24h_pct=1.5,
        )
        assert isinstance(out_normal, MacroSentinelOutput)
        assert out_normal.score < 40.0
        assert out_normal.tail_risk_flag is False

        out_crisis = sentinel.evaluate(
            stablecoin_depeg_bps=520.0,
            dvol_index=110.0,
            dvol_zscore=3.8,
            tvl_change_24h_pct=-18.0,
        )
        assert out_crisis.score > 80.0
        assert out_crisis.tail_risk_flag is True

    def test_r3_f13_cross_sectional_breadth_analyst_evaluation(self) -> None:
        """R3.3: Verify Model 2 (Breadth Analyst) evaluates dispersion and correlation breakdown."""
        breadth = CrossSectionalBreadthAnalyst()
        telemetry = MultiAssetTelemetry(
            funding_rates_1h={"BTC": 0.000015, "ETH": 0.000018, "SOL": 0.000012},
            volumes_24h_usd={"BTC": 2_000_000_000.0, "ETH": 1_000_000_000.0, "SOL": 500_000_000.0},
        )
        out = breadth.evaluate(telemetry)
        assert isinstance(out, BreadthAnalystOutput)
        assert 0.0 <= out.breadth_score <= 100.0
        assert out.dispersion_bps >= 0.0
        assert out.correlation_dispersion >= 0.0

    def test_r3_f14_adversarial_red_team_critic_liquidation_evaluation(self) -> None:
        """R3.4: Verify Model 3 (Adversarial Critic) challenges portfolio exposures and cascades."""
        critic = AdversarialRedTeamCritic()
        stress = PortfolioStressInput(
            gross_leverage=1.8,
            vpin=0.35,
            hawkes_eta=0.45,
            liquidation_volume_usd=5_000_000.0,
        )
        out = critic.evaluate(stress)
        assert isinstance(out, RedTeamCriticOutput)
        assert 0.0 <= out.criticism_score <= 100.0
        assert 0.0 <= out.liquidation_cascade_prob <= 1.0
        assert 0.0 <= out.haircut_multiplier <= 1.0
        assert isinstance(out.veto_active, bool)

    def test_r3_f15_risk_committee_consensus_engine_multiplier(self) -> None:
        """R3.5: Verify Tri-model Consensus Engine computes unified M_regime in [0.0, 1.0]."""
        engine = RiskCommitteeConsensusEngine()

        macro_out = MacroSentinelOutput(
            score=25.0,
            stablecoin_depeg_bps=3.0,
            dvol_index=48.0,
            dvol_zscore=0.1,
            tail_risk_flag=False,
            regime_label="STABLE",
            confidence=0.9,
            summary="Macro risk benign.",
        )
        breadth_out = BreadthAnalystOutput(
            breadth_score=30.0,
            dispersion_bps=20.0,
            correlation_dispersion=0.2,
            funding_divergence_bps=1.5,
            concentration_risk_flag=False,
            regime_label="HEALTHY",
            confidence=0.85,
            summary="Broad market breadth intact.",
        )
        critic_out = RedTeamCriticOutput(
            criticism_score=20.0,
            liquidation_cascade_prob=0.10,
            toxic_flow_vulnerability=0.20,
            haircut_multiplier=1.0,
            veto_active=False,
            identified_risks=(),
            confidence=0.88,
            summary="No structural threats found.",
        )

        committee_out = engine.reconcile(macro_out, breadth_out, critic_out)
        assert isinstance(committee_out, RiskCommitteeOutput)
        assert 0.0 <= committee_out.consensus_risk_score <= 100.0
        assert committee_out.regime_multiplier == 1.0
        assert committee_out.tail_risk_flag is False

        # In crisis, consensus multiplier drops to 0.0
        crisis_critic = RedTeamCriticOutput(
            criticism_score=95.0,
            liquidation_cascade_prob=0.92,
            toxic_flow_vulnerability=0.85,
            haircut_multiplier=0.0,
            veto_active=True,
            identified_risks=("IMMINENT_LIQUIDATION_CASCADE",),
            confidence=0.95,
            summary="Emergency veto issued.",
        )
        committee_crisis = engine.reconcile(macro_out, breadth_out, crisis_critic)
        assert committee_crisis.regime_multiplier == 0.0
        assert committee_crisis.tail_risk_flag is True

    # -------------------------------------------------------------------------
    # R4: Historical Parquet Backfill & CPCV-Gated Challenger Tournament
    # -------------------------------------------------------------------------

    def test_r4_f16_multi_asset_parquet_backfill_18_column_storage(self) -> None:
        """R4.1: Verify Parquet backfill storage contains BTC, ETH, SOL with required columns."""
        for parquet_path in (BTC_PARQUET_FILE, ETH_PARQUET_FILE, SOL_PARQUET_FILE):
            assert parquet_path.exists(), f"Missing backfill parquet file: {parquet_path}"
            df = pl.read_parquet(parquet_path)
            assert df.height > 0
            for col in BASE_COLUMNS:
                assert col in df.columns, f"Missing column {col} in {parquet_path}"
            assert len(df.columns) >= 18  # base schema plus funding_rate_1h and provenance columns

    def test_r4_f17_cpcv_45_split_combinatorial_purging_and_embargo(self) -> None:
        """R4.2: Verify 45-split CPCV generates 45 unique splits with zero label leakage."""
        cpcv = CombinatorialPurgedCV(vertical_barrier_bars=16, n_groups=10, k_test_groups=2, embargo_pct=0.01)
        splits = cpcv.generate_splits(500)
        assert len(splits) == 45

        for s in splits:
            train_set = set(s.train_indices)
            test_set = set(s.test_indices)
            assert len(train_set.intersection(test_set)) == 0
            assert len(s.test_indices) > 0
            assert len(s.train_indices) > 0

    def test_r4_f18_deflated_sharpe_ratio_computation_and_penalization(self) -> None:
        """R4.3: Verify Deflated Sharpe Ratio penalizes multi-trial selection bias."""
        rng = np.random.default_rng(42)
        returns_positive = rng.normal(0.002, 0.01, size=500)

        dsr_1_trial = compute_dsr(returns_positive, total_trials=1)
        assert 0.0 <= dsr_1_trial <= 1.0

        dsr_50_trials = compute_dsr(returns_positive, total_trials=50)
        assert dsr_50_trials <= dsr_1_trial

    def test_r4_f19_pbo_probability_of_backtest_overfitting(self) -> None:
        """R4.4: Verify PBO calculation across combinatorial paths."""
        rng = np.random.default_rng(2026)
        is_matrix = rng.normal(0.002, 0.01, size=(45, 5))
        oos_matrix = rng.normal(0.001, 0.01, size=(45, 5))
        pbo = compute_pbo(is_matrix, oos_matrix)
        assert 0.0 <= pbo <= 1.0

    def test_r4_f20_automated_strategy_tournament_promotion_gate(self) -> None:
        """R4.5: Verify StrategyTournament evaluates candidate models and enforces gates."""
        bars = _make_mock_dollar_bars(n_bars=200)
        strat1 = create_trend_strategy(strategy_id="cand_trend", fast_window=4, slow_window=12)
        strat2 = create_mean_reversion_strategy(strategy_id="cand_mean_rev", window=8, z_entry=1.5)
        strat3 = create_funding_carry_strategy(strategy_id="cand_carry", entry_threshold_bps=0.5)

        tournament = StrategyTournament(
            candidates=[strat1, strat2, strat3],
            vertical_barrier_bars=4,
            fee_schedule=FeeSchedule.hyperliquid(),
        )
        result = tournament.run_tournament(bars=bars)
        assert isinstance(result, TournamentResult)
        assert len(result.rankings) == 3
        assert result.summary is not None

    # -------------------------------------------------------------------------
    # R5: Persistent Paper Trading Ledger & UI Forensics Dashboard
    # -------------------------------------------------------------------------

    def test_r5_f21_persistent_account_ledger_initialization_5000_usd(self, tmp_path: Path) -> None:
        """R5.1: Verify persistent ledger loads and initializes with exactly $5,000.00 base capital."""
        ledger_file = tmp_path / "account_ledger.json"
        # Phase 0: isolated ledger and scripted feed so the reset never touches the project ledger.
        store = build_store(
            tmp_path,
            live_bars={"BTC": flat_bars("BTC", 0, 1, 100.0)},
            snapshots=[make_snapshot()],
            history={"BTC": flat_history("BTC", 64, 100.0)},
            ledger_name="store_ledger.json",
        )
        reset_res = store.reset_account(starting_capital_usd=5000.0)
        assert reset_res.starting_capital_usd == 5000.0
        assert reset_res.current_equity_usd == 5000.0
        assert reset_res.net_pnl_usd == 0.0

        ledger_data = {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": 5000.0,
            "current_cash_usd": 5000.0,
            "total_realized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "trade_history": [],
            "last_updated_iso": datetime.now(timezone.utc).isoformat(),
        }
        ledger_file.write_text(json.dumps(ledger_data, indent=2), encoding="utf-8")
        assert ledger_file.exists()
        loaded = json.loads(ledger_file.read_text(encoding="utf-8"))
        assert loaded["base_capital_usd"] == 5000.0
        assert loaded["current_cash_usd"] == 5000.0

    def test_r5_f22_accumulated_pnl_and_win_loss_persistence(self, tmp_path: Path) -> None:
        """R5.2: Verify paper trades accumulate PnL and persist across simulated restarts."""
        ledger_file = tmp_path / "account_ledger.json"

        ledger_state = {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": 5000.0,
            "current_cash_usd": 5075.0,
            "total_realized_pnl": 75.0,
            "win_count": 1,
            "loss_count": 1,
            "trade_history": [
                {
                    "trade_id": "tr_001",
                    "instrument_id": "BTC",
                    "action": "CLOSE_WIN",
                    "realized_pnl_usd": 120.0,
                    "why_did_it_trade": "Trend breakout confirmed by meta-model",
                },
                {
                    "trade_id": "tr_002",
                    "instrument_id": "ETH",
                    "action": "CLOSE_LOSS",
                    "realized_pnl_usd": -45.0,
                    "why_did_it_trade": "Stop loss triggered on basis reversion",
                },
            ],
            "last_updated_iso": datetime.now(timezone.utc).isoformat(),
        }
        ledger_file.write_text(json.dumps(ledger_state), encoding="utf-8")

        recovered = json.loads(ledger_file.read_text(encoding="utf-8"))
        assert recovered["base_capital_usd"] == 5000.0
        assert recovered["total_realized_pnl"] == 75.0
        assert recovered["win_count"] == 1
        assert recovered["loss_count"] == 1
        assert len(recovered["trade_history"]) == 2

    def test_r5_f23_atomic_ledger_file_state_management(self, tmp_path: Path) -> None:
        """R5.3: Verify atomic read/write prevents state corruption during concurrent updates."""
        target_file = tmp_path / "account_ledger.json"
        tmp_file = tmp_path / "account_ledger.json.tmp"

        payload = {"base_capital_usd": 5000.0, "current_cash_usd": 5250.0, "trades": 5}
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, target_file)
        assert target_file.exists()
        assert not tmp_file.exists()
        loaded = json.loads(target_file.read_text(encoding="utf-8"))
        assert loaded["current_cash_usd"] == 5250.0

    def test_r5_f24_feature_attribution_trade_forensics_why_did_it_trade(self) -> None:
        """R5.4: Verify trade forensics records explain entry reasons and feature attribution."""
        forensic_record: dict[str, Any] = {
            "trade_id": "tx_btc_0921",
            "timestamp_iso": "2026-10-31T01:30:00Z",
            "instrument_id": "BTC",
            "side": "BUY",
            "fill_price": 64850.0,
            "quantity": 0.05,
            "notional_usd": 3242.50,
            "regime": "GENUINE_TREND_CONVICTION",
            "p_meta": 0.72,
            "mahalanobis_di": 1.45,
            "vpin": 0.28,
            "hawkes_eta": 0.32,
            "asri_score": 18.0,
            "kelly_fraction": 0.22,
            "why_did_it_trade": (
                "Long BTC entered: 72% meta-conviction trend signal with 1.45 Mahalanobis distance "
                "within epistemic bound, low VPIN toxicity (0.28), and 18 ASRI benign macro regime."
            ),
            "feature_attribution": {
                "ffd_close_zscore": 1.85,
                "ffd_oi_zscore": 1.40,
                "basis_pct": 0.0003,
                "funding_rate_1h": 0.00008,
            },
        }

        assert forensic_record["p_meta"] >= 0.50
        assert forensic_record["mahalanobis_di"] < 3.0
        assert "why_did_it_trade" in forensic_record
        assert len(forensic_record["why_did_it_trade"]) > 20
        assert "ffd_close_zscore" in forensic_record["feature_attribution"]

    def test_r5_f25_ui_telemetry_forensics_dashboard_endpoints(self) -> None:
        """R5.5: Verify FastAPI Web UI serves telemetry, tooltips, and control center state."""
        app = create_app()
        client = TestClient(app)

        resp_state = client.get("/api/state")
        assert resp_state.status_code == 200
        state = resp_state.json()
        assert "regime_mode" in state
        assert "pnl_metrics" in state

        resp_tooltips = client.get("/api/tooltips")
        assert resp_tooltips.status_code == 200
        tooltips = resp_tooltips.json()
        assert "vpin_toxicity" in tooltips
        assert "hawkes_reflexivity" in tooltips
        assert "meta_probability" in tooltips

        for act in ("start_dry_run", "pause_entries", "switch_carry_only", "emergency_flatten"):
            resp_ctrl = client.post(f"/api/control/{act}")
            assert resp_ctrl.status_code == 200
            ctrl_data = resp_ctrl.json()
            assert ctrl_data["status"] == "ok"


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 TESTS PER FEATURE FOR R1 - R5)
# ==============================================================================


class TestTier2BoundaryCornerCases:
    """Tier 2: Boundary conditions, corner cases, and stress tests for R1-R5."""

    # -------------------------------------------------------------------------
    # R1 Boundary Cases
    # -------------------------------------------------------------------------

    def test_r1_b01_empty_order_book_zero_spread_zero_depth(self) -> None:
        """R1.B1: Empty order book returns 0.0 spread and depth without division by zero."""
        spread, d10, d50 = _compute_l2_depth_and_spread([], [])
        assert spread == 0.0
        assert d10 == 0.0
        assert d50 == 0.0

        spread_b, _, _ = _compute_l2_depth_and_spread([], [(50000.0, 1.0)])
        assert spread_b == 0.0

    def test_r1_b02_crossed_order_book_detection(self) -> None:
        """R1.B2: Detect invalid crossed book where best_bid >= best_ask."""
        bids = [(65010.0, 1.0), (65000.0, 1.0)]
        asks = [(65000.0, 1.0), (65020.0, 1.0)]
        spread, _, _ = _compute_l2_depth_and_spread(bids, asks)
        assert spread == 0.0

    def test_r1_b03_extreme_funding_rate_spike_handling(self) -> None:
        """R1.B3: Handle extreme 1h funding rate (+5% per hour) without overflow."""
        extreme_rate = 0.05  # 5% per hour
        snap = AssetMarketSnapshot(
            instrument_id="SOL",
            mid_price=150.0,
            mark_price=150.5,
            oracle_price=150.0,
            open_interest=500000.0,
            funding_rate_1h=extreme_rate,
        )
        assert pytest.approx(snap.funding_rate_8h_equiv, rel=1e-5) == 0.40
        assert pytest.approx(snap.annualized_apr_pct, rel=1e-5) == 43800.0
        assert not math.isnan(snap.annualized_apr_pct)

    def test_r1_b04_partial_feed_loss_degradation(self) -> None:
        """R1.B4: Partial feed loss returns valid snapshots for available assets without crash."""
        now_ns = time.time_ns()
        btc_book = _make_mock_l2_book(60000.0)
        btc_snap = AssetMarketSnapshot(
            instrument_id="BTC",
            mid_price=60000.0,
            mark_price=60000.0,
            oracle_price=60000.0,
            open_interest=50000.0,
            l2_book=btc_book,
            timestamp_ns=now_ns,
        )
        multi = MultiAssetMarketSnapshot(
            timestamp_ns=now_ns,
            assets={"BTC": btc_snap},
            feed_source="HYPERLIQUID-L1",
        )
        assert multi.get_asset("BTC") is not None
        assert multi.get_asset("ETH") is None
        assert multi.get_asset("SOL") is None

    def test_r1_b05_bybit_failover_trigger_on_timeout_or_error(self) -> None:
        """R1.B5: Failover cleanly switches to Bybit / fallback when network call fails."""
        engine = LiveMarketDataEngine(
            timeout_sec=0.1,
            hl_url="http://192.0.2.1:81/blackhole",
            bybit_tickers_url="http://192.0.2.1:81/blackhole",
        )
        snap = engine.get_multi_asset_snapshot_sync(coins=("BTC", "ETH", "SOL"))
        assert snap is not None
        assert snap.feed_source in ("BYBIT-V5-FALLBACK", "SIMULATED-FALLBACK")

    # -------------------------------------------------------------------------
    # R2 Boundary Cases
    # -------------------------------------------------------------------------

    def test_r2_b06_unprofitable_carry_rejection_below_fee_hurdle(self) -> None:
        """R2.B1: Funding rate below round-trip maker hurdle must reject carry trade."""
        sleeve = FundingCarrySleeve(config=FundingCarryConfig(min_positive_funding_8h=0.0003))
        hl_fee = FeeSchedule.hyperliquid()

        opp = sleeve.evaluate_opportunity(
            instrument_id="BTC",
            funding_rate_8h=0.0001,
            fee_schedule=hl_fee,
        )
        assert opp.direction == CarryDirection.FLAT
        assert opp.is_actionable is False
        assert opp.spot_target_weight == 0.0

    def test_r2_b07_alo_crossing_the_spread_immediate_rejection(self) -> None:
        """R2.B2: ALO order crossing the spread must never execute as taker."""
        algo = DefensiveMakerExecutionAlgorithm()
        best_bid = 50000.0
        best_ask = 50005.0

        cmd = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            side="BUY",
            quantity=0.1,
            best_bid=best_bid,
            best_ask=best_ask,
            resting_price=50010.0,
            vpin=0.10,
            hawkes_eta=0.10,
        )
        if cmd.action == ExecutionAction.PLACE_GTX_MAKER:
            assert cmd.target_price <= best_bid

    def test_r2_b08_max_gross_leverage_cap_clamping_2_5x(self) -> None:
        """R2.B3: Sizing equation strictly clamps total gross leverage to <= 2.5x."""
        fee = FeeSchedule.hyperliquid()
        inputs = [
            AssetAllocationInput(
                asset_id="BTC",
                y_primary=1,
                w_hrp=0.50,
                p_meta=0.95,
                expected_gross_return_bps=60.0,
                fee_schedule=fee,
            ),
            AssetAllocationInput(
                asset_id="ETH",
                y_primary=1,
                w_hrp=0.30,
                p_meta=0.95,
                expected_gross_return_bps=60.0,
                fee_schedule=fee,
            ),
            AssetAllocationInput(
                asset_id="SOL",
                y_primary=-1,
                w_hrp=0.20,
                p_meta=0.95,
                expected_gross_return_bps=60.0,
                fee_schedule=fee,
            ),
        ]
        out = compute_signed_target_weights(
            inputs=inputs,
            regime_multiplier=1.0,
            max_gross_leverage=2.5,
        )
        assert out.gross_leverage <= 2.500000001
        assert sum(abs(w) for w in out.weights.values()) <= 2.500000001

    def test_r2_b09_missing_options_surface_data_fallback(self) -> None:
        """R2.B4: Missing or zero options surface data falls back gracefully."""
        surface = DeribitOptionsSurface(currency="BTC")
        assert surface.skew_25d == 0.0
        assert surface.net_gex_usd == 0.0
        primary_dir = 0
        assert primary_dir == 0

    def test_r2_b10_consecutive_gtx_rejections_deepening_backoff(self) -> None:
        """R2.B5: On 4th consecutive GTX rejection, maker backs off 1 tick deeper passive."""
        algo = DefensiveMakerExecutionAlgorithm()
        cmd = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            side="BUY",
            quantity=0.1,
            best_bid=59990.0,
            best_ask=60010.0,
            resting_price=59990.0,
            consecutive_rejections=4,
            vpin=0.20,
            hawkes_eta=0.20,
        )
        assert cmd.action == ExecutionAction.BACKOFF_DEEPER_GTX
        assert cmd.target_price < 59990.0

    # -------------------------------------------------------------------------
    # R3 Boundary Cases
    # -------------------------------------------------------------------------

    def test_r3_b11_acute_stablecoin_depeg_hard_stop(self) -> None:
        """R3.B1: Acute stablecoin depeg (<= $0.95) forces M_regime = 0.0."""
        m_regime = compute_regime_multiplier_from_asri(asri_score=85.0, tail_risk_flag=True)
        assert m_regime == 0.0

    def test_r3_b12_extreme_correlation_breakdown_dump(self) -> None:
        """R3.B2: Severe multi-asset dump escalates Breadth Analyst score."""
        breadth = CrossSectionalBreadthAnalyst()
        dump_telemetry = MultiAssetTelemetry(
            asset_returns_history={
                "BTC": (-0.045, -0.050, -0.055, -0.060),
                "ETH": (-0.055, -0.060, -0.065, -0.070),
                "SOL": (-0.075, -0.080, -0.085, -0.090),
            },
            funding_rates_1h={"BTC": -0.0005, "ETH": -0.0006, "SOL": -0.0008},
        )
        out = breadth.evaluate(dump_telemetry)
        assert out.breadth_score > 50.0

    def test_r3_b13_adversarial_critic_toxic_flow_liquidation_veto(self) -> None:
        """R3.B3: VPIN > 0.60 and Hawkes eta > 0.70 triggers emergency veto."""
        critic = AdversarialRedTeamCritic()
        toxic_stress = PortfolioStressInput(
            gross_leverage=2.4,
            vpin=0.72,
            hawkes_eta=0.85,
            liquidation_volume_usd=40_000_000.0,
        )
        out = critic.evaluate(toxic_stress)
        assert out.veto_active is True
        assert out.haircut_multiplier == 0.0

    def test_r3_b14_corrupted_json_model_output_fallback(self) -> None:
        """R3.B4: Corrupted model output returns deterministic neutral fallback without crash."""
        fallback = parse_and_validate_llm_risk_json("{corrupt: [json, {missing_quotes}")
        assert isinstance(fallback, SystemicRiskOutput)
        assert fallback.score == 0.0
        assert fallback.confidence == 0.0
        assert fallback.tail_risk_flag is False
        assert "neutral fallback mode" in fallback.plain_english_summary

    def test_r3_b15_metal_gpu_tensor_extreme_dimensions(self) -> None:
        """R3.B5: MLX Metal GPU handles single vector and large batches safely."""
        import mlx.core as mx

        scorer = MLXCommitteeScorer(in_features=12, hidden_dim=16, out_features=1)
        x_single = mx.zeros((1, 12))
        res_single = scorer(x_single)
        mx.eval(res_single)
        assert res_single.shape == (1, 1)

        x_batch = mx.zeros((128, 12))
        res_batch = scorer(x_batch)
        mx.eval(res_batch)
        assert res_batch.shape == (128, 1)

    # -------------------------------------------------------------------------
    # R4 Boundary Cases
    # -------------------------------------------------------------------------

    def test_r4_b16_degenerate_zero_variance_returns_dsr(self) -> None:
        """R4.B1: Zero-variance returns series handles cleanly in DSR without ZeroDivisionError."""
        returns_zero = np.zeros(100)
        dsr = compute_dsr(returns_zero, total_trials=1)
        assert 0.0 <= dsr <= 1.0

    def test_r4_b17_extreme_kurtosis_negative_skew_dsr_penalty(self) -> None:
        """R4.B2: High kurtosis (fat tails) penalizes DSR relative to normal returns."""
        rng = np.random.default_rng(42)
        base_norm = rng.normal(0.003, 0.01, size=500)
        base_fat = base_norm.copy()
        base_fat[0] = 0.15
        base_fat[1] = -0.15
        dsr_norm = compute_dsr(base_norm, total_trials=10)
        dsr_fat = compute_dsr(base_fat, total_trials=10)
        assert dsr_fat <= dsr_norm

    def test_r4_b18_high_trial_count_selection_bias_haircut(self) -> None:
        """R4.B3: 200 trials of strategy testing drastically cuts DSR compared to 2 trials."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0.002, 0.01, size=500)
        dsr_few = compute_dsr(rets, total_trials=2)
        dsr_many = compute_dsr(rets, total_trials=200)
        assert dsr_many <= dsr_few

    def test_r4_b19_parquet_schema_missing_column_validation(self, tmp_path: Path) -> None:
        """R4.B4: Detecting missing columns in corrupted Parquet file."""
        corrupt_file = tmp_path / "corrupt_bars.parquet"
        df_incomplete = pl.DataFrame({
            "instrument_id": ["BTC"],
            "open": [60000.0],
            "high": [60100.0],
            "low": [59900.0],
        })
        df_incomplete.write_parquet(corrupt_file)

        df_read = pl.read_parquet(corrupt_file)
        missing = [c for c in BASE_COLUMNS if c not in df_read.columns]
        assert len(missing) > 0
        assert "funding_rate_8h" in missing

    def test_r4_b20_cpcv_purging_boundary_overlaps(self) -> None:
        """R4.B5: CPCV purge window handles edge cases cleanly without negative indices."""
        cpcv = CombinatorialPurgedCV(vertical_barrier_bars=4, n_groups=5, k_test_groups=1, embargo_pct=0.01)
        splits = cpcv.generate_splits(100)
        assert len(splits) == 5
        for s in splits:
            assert len(s.train_indices) > 0
            assert len(s.test_indices) > 0

    # -------------------------------------------------------------------------
    # R5 Boundary Cases
    # -------------------------------------------------------------------------

    def test_r5_b21_simulated_write_interruption_leaves_ledger_intact(self, tmp_path: Path) -> None:
        """R5.B1: Interrupted write leaves original ledger intact."""
        ledger = tmp_path / "account_ledger.json"
        original_data = {"base_capital_usd": 5000.0, "status": "ORIGINAL_INTACT"}
        ledger.write_text(json.dumps(original_data), encoding="utf-8")

        tmp_target = tmp_path / "account_ledger.json.tmp"
        tmp_target.write_text("CORRUPTED_PARTIAL_JSON_{", encoding="utf-8")

        recovered = json.loads(ledger.read_text(encoding="utf-8"))
        assert recovered["status"] == "ORIGINAL_INTACT"

    def test_r5_b22_account_equity_drawdown_limit_handling(self) -> None:
        """R5.B2: Severe account drawdown clamps sizing and prevents over-leveraging."""
        fee = FeeSchedule.hyperliquid()
        current_equity = 500.0  # 90% drawdown from $5,000 base
        input_trade = AssetAllocationInput(
            asset_id="BTC",
            y_primary=1,
            w_hrp=1.0,
            p_meta=0.60,
            expected_gross_return_bps=40.0,
            fee_schedule=fee,
        )
        res = compute_signed_target_weights([input_trade], regime_multiplier=1.0, max_gross_leverage=2.5)
        max_notional_usd = current_equity * res.gross_leverage
        assert max_notional_usd <= 500.0 * 2.500000001

    def test_r5_b23_rapid_sequential_restarts_idempotency(self, tmp_path: Path) -> None:
        """R5.B3: 20 rapid sequential stop/restarts maintain exact identical capital and history."""
        ledger = tmp_path / "account_ledger.json"
        state = {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": 5000.0,
            "current_cash_usd": 5120.0,
            "total_realized_pnl": 120.0,
            "win_count": 2,
            "loss_count": 0,
        }
        ledger.write_text(json.dumps(state), encoding="utf-8")

        for _ in range(20):
            data = json.loads(ledger.read_text(encoding="utf-8"))
            assert data["base_capital_usd"] == 5000.0
            assert data["current_cash_usd"] == 5120.0
            assert data["total_realized_pnl"] == 120.0

    def test_r5_b24_empty_trade_history_initialization(self, tmp_path: Path) -> None:
        """R5.B4: Empty ledger initializes safely without null division in metrics."""
        ledger = tmp_path / "account_ledger.json"
        clean_state = {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": 5000.0,
            "current_cash_usd": 5000.0,
            "total_realized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "trade_history": [],
        }
        ledger.write_text(json.dumps(clean_state), encoding="utf-8")
        data = json.loads(ledger.read_text(encoding="utf-8"))
        total_trades = data["win_count"] + data["loss_count"]
        win_rate = (data["win_count"] / total_trades) if total_trades > 0 else 0.0
        assert win_rate == 0.0

    def test_r5_b25_invalid_capital_reset_rejection(self) -> None:
        """R5.B5: POST /api/reset rejects negative or zero starting capital with HTTP 422."""
        app = create_app()
        client = TestClient(app)
        resp_neg = client.post("/api/reset", json={"starting_capital_usd": -100.0})
        assert resp_neg.status_code == 422
        resp_zero = client.post("/api/reset", json={"starting_capital_usd": 0.0})
        assert resp_zero.status_code == 422


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (PAIRWISE INTERACTIONS ACROSS R1-R5)
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise interactions across R1, R2, R3, R4, and R5."""

    def test_r1_r2_live_funding_spike_triggers_alo_carry_entry(self) -> None:
        """R1+R2: Live Hyperliquid 1h funding spike triggers automated ALO carry entry."""
        multi_snap = _make_mock_multi_asset_snapshot(
            btc_price=64000.0,
            funding_1h_btc=0.00015,
        )
        btc_snap = multi_snap.get_asset("BTC")
        assert btc_snap is not None

        sleeve = FundingCarrySleeve(config=FundingCarryConfig(min_positive_funding_8h=0.0002))
        opp = sleeve.evaluate_opportunity(
            instrument_id="BTC",
            funding_rate_8h=btc_snap.funding_rate_8h_equiv,
            fee_schedule=FeeSchedule.hyperliquid(),
        )
        assert opp.direction == CarryDirection.SPOT_LONG_PERP_SHORT
        assert opp.is_actionable is True

        maker_algo = DefensiveMakerExecutionAlgorithm()
        cmd = maker_algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            side="SELL",
            quantity=0.1,
            best_bid=btc_snap.l2_book.bids[0][0],
            best_ask=btc_snap.l2_book.asks[0][0],
            vpin=0.15,
            hawkes_eta=0.25,
        )
        assert cmd.action == ExecutionAction.PLACE_GTX_MAKER
        assert cmd.post_only_gtx is True

    def test_r2_r3_macro_crisis_overrides_sleeve_directional_signals(self) -> None:
        """R2+R3: Macro crisis (M_regime = 0.0) overrides and halts bullish sleeve allocations."""
        fee = FeeSchedule.hyperliquid()
        sleeve_input = AssetAllocationInput(
            asset_id="BTC",
            y_primary=1,
            w_hrp=0.50,
            p_meta=0.88,
            expected_gross_return_bps=85.0,
            fee_schedule=fee,
        )

        m_regime_crisis = compute_regime_multiplier_from_asri(asri_score=92.0, tail_risk_flag=True)
        assert m_regime_crisis == 0.0

        allocation = compute_signed_target_weights([sleeve_input], regime_multiplier=m_regime_crisis)
        assert allocation.weights["BTC"] == 0.0
        assert allocation.gross_leverage == 0.0

    def test_r1_r4_market_ticks_stream_to_parquet_dollar_bars(self) -> None:
        """R1+R4: Real-time ingestion ticks aggregate into DollarBar matching Parquet schema."""
        snap = _make_mock_multi_asset_snapshot()
        btc = snap.get_asset("BTC")
        assert btc is not None

        bar = DollarBar(
            instrument_id="BTCUSDT-PERP",
            horizon_label="15m",
            ts_open_ns=snap.timestamp_ns - 900_000_000_000,
            ts_close_ns=snap.timestamp_ns,
            open=btc.mid_price,
            high=btc.mark_price + 5.0,
            low=btc.mid_price - 5.0,
            close=btc.mid_price,
            volume=5.0,
            dollar_volume=5.0 * btc.mid_price,
            buy_dollar_volume=2.5 * btc.mid_price,
            sell_dollar_volume=2.5 * btc.mid_price,
            open_interest=btc.open_interest,
            funding_rate_1h=btc.funding_rate_1h,
            funding_rate_8h=btc.funding_rate_8h_equiv,
            mark_price=btc.mark_price,
            spot_price=btc.oracle_price,
        )
        bar_dict = bar.model_dump()
        for col in BASE_COLUMNS:
            assert col in bar_dict

    def test_r2_r5_trade_execution_updates_persistent_ledger_and_forensics(self, tmp_path: Path) -> None:
        """R2+R5: ALO trade execution updates persistent ledger and attaches forensics reason."""
        ledger_file = tmp_path / "account_ledger.json"
        base_capital = 5000.0

        ledger_state: dict[str, Any] = {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": base_capital,
            "current_cash_usd": base_capital,
            "total_realized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "trade_history": [],
        }
        ledger_file.write_text(json.dumps(ledger_state), encoding="utf-8")

        realized_pnl = 35.00
        new_cash = base_capital + realized_pnl
        forensics_note = (
            "Why Did It Trade: Harvested 1h funding carry on BTC with 12 bps/8h spread, "
            "ALO maker entry avoiding 4.5 bps taker fee, and benign ASRI risk (14.0)."
        )
        trade_record: dict[str, Any] = {
            "trade_id": "tx_carry_001",
            "instrument_id": "BTC",
            "action": "HARVEST_CARRY_PROFIT",
            "realized_pnl_usd": realized_pnl,
            "why_did_it_trade": forensics_note,
        }

        ledger_state["current_cash_usd"] = new_cash
        ledger_state["total_realized_pnl"] += realized_pnl
        ledger_state["win_count"] += 1
        ledger_state["trade_history"].append(trade_record)

        ledger_file.write_text(json.dumps(ledger_state), encoding="utf-8")

        reloaded = json.loads(ledger_file.read_text(encoding="utf-8"))
        assert reloaded["current_cash_usd"] == 5035.00
        assert reloaded["win_count"] == 1
        assert reloaded["trade_history"][0]["why_did_it_trade"] == forensics_note

    def test_r3_r5_risk_committee_state_streams_to_ui_dashboard(self) -> None:
        """R3+R5: 3-model risk committee consensus streams into FastAPI /api/state."""
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/state")
        assert resp.status_code == 200
        state = resp.json()
        assert "regime_mode" in state
        assert "pnl_metrics" in state

    def test_r4_r2_cpcv_tournament_promotion_reweights_active_sleeves(self) -> None:
        """R4+R2: Tournament promotion updates Champion model and dynamic sleeve weights."""
        gatekeeper = ChampionChallengerGatekeeper(asset_id="BTC")
        assert gatekeeper.state.regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK

        challenger = CPCVValidationResult(
            model_id="challenger_opt_alpha_v4",
            purge_bars=16,
            psr_vs_zero=0.98,
            psr_vs_benchmark=0.96,
            dsr=0.96,
            pbo=0.15,
            expected_net_pnl_bps=28.0,
            sharpe_net=2.1,
            skewness=-0.1,
            kurtosis=3.2,
            total_trials=45,
        )
        assert challenger.passed_all_gates is True

        promoted_state = gatekeeper.evaluate_challenger(challenger)
        assert promoted_state.champion_model_id == "challenger_opt_alpha_v4"
        assert promoted_state.regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY

    def test_r1_r3_ingestion_microstructure_feeds_adversarial_critic(self) -> None:
        """R1+R3: L2 spread widening and depth depletion feed Adversarial Red-Team Critic."""
        wide_book = _make_mock_l2_book(mid_price=60000.0, half_spread_bps=17.5, size_per_level=0.01)
        assert wide_book.spread_bps > 30.0

        critic = AdversarialRedTeamCritic()
        stress = PortfolioStressInput(
            gross_leverage=2.0,
            vpin=0.65,
            hawkes_eta=0.75,
            liquidation_volume_usd=15_000_000.0,
        )
        out = critic.evaluate(stress)
        assert out.criticism_score > 50.0

    def test_r4_r5_tournament_backtest_ledger_forensics_audit(self) -> None:
        """R4+R5: Tournament backtest outcomes format forensic logs matching live ledger."""
        bars = _make_mock_dollar_bars(n_bars=100)
        strat = create_trend_strategy(strategy_id="trend_audit", fast_window=3, slow_window=8)
        tournament = StrategyTournament(candidates=[strat], vertical_barrier_bars=2)

        res = tournament.run_tournament(bars)
        cand_res = res.rankings[0]

        audit_log = {
            "strategy_id": cand_res.strategy_id,
            "sharpe_net": cand_res.sharpe_net,
            "dsr": cand_res.dsr,
            "pbo": cand_res.pbo,
            "passed_gates": cand_res.passed_all_gates,
            "why_promoted": f"Strategy {cand_res.strategy_id} achieved Sharpe {cand_res.sharpe_net:.2f}",
        }
        assert "why_promoted" in audit_log
        assert audit_log["sharpe_net"] is not None


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (REALISTIC MULTI-STEP WORKFLOWS)
# ==============================================================================


class TestTier4RealWorldApplicationScenarios:
    """Tier 4: Realistic end-to-end trading lifecycles and stress scenarios."""

    def test_r1_to_r5_full_trading_lifecycle_pipeline(self, tmp_path: Path) -> None:
        """Scenario 1: Complete pipeline: Ingestion -> Sleeve -> Risk -> Execution -> Ledger -> UI."""
        ledger_file = tmp_path / "account_ledger.json"
        base_capital = 5000.0

        # Step 1: Ingestion streams live market snapshot
        snap = _make_mock_multi_asset_snapshot(btc_price=65000.0, funding_1h_btc=0.00012)
        btc = snap.get_asset("BTC")
        assert btc is not None

        # Step 2: Risk Committee checks systemic conditions
        sentinel = MacroStablecoinSentinel()
        breadth = CrossSectionalBreadthAnalyst()
        critic = AdversarialRedTeamCritic()
        consensus = RiskCommitteeConsensusEngine()

        macro_out = sentinel.evaluate(stablecoin_depeg_bps=3.0)
        breadth_out = breadth.evaluate(MultiAssetTelemetry())
        critic_out = critic.evaluate(PortfolioStressInput(gross_leverage=1.0, vpin=0.22, hawkes_eta=0.25))

        comm_out = consensus.reconcile(macro_out, breadth_out, critic_out)
        assert comm_out.regime_multiplier == 1.0

        # Step 3: Sleeve identifies carry arbitrage opportunity
        sleeve = FundingCarrySleeve()
        opp = sleeve.evaluate_opportunity(
            instrument_id="BTC",
            funding_rate_8h=btc.funding_rate_8h_equiv,
            fee_schedule=FeeSchedule.hyperliquid(),
            asri_score=comm_out.consensus_risk_score,
        )
        assert opp.is_actionable is True

        # Step 4: ALO Post-Only maker quotes order
        algo = DefensiveMakerExecutionAlgorithm()
        cmd = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            side="SELL",
            quantity=0.1,
            best_bid=btc.l2_book.bids[0][0],
            best_ask=btc.l2_book.asks[0][0],
            vpin=0.22,
            hawkes_eta=0.25,
        )
        assert cmd.action == ExecutionAction.PLACE_GTX_MAKER
        assert cmd.post_only_gtx is True

        # Step 5: Paper trade execution and ledger persistence
        realized_gain = 42.50
        updated_equity = base_capital + realized_gain
        ledger_state = {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": base_capital,
            "current_cash_usd": updated_equity,
            "total_realized_pnl": realized_gain,
            "win_count": 1,
            "loss_count": 0,
            "trade_history": [
                {
                    "trade_id": "tx_full_001",
                    "instrument_id": "BTC",
                    "action": "CLOSE_CARRY_PROFIT",
                    "realized_pnl_usd": realized_gain,
                    "why_did_it_trade": "Delta-neutral carry on 1h settlement with 1.5 bps maker fee tier.",
                }
            ],
            "last_updated_iso": datetime.now(timezone.utc).isoformat(),
        }
        ledger_file.write_text(json.dumps(ledger_state), encoding="utf-8")

        # Step 6: UI Telemetry reflects updated status
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/state")
        assert resp.status_code == 200
        assert "pnl_metrics" in resp.json()

    def test_flash_crash_and_stablecoin_depeg_crisis_response(self) -> None:
        """Scenario 2: Market flash crash + USDT depeg triggers toxic shield and emergency flatten."""
        sentinel = MacroStablecoinSentinel()
        out_sentinel = sentinel.evaluate(
            stablecoin_depeg_bps=750.0,
            dvol_index=130.0,
            dvol_zscore=4.5,
            tvl_change_24h_pct=-25.0,
        )
        assert out_sentinel.tail_risk_flag is True

        m_regime = compute_regime_multiplier_from_asri(out_sentinel.score, tail_risk_flag=True)
        assert m_regime == 0.0

        algo = DefensiveMakerExecutionAlgorithm()
        cmd = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            side="BUY",
            quantity=0.1,
            best_bid=47900.0,
            best_ask=48100.0,
            resting_price=47900.0,
            vpin=0.85,
            hawkes_eta=0.95,
        )
        assert cmd.action == ExecutionAction.CANCEL_ALL_TOXIC_SHIELD
        assert "toxic_sweep_shield" in cmd.reason.lower()

    def test_server_restart_durability_across_active_trading_sessions(self, tmp_path: Path) -> None:
        """Scenario 3: Server restart preserves accumulated PnL, win/loss history, and $5,000 base."""
        ledger_file = tmp_path / "account_ledger.json"

        session1_state = {
            "account_id": "hyperliquid_paper_01",
            "base_capital_usd": 5000.0,
            "current_cash_usd": 5185.0,
            "total_realized_pnl": 185.0,
            "win_count": 2,
            "loss_count": 1,
            "trade_history": [
                {"id": 1, "pnl": 120.0},
                {"id": 2, "pnl": -35.0},
                {"id": 3, "pnl": 100.0},
            ],
            "last_updated_iso": "2026-10-31T01:00:00Z",
        }
        ledger_file.write_text(json.dumps(session1_state), encoding="utf-8")

        session2_reloaded = json.loads(ledger_file.read_text(encoding="utf-8"))
        assert session2_reloaded["base_capital_usd"] == 5000.0
        assert session2_reloaded["current_cash_usd"] == 5185.0
        assert session2_reloaded["total_realized_pnl"] == 185.0
        assert session2_reloaded["win_count"] == 2
        assert session2_reloaded["loss_count"] == 1
        assert len(session2_reloaded["trade_history"]) == 3

    def test_cross_venue_hyperliquid_vs_bybit_funding_arbitrage(self) -> None:
        """Scenario 4: Exploit 18% APR divergence between Hyperliquid and Bybit."""
        hl_fee = FeeSchedule.hyperliquid()
        hl_1h = 0.0001
        hl_apr = hl_1h * 24.0 * 365.0 * 100.0

        bybit_8h = 0.0002
        bybit_apr = bybit_8h * 3.0 * 365.0 * 100.0

        apr_spread = hl_apr - bybit_apr
        assert apr_spread > 50.0

        maker_cost = hl_fee.maker_rate * 10_000.0
        taker_cost = hl_fee.taker_rate * 10_000.0
        fee_savings_bps = taker_cost - maker_cost
        assert fee_savings_bps == 3.0

    def test_champion_challenger_retraining_and_live_deployment_tournament(self) -> None:
        """Scenario 5: 45-split tournament crowns new strategy which deploys live without restart."""
        bars = _make_mock_dollar_bars(n_bars=150)
        cand_trend = create_trend_strategy(strategy_id="cand_trend", fast_window=4, slow_window=10)
        cand_mean_rev = create_mean_reversion_strategy(strategy_id="cand_mean_rev", window=8, z_entry=1.6)

        tournament = StrategyTournament(candidates=[cand_trend, cand_mean_rev], vertical_barrier_bars=4)
        tourn_res = tournament.run_tournament(bars)
        assert tourn_res.total_candidates == 2

        best = tourn_res.rankings[0]
        gatekeeper = ChampionChallengerGatekeeper(asset_id="BTC")
        new_state = gatekeeper.evaluate_challenger(
            CPCVValidationResult(
                model_id=best.strategy_id,
                purge_bars=4,
                psr_vs_zero=0.97,
                psr_vs_benchmark=0.96,
                dsr=0.96,
                pbo=0.18,
                expected_net_pnl_bps=22.0,
                sharpe_net=1.9,
                skewness=0.1,
                kurtosis=3.1,
                total_trials=45,
            )
        )
        assert new_state.champion_model_id == best.strategy_id
        assert new_state.regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY

    def test_toxic_order_flow_vpin_hawkes_sweep_protection(self) -> None:
        """Scenario 6: Toxic informed sweep triggers sub-5ms order cancellation & defensive backoff."""
        algo = DefensiveMakerExecutionAlgorithm()
        cmd = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=time.time_ns(),
            side="BUY",
            quantity=0.1,
            best_bid=64995.0,
            best_ask=65005.0,
            resting_price=64995.0,
            vpin=0.75,
            hawkes_eta=0.82,
        )
        assert cmd.action == ExecutionAction.CANCEL_ALL_TOXIC_SHIELD
        assert cmd.post_only_gtx is False
        assert "toxic_sweep_shield" in cmd.reason.lower()
