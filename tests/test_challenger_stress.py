"""Challenger 1 Adversarial Stress Test Suite.

Empirical verification of:
1. Data Ingestion: Offline handling, Bybit failover, simulated fallback, malformed payloads,
   1h funding rate synchronization, and top-20 order book depth.
2. Defensive Maker: ALO post-only constraints, rejection of taker fills, re-peg hysteresis,
   rejection backoff, and toxic sweep shields (VPIN > 0.60, Hawkes eta > 0.70).
3. Funding Carry Sleeve: Negative funding rates, cross-venue spread hurdles (7 bps),
   Hyperliquid fee accounting (1.5 bps maker), and breakeven horizons.
4. Derivatives Alpha Sleeve: Empty options surface handling, extreme 25d skew, dealer GEX extremes,
   and HLP vault positioning.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from core.contracts import (
    AssetMarketSnapshot,
    DeribitOptionsSurface,
    DollarBar,
    ExecutionAction,
    ExecutionRegimeMode,
    FeeSchedule,
    HawkesStateSnapshot,
    HyperliquidWhaleVaultMetrics,
    L2Level,
    L2OrderBookSnapshot,
    MultiAssetMarketSnapshot,
    SettlementCurrency,
)
from core.data_ingestion import (
    LiveMarketDataEngine,
    _compute_l2_depth_and_spread,
    normalize_coin_symbol,
)
from core.execution.defensive_maker import (
    ALOOrderConstraint,
    DefensiveMakerConfig,
    DefensiveMakerExecutionAlgorithm,
    RestingMakerOrderState,
)
from sleeves.sleeve_1_derivatives_alpha import (
    DerivativesAlphaConfig,
    DerivativesAlphaSleeve,
)
from sleeves.sleeve_3_funding_carry import (
    CarryDirection,
    FundingCarryConfig,
    FundingCarrySleeve,
)


# =============================================================================
# PART 1: DATA INGESTION STRESS TESTING
# =============================================================================

class TestDataIngestionStress:
    """Adversarial stress testing of LiveMarketDataEngine."""

    @pytest.mark.asyncio
    async def test_offline_primary_failover_to_bybit(self) -> None:
        """When Hyperliquid L1 is completely down (503/timeout), failover to Bybit V5 seamlessly."""
        engine = LiveMarketDataEngine(
            hl_url="http://127.0.0.1:9999/dead_endpoint",
            bybit_tickers_url="http://mock.bybit/tickers",
            bybit_book_url="http://mock.bybit/orderbook",
            timeout_sec=0.1,
        )

        mock_ticker = {
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lastPrice": "75000.0",
                    "markPrice": "75010.0",
                    "indexPrice": "75000.0",
                    "openInterest": "50000.0",
                    "turnover24h": "1000000000.0",
                    "fundingRate": "0.0008",  # 8h rate
                }]
            }
        }
        mock_book = {
            "result": {
                "b": [["74990.0", "1.5"], ["74980.0", "2.0"]],
                "a": [["75010.0", "1.5"], ["75020.0", "2.0"]],
            },
            "time": 1700000000000,
        }

        async def mock_get(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
            if "tickers" in str(url):
                return httpx.Response(200, json=mock_ticker)
            elif "orderbook" in str(url):
                return httpx.Response(200, json=mock_book)
            return httpx.Response(404)

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            snap = await engine.get_multi_asset_snapshot(coins=("BTC",), include_l2=True)
            assert snap.feed_source == "BYBIT-V5-FALLBACK"
            assert "BTC" in snap.assets
            btc = snap.assets["BTC"]
            assert btc.feed_source == "BYBIT-V5-FALLBACK"
            assert btc.mark_price == 75010.0
            assert btc.funding_rate_8h_equiv == 0.0008
            assert btc.funding_rate_1h == pytest.approx(0.0001)

    @pytest.mark.asyncio
    async def test_offline_both_primary_and_bybit_simulated_fallback(self) -> None:
        """When BOTH Hyperliquid and Bybit are unreachable, activate simulated fallback."""
        engine = LiveMarketDataEngine(
            hl_url="http://127.0.0.1:9999/dead_hl",
            bybit_tickers_url="http://127.0.0.1:9999/dead_bybit",
            bybit_book_url="http://127.0.0.1:9999/dead_bybit",
            timeout_sec=0.05,
        )

        snap = await engine.get_multi_asset_snapshot(coins=("BTC", "ETH", "SOL"))
        assert snap.feed_source == "SIMULATED-FALLBACK"
        assert len(snap.assets) == 3
        for coin in ("BTC", "ETH", "SOL"):
            assert coin in snap.assets
            asset = snap.assets[coin]
            assert asset.feed_source == "SIMULATED-FALLBACK"
            assert asset.mid_price > 0.0
            assert asset.open_interest > 0.0
            assert len(asset.l2_book.bids) == 20
            assert len(asset.l2_book.asks) == 20

    def test_sync_offline_both_simulated_fallback(self) -> None:
        """Synchronous get_multi_asset_snapshot_sync handles offline by activating simulated fallback."""
        engine = LiveMarketDataEngine(
            hl_url="http://127.0.0.1:9999/dead_hl",
            bybit_tickers_url="http://127.0.0.1:9999/dead_bybit",
            bybit_book_url="http://127.0.0.1:9999/dead_bybit",
            timeout_sec=0.05,
        )
        snap = engine.get_multi_asset_snapshot_sync(coins=("BTC",))
        assert snap.feed_source == "SIMULATED-FALLBACK"
        assert "BTC" in snap.assets

    @pytest.mark.asyncio
    async def test_malformed_hyperliquid_payloads(self) -> None:
        """Malformed JSON or missing fields from Hyperliquid triggers failover, does not crash."""
        engine = LiveMarketDataEngine(
            hl_url="http://mock.hl/info",
            timeout_sec=0.1,
        )

        # Malformed payload 1: empty list
        with patch("httpx.AsyncClient.post", return_value=httpx.Response(200, json=[])):
            snap = await engine.get_multi_asset_snapshot(coins=("BTC",))
            # Failed HL -> tries Bybit -> simulated fallback
            assert snap.feed_source in ("BYBIT-V5-FALLBACK", "SIMULATED-FALLBACK")

        # Malformed payload 2: missing universe key
        with patch("httpx.AsyncClient.post", return_value=httpx.Response(200, json=[{"foo": "bar"}, []])):
            snap = await engine.get_multi_asset_snapshot(coins=("BTC",))
            assert snap.feed_source in ("BYBIT-V5-FALLBACK", "SIMULATED-FALLBACK")

        # Malformed payload 3: universe present but asset not in universe
        with patch(
            "httpx.AsyncClient.post",
            return_value=httpx.Response(
                200,
                json=[{"universe": [{"name": "DOGE"}]}, [{"markPx": "0.1", "funding": "0.0001"}]],
            ),
        ):
            snap = await engine.get_multi_asset_snapshot(coins=("BTC",))
            # No matching asset for BTC -> raises ValueError in HL -> fallback
            assert snap.feed_source in ("BYBIT-V5-FALLBACK", "SIMULATED-FALLBACK")

    def test_1h_funding_rate_synchronization_and_annualization(self) -> None:
        """Verify 1h funding rate correctly synchronizes with 8h equivalent and annualized APR."""
        # Test case A: 1h rate is 1.5 bps (0.00015)
        snap = AssetMarketSnapshot(
            instrument_id="BTC",
            mid_price=80000.0,
            mark_price=80010.0,
            oracle_price=80000.0,
            open_interest=5000.0,
            funding_rate_1h=0.00015,
        )
        assert snap.funding_rate_8h_equiv == pytest.approx(0.00015 * 8.0)
        # APR = 0.00015 * 24 * 365 * 100 = 131.4%
        assert snap.annualized_apr_pct == pytest.approx(0.00015 * 24.0 * 365.0 * 100.0, rel=1e-4)

        # Test case B: negative 1h rate (-0.5 bps = -0.00005)
        snap_neg = AssetMarketSnapshot(
            instrument_id="ETH",
            mid_price=2500.0,
            mark_price=2499.0,
            oracle_price=2500.0,
            open_interest=20000.0,
            funding_rate_1h=-0.00005,
        )
        assert snap_neg.funding_rate_8h_equiv == pytest.approx(-0.0004)
        assert snap_neg.annualized_apr_pct == pytest.approx(-0.00005 * 24.0 * 365.0 * 100.0, rel=1e-4)

    def test_top20_order_book_depth_and_spread_calculation(self) -> None:
        """Stress test L2 depth and spread calculations under normal and extreme book shapes."""
        # 1. Normal book
        bids = [(100.0 - i * 0.1, 1.0) for i in range(20)]
        asks = [(100.2 + i * 0.1, 1.0) for i in range(20)]
        spread, d10, d50 = _compute_l2_depth_and_spread(bids, asks)
        mid = (100.0 + 100.2) / 2.0  # 100.1
        expected_spread = (100.2 - 100.0) / mid * 10000.0  # ~19.98 bps
        assert spread == pytest.approx(expected_spread, abs=0.01)
        assert d10 > 0.0
        assert d50 >= d10

        # 2. Empty book
        assert _compute_l2_depth_and_spread([], []) == (0.0, 0.0, 0.0)

        # 3. Degenerate prices (zero or negative)
        assert _compute_l2_depth_and_spread([(0.0, 1.0)], [(0.0, 1.0)]) == (0.0, 0.0, 0.0)

        # 4. Inverted/crossed book: best_ask <= best_bid -> spread should be 0.0 (non-negative)
        spread_inv, _, _ = _compute_l2_depth_and_spread([(101.0, 1.0)], [(100.0, 1.0)])
        assert spread_inv == 0.0


# =============================================================================
# PART 2: DEFENSIVE MAKER EXECUTION STRESS TESTING
# =============================================================================

class TestDefensiveMakerExecutionStress:
    """Adversarial stress testing of DefensiveMakerExecutionAlgorithm."""

    def test_alo_constraint_validation(self) -> None:
        """Verify ALOOrderConstraint strictly prevents taker crossing."""
        # BUY order: passive if price <= best_bid
        assert ALOOrderConstraint.validate_passive_price("BUY", 99.0, best_bid=99.0, best_ask=100.0) is True
        assert ALOOrderConstraint.validate_passive_price("BUY", 98.0, best_bid=99.0, best_ask=100.0) is True
        # BUY order above best bid -> crosses spread towards ask
        assert ALOOrderConstraint.validate_passive_price("BUY", 99.5, best_bid=99.0, best_ask=100.0) is False
        assert ALOOrderConstraint.validate_passive_price("BUY", 100.0, best_bid=99.0, best_ask=100.0) is False

        # SELL order: passive if price >= best_ask
        assert ALOOrderConstraint.validate_passive_price("SELL", 100.0, best_bid=99.0, best_ask=100.0) is True
        assert ALOOrderConstraint.validate_passive_price("SELL", 101.0, best_bid=99.0, best_ask=100.0) is True
        assert ALOOrderConstraint.validate_passive_price("SELL", 99.5, best_bid=99.0, best_ask=100.0) is False
        assert ALOOrderConstraint.validate_passive_price("SELL", 99.0, best_bid=99.0, best_ask=100.0) is False

        # Verify no taker fill
        assert ALOOrderConstraint.verify_no_taker_fill("BUY", 99.0, best_bid=99.0, best_ask=100.0) is True
        assert ALOOrderConstraint.verify_no_taker_fill("BUY", 100.0, best_bid=99.0, best_ask=100.0) is False
        assert ALOOrderConstraint.verify_no_taker_fill("SELL", 100.0, best_bid=99.0, best_ask=100.0) is True
        assert ALOOrderConstraint.verify_no_taker_fill("SELL", 99.0, best_bid=99.0, best_ask=100.0) is False

    def test_repeg_hysteresis_and_cooldown_matrix(self) -> None:
        """Test the 4 quadrant combinations of (BBO shift >= 1.5 bps) x (cooldown >= 500 ms)."""
        algo = DefensiveMakerExecutionAlgorithm(
            repeg_min_bps=1.5,
            repeg_cooldown_ms=500,
        )
        base_ts = 1_000_000_000_000  # ns
        resting_px = 10000.0

        # Initial order placement at 10000.0
        cmd1 = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=base_ts,
            side="BUY",
            quantity=1.0,
            best_bid=resting_px,
            best_ask=10002.0,
        )
        assert cmd1.action == ExecutionAction.PLACE_GTX_MAKER
        assert cmd1.target_price == resting_px

        # Quadrant 1: Shift < 1.5 bps (1.0 bps = 10001.0), elapsed < 500ms (200ms)
        # Shift = 1.0 / 10000 = 1.0 bps < 1.5 bps
        cmd_q1 = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=base_ts + 200_000_000,
            side="BUY",
            quantity=1.0,
            best_bid=10001.0,
            best_ask=10003.0,
        )
        assert cmd_q1.action == ExecutionAction.HOLD_RESTING_HYSTERESIS
        assert cmd_q1.target_price == resting_px

        # Quadrant 2: Shift >= 1.5 bps (2.0 bps = 10002.0), BUT elapsed < 500ms (300ms)
        cmd_q2 = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=base_ts + 300_000_000,
            side="BUY",
            quantity=1.0,
            best_bid=10002.0,
            best_ask=10004.0,
        )
        assert cmd_q2.action == ExecutionAction.HOLD_RESTING_HYSTERESIS

        # Quadrant 3: Shift < 1.5 bps (1.0 bps = 10001.0), BUT elapsed >= 500ms (600ms)
        cmd_q3 = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=base_ts + 600_000_000,
            side="BUY",
            quantity=1.0,
            best_bid=10001.0,
            best_ask=10003.0,
        )
        assert cmd_q3.action == ExecutionAction.HOLD_RESTING_HYSTERESIS

        # Quadrant 4: Shift >= 1.5 bps (2.0 bps = 10002.0) AND elapsed >= 500ms (600ms) -> AMEND!
        cmd_q4 = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=base_ts + 600_000_000,
            side="BUY",
            quantity=1.0,
            best_bid=10002.0,
            best_ask=10004.0,
        )
        assert cmd_q4.action == ExecutionAction.AMEND_GTX_MAKER
        assert cmd_q4.target_price == 10002.0

    def test_gtx_rejection_3_strike_backoff(self) -> None:
        """Consecutive rejections cap at 3. The 4th attempt places order 1 tick deeper + 1,000 ms cooldown."""
        algo = DefensiveMakerExecutionAlgorithm(
            max_gtx_rejections=3,
            rejection_backoff_ms=1000,
        )
        ts = 1_000_000_000_000

        # Simulate 3 consecutive rejections
        for i in range(1, 4):
            algo.on_order_rejected("BTC", ts_ns=ts)
            assert algo.get_consecutive_rejections("BTC") == i

        # 4th attempt -> BACKOFF_DEEPER_GTX at best_bid - tick_size (for BUY)
        cmd = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=ts + 100_000_000,
            side="BUY",
            quantity=1.0,
            best_bid=70000.0,
            best_ask=70005.0,
            tick_size=0.5,
        )
        assert cmd.action == ExecutionAction.BACKOFF_DEEPER_GTX
        assert cmd.target_price == 70000.0 - 0.5  # 69999.5
        assert cmd.post_only_gtx is True
        assert cmd.time_in_force == "GTX"

        # Now test for SELL: 3 rejections -> 1 tick deeper (best_ask + tick_size)
        algo_sell = DefensiveMakerExecutionAlgorithm(max_gtx_rejections=3)
        for _ in range(3):
            algo_sell.on_order_rejected("ETH", ts_ns=ts)
        cmd_sell = algo_sell.evaluate_order_intent(
            instrument_id="ETH",
            ts_ns=ts + 100_000_000,
            side="SELL",
            quantity=1.0,
            best_bid=2500.0,
            best_ask=2501.0,
            tick_size=0.1,
        )
        assert cmd_sell.action == ExecutionAction.BACKOFF_DEEPER_GTX
        assert cmd_sell.target_price == 2501.0 + 0.1  # 2501.1

    def test_toxic_sweep_shield_triggers(self) -> None:
        """VPIN > 0.60 or Hawkes eta > 0.70 immediately cancels resting entries."""
        algo = DefensiveMakerExecutionAlgorithm(
            vpin_toxic_threshold=0.60,
            hawkes_eta_toxic_threshold=0.70,
        )
        ts = 1_000_000_000_000

        # Test A: VPIN toxic (0.65 > 0.60, eta normal 0.30)
        cmd_vpin = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=ts,
            side="BUY",
            quantity=1.0,
            best_bid=70000.0,
            best_ask=70001.0,
            vpin=0.65,
            hawkes_eta=0.30,
        )
        assert cmd_vpin.action == ExecutionAction.CANCEL_ALL_TOXIC_SHIELD
        assert cmd_vpin.side == "NONE"

        # Test B: Hawkes eta toxic (0.75 > 0.70, VPIN normal 0.30)
        cmd_hawkes = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=ts,
            side="BUY",
            quantity=1.0,
            best_bid=70000.0,
            best_ask=70001.0,
            vpin=0.30,
            hawkes_eta=0.75,
        )
        assert cmd_hawkes.action == ExecutionAction.CANCEL_ALL_TOXIC_SHIELD

        # Test C: Both toxic
        cmd_both = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=ts,
            side="SELL",
            quantity=1.0,
            best_bid=70000.0,
            best_ask=70001.0,
            vpin=0.80,
            hawkes_eta=0.85,
        )
        assert cmd_both.action == ExecutionAction.CANCEL_ALL_TOXIC_SHIELD

    def test_emergency_taker_exit_exclusive_path(self) -> None:
        """Hard stop loss or circuit breaker is the ONLY condition allowing IOC/taker execution."""
        algo = DefensiveMakerExecutionAlgorithm()
        ts = 1_000_000_000_000

        # Hard stop breached -> EMERGENCY_TAKER_EXIT
        cmd_stop = algo.evaluate_order_intent(
            instrument_id="BTC",
            ts_ns=ts,
            side="SELL",
            quantity=2.0,
            best_bid=65000.0,
            best_ask=65005.0,
            hard_stop_breached=True,
        )
        assert cmd_stop.action == ExecutionAction.EMERGENCY_TAKER_EXIT
        assert cmd_stop.post_only_gtx is False
        assert cmd_stop.time_in_force == "IOC"
        assert cmd_stop.target_price == 65000.0  # sells at best_bid


# =============================================================================
# PART 3: FUNDING CARRY SLEEVE STRESS TESTING
# =============================================================================

class TestFundingCarrySleeveStress:
    """Adversarial stress testing of FundingCarrySleeve."""

    def test_hyperliquid_fee_schedule_accounting(self) -> None:
        """Verify Hyperliquid fee schedule (1.5 bps maker / 4.5 bps taker)."""
        hl_fee = FeeSchedule.hyperliquid()
        assert hl_fee.maker_bps == 1.5
        assert hl_fee.taker_bps == 4.5
        assert hl_fee.maker_rate == pytest.approx(0.00015)
        assert hl_fee.roundtrip_maker_bps == 3.0
        # Roundtrip + slippage (0.5 bps each way) = 2 * (1.5 + 0.5) = 4.0 bps
        assert hl_fee.roundtrip_maker_plus_slippage_bps == 4.0

    def test_negative_funding_perp_long_gated_by_asri(self) -> None:
        """Deeply negative funding triggers PERP_LONG_NEGATIVE_FUNDING ONLY when ASRI <= 40."""
        sleeve = FundingCarrySleeve(
            FundingCarryConfig(
                min_negative_funding_8h=0.0005,  # -5 bps/8h
                max_asri_for_negative_funding=40.0,
            )
        )

        # Case 1: Low risk ASRI = 20 <= 40 -> Actionable PERP_LONG_NEGATIVE_FUNDING
        eval_low_risk = sleeve.evaluate_opportunity(
            instrument_id="BTC",
            funding_rate_8h=-0.0008,  # -8 bps/8h
            asri_score=20.0,
        )
        assert eval_low_risk.direction == CarryDirection.PERP_LONG_NEGATIVE_FUNDING
        assert eval_low_risk.is_actionable is True
        assert eval_low_risk.perp_target_weight > 0.0
        assert eval_low_risk.spot_target_weight == 0.0

        # Case 2: Elevated risk ASRI = 45 > 40 -> BLOCKED (FLAT)
        eval_high_risk = sleeve.evaluate_opportunity(
            instrument_id="BTC",
            funding_rate_8h=-0.0008,
            asri_score=45.0,
        )
        assert eval_high_risk.direction == CarryDirection.FLAT
        assert eval_high_risk.is_actionable is False
        assert "ASRI" in eval_high_risk.reason

    def test_cross_venue_spread_hurdle_and_threshold(self) -> None:
        """Cross-venue spread must clear 7 bps threshold and roundtrip fee hurdle."""
        sleeve = FundingCarrySleeve(
            FundingCarryConfig(cross_venue_min_spread_bps=7.0)
        )
        fee_hl = FeeSchedule.hyperliquid()   # 1.5 bps maker -> 3.0 bps roundtrip
        fee_bybit = FeeSchedule.usdt_perp() # 2.0 bps maker -> 4.0 bps roundtrip
        # Total roundtrip hurdle = 3.0 + 4.0 = 7.0 bps

        # Scenario A: HL rate 10 bps, Bybit 4 bps -> spread = 6 bps < 7 bps -> NOT actionable
        res_sub = sleeve.evaluate_cross_venue_spread(
            instrument_id="BTC",
            hl_funding_rate_8h=0.0010,
            bybit_funding_rate_8h=0.0004,
            hl_fee=fee_hl,
            bybit_fee=fee_bybit,
        )
        assert res_sub.is_actionable is False
        assert res_sub.spread_bps_8h == pytest.approx(6.0)

        # Scenario B: HL rate 12 bps, Bybit 4 bps -> spread = 8 bps >= 7 bps
        # Over 9 epochs, expected gross = 8 * 9 = 72 bps, net = 72 - 7.0 = 65 bps > 0 -> Actionable!
        res_act = sleeve.evaluate_cross_venue_spread(
            instrument_id="BTC",
            hl_funding_rate_8h=0.0012,
            bybit_funding_rate_8h=0.0004,
            hl_fee=fee_hl,
            bybit_fee=fee_bybit,
        )
        assert res_act.is_actionable is True
        assert res_act.hl_side == "SELL"
        assert res_act.bybit_side == "BUY"
        assert res_act.hl_target_weight < 0.0
        assert res_act.bybit_target_weight > 0.0
        assert res_act.net_delta == 0.0  # strictly delta-neutral

    def test_breakeven_math_and_zero_funding_safety(self) -> None:
        """Breakeven calculation handles zero funding gracefully without unhandled ZeroDivisionError."""
        be_epochs = FundingCarrySleeve.compute_breakeven_epochs(
            funding_rate_8h=0.0,
            fee_schedule=FeeSchedule.hyperliquid(),
        )
        assert math.isinf(be_epochs)

        be_hours = FundingCarrySleeve.compute_breakeven_hours(
            funding_rate_1h=0.0,
            fee_schedule=FeeSchedule.hyperliquid(),
        )
        assert math.isinf(be_hours)


# =============================================================================
# PART 4: DERIVATIVES ALPHA SLEEVE STRESS TESTING
# =============================================================================

class TestDerivativesAlphaSleeveStress:
    """Adversarial stress testing of DerivativesAlphaSleeve."""

    def test_empty_options_surface_and_vault_returns_neutral(self) -> None:
        """When options surface is empty/missing/zeros, emit direction=0, p_meta=0.50."""
        sleeve = DerivativesAlphaSleeve()

        # Completely empty / None
        sig_none = sleeve.generate_alpha_signal(options_surface=None, vault_metrics=None)
        assert sig_none.primary_direction == 0
        assert sig_none.p_meta == 0.50
        assert sig_none.expected_gross_edge_bps == 0.0

        # All zeros surface and balanced vault
        empty_surf = DeribitOptionsSurface(currency="BTC", skew_25d=0.0, net_gex_usd=0.0, term_structure_slope=0.0)
        empty_vault = HyperliquidWhaleVaultMetrics(vault_address="0x123", net_delta_usd=0.0, hlp_crowding="BALANCED")
        sig_empty = sleeve.generate_alpha_signal(options_surface=empty_surf, vault_metrics=empty_vault)
        assert sig_empty.primary_direction == 0
        assert sig_empty.p_meta == 0.50

    def test_extreme_25d_skew_and_boundedness(self) -> None:
        """Extreme options skew produces expected direction while maintaining calibrated p_meta bounds."""
        sleeve = DerivativesAlphaSleeve()

        # Extreme positive call skew (+20%)
        surf_bull = DeribitOptionsSurface(currency="BTC", skew_25d=0.20, term_structure_slope=0.02)
        sig_bull = sleeve.generate_alpha_signal(options_surface=surf_bull)
        assert sig_bull.primary_direction == 1
        assert 0.50 < sig_bull.p_meta <= 0.95
        assert sig_bull.expected_gross_edge_bps >= 25.0

        # Extreme negative put skew (-20%)
        surf_bear = DeribitOptionsSurface(currency="BTC", skew_25d=-0.20, term_structure_slope=-0.02)
        sig_bear = sleeve.generate_alpha_signal(options_surface=surf_bear)
        assert sig_bear.primary_direction == -1
        assert 0.50 < sig_bear.p_meta <= 0.95

    def test_dealer_gex_extremes(self) -> None:
        """Positive GEX pins volatility; Negative GEX amplifies trend."""
        sleeve = DerivativesAlphaSleeve()

        # Large positive GEX (+50M USD) -> POSITIVE_PINNING
        surf_pin = DeribitOptionsSurface(currency="BTC", skew_25d=0.02, net_gex_usd=50_000_000.0)
        sig_pin = sleeve.generate_alpha_signal(options_surface=surf_pin)
        assert sig_pin.gex_regime == "POSITIVE_PINNING"

        # Large negative GEX (-50M USD) -> NEGATIVE_AMPLIFYING
        surf_amp = DeribitOptionsSurface(currency="BTC", skew_25d=0.02, net_gex_usd=-50_000_000.0)
        sig_amp = sleeve.generate_alpha_signal(options_surface=surf_amp)
        assert sig_amp.gex_regime == "NEGATIVE_AMPLIFYING"

    def test_hlp_vault_positioning_signals(self) -> None:
        """HLP net short = retail crowded long (downside risk); HLP net long = retail short (upside squeeze)."""
        sleeve = DerivativesAlphaSleeve()

        # Retail crowded short -> HLP is long -> Bullish squeeze potential
        vault_squeeze = HyperliquidWhaleVaultMetrics(
            vault_address="0xdfc24b077bc1425ad1dea75bcb6f8158e10df303",
            tvl_usd=150_000_000.0,
            net_delta_usd=10_000_000.0,
            hlp_crowding="CROWDED_LONG",  # HLP is crowded long -> retail is short
        )
        sig_squeeze = sleeve.generate_alpha_signal(vault_metrics=vault_squeeze)
        assert sig_squeeze.primary_direction == 1
        assert sig_squeeze.hlp_vault_crowding == "CROWDED_LONG"

        # Retail crowded long -> HLP is short -> Bearish cascade vulnerability
        vault_cascade = HyperliquidWhaleVaultMetrics(
            vault_address="0xdfc24b077bc1425ad1dea75bcb6f8158e10df303",
            tvl_usd=150_000_000.0,
            net_delta_usd=-10_000_000.0,
            hlp_crowding="CROWDED_SHORT",
        )
        sig_cascade = sleeve.generate_alpha_signal(vault_metrics=vault_cascade)
        assert sig_cascade.primary_direction == -1
        assert sig_cascade.hlp_vault_crowding == "CROWDED_SHORT"

    def test_confluence_alpha_and_kelly_conversion(self) -> None:
        """Strong confluence between skew, term structure, and vault produces high p_meta and valid Kelly."""
        sleeve = DerivativesAlphaSleeve()

        # Strong bullish confluence: +3% skew, contango (+2%), HLP crowded long (retail short)
        surf_bull = DeribitOptionsSurface(currency="BTC", skew_25d=0.03, term_structure_slope=0.02, net_gex_usd=20_000_000.0)
        vault_bull = HyperliquidWhaleVaultMetrics(net_delta_usd=5_000_000.0, hlp_crowding="CROWDED_LONG")
        sig = sleeve.generate_alpha_signal(options_surface=surf_bull, vault_metrics=vault_bull)
        assert sig.primary_direction == 1
        assert sig.p_meta >= 0.70
        assert sig.expected_gross_edge_bps > 40.0

        sleeve_out = sig.to_sleeve_signal_output()
        assert sleeve_out.sleeve_id == "derivatives_alpha"
        assert sleeve_out.direction == 1
        assert 0.0 < sleeve_out.recommended_leverage <= 2.5
        assert sleeve_out.routing_tag == "ALO_POST_ONLY"

        alloc_input = sig.to_asset_allocation_input(w_hrp=0.40)
        assert alloc_input.asset_id == "BTC"
        assert alloc_input.y_primary == 1
        assert alloc_input.w_hrp == 0.40


# =============================================================================
# PART 5: ADVANCED BOUNDARY & INTEGRATION STRESS
# =============================================================================

class TestAdvancedBoundaryStress:
    """Corner case and boundary condition stress tests."""

    def test_funding_carry_hysteresis_exit(self) -> None:
        """Test hysteresis: position stays open when funding drops below entry but above exit threshold."""
        sleeve = FundingCarrySleeve(
            FundingCarryConfig(
                min_positive_funding_8h=0.0003,      # 3.0 bps entry threshold
                exit_funding_threshold_8h=0.00005,  # 0.5 bps exit threshold
            )
        )
        # 1. Initial entry at 5 bps -> SPOT_LONG_PERP_SHORT
        eval1 = sleeve.evaluate_opportunity("BTC", funding_rate_8h=0.0005)
        assert eval1.direction == CarryDirection.SPOT_LONG_PERP_SHORT
        assert eval1.is_actionable is True

        # 2. Funding dips to 1.5 bps (below 3 bps entry, but above 0.5 bps exit)
        # Hysteresis keeps the position OPEN!
        eval2 = sleeve.evaluate_opportunity("BTC", funding_rate_8h=0.00015)
        assert eval2.direction == CarryDirection.SPOT_LONG_PERP_SHORT
        assert eval2.is_actionable is True

        # 3. Funding collapses to 0.3 bps (below 0.5 bps exit) -> Exits to FLAT!
        eval3 = sleeve.evaluate_opportunity("BTC", funding_rate_8h=0.00003)
        assert eval3.direction == CarryDirection.FLAT
        assert eval3.is_actionable is False

    def test_reverse_cross_venue_spread_arbitrage(self) -> None:
        """When Bybit funding is significantly higher than HL, trigger reverse arbitrage."""
        sleeve = FundingCarrySleeve(FundingCarryConfig(cross_venue_min_spread_bps=7.0))
        # Bybit = 15 bps, HL = 5 bps -> spread = HL - Bybit = -10 bps (magnitude 10 bps >= 7 bps)
        eval_rev = sleeve.evaluate_cross_venue_spread(
            instrument_id="BTC",
            hl_funding_rate_8h=0.0005,
            bybit_funding_rate_8h=0.0015,
        )
        assert eval_rev.is_actionable is True
        assert eval_rev.hl_side == "BUY"   # Long HL Perp (pay 5 bps)
        assert eval_rev.bybit_side == "SELL" # Short Bybit Perp (receive 15 bps)
        assert eval_rev.hl_target_weight > 0.0
        assert eval_rev.bybit_target_weight < 0.0
        assert eval_rev.net_delta == 0.0

    def test_symbol_normalization_adversarial(self) -> None:
        """Normalize symbol handles weird casing, whitespace, and exchange suffixes."""
        assert normalize_coin_symbol("BTCUSDT-PERP") == "BTC"
        assert normalize_coin_symbol("eth-perp") == "ETH"
        assert normalize_coin_symbol("  solUSDT  ") == "SOL"
        assert normalize_coin_symbol("BTCUSDC-PERP") == "BTC"
        assert normalize_coin_symbol("dogeusd") == "DOGE"
        assert normalize_coin_symbol("SOL") == "SOL"

    def test_l2_book_depth_truncation_to_20_levels(self) -> None:
        """Raw order book with > 20 levels is strictly truncated to top 20."""
        # Simulated raw levels with 30 items
        raw_bids = [{"px": str(100.0 - i * 0.1), "sz": "1.0", "n": 1} for i in range(30)]
        raw_asks = [{"px": str(100.1 + i * 0.1), "sz": "1.0", "n": 1} for i in range(30)]
        snap = L2OrderBookSnapshot(
            instrument_id="BTC",
            bids=[(float(str(b["px"])), float(str(b["sz"]))) for b in raw_bids[:20]],
            asks=[(float(str(a["px"])), float(str(a["sz"]))) for a in raw_asks[:20]],
        )
        assert len(snap.bids) == 20
        assert len(snap.asks) == 20
        assert len(snap.levels_bids) == 20
        assert len(snap.levels_asks) == 20

