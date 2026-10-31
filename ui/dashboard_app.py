"""Interactive Web UI, Plain-English Tooltip Registry, and Plotly Chart Builders.

Provides the layman-accessible visual layer for the Gemini Argon 4 quantitative
derivatives engine on Apple M1 Max (64GB), including:
1. ``PLAIN_ENGLISH_TOOLTIPS`` covering every metric badge, chart overlay, and control.
2. Interactive Plotly chart builders (``build_candlestick_dollar_bar_figure``,
   ``build_portfolio_weights_figure``, ``build_validation_gates_figure``).
3. Complete 3-tab HTML Control Center renderer (``render_full_dashboard_html``).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Literal, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import (
    AssetAllocationInput,
    ChampionChallengerState,
    CPCVValidationResult,
    DollarBar,
    ExecutionAction,
    ExecutionOrderCommand,
    ExecutionRegimeMode,
    FeeSchedule,
    FFDWeightsConfig,
    HawkesStateSnapshot,
    MetaPredictionOutput,
    PortfolioAllocationOutput,
    PositioningRegime,
    PositioningSignalSnapshot,
    SystemicRiskOutput,
    TripleBarrierConfig,
    TripleBarrierLabel,
    compute_regime_multiplier_from_asri,
    compute_signed_target_weights,
)

DOCS_DIR: Path = Path(__file__).resolve().parent.parent / "docs"

DOC_FILENAMES: dict[str, str] = {
    "LAYMANS_ARCHITECTURE_GUIDE": "LAYMANS_ARCHITECTURE_GUIDE.md",
    "UI_AND_OPERATIONS_MANUAL": "UI_AND_OPERATIONS_MANUAL.md",
    "PLAIN_ENGLISH_GLOSSARY": "PLAIN_ENGLISH_GLOSSARY.md",
    "TROUBLESHOOTING_AND_SAFETY": "TROUBLESHOOTING_AND_SAFETY.md",
}

PLAIN_ENGLISH_TOOLTIPS: dict[str, str] = {
    "vpin_toxicity": (
        "VPIN Toxicity (Shark Detector, 0.00 to 1.00): Measures whether order flow is calm two-way "
        "retail trading (healthy <= 0.45) or one-sided institutional dumping/sweeping (toxic > 0.60). "
        "When VPIN exceeds 0.60, the bot automatically pulls resting limit orders in under 5ms so you "
        "never catch a falling knife."
    ),
    "hawkes_reflexivity": (
        "O(1) Hawkes Reflexivity (Avalanche Meter, eta 0.00 to 1.00): Measures what fraction of trades "
        "are triggering further trades in a self-reinforcing chain reaction. Healthy calm markets stay "
        "below 0.50; if eta climbs above 0.70, an order-book avalanche is underway and the bot steps aside."
    ),
    "triple_barrier": (
        "Fee-Adjusted Triple-Barrier Bands: Places every trade inside a 3-walled box with a Profit Target "
        "on top (always >= 80 bps and >= 6x round-trip fees + slippage), a Volatility Stop-Loss on the "
        "bottom, and a 16-bar expiration timer on the right. Ensures the bot never chases tiny moves that "
        "lose money to exchange fees."
    ),
    "meta_probability": (
        "LightGBM Meta-Probability (p_meta, 0% to 100%): Our Stage-2 'Sniper Quality Inspector' score "
        "predicting the probability that the primary trade setup will actually hit its profit target "
        "after subtracting all fees and funding. If p_meta is 50% or lower, position size is automatically 0%."
    ),
    "ffd_stationarity": (
        "Fractional Differentiation (FFD Stationarity): Acts like polarized sunglasses for price charts—"
        "removing the raw dollar price level (so $90k BTC is comparable to $30k BTC, p_ADF <= 0.01) while "
        "preserving over 80% of long-term trend memory without looking into the future."
    ),
    "squeeze_regime": (
        "Perpetual Positioning & Squeeze Regime: Combines Open Interest, 8-hour Funding Rates, and "
        "Liquidation spikes to detect whether over-leveraged traders are trapped (e.g. CROWDED_SHORT_SQUEEZE "
        "or CROWDED_LONG_VULNERABILITY) or moving in a healthy GENUINE_TREND_CONVICTION."
    ),
    "asri_macro": (
        "ASRI Macro Weather Score (0 to 100): Generated locally on your M1 Max GPU by the Qwen3-8B-4bit AI "
        "reading global crypto/macro news. Scores <= 40 mean Clear Skies (100% normal sizing); 40-70 tapers "
        "risk down to 50%; >= 95 or a Tail-Risk Flag triggers an immediate 0% directional halt."
    ),
    "mahalanobis_di": (
        "Mahalanobis Distance (DI_values Uncharted Territory Alarm): Measures how unusual today's market "
        "indicators look compared to historical training data. If DI_values rises above the safety threshold, "
        "the bot admits 'I have never seen a market like this' and blocks new trades immediately."
    ),
    "cpcv_gates": (
        "45-Split CPCV Promotion Gates: Before any new AI model ('Challenger') can replace the active model "
        "('Champion'), it must pass 45 time-travel backtest splits with purged/embargoed borders to prove it "
        "did not cheat or memorize noise."
    ),
    "psr_dsr_pbo": (
        "PSR, DSR & PBO Statistical Lie-Detectors: Probabilistic Sharpe Ratio (PSR >= 95%) and Deflated Sharpe "
        "Ratio (DSR >= 95%) confirm profits are statistically real after penalizing for multiple test trials, "
        "while Probability of Backtest Overfitting (PBO < 30%) confirms low curve-fitting risk."
    ),
    "hrp_cvar_weights": (
        "Signed HRP-CVaR Target Weights (w_final,i): Sizes each coin by combining crash-diversified "
        "Hierarchical Risk Parity (w_HRP), Half-Kelly AI conviction (f_Kelly), and the ASRI macro weather "
        "dial (M_regime), strictly capped at 2.5x maximum gross leverage across the portfolio."
    ),
    "funding_carry": (
        "Delta-Neutral Funding Carry Sleeve: Buys Spot coin and shorts an equal amount of Perpetual Futures "
        "so price swings cancel out (zero directional risk), collecting 8-hour funding fee checks paid by "
        "leveraged traders like rent on an apartment."
    ),
    "start_dry_run": (
        "Start Dry-Run (Paper Trading / Full Active Mode): Runs the complete Directional + Carry engine on "
        "live market feeds in safe simulation mode (DIRECTIONAL_PLUS_CARRY) so you can verify execution and "
        "signals without risking real capital."
    ),
    "pause_entries": (
        "Pause Entries: Immediately cancels all resting unfilled entry orders and prevents new trades from "
        "opening (PAUSED_ENTRIES), while keeping protective stop-losses and profit targets active on existing positions."
    ),
    "switch_carry_only": (
        "Switch to Carry-Only: Turns off directional Long/Short predictions and restricts the portfolio to "
        "low-risk Delta-Neutral Funding Carry and Cash (CARRY_ONLY_FALLBACK)."
    ),
    "emergency_flatten": (
        "Emergency Flatten All: Immediate panic button (EMERGENCY_FLATTENED). Cancels every open order and "
        "closes all positions across all coins to 100% cash within seconds."
    ),
    "net_of_fee_pnl": (
        "Live Net-of-Fee PnL: Your true bottom-line profit after subtracting all exchange Maker/Taker fees, "
        "slippage, and 8-hour funding payments."
    ),
    "maker_taker_ratio": (
        "Maker vs. Taker Fill Ratio: Percentage of orders filled passively as low-fee Post-Only GTX Maker "
        "orders (2.0 bps on USDT perps / 0.0 bps on USDC perps) vs. expensive Taker market orders (5.5 bps)."
    ),
    "persistent_ledger": (
        "Persistent Paper Trading Ledger ($5,000 Base Capital): Atomic disk-backed ledger preserving base "
        "capital ($5,000.00), accumulated paper PnL, win/loss stats, and trade audit history across server restarts."
    ),
    "multi_asset_feeds": (
        "Real-Time Multi-Asset Hyperliquid L1 Feeds: Simultaneous sub-second streaming of BTC, ETH, and SOL mark, "
        "oracle index, open interest, and 1-hour funding rates directly from Hyperliquid L1 (with Bybit failover)."
    ),
    "funding_carry_sleeve": (
        "Funding Carry Sleeve: Delta-neutral 1-hour funding carry settlements on Hyperliquid, Add-Liquidity-Only "
        "(ALO) Post-Only Maker routing (1.5 bps maker fee vs 4.5 bps taker avoided), and Bybit/OKX spread arbitrage."
    ),
    "derivatives_alpha_sleeve": (
        "Derivatives Alpha Sleeve: Ingests Deribit options 25-delta skew, term structure slope, dealer net gamma "
        "exposure (GEX), and Hyperliquid HLP whale vault (0xdfc24b...) positioning signals."
    ),
    "risk_committee_models": (
        "Apple Silicon (MLX) 3-Model Risk Committee: Metal GPU accelerated consensus across Macro Sentinel, "
        "Cross-Sectional Breadth Analyst, and Adversarial Red-Team Critic computing portfolio multiplier M_regime."
    ),
    "trade_forensics": (
        "Why Did It Trade? (Trade Forensics): Explains the exact rationale and quantitative feature attribution "
        "behind each trade decision, including regime setup, LightGBM p_meta, Mahalanobis distance, VPIN, Hawkes eta, and Kelly conviction."
    ),
}


class PnLAndExecutionMetrics(BaseModel):
    """Live net-of-fee PnL, starting paper capital, Maker/Taker ratio, and fee-saving telemetry."""

    model_config = ConfigDict(frozen=True)

    starting_capital_usd: float = Field(default=5_000.00)
    net_pnl_usd: float = Field(default=0.00)
    net_pnl_bps: float = Field(default=0.0)
    gross_pnl_usd: float = Field(default=0.00)
    total_fees_paid_usd: float = Field(default=0.00)
    maker_fill_ratio: float = Field(default=0.914, ge=0.0, le=1.0)
    taker_fill_ratio: float = Field(default=0.086, ge=0.0, le=1.0)
    fee_savings_usd: float = Field(default=0.00, ge=0.0)
    fee_savings_bps: float = Field(default=31.5, ge=0.0)
    annualized_carry_apr_pct: float = Field(default=14.8)
    is_dry_run: bool = Field(default=True)

    @property
    def current_equity_usd(self) -> float:
        """Current total paper account equity ($5,000 starting capital + net PnL)."""
        return self.starting_capital_usd + self.net_pnl_usd


class DashboardTelemetryState(BaseModel):
    """Complete snapshot of the Control Center state using core Pydantic v2 contracts."""

    model_config = ConfigDict(frozen=True)

    active_instrument: str = "BTCUSDT-PERP"
    regime_mode: ExecutionRegimeMode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
    is_dry_run: bool = True
    pnl_metrics: PnLAndExecutionMetrics
    ffd_config: FFDWeightsConfig
    asri_risk: SystemicRiskOutput
    hawkes_state: HawkesStateSnapshot
    last_order_command: ExecutionOrderCommand
    dollar_bars: list[DollarBar]
    positioning_signals: list[PositioningSignalSnapshot]
    barrier_labels: list[TripleBarrierLabel]
    meta_predictions: list[MetaPredictionOutput]
    hawkes_history: list[HawkesStateSnapshot]
    champion_cpcv: CPCVValidationResult
    challenger_cpcv: CPCVValidationResult
    champion_state: ChampionChallengerState
    portfolio_allocation: PortfolioAllocationOutput
    last_control_action: str = "start_dry_run"
    last_control_message: str = (
        "Dry-Run Mode active (DIRECTIONAL_PLUS_CARRY): All safety shields and maker execution online."
    )


def build_default_dashboard_state(
    regime_mode: ExecutionRegimeMode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY,
    is_dry_run: bool = True,
    last_control_action: str = "start_dry_run",
    last_control_message: str | None = None,
) -> DashboardTelemetryState:
    """Construct a rich, realistic, strictly-typed state snapshot for the Control Center."""
    base_ts_ns = 1_730_000_000_000_000_000
    step_ns = 900_000_000_000  # 15m in nanoseconds

    prices = [
        67850.0,
        67920.0,
        68110.0,
        68040.0,
        68290.0,
        68550.0,
        68490.0,
        68780.0,
        69120.0,
        69340.0,
        69210.0,
        69580.0,
    ]
    regimes = [
        PositioningRegime.NEUTRAL,
        PositioningRegime.NEUTRAL,
        PositioningRegime.CROWDED_SHORT_SQUEEZE,
        PositioningRegime.CROWDED_SHORT_SQUEEZE,
        PositioningRegime.GENUINE_TREND_CONVICTION,
        PositioningRegime.GENUINE_TREND_CONVICTION,
        PositioningRegime.GENUINE_TREND_CONVICTION,
        PositioningRegime.CROWDED_SHORT_SQUEEZE,
        PositioningRegime.GENUINE_TREND_CONVICTION,
        PositioningRegime.CROWDED_LONG_VULNERABILITY,
        PositioningRegime.CASCADE_EXHAUSTION_REVERSAL,
        PositioningRegime.GENUINE_TREND_CONVICTION,
    ]
    p_metas = [0.46, 0.49, 0.68, 0.72, 0.76, 0.79, 0.71, 0.82, 0.78, 0.54, 0.69, 0.77]
    di_vals = [1.42, 1.55, 1.88, 1.91, 1.74, 1.69, 1.83, 2.05, 2.18, 2.75, 2.40, 1.82]
    vpins = [0.24, 0.27, 0.31, 0.29, 0.33, 0.35, 0.38, 0.41, 0.44, 0.58, 0.46, 0.32]
    etas = [0.28, 0.31, 0.36, 0.34, 0.39, 0.42, 0.45, 0.48, 0.51, 0.64, 0.49, 0.35]

    dollar_bars: list[DollarBar] = []
    positioning_signals: list[PositioningSignalSnapshot] = []
    barrier_labels: list[TripleBarrierLabel] = []
    meta_predictions: list[MetaPredictionOutput] = []
    hawkes_history: list[HawkesStateSnapshot] = []

    tb_cfg = TripleBarrierConfig()
    barrier_frac = tb_cfg.effective_profit_barrier_decimal(0.0045)  # >= 80 bps

    for idx, close_px in enumerate(prices):
        ts_open = base_ts_ns + idx * step_ns
        ts_close = ts_open + step_ns
        open_px = prices[idx - 1] if idx > 0 else close_px - 60.0
        high_px = max(open_px, close_px) + 95.0
        low_px = min(open_px, close_px) - 85.0
        reg = regimes[idx]
        prim_dir: int = 0 if p_metas[idx] < 0.50 else (1 if reg != PositioningRegime.CROWDED_LONG_VULNERABILITY else -1)
        typed_dir: Literal[-1, 0, 1] = 1 if prim_dir > 0 else (-1 if prim_dir < 0 else 0)

        dollar_bars.append(
            DollarBar(
                instrument_id="BTCUSDT-PERP",
                horizon_label="15m",
                ts_open_ns=ts_open,
                ts_close_ns=ts_close,
                open=open_px,
                high=high_px,
                low=low_px,
                close=close_px,
                volume=380.0 + idx * 15.0,
                dollar_volume=25_800_000.0 + idx * 750_000.0,
                buy_dollar_volume=14_200_000.0 + idx * 450_000.0,
                sell_dollar_volume=11_600_000.0 + idx * 300_000.0,
                open_interest=84_500.0 + idx * 320.0,
                funding_rate_8h=0.00035 + (idx * 0.00001),
                mark_price=close_px + 12.5,
                spot_price=close_px,
                liquidation_volume_usd=1_850_000.0 if reg == PositioningRegime.CASCADE_EXHAUSTION_REVERSAL else 220_000.0,
                tick_count=4200 + idx * 110,
            )
        )

        positioning_signals.append(
            PositioningSignalSnapshot(
                instrument_id="BTCUSDT-PERP",
                ts_close_ns=ts_close,
                ffd_close=18.4 + idx * 0.65,
                ffd_oi=112.0 + idx * 1.8,
                ffd_close_zscore=0.45 + idx * 0.12,
                ffd_oi_zscore=0.38 + idx * 0.10,
                oi_funding_basis_zscore=0.82 + idx * 0.08,
                basis_pct=0.00018,
                funding_rate_8h=0.00035 + (idx * 0.00001),
                oi_pct_change=0.012,
                liquidation_zscore=2.85 if reg == PositioningRegime.CASCADE_EXHAUSTION_REVERSAL else 0.40,
                liquidation_exhaustion_flag=1 if reg == PositioningRegime.CASCADE_EXHAUSTION_REVERSAL else 0,
                regime=reg,
                primary_direction=typed_dir,
            )
        )

        upper_b = close_px * (1.0 + barrier_frac)
        lower_b = close_px * (1.0 - 0.00675)
        barrier_labels.append(
            TripleBarrierLabel(
                bar_index=idx,
                primary_direction=typed_dir,
                gross_return=0.0092 if p_metas[idx] >= 0.55 else 0.0015,
                net_return=0.0086 if p_metas[idx] >= 0.55 else -0.0004,
                fee_cost=0.0004,
                funding_cost=0.0002,
                upper_barrier=upper_b,
                lower_barrier=lower_b,
                hit_bar_offset=6,
                barrier_touched="PROFIT" if p_metas[idx] >= 0.55 else "VERTICAL",
                meta_label=1 if p_metas[idx] >= 0.55 else 0,
            )
        )

        meta_predictions.append(
            MetaPredictionOutput(
                instrument_id="BTCUSDT-PERP",
                ts_close_ns=ts_close,
                primary_direction=typed_dir,
                p_meta=p_metas[idx],
                mahalanobis_di=di_vals[idx],
                di_threshold=3.50,
                epistemic_gate_passed=di_vals[idx] < 3.50,
                svd_factors=(0.64, -0.21, 0.15),
                liquidation_exhaustion_flag=1 if reg == PositioningRegime.CASCADE_EXHAUSTION_REVERSAL else 0,
                expected_net_edge_bps=round((p_metas[idx] - 0.45) * 85.0, 2),
            )
        )

        hawkes_history.append(
            HawkesStateSnapshot(
                ts_ns=ts_close,
                r_state=0.42 + idx * 0.02,
                lambda_intensity=1.85 + idx * 0.08,
                mu=1.10,
                alpha=0.38,
                beta=0.95,
                branching_ratio_eta=etas[idx],
                vpin=vpins[idx],
                toxic_sweep_active=(vpins[idx] > 0.60 or etas[idx] > 0.70),
            )
        )

    asri = SystemicRiskOutput(
        score=24.5,
        confidence=0.89,
        tail_risk_flag=False,
        plain_english_summary=(
            "Clear Skies (ASRI 24.5/100): Stablecoin pegs are rock-solid, exchange reserves are healthy, "
            "and macro liquidity supports full 100% normal risk sizing (M_regime = 1.00)."
        ),
    )

    m_regime = compute_regime_multiplier_from_asri(asri.score, asri.tail_risk_flag)
    if regime_mode in (ExecutionRegimeMode.PAUSED_ENTRIES, ExecutionRegimeMode.EMERGENCY_FLATTENED):
        effective_m_regime = 0.0
    elif regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK:
        effective_m_regime = 0.25
    else:
        effective_m_regime = m_regime

    alloc_inputs = [
        AssetAllocationInput(
            asset_id="BTCUSDT-PERP",
            y_primary=1 if regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY else 0,
            w_hrp=0.45,
            p_meta=0.77,
            expected_gross_return_bps=94.0,
            fee_schedule=FeeSchedule.usdt_perp(),
            epistemic_gate_passed=True,
        ),
        AssetAllocationInput(
            asset_id="ETHUSDT-PERP",
            y_primary=1 if regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY else 0,
            w_hrp=0.35,
            p_meta=0.71,
            expected_gross_return_bps=86.0,
            fee_schedule=FeeSchedule.usdt_perp(),
            epistemic_gate_passed=True,
        ),
        AssetAllocationInput(
            asset_id="SOLUSDT-PERP",
            y_primary=-1 if regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY else 0,
            w_hrp=0.20,
            p_meta=0.68,
            expected_gross_return_bps=110.0,
            fee_schedule=FeeSchedule.usdt_perp(),
            epistemic_gate_passed=True,
        ),
    ]
    portfolio_alloc = compute_signed_target_weights(
        inputs=alloc_inputs,
        regime_multiplier=effective_m_regime,
        gamma_frac=0.5,
        max_gross_leverage=2.5,
    )

    champion_cpcv = CPCVValidationResult(
        model_id="lgbm_meta_champion_v4_2",
        n_groups=10,
        k_test_groups=2,
        num_splits=45,
        n_oos_paths=9,
        purge_bars=16,
        embargo_pct=0.01,
        psr_vs_zero=0.984,
        psr_vs_benchmark=0.968,
        dsr=0.961,
        pbo=0.142,
        expected_net_pnl_bps=24.6,
        sharpe_net=2.48,
        skewness=0.34,
        kurtosis=3.42,
        total_trials=12,
    )

    challenger_cpcv = CPCVValidationResult(
        model_id="lgbm_meta_challenger_v4_3_rc1",
        n_groups=10,
        k_test_groups=2,
        num_splits=45,
        n_oos_paths=9,
        purge_bars=16,
        embargo_pct=0.01,
        psr_vs_zero=0.991,
        psr_vs_benchmark=0.975,
        dsr=0.972,
        pbo=0.118,
        expected_net_pnl_bps=27.4,
        sharpe_net=2.66,
        skewness=0.41,
        kurtosis=3.28,
        total_trials=14,
    )

    champion_state = ChampionChallengerState(
        asset_id="BTCUSDT-PERP",
        champion_model_id=champion_cpcv.model_id,
        champion_live_30d_psr=0.94,
        consecutive_failed_retrains=0,
        regime_mode=regime_mode,
        last_promotion_reason=(
            "Challenger v4.3_rc1 passed all 45 CPCV splits (PSR=99.1%, DSR=97.2%, PBO=11.8%, Net Edge=+27.4 bps)."
        ),
    )

    order_action = (
        ExecutionAction.EMERGENCY_TAKER_EXIT
        if regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED
        else (
            ExecutionAction.HOLD_RESTING_HYSTERESIS
            if regime_mode == ExecutionRegimeMode.PAUSED_ENTRIES
            else ExecutionAction.PLACE_GTX_MAKER
        )
    )

    return DashboardTelemetryState(
        active_instrument="BTCUSDT-PERP",
        regime_mode=regime_mode,
        is_dry_run=is_dry_run,
        pnl_metrics=PnLAndExecutionMetrics(is_dry_run=is_dry_run),
        ffd_config=FFDWeightsConfig(
            split_id="cpcv_fold_45_prod",
            d_star=0.28,
            threshold_tau=1e-5,
            weights=(1.0, -0.28, -0.1008, -0.0571, -0.0383),
            adf_pvalue=0.0024,
            level_correlation=0.884,
        ),
        asri_risk=asri,
        hawkes_state=hawkes_history[-1],
        last_order_command=ExecutionOrderCommand(
            instrument_id="BTCUSDT-PERP",
            ts_ns=dollar_bars[-1].ts_close_ns,
            action=order_action,
            side="NONE" if regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED else "BUY",
            target_price=dollar_bars[-1].close - 5.0,
            quantity=0.0 if regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED else 0.42,
            post_only_gtx=regime_mode != ExecutionRegimeMode.EMERGENCY_FLATTENED,
            time_in_force="IOC" if regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED else "GTX",
            consecutive_rejections=0,
            reason=last_control_message
            or "Resting Post-Only GTX Maker order inside healthy VPIN/Hawkes shield.",
        ),
        dollar_bars=dollar_bars,
        positioning_signals=positioning_signals,
        barrier_labels=barrier_labels,
        meta_predictions=meta_predictions,
        hawkes_history=hawkes_history,
        champion_cpcv=champion_cpcv,
        challenger_cpcv=challenger_cpcv,
        champion_state=champion_state,
        portfolio_allocation=portfolio_alloc,
        last_control_action=last_control_action,
        last_control_message=last_control_message
        or "Dry-Run Mode active (DIRECTIONAL_PLUS_CARRY): All safety shields and maker execution online.",
    )


def build_candlestick_dollar_bar_figure(
    state: DashboardTelemetryState | None = None,
    *,
    bars: Sequence[DollarBar] | None = None,
    barriers: Sequence[TripleBarrierLabel] | None = None,
    meta_outputs: Sequence[MetaPredictionOutput] | None = None,
    hawkes_snapshots: Sequence[HawkesStateSnapshot] | None = None,
    positioning_snapshots: Sequence[PositioningSignalSnapshot] | None = None,
    asri_output: SystemicRiskOutput | None = None,
) -> go.Figure:
    """Build the interactive 3-row Plotly Dollar-Bar figure with toggleable overlays.

    Overlays included:
    - Dynamic Dollar-Bar OHLC candlesticks
    - Triple-Barrier Profit Target & Stop-Loss bands (``triple_barrier``)
    - Perpetual Positioning & Squeeze Regime markers (``squeeze_regime``)
    - LightGBM Meta-Probability $p_{\\text{meta}}$ (``meta_probability``)
    - VPIN Toxicity & $O(1)$ Hawkes Reflexivity $\\eta$ (``vpin_toxicity``, ``hawkes_reflexivity``)
    - Mahalanobis Outlier Distance ``DI_values`` & threshold (``mahalanobis_di``)
    - ASRI Macro Score & ``plain_english_summary`` banner annotation (``asri_macro``)
    """
    resolved = state or build_default_dashboard_state()
    use_bars = list(bars) if bars is not None else resolved.dollar_bars
    use_barriers = list(barriers) if barriers is not None else resolved.barrier_labels
    use_meta = list(meta_outputs) if meta_outputs is not None else resolved.meta_predictions
    use_hawkes = list(hawkes_snapshots) if hawkes_snapshots is not None else resolved.hawkes_history
    use_pos = list(positioning_snapshots) if positioning_snapshots is not None else resolved.positioning_signals
    use_asri = asri_output if asri_output is not None else resolved.asri_risk

    x_labels = [f"Bar {i + 1}" for i in range(len(use_bars))]
    opens = [b.open for b in use_bars]
    highs = [b.high for b in use_bars]
    lows = [b.low for b in use_bars]
    closes = [b.close for b in use_bars]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.54, 0.23, 0.23],
        subplot_titles=(
            f"{resolved.active_instrument} Price Bars + Profit Target (Green) & Stop-Loss (Red)",
            "AI Win Probability (p_meta) & Order-Book Toxicity Shield (VPIN / Hawkes)",
            "Unfamiliar Market Check (Mahalanobis DI) & Macro Weather (ASRI)",
        ),
    )

    # Trace 0: Dollar-Bar Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=x_labels,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price (OHLC)",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # Trace 1 & 2: Triple-Barrier Profit & Stop Bands
    if use_barriers:
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_barriers)],
                y=[b.upper_barrier for b in use_barriers],
                mode="lines",
                name="Profit Target (>=80 bps)",
                line={"color": "#34d399", "width": 2, "dash": "dash"},
                hovertemplate="Profit Target: $%{y:,.2f}<extra>Triple-Barrier Top Wall</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_barriers)],
                y=[b.lower_barrier for b in use_barriers],
                mode="lines",
                name="Stop-Loss Band",
                line={"color": "#f87171", "width": 2, "dash": "dot"},
                hovertemplate="Stop-Loss: $%{y:,.2f}<extra>Triple-Barrier Bottom Wall</extra>",
            ),
            row=1,
            col=1,
        )

    # Trace 3: Funding / OI Squeeze Regime Markers (clean markers only, text on hover so zero overlap!)
    if use_pos:
        regime_colors = {
            PositioningRegime.CROWDED_SHORT_SQUEEZE: "#22c55e",
            PositioningRegime.GENUINE_TREND_CONVICTION: "#38bdf8",
            PositioningRegime.CROWDED_LONG_VULNERABILITY: "#f97316",
            PositioningRegime.CASCADE_EXHAUSTION_REVERSAL: "#a855f7",
            PositioningRegime.NEUTRAL: "#94a3b8",
        }
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_pos)],
                y=[highs[i] + 65.0 for i in range(min(len(use_pos), len(highs)))],
                mode="markers",
                name="Regime Signal (Hover)",
                text=[p.regime.value.replace("_", " ") for p in use_pos],
                marker={
                    "size": 9,
                    "color": [regime_colors.get(p.regime, "#94a3b8") for p in use_pos],
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "#0f172a"},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>Marker Level: $%{y:,.2f}<extra>Positioning Regime</extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Trace 4: LightGBM Meta-Probability (p_meta)
    if use_meta:
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_meta)],
                y=[m.p_meta for m in use_meta],
                mode="lines+markers",
                name="AI Win Prob (p_meta)",
                line={"color": "#60a5fa", "width": 2.5},
                marker={"size": 6},
                hovertemplate="AI Confidence (p_meta): %{y:.1%}<extra>Pass > 50%</extra>",
            ),
            row=2,
            col=1,
        )

    # Trace 5 & 6: VPIN Toxicity & Hawkes Reflexivity Eta
    if use_hawkes:
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_hawkes)],
                y=[h.vpin for h in use_hawkes],
                mode="lines",
                name="VPIN Toxicity (<=0.60)",
                line={"color": "#f59e0b", "width": 2},
                hovertemplate="VPIN Toxicity: %{y:.2f}<extra>Safe <= 0.60</extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_hawkes)],
                y=[h.branching_ratio_eta for h in use_hawkes],
                mode="lines",
                name="Hawkes Avalanche (<=0.70)",
                line={"color": "#ec4899", "width": 2, "dash": "dashdot"},
                hovertemplate="Hawkes eta: %{y:.2f}<extra>Safe <= 0.70</extra>",
            ),
            row=2,
            col=1,
        )

    # Trace 7, 8 & 9: Mahalanobis DI_values, DI Threshold, and ASRI Macro Score
    if use_meta:
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_meta)],
                y=[m.mahalanobis_di for m in use_meta],
                mode="lines+markers",
                name="Outlier Distance (DI)",
                line={"color": "#c084fc", "width": 2.5},
                marker={"size": 6},
                hovertemplate="Mahalanobis DI: %{y:.2f}<extra>Unfamiliar Market Check</extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_meta)],
                y=[m.di_threshold for m in use_meta],
                mode="lines",
                name="Max Safe DI Limit (3.50)",
                line={"color": "#ef4444", "width": 1.5, "dash": "dash"},
                hovertemplate="Max Safe DI: %{y:.2f}<extra>Safety Ceiling</extra>",
            ),
            row=3,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=[use_asri.score / 25.0 for _ in x_labels],
            mode="lines",
            name=f"Macro Risk ASRI ({use_asri.score:.0f}/100)",
            line={"color": "#2dd4bf", "width": 2, "dash": "dot"},
            customdata=[use_asri.plain_english_summary for _ in x_labels],
            hovertemplate="ASRI Score: " + f"{use_asri.score:.1f}/100" + "<br>%{customdata}<extra>Macro Weather</extra>",
        ),
        row=3,
        col=1,
    )

    fig.update_annotations(font_size=12, font_color="#cbd5e1")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#152033",
        plot_bgcolor="#0f172a",
        height=680,
        margin={"l": 55, "r": 25, "t": 38, "b": 85},
        xaxis_rangeslider_visible=False,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.07,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11, "color": "#cbd5e1"},
            "bgcolor": "rgba(15, 23, 42, 0.7)",
        },
    )
    return fig


def build_portfolio_weights_figure(
    allocation: PortfolioAllocationOutput | DashboardTelemetryState | None = None,
) -> go.Figure:
    """Build an interactive Plotly chart of signed ``HRP-CVaR`` target weights $w_{\\text{final}, i}$."""
    if isinstance(allocation, DashboardTelemetryState):
        alloc = allocation.portfolio_allocation
    elif isinstance(allocation, PortfolioAllocationOutput):
        alloc = allocation
    else:
        alloc = build_default_dashboard_state().portfolio_allocation

    assets = list(alloc.weights.keys())
    signed_weights = [alloc.weights[a] for a in assets]
    hrp_base = [alloc.hrp_unsigned_weights.get(a, 0.0) for a in assets]
    kelly_vals = [alloc.kelly_scalars.get(a, 0.0) for a in assets]

    colors = ["#10b981" if w >= 0 else "#f43f5e" for w in signed_weights]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=assets,
            y=signed_weights,
            name="Signed Final Target Weight (w_final,i)",
            marker_color=colors,
            text=[f"{w:+.2%}" for w in signed_weights],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>Signed Target Weight (w_final,i): %{y:+.4f}<extra>HRP-CVaR Final Allocation</extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=assets,
            y=hrp_base,
            name="Unsigned HRP-CVaR Base Weight (w_HRP,i)",
            marker_color="#38bdf8",
            opacity=0.65,
            hovertemplate="<b>%{x}</b><br>Base HRP-CVaR Weight: %{y:.4f}<extra>Crash-Diversified Base</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=assets,
            y=kelly_vals,
            mode="lines+markers",
            name="Fractional Kelly Conviction Scalar f_Kelly(p_meta)",
            line={"color": "#facc15", "width": 2.5},
            hovertemplate="<b>%{x}</b><br>Half-Kelly Scalar: %{y:.2%}<extra>Meta-Model Sizing</extra>",
        )
    )

    fig.update_layout(
        title={
            "text": (
                f"<b>Target Portfolio Weights</b> — "
                f"Total Leverage: {alloc.gross_leverage:.2f}x / 2.50x Cap"
            ),
            "font": {"size": 14, "color": "#f8fafc"},
        },
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#152033",
        plot_bgcolor="#0f172a",
        height=420,
        margin={"l": 55, "r": 25, "t": 55, "b": 75},
        yaxis_title="Portfolio Weight",
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5},
    )
    return fig


def build_validation_gates_figure(
    champion: CPCVValidationResult | DashboardTelemetryState | None = None,
    challenger: CPCVValidationResult | None = None,
) -> go.Figure:
    """Build an interactive Plotly comparison of Champion vs. Challenger 45-split CPCV gates."""
    if isinstance(champion, DashboardTelemetryState):
        champ = champion.champion_cpcv
        chall = challenger or champion.challenger_cpcv
    elif isinstance(champion, CPCVValidationResult):
        champ = champion
        chall = challenger or build_default_dashboard_state().challenger_cpcv
    else:
        default_state = build_default_dashboard_state()
        champ = default_state.champion_cpcv
        chall = challenger or default_state.challenger_cpcv

    gate_names = [
        "PSR vs Zero (>=0.95)",
        "PSR vs Benchmark (>=0.95)",
        "Deflated Sharpe DSR (>=0.95)",
        "Non-Overfit 1 - PBO (>0.70)",
        "Net PnL Hurdle Ratio (E[PnL]/15bps >1.0)",
    ]
    champ_scores = [
        champ.psr_vs_zero,
        champ.psr_vs_benchmark,
        champ.dsr,
        1.0 - champ.pbo,
        champ.expected_net_pnl_bps / 15.0,
    ]
    chall_scores = [
        chall.psr_vs_zero,
        chall.psr_vs_benchmark,
        chall.dsr,
        1.0 - chall.pbo,
        chall.expected_net_pnl_bps / 15.0,
    ]
    thresholds = [0.95, 0.95, 0.95, 0.70, 1.00]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=gate_names,
            y=champ_scores,
            name=f"Champion ({champ.model_id}) — {'PASSED' if champ.passed_all_gates else 'FAILED'}",
            marker_color="#3b82f6",
            text=[f"{v:.3f}" for v in champ_scores],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=gate_names,
            y=chall_scores,
            name=f"Challenger ({chall.model_id}) — {'PASSED' if chall.passed_all_gates else 'FAILED'}",
            marker_color="#10b981" if chall.passed_all_gates else "#f59e0b",
            text=[f"{v:.3f}" for v in chall_scores],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gate_names,
            y=thresholds,
            mode="lines+markers",
            name="Hard Promotion Gate Minimum Threshold",
            line={"color": "#ef4444", "width": 2.5, "dash": "dash"},
        )
    )

    fig.update_layout(
        title={
            "text": (
                f"<b>45-Split CPCV Promotion Gatekeeper</b> — "
                f"Champion Sharpe: {champ.sharpe_net:.2f} vs. Challenger Sharpe: {chall.sharpe_net:.2f}"
            ),
            "font": {"size": 14, "color": "#f8fafc"},
        },
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#152033",
        plot_bgcolor="#0f172a",
        height=420,
        margin={"l": 55, "r": 25, "t": 55, "b": 75},
        yaxis_title="Gate Score / Ratio",
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5},
    )
    return fig


def load_layman_document_markdown(doc_name: str) -> str:
    """Load any of the four Layman's Documentation Suite manuals from ``docs/``."""
    clean_key = doc_name.strip().removesuffix(".md").upper()
    filename = DOC_FILENAMES.get(clean_key)
    if filename is None:
        # Check case-insensitive match against filenames
        for key, candidate in DOC_FILENAMES.items():
            if doc_name.strip().lower() in (key.lower(), candidate.lower()):
                filename = candidate
                break
    if filename is None:
        raise KeyError(
            f"Unknown manual '{doc_name}'. Available manuals: {', '.join(DOC_FILENAMES.values())}"
        )
    path = DOCS_DIR / filename
    if not path.exists():
        return f"# {filename}\n\nDocumentation file not found at `{path}`."
    return path.read_text(encoding="utf-8")


def render_full_dashboard_html(state: DashboardTelemetryState | None = None) -> str:
    """Render the simplified, high-clarity 3-tab HTML Control Center with live Dry-Run feedback."""
    resolved = state or build_default_dashboard_state()
    candlestick_fig = build_candlestick_dollar_bar_figure(resolved)
    weights_fig = build_portfolio_weights_figure(resolved.portfolio_allocation)
    gates_fig = build_validation_gates_figure(resolved.champion_cpcv, resolved.challenger_cpcv)

    candlestick_json = json.dumps(candlestick_fig.to_dict())
    weights_json = json.dumps(weights_fig.to_dict())
    gates_json = json.dumps(gates_fig.to_dict())

    docs_html_blocks: list[str] = []
    for doc_key, doc_file in DOC_FILENAMES.items():
        raw_md = load_layman_document_markdown(doc_key)
        escaped_md = html.escape(raw_md)
        docs_html_blocks.append(
            f'<div class="doc-card" id="doc-{doc_key}">'
            f'<h3>📘 {html.escape(doc_file)}</h3>'
            f'<pre class="markdown-pre">{escaped_md}</pre>'
            f"</div>"
        )

    tooltip_rows = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(v)}</td></tr>"
        for k, v in PLAIN_ENGLISH_TOOLTIPS.items()
    )

    latest_bar = resolved.dollar_bars[-1]
    latest_meta = resolved.meta_predictions[-1]
    latest_pos = resolved.positioning_signals[-1]
    shield_status = (
        "TOXIC SWEEP ACTIVE (ORDERS PULLED)"
        if resolved.hawkes_state.toxic_sweep_active
        else "ALL 4 SHIELDS GREEN (SAFE)"
    )

    weights_rows = "".join(
        f"<tr>"
        f"<td><strong>{html.escape(asset)}</strong></td>"
        f"<td><span class='pos-pill {'pos-long' if w > 0 else ('pos-short' if w < 0 else 'pos-flat')}'>"
        f"{'LONG (BUY)' if w > 0 else ('SHORT (SELL)' if w < 0 else 'FLAT (CASH)')}</span></td>"
        f"<td><strong>{w:+.1%}</strong> ({w:+.4f})</td>"
        f"<td>{resolved.portfolio_allocation.hrp_unsigned_weights.get(asset, 0.0):.0%}</td>"
        f"<td>{resolved.portfolio_allocation.kelly_scalars.get(asset, 0.0):.0%}</td>"
        f"</tr>"
        for asset, w in resolved.portfolio_allocation.weights.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Project Phoenix — Simple Trading Control Center</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    :root {{
      --bg: #0b1120;
      --panel: #152033;
      --panel-alt: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
      --blue: #38bdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    /* TOP CONTROL BAR */
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: #0f172a;
      border-bottom: 1px solid var(--border);
      gap: 16px;
    }}
    .brand-row {{
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .brand h1 {{ margin: 0; font-size: 1.3rem; letter-spacing: -0.01em; }}
    .brand p {{ margin: 3px 0 0; color: var(--muted); font-size: 0.86rem; }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 14px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.85rem;
      background: rgba(16, 185, 129, 0.16);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.45);
    }}
    .status-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 1.4s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 1; }}
      50% {{ transform: scale(1.3); opacity: 0.55; }}
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      border: 2px solid transparent;
      padding: 10px 16px;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      font-size: 0.88rem;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      opacity: 0.82;
    }}
    .btn:hover {{ opacity: 1; transform: translateY(-1px); }}
    .btn.active-mode {{
      opacity: 1;
      border-color: #ffffff;
      box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.22);
    }}
    .btn-dryrun {{ background: #059669; }}
    .btn-step {{ background: #2563eb; border: 1px solid #60a5fa; opacity: 1; }}
    .btn-pause {{ background: #d97706; }}
    .btn-carry {{ background: #0284c7; }}
    .btn-flatten {{ background: #dc2626; }}

    /* LIVE ACTION FEEDBACK BANNER */
    .action-feedback-banner {{
      margin: 14px 24px 10px;
      padding: 12px 18px;
      background: #172554;
      border-left: 5px solid var(--green);
      border-radius: 10px;
      font-size: 0.94rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      transition: background 0.25s ease;
    }}
    .flash-highlight {{
      animation: flashBg 0.7s ease;
    }}
    @keyframes flashBg {{
      0% {{ background: rgba(16, 185, 129, 0.35); }}
      100% {{ background: #172554; }}
    }}

    /* 4 SIMPLE TOP CARDS */
    .simple-kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      padding: 6px 24px 16px;
    }}
    .kpi-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .kpi-label {{
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      cursor: help;
    }}
    .kpi-main {{
      font-size: 1.45rem;
      font-weight: 800;
      margin: 6px 0 4px;
      color: #fff;
    }}
    .kpi-sub {{
      font-size: 0.82rem;
      color: var(--muted);
    }}

    /* TABS */
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 0 24px;
      border-bottom: 1px solid var(--border);
      background: #0f172a;
    }}
    .tab-btn {{
      background: transparent;
      color: var(--muted);
      border: none;
      border-bottom: 3px solid transparent;
      padding: 13px 18px;
      cursor: pointer;
      font-weight: 700;
      font-size: 0.95rem;
    }}
    .tab-btn.active {{
      color: var(--blue);
      border-bottom-color: var(--blue);
    }}
    .tab-panel {{ display: none; padding: 18px 24px; }}
    .tab-panel.active {{ display: block; }}

    /* LIVE SPLIT LAYOUT */
    .live-grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      align-items: start;
    }}
    @media (max-width: 1024px) {{
      .live-grid {{ grid-template-columns: 1fr; }}
    }}
    .chart-box {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 16px;
    }}
    .feed-box {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .feed-box h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .activity-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      max-height: 290px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .activity-item {{
      background: #0f172a;
      border-left: 3px solid var(--blue);
      padding: 9px 12px;
      border-radius: 6px;
      font-size: 0.83rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .pos-pill {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 6px;
      font-size: 0.76rem;
      font-weight: 700;
    }}
    .pos-long {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .pos-short {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
    .pos-flat {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }}

    /* ADVANCED DRAWER */
    details.advanced-drawer {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 18px;
      margin-top: 12px;
    }}
    details.advanced-drawer summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--blue);
      font-size: 0.92rem;
    }}
    .badge-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .metric-card {{
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
    }}
    .metric-title {{
      font-size: 0.78rem;
      color: var(--muted);
      border-bottom: 1px dotted var(--muted);
      display: inline-block;
      cursor: help;
    }}
    .metric-val {{
      font-size: 1.05rem;
      font-weight: 700;
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 16px;
    }}
    th, td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 0.88rem;
    }}
    th {{ background: #0f172a; color: var(--blue); }}
    .markdown-pre {{
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: #0f172a;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-size: 0.85rem;
      max-height: 440px;
      overflow-y: auto;
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand-row">
      <div class="brand">
        <h1>🔥 Project Phoenix — Control Center</h1>
        <p>Paper Trading &amp; Risk Control • Mode: <strong id="mode-indicator">{html.escape(resolved.regime_mode.value)}</strong></p>
      </div>
      <div id="live-status-pill" class="status-pill">
        <span id="status-dot" class="status-dot"></span>
        <span id="status-pill-text">DRY-RUN RUNNING LIVE (Auto-Updating)</span>
      </div>
    </div>

    <div class="controls">
      <button id="btn-start_dry_run" class="btn btn-dryrun active-mode" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['start_dry_run'])}" onclick="sendControl('start_dry_run')">
        ▶ Start Dry-Run
      </button>
      <button id="btn-step_bar" class="btn btn-step" title="Immediately simulate 1 new 15-minute market bar and update chart &amp; profit" onclick="stepSingleBar()">
        ⚡ Poll Next 15m Bar
      </button>
      <button id="btn-pause_entries" class="btn btn-pause" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['pause_entries'])}" onclick="sendControl('pause_entries')">
        ⏸ Pause Entries
      </button>
      <button id="btn-switch_carry_only" class="btn btn-carry" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['switch_carry_only'])}" onclick="sendControl('switch_carry_only')">
        🏦 Switch to Carry-Only
      </button>
      <button id="btn-emergency_flatten" class="btn btn-flatten" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['emergency_flatten'])}" onclick="sendControl('emergency_flatten')">
        🛑 Emergency Flatten All
      </button>
    </div>
  </header>

  <!-- CLEAR FEEDBACK BANNER SO EVERY BUTTON CLICK IS IMMEDIATELY VISIBLE -->
  <div id="feedback-banner" class="action-feedback-banner" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['asri_macro'])}">
    <div>
      <strong id="banner-headline">📢 Live Status:</strong>
      <span id="banner-message">{html.escape(resolved.last_control_message)}</span>
    </div>
    <div style="font-size: 0.82rem; color: #cbd5e1; white-space: nowrap;">
      Bar #<span id="bar-counter">{len(resolved.dollar_bars)}</span> • BTC: <strong id="live-btc-price">${latest_bar.close:,.2f}</strong>
    </div>
  </div>

  <!-- 4 SIMPLE, PLAIN-ENGLISH SUMMARY CARDS -->
  <div class="simple-kpi-grid">
    <div class="kpi-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['persistent_ledger'])}">
      <div id="kpi-pnl-label" class="kpi-label">1. Persistent Paper Ledger (On ${resolved.pnl_metrics.starting_capital_usd:,.0f} Base) (?)</div>
      <div id="kpi-pnl-val" class="kpi-main" style="color: {'var(--green)' if resolved.pnl_metrics.net_pnl_usd >= 0 else 'var(--red)'};">{'+' if resolved.pnl_metrics.net_pnl_usd >= 0 else ''}${resolved.pnl_metrics.net_pnl_usd:,.2f}</div>
      <div id="kpi-pnl-sub" class="kpi-sub">Equity: <strong>${resolved.pnl_metrics.current_equity_usd:,.2f}</strong> ({resolved.pnl_metrics.net_pnl_bps:+.1f} bps net) | Record: <strong>0W / 0L</strong> (0.0% win rate) | Saved $0.00 in fees</div>
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; gap: 6px; align-items: center;">
        <label for="starting-capital-input" style="font-size: 0.76rem; color: var(--muted); font-weight: 600;">Start Capital ($):</label>
        <input id="starting-capital-input" type="number" min="100" step="1000" value="{int(resolved.pnl_metrics.starting_capital_usd)}" style="width: 105px; padding: 5px 8px; border-radius: 6px; border: 1px solid var(--border); background: #0f172a; color: #fff; font-weight: 700; font-size: 0.82rem;" />
        <button class="btn" style="background: #475569; padding: 5px 10px; font-size: 0.76rem; border-radius: 6px; opacity: 1;" onclick="resetCapital()" title="Clear accumulated profit back to $0.00 and set your custom paper starting capital">
          🔄 Set Capital &amp; Clear Profit ($0)
        </button>
      </div>
    </div>

    <div class="kpi-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['hrp_cvar_weights'])}">
      <div class="kpi-label">2. What The Bot Is Doing Now (?)</div>
      <div id="kpi-action-val" class="kpi-main">Buying BTC &amp; ETH</div>
      <div id="kpi-action-sub" class="kpi-sub">Total Leverage: {resolved.portfolio_allocation.gross_leverage:.2f}x (Safe Cap: 2.50x)</div>
    </div>

    <div class="kpi-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['risk_committee_models'])}">
      <div class="kpi-label">3. 3-Model MLX Risk Committee (?)</div>
      <div id="kpi-shield-val" class="kpi-main" style="color: var(--green);">M_regime: 1.00 (Risk: {resolved.asri_risk.score:.0f}/100)</div>
      <div id="kpi-shield-sub" class="kpi-sub">Sentinel: 18 • Breadth: 15 • Red-Team Critic: 12 (Safe)</div>
    </div>

    <div class="kpi-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['funding_carry_sleeve'])}">
      <div class="kpi-label">4. Quantitative Sleeve Yields (?)</div>
      <div class="kpi-main" style="color: var(--blue);">Carry: +{resolved.pnl_metrics.annualized_carry_apr_pct:.1f}% APR</div>
      <div class="kpi-sub">Hyperliquid 1h ALO Maker | Deribit Skew: +2.4% Call Demand</div>
    </div>
  </div>

  <nav class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-live', this)">1. Live Trading &amp; Activity</button>
    <button class="tab-btn" onclick="switchTab('tab-portfolio', this)">2. Portfolio &amp; AI Validation</button>
    <button class="tab-btn" onclick="switchTab('tab-guide', this)">3. Plain-English Guide &amp; Manuals</button>
  </nav>

  <!-- TAB 1: SIMPLE LIVE TRADING & ACTIVITY FEED -->
  <section id="tab-live" class="tab-panel active">
    <!-- MULTI-ASSET REAL-TIME DATA FEEDS TABLE (BTC, ETH, SOL) -->
    <div class="feed-box" style="margin-bottom: 20px;" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['multi_asset_feeds'])}">
      <h3>
        <span>⚡ Real-Time Multi-Asset Hyperliquid L1 Feeds (Sub-Second Ingestion)</span>
        <span id="market-feed-source-pill" class="pos-pill pos-long">● Hyperliquid L1 Active</span>
      </h3>
      <div style="overflow-x: auto;">
        <table id="multi-asset-feed-table">
          <thead>
            <tr>
              <th>Asset</th>
              <th>Mid Price</th>
              <th>Mark Price</th>
              <th>Spot Oracle</th>
              <th>Basis (bps)</th>
              <th>Open Interest</th>
              <th>1h Funding Rate</th>
              <th>Annual APR</th>
              <th>L2 Spread</th>
              <th>Top-20 Depth</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="multi-asset-tbody">
            <tr id="row-BTC">
              <td><strong>BTC</strong></td>
              <td id="btc-mid">${latest_bar.close:,.2f}</td>
              <td id="btc-mark">${latest_bar.close + 1.50:,.2f}</td>
              <td id="btc-oracle">${latest_bar.close:,.2f}</td>
              <td id="btc-basis">+0.2 bps</td>
              <td id="btc-oi">{latest_bar.open_interest:,.0f}</td>
              <td id="btc-funding" style="color: var(--blue);">+0.0012%</td>
              <td id="btc-apr" style="font-weight: 700; color: var(--green);">+10.5%</td>
              <td id="btc-spread">1.2 bps</td>
              <td id="btc-depth">$4.8M</td>
              <td><span class="pos-pill pos-long">STREAMING</span></td>
            </tr>
            <tr id="row-ETH">
              <td><strong>ETH</strong></td>
              <td id="eth-mid">$2,244.50</td>
              <td id="eth-mark">$2,244.75</td>
              <td id="eth-oracle">$2,244.50</td>
              <td id="eth-basis">+1.1 bps</td>
              <td id="eth-oi">142,500</td>
              <td id="eth-funding" style="color: var(--blue);">+0.0015%</td>
              <td id="eth-apr" style="font-weight: 700; color: var(--green);">+13.1%</td>
              <td id="eth-spread">1.5 bps</td>
              <td id="eth-depth">$2.6M</td>
              <td><span class="pos-pill pos-long">STREAMING</span></td>
            </tr>
            <tr id="row-SOL">
              <td><strong>SOL</strong></td>
              <td id="sol-mid">$96.65</td>
              <td id="sol-mark">$96.70</td>
              <td id="sol-oracle">$96.65</td>
              <td id="sol-basis">+5.2 bps</td>
              <td id="sol-oi">580,000</td>
              <td id="sol-funding" style="color: var(--blue);">+0.0028%</td>
              <td id="sol-apr" style="font-weight: 700; color: var(--green);">+24.5%</td>
              <td id="sol-spread">2.0 bps</td>
              <td id="sol-depth">$1.2M</td>
              <td><span class="pos-pill pos-long">STREAMING</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- QUANTITATIVE SLEEVE TELEMETRY (FUNDING CARRY & DERIVATIVES ALPHA) -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 20px;">
      <div class="feed-box" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['funding_carry_sleeve'])}">
        <h3>🏦 Sleeve 3: Funding Carry Arbitrage (Hyperliquid 1h)</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.86rem;">
          <div>Strategy: <strong>Delta-Neutral Spot-Perp</strong></div>
          <div>Settlement: <span class="pos-pill pos-long">1h Epoch Active</span></div>
          <div>Carry Yield: <strong id="sleeve-carry-apr" style="color: var(--green);">+14.8% APR</strong></div>
          <div>Routing: <strong>ALO Post-Only Maker</strong></div>
          <div>Maker Fee: <strong>1.5 bps</strong> (Rebate Tier)</div>
          <div>Taker Fee Avoided: <strong style="color: var(--green);">4.5 bps</strong></div>
          <div>Cross-Venue Spread: <strong id="sleeve-cross-spread">+8.4 bps (HL vs Bybit)</strong></div>
          <div>Breakeven Amortization: <strong id="sleeve-breakeven">12.0 hours</strong></div>
        </div>
      </div>

      <div class="feed-box" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['derivatives_alpha_sleeve'])}">
        <h3>📈 Sleeve 1: Derivatives Alpha (Deribit Skew &amp; Vaults)</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.86rem;">
          <div>Deribit 25d Skew: <strong id="sleeve-skew" style="color: var(--green);">+2.4% (Bullish Call Demand)</strong></div>
          <div>Term Structure: <strong id="sleeve-term">Contango (Slope: +0.008)</strong></div>
          <div>Dealer Net GEX: <strong id="sleeve-gex">+$42.5M (Pinning)</strong></div>
          <div>HLP Vault Address: <code style="font-size: 0.75rem;">0xdfc24b...</code></div>
          <div>Vault Crowding: <strong id="sleeve-vault">Balanced (Low Vuln)</strong></div>
          <div>Counterparty State: <span class="pos-pill pos-long">Safe Risk Profile</span></div>
        </div>
      </div>
    </div>

    <!-- LIVE FEED AVAILABILITY -->
    <div class="feed-box" style="margin-bottom: 20px;" title="Which inputs come from a live feed right now. Anything marked missing is zero-filled or ignored, never invented.">
      <h3>📡 Live Feed Availability</h3>
      <div id="feed-availability" style="display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.8rem; margin-top: 8px;">
        <span class="pos-pill">waiting for first poll…</span>
      </div>
      <div id="feed-availability-sub" style="margin-top: 8px; font-size: 0.78rem; color: var(--muted);"></div>
    </div>

    <!-- 3-MODEL MLX RISK COMMITTEE TELEMETRY -->
    <div class="feed-box" style="margin-bottom: 20px;" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['risk_committee_models'])}">
      <h3>
        <span>🛡️ Apple Silicon (MLX) 3-Model Risk Committee (Apple Metal GPU Consensus)</span>
        <span id="risk-consensus-badge" class="pos-pill pos-long">Consensus M_regime: 1.00</span>
      </h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 10px;">
        <div class="metric-card">
          <span class="metric-title">1. Macro / Stablecoin Sentinel</span>
          <div id="risk-sentinel-score" class="metric-val" style="color: var(--green);">18.0 / 100</div>
          <div id="risk-sentinel-sub" class="metric-sub" style="font-size: 0.78rem; color: var(--muted);">Peg Dev: 1.5 bps • DVOL: 51.0 (Calm)</div>
        </div>
        <div class="metric-card">
          <span class="metric-title">2. Cross-Sectional Breadth Analyst</span>
          <div id="risk-breadth-score" class="metric-val" style="color: var(--green);">15.0 / 100</div>
          <div id="risk-breadth-sub" class="metric-sub" style="font-size: 0.78rem; color: var(--muted);">Dispersion: 42.0 bps • Corr Dist: 0.85</div>
        </div>
        <div class="metric-card">
          <span class="metric-title">3. Adversarial Red-Team Critic</span>
          <div id="risk-critic-score" class="metric-val" style="color: var(--green);">12.0 / 100</div>
          <div id="risk-critic-sub" class="metric-sub" style="font-size: 0.78rem; color: var(--muted);">Cascade Prob: 8.0% • Veto: None (Safe)</div>
        </div>
        <div class="metric-card" style="border: 1px solid var(--blue);">
          <span class="metric-title">Consensus Portfolio Multiplier M_regime</span>
          <div id="risk-consensus-multiplier" class="metric-val" style="color: var(--blue);">M_regime = 1.00</div>
          <div id="risk-consensus-sub" class="metric-sub" style="font-size: 0.78rem; color: var(--muted);">Unified ASRI: {resolved.asri_risk.score:.0f} • 100% Risk Budget</div>
        </div>
      </div>
      <div id="risk-committee-rationale" style="margin-top: 10px; font-size: 0.84rem; background: #0f172a; padding: 10px 14px; border-radius: 6px; border: 1px solid var(--border); color: #cbd5e1;">
        <strong>Committee Rationale:</strong> 3-Model MLX Risk Committee consensus: Benign macro, cross-sectional breadth, and low adversarial risk.
      </div>
    </div>

    <!-- WHY DID IT TRADE? (TRADE FORENSICS & FEATURE ATTRIBUTION) -->
    <div class="feed-box" style="margin-bottom: 20px;" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['trade_forensics'])}">
      <h3>
        <span>🔍 Why Did It Trade? — Trade Forensics &amp; Feature Attribution</span>
        <span id="forensic-verdict-badge" class="pos-pill pos-long">Trade Attribution Active</span>
      </h3>
      <div id="forensic-rationale-box" style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 14px 18px; margin: 10px 0; font-size: 0.92rem; line-height: 1.6;">
        <strong style="color: var(--green);">📢 Decision Rationale:</strong>
        <span id="forensic-rationale-text">Long BTC entered: 77% meta-conviction trend signal with 1.82 Mahalanobis distance within epistemic bound, low VPIN toxicity (0.32), and 15 ASRI benign macro regime.</span>
      </div>

      <h4 style="margin: 14px 0 8px; color: var(--blue); font-size: 0.9rem;">10-Factor Feature Attribution Breakdown:</h4>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Feature Name</th>
              <th>Description</th>
              <th>Normalized Value</th>
              <th>Significance</th>
            </tr>
          </thead>
          <tbody id="feature-attribution-tbody">
            <tr><td><code>ffd_close_zscore</code></td><td>Fractional Diff Price Trend Memory</td><td id="fa-ffd-close">+1.85</td><td><span class="pos-pill pos-long">Strong Trend Support</span></td></tr>
            <tr><td><code>ffd_oi_zscore</code></td><td>Stationary Open Interest Accumulation</td><td id="fa-ffd-oi">+1.40</td><td><span class="pos-pill pos-long">Informed Flow</span></td></tr>
            <tr><td><code>oi_funding_basis_zscore</code></td><td>OI / Funding Basis Divergence</td><td id="fa-basis-z">+0.82</td><td><span class="pos-pill pos-long">Positive Basis</span></td></tr>
            <tr><td><code>basis_pct</code></td><td>Mark-to-Oracle Basis</td><td id="fa-basis-pct">+0.018%</td><td><span class="pos-pill pos-long">Premium</span></td></tr>
            <tr><td><code>funding_rate_1h</code></td><td>Hyperliquid 1h Funding Rate</td><td id="fa-funding-1h">+0.0012%</td><td><span class="pos-pill pos-long">Yield Positive</span></td></tr>
            <tr><td><code>liquidation_zscore</code></td><td>Liquidation Volume Spike</td><td id="fa-liq-z">0.40</td><td><span class="pos-pill pos-flat">Normal</span></td></tr>
            <tr><td><code>regime_weight_bps</code></td><td>Stage-2 Expected Net Edge</td><td id="fa-edge-bps">+27.4 bps</td><td><span class="pos-pill pos-long">Above Hurdle</span></td></tr>
            <tr><td><code>kelly_conviction</code></td><td>Half-Kelly Fraction f_Kelly(p_meta)</td><td id="fa-kelly">26.5%</td><td><span class="pos-pill pos-long">High Conviction</span></td></tr>
            <tr><td><code>m_regime_multiplier</code></td><td>MLX Committee Risk Multiplier</td><td id="fa-m-regime">1.00</td><td><span class="pos-pill pos-long">Full Budget</span></td></tr>
            <tr><td><code>mahalanobis_gate</code></td><td>Epistemic OOD Distance (DI &le; 3.50)</td><td id="fa-di">DI = 1.82 (PASS)</td><td><span class="pos-pill pos-long">Safe In-Distribution</span></td></tr>
          </tbody>
        </table>
      </div>

      <details style="margin-top: 14px;">
        <summary style="cursor: pointer; color: var(--blue); font-weight: 600; font-size: 0.88rem;">📜 View Historical Forensics Audit Trail (Past Trades)</summary>
        <div style="overflow-x: auto; margin-top: 10px;">
          <table>
            <thead>
              <tr>
                <th>Trade ID</th>
                <th>Time (UTC)</th>
                <th>Action</th>
                <th>Fill Px</th>
                <th>Realized PnL</th>
                <th>Why Did It Trade?</th>
              </tr>
            </thead>
            <tbody id="forensics-audit-tbody">
              <tr>
                <td><code>tr_init_001</code></td>
                <td>01:30:00 UTC</td>
                <td><span class="pos-pill pos-long">ENTER_LONG</span></td>
                <td>$68,110.00</td>
                <td style="color: var(--green);">+$35.00</td>
                <td>Harvested 1h funding carry on BTC with ALO maker entry avoiding 4.5 bps taker fee and benign ASRI risk.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
    </div>

    <div class="live-grid">
      <div class="chart-box">
        <div id="candlestick-chart"></div>
      </div>

      <div>
        <div class="feed-box">
          <h3>
            <span>📡 Live Bot Activity Log</span>
            <span id="tick-badge" style="font-size: 0.78rem; color: var(--green);">● Ready</span>
          </h3>
          <ul id="activity-log-list" class="activity-list">
            <li class="activity-item">Click <strong>"▶ Start Dry-Run"</strong> above to paper-trade each closed Hyperliquid 15-minute candle as it arrives, or click <strong>"⚡ Poll Next 15m Bar"</strong> to check for the next closed candle now. A tick with no new candle changes nothing; wins and losses are counted only when a position closes.</li>
            <li class="activity-item">[Initial State] BTC @ ${latest_bar.close:,.2f} | AI Confidence: {latest_meta.p_meta:.0%} | All 4 Safety Shields Green.</li>
          </ul>
        </div>

        <div class="feed-box">
          <h3>📊 Current Target Positions</h3>
          <table>
            <thead>
              <tr>
                <th>Coin</th>
                <th>Direction</th>
                <th>Target Size</th>
              </tr>
            </thead>
            <tbody id="compact-weights-tbody">
              {"".join(
                  f"<tr><td><strong>{html.escape(a)}</strong></td>"
                  f"<td><span class='pos-pill {'pos-long' if w > 0 else ('pos-short' if w < 0 else 'pos-flat')}'>"
                  f"{'LONG' if w > 0 else ('SHORT' if w < 0 else 'CASH')}</span></td>"
                  f"<td><strong>{w:+.1%}</strong></td></tr>"
                  for a, w in resolved.portfolio_allocation.weights.items()
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- COLLAPSIBLE ADVANCED QUANT DRAWER SO IT DOES NOT CLUTTER THE SCREEN -->
    <details class="advanced-drawer">
      <summary>🔬 Show Under-the-Hood Quant Gauges (8 Advanced Technical Metrics — Hover Any Card for Plain-English Tooltip)</summary>
      <div class="badge-grid">
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['vpin_toxicity'])}">
          <span class="metric-title">VPIN Toxicity &amp; Shield (?)</span>
          <div id="adv-vpin" class="metric-val">{resolved.hawkes_state.vpin:.2f} — {shield_status}</div>
        </div>
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['hawkes_reflexivity'])}">
          <span class="metric-title">O(1) Hawkes Reflexivity eta (?)</span>
          <div id="adv-eta" class="metric-val">{resolved.hawkes_state.branching_ratio_eta:.2f} (Safe &le; 0.70)</div>
        </div>
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['meta_probability'])}">
          <span class="metric-title">LightGBM Meta-Probability p_meta (?)</span>
          <div id="adv-pmeta" class="metric-val">{latest_meta.p_meta:.1%} (+{latest_meta.expected_net_edge_bps:.1f} bps net edge)</div>
        </div>
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['mahalanobis_di'])}">
          <span class="metric-title">Mahalanobis DI_values Gate (?)</span>
          <div id="adv-di" class="metric-val">DI = {latest_meta.mahalanobis_di:.2f} / {latest_meta.di_threshold:.2f} ({'PASS' if latest_meta.epistemic_gate_passed else 'BLOCKED'})</div>
        </div>
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['squeeze_regime'])}">
          <span class="metric-title">Perpetual Squeeze Regime (?)</span>
          <div id="adv-regime" class="metric-val">{html.escape(latest_pos.regime.value)}</div>
        </div>
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['ffd_stationarity'])}">
          <span class="metric-title">FFD Stationarity (d* / ADF) (?)</span>
          <div class="metric-val">d*={resolved.ffd_config.d_star:.2f}, p_ADF={resolved.ffd_config.adf_pvalue:.4f}</div>
        </div>
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['triple_barrier'])}">
          <span class="metric-title">Triple-Barrier Profit Hurdle (?)</span>
          <div class="metric-val">&ge; 80.0 bps Net-of-Fee Target</div>
        </div>
        <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['asri_macro'])}">
          <span class="metric-title">Local Qwen3-8B ASRI Macro (?)</span>
          <div class="metric-val">{resolved.asri_risk.score:.1f}/100 (M_regime={resolved.portfolio_allocation.regime_multiplier:.2f})</div>
        </div>
      </div>
    </details>
  </section>

  <!-- TAB 2: PORTFOLIO & VALIDATION -->
  <section id="tab-portfolio" class="tab-panel">
    <div class="badge-grid" style="margin-bottom: 16px;">
      <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['net_of_fee_pnl'])}">
        <span class="metric-title">Live Net-of-Fee PnL (?)</span>
        <div class="metric-val">${resolved.pnl_metrics.net_pnl_usd:,.2f} (+{resolved.pnl_metrics.net_pnl_bps:.1f} bps)</div>
      </div>
      <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['maker_taker_ratio'])}">
        <span class="metric-title">Maker vs. Taker Fill Ratio (?)</span>
        <div class="metric-val">{resolved.pnl_metrics.maker_fill_ratio:.1%} Maker / {resolved.pnl_metrics.taker_fill_ratio:.1%} Taker</div>
      </div>
      <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['maker_taker_ratio'])}">
        <span class="metric-title">GTX Maker Fee Savings (?)</span>
        <div class="metric-val">${resolved.pnl_metrics.fee_savings_usd:,.2f} (+{resolved.pnl_metrics.fee_savings_bps:.1f} bps saved)</div>
      </div>
      <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['cpcv_gates'])}">
        <span class="metric-title">Champion vs. Challenger CPCV (?)</span>
        <div class="metric-val">PSR={resolved.champion_cpcv.psr_vs_zero:.1%} | DSR={resolved.champion_cpcv.dsr:.1%} | PBO={resolved.champion_cpcv.pbo:.1%}</div>
      </div>
    </div>

    <div class="chart-box" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['hrp_cvar_weights'])}">
      <div id="weights-chart"></div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Coin</th>
          <th>Plain-English Action</th>
          <th>Signed Target Weight (w_final,i)</th>
          <th>Base Risk Parity Weight (w_HRP,i)</th>
          <th>AI Confidence Sizing f_Kelly(p_meta)</th>
        </tr>
      </thead>
      <tbody id="full-weights-tbody">
        {weights_rows}
      </tbody>
    </table>

    <div class="chart-box" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['psr_dsr_pbo'])}">
      <div id="gates-chart"></div>
    </div>
  </section>

  <!-- TAB 3: HOW IT WORKS (PLAIN-ENGLISH GUIDE) -->
  <section id="tab-guide" class="tab-panel">
    <h2>Plain-English Operator Manuals &amp; Tooltip Encyclopedia</h2>
    <p>Every manual below is written in plain English and loaded from the <code>docs/</code> folder.</p>
    {"".join(docs_html_blocks)}
    <h3>Complete Plain-English Tooltip Registry</h3>
    <table>
      <thead>
        <tr><th>Metric / Control ID</th><th>Plain-English Explanation</th></tr>
      </thead>
      <tbody>
        {tooltip_rows}
      </tbody>
    </table>
  </section>

  <script>
    const candlestickSpec = {candlestick_json};
    const weightsSpec = {weights_json};
    const gatesSpec = {gates_json};
    Plotly.newPlot('candlestick-chart', candlestickSpec.data, candlestickSpec.layout, {{responsive: true}});
    Plotly.newPlot('weights-chart', weightsSpec.data, weightsSpec.layout, {{responsive: true}});
    Plotly.newPlot('gates-chart', gatesSpec.data, gatesSpec.layout, {{responsive: true}});

    let dryRunTimer = null;

    function switchTab(tabId, btnEl) {{
      document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btnEl.classList.add('active');
      window.dispatchEvent(new Event('resize'));
    }}

    function highlightActiveButton(action) {{
      ['start_dry_run', 'pause_entries', 'switch_carry_only', 'emergency_flatten'].forEach(id => {{
        const btn = document.getElementById('btn-' + id);
        if (btn) btn.classList.remove('active-mode');
      }});
      const activeBtn = document.getElementById('btn-' + action);
      if (activeBtn) activeBtn.classList.add('active-mode');

      const pill = document.getElementById('live-status-pill');
      const pillText = document.getElementById('status-pill-text');
      const dot = document.getElementById('status-dot');
      const tickBadge = document.getElementById('tick-badge');

      if (action === 'start_dry_run') {{
        pill.style.background = 'rgba(16, 185, 129, 0.2)';
        pill.style.color = '#34d399';
        pill.style.borderColor = '#10b981';
        dot.style.background = '#10b981';
        pillText.textContent = 'DRY-RUN RUNNING LIVE (Auto-Ticking Every 4s)';
        tickBadge.textContent = '● Live Auto-Ticking';
        tickBadge.style.color = '#10b981';
      }} else if (action === 'pause_entries') {{
        pill.style.background = 'rgba(245, 158, 11, 0.2)';
        pill.style.color = '#fbbf24';
        pill.style.borderColor = '#f59e0b';
        dot.style.background = '#f59e0b';
        pillText.textContent = 'PAUSED (No New Entries — Stop-Losses Active)';
        tickBadge.textContent = '⏸ Paused';
        tickBadge.style.color = '#fbbf24';
      }} else if (action === 'switch_carry_only') {{
        pill.style.background = 'rgba(56, 189, 248, 0.2)';
        pill.style.color = '#38bdf8';
        pill.style.borderColor = '#0284c7';
        dot.style.background = '#38bdf8';
        pillText.textContent = 'CARRY-ONLY MODE (Safe 8h Funding Yield Only)';
        tickBadge.textContent = '🏦 Carry-Only';
        tickBadge.style.color = '#38bdf8';
      }} else if (action === 'emergency_flatten') {{
        pill.style.background = 'rgba(239, 68, 68, 0.22)';
        pill.style.color = '#f87171';
        pill.style.borderColor = '#ef4444';
        dot.style.background = '#ef4444';
        pillText.textContent = 'STOPPED & FLATTENED (100% Cash)';
        tickBadge.textContent = '🛑 Stopped (100% Cash)';
        tickBadge.style.color = '#f87171';
      }}
    }}

    function applyLiveStateToDOM(state, explanationText) {{
      if (!state) return;

      document.getElementById('mode-indicator').textContent = state.regime_mode;
      const banner = document.getElementById('feedback-banner');
      banner.classList.remove('flash-highlight');
      void banner.offsetWidth;
      banner.classList.add('flash-highlight');

      if (explanationText) {{
        document.getElementById('banner-message').textContent = explanationText;
      }} else if (state.last_control_message) {{
        document.getElementById('banner-message').textContent = state.last_control_message;
      }}

      if (state.tick_count !== undefined) {{
        document.getElementById('bar-counter').textContent = state.tick_count;
      }}
      if (state.latest_price !== undefined) {{
        document.getElementById('live-btc-price').textContent = '$' + Number(state.latest_price).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
      }}

      // Update KPI Card 1: PnL on Persistent Paper Ledger ($5,000 base default)
      if (state.net_pnl_usd !== undefined || (state.account_ledger && state.account_ledger.total_realized_pnl !== undefined)) {{
        const startCap = Number(state.starting_capital_usd || (state.account_ledger && state.account_ledger.base_capital_usd) || 5000);
        const netPnl = state.net_pnl_usd !== undefined ? Number(state.net_pnl_usd) : Number(state.account_ledger.total_realized_pnl);
        const equity = state.current_equity_usd !== undefined ? Number(state.current_equity_usd) : (startCap + netPnl);
        const pnlSign = netPnl >= 0 ? '+' : '-';
        const absPnl = Math.abs(netPnl);
        const pnlEl = document.getElementById('kpi-pnl-val');
        const lblEl = document.getElementById('kpi-pnl-label');
        if (lblEl) {{
          lblEl.textContent = '1. Persistent Paper Ledger ($' + startCap.toLocaleString(undefined, {{maximumFractionDigits: 0}}) + ' Base) (?)';
        }}
        if (pnlEl) {{
          pnlEl.textContent = pnlSign + '$' + absPnl.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
          pnlEl.style.color = netPnl >= 0 ? 'var(--green)' : 'var(--red)';
        }}
        const netBps = state.net_pnl_bps !== undefined ? Number(state.net_pnl_bps) : (netPnl / startCap * 10000);
        const subEl = document.getElementById('kpi-pnl-sub');
        if (subEl) {{
          const ledger = state.account_ledger || {{}};
          const tradeCount = ledger.trade_count !== undefined ? ledger.trade_count : (state.tick_count || 0);
          const wins = ledger.win_count !== undefined ? ledger.win_count : (state.win_count || 0);
          const losses = ledger.loss_count !== undefined ? ledger.loss_count : (state.loss_count || 0);
          const winRate = ledger.win_rate_pct !== undefined ? Number(ledger.win_rate_pct).toFixed(1) : ((wins + losses > 0) ? ((wins / (wins + losses)) * 100).toFixed(1) : '0.0');
          subEl.innerHTML =
            'Equity: <strong>$' + equity.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '</strong> (' +
            (netBps >= 0 ? '+' : '') + netBps.toFixed(1) + ' bps net | Saved $' +
            Number(state.fee_savings_usd || 0).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + ' in fees)<br>' +
            '<span id="kpi-ledger-stats" style="color: var(--blue);">Ledger: ' + tradeCount + ' trades | Wins: ' + wins + ' / Losses: ' + losses + ' (' + winRate + '% win rate)</span>';
        }}
      }}

      // Update KPI Card 2 & Weights tables
      const weights = state.signed_hrp_cvar_weights || (state.portfolio_allocation && state.portfolio_allocation.weights) || {{}};
      const grossLev = state.gross_leverage !== undefined ? state.gross_leverage : 0;
      const activeCoins = Object.entries(weights).filter(([_, w]) => Math.abs(w) > 0.001);
      const actionValEl = document.getElementById('kpi-action-val');
      if (actionValEl) {{
        if (state.regime_mode === 'EMERGENCY_FLATTENED' || activeCoins.length === 0) {{
          actionValEl.textContent = '100% Cash (All Flat)';
          actionValEl.style.color = '#f87171';
        }} else if (state.regime_mode === 'PAUSED_ENTRIES') {{
          actionValEl.textContent = 'Paused (Holding Existing)';
          actionValEl.style.color = '#fbbf24';
        }} else if (state.regime_mode === 'CARRY_ONLY_FALLBACK') {{
          actionValEl.textContent = 'Hedged Carry Only';
          actionValEl.style.color = '#38bdf8';
        }} else {{
          actionValEl.textContent = activeCoins.map(([c, w]) => (w > 0 ? 'Long ' : 'Short ') + c.replace('USDT-PERP', '') + ' (' + (w * 100).toFixed(0) + '%)').join(', ');
          actionValEl.style.color = '#34d399';
        }}
      }}
      const actionSubEl = document.getElementById('kpi-action-sub');
      if (actionSubEl) {{
        actionSubEl.textContent = 'Total Leverage: ' + Number(grossLev).toFixed(2) + 'x (Safe Cap: 2.50x)';
      }}

      // Update KPI Card 3: 3-Model Risk Committee
      const rc = state.risk_committee;
      if (rc) {{
        const kpiRiskVal = document.getElementById('kpi-risk-val');
        if (kpiRiskVal) {{
          kpiRiskVal.textContent = 'M_regime = ' + Number(rc.regime_multiplier).toFixed(2);
        }}
        const kpiRiskSub = document.getElementById('kpi-risk-sub');
        if (kpiRiskSub) {{
          const sentStatus = rc.macro_sentinel_score < 40 ? 'Safe' : 'Elevated';
          const criticStatus = rc.red_team_critic_score < 40 ? 'Green' : 'Caution';
          kpiRiskSub.textContent = `Sentinel: ${{sentStatus}} (${{rc.macro_sentinel_score.toFixed(0)}}) | Critic: ${{criticStatus}} (${{rc.red_team_critic_score.toFixed(0)}})`;
        }}
      }}

      // Update KPI Card 4: Sleeve Carry & Vol Alpha
      const st = state.sleeve_telemetry;
      if (st && st.funding_carry) {{
        const fc = st.funding_carry;
        const kpiSleeveVal = document.getElementById('kpi-sleeve-val');
        if (kpiSleeveVal) {{
          kpiSleeveVal.textContent = Number(fc.annualized_apr_pct).toFixed(1) + '% APR';
        }}
        const kpiSleeveSub = document.getElementById('kpi-sleeve-sub');
        if (kpiSleeveSub) {{
          kpiSleeveSub.textContent = `ALO Maker: ${{fc.maker_rebate_bps}} bps (avoided ${{fc.taker_fee_avoided_bps}} bps taker)`;
        }}
      }}

      // Update Multi-Asset Feeds Table (BTC, ETH, SOL)
      if (state.multi_asset_snapshot && state.multi_asset_snapshot.assets) {{
        const assets = state.multi_asset_snapshot.assets;
        ['BTC', 'ETH', 'SOL'].forEach(coin => {{
          const a = assets[coin];
          if (!a) return;
          const prefix = coin.toLowerCase();
          const midEl = document.getElementById(`${{prefix}}-mid`);
          if (midEl) midEl.textContent = '$' + Number(a.mid_price).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
          const markEl = document.getElementById(`${{prefix}}-mark`);
          if (markEl) markEl.textContent = '$' + Number(a.mark_price).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
          const oracleEl = document.getElementById(`${{prefix}}-oracle`);
          if (oracleEl) oracleEl.textContent = '$' + Number(a.oracle_price).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
          const basisEl = document.getElementById(`${{prefix}}-basis`);
          if (basisEl) basisEl.textContent = (a.basis_bps >= 0 ? '+' : '') + Number(a.basis_bps).toFixed(1) + ' bps';
          const oiEl = document.getElementById(`${{prefix}}-oi`);
          if (oiEl) oiEl.textContent = Number(a.open_interest).toLocaleString(undefined, {{maximumFractionDigits: 0}});
          const fEl = document.getElementById(`${{prefix}}-funding`);
          if (fEl) fEl.textContent = (a.funding_rate_1h >= 0 ? '+' : '') + (a.funding_rate_1h * 100).toFixed(4) + '%';
          const aprEl = document.getElementById(`${{prefix}}-apr`);
          if (aprEl) aprEl.textContent = (a.annualized_funding_apr_pct >= 0 ? '+' : '') + Number(a.annualized_funding_apr_pct).toFixed(1) + '%';
          const spreadEl = document.getElementById(`${{prefix}}-spread`);
          if (spreadEl) spreadEl.textContent = Number(a.bid_ask_spread_bps).toFixed(1) + ' bps';
          const depthEl = document.getElementById(`${{prefix}}-depth`);
          if (depthEl) depthEl.textContent = '$' + (Number(a.depth_usd) / 1e6).toFixed(1) + 'M';
        }});
      }} else if (state.latest_price !== undefined) {{
        const btcMid = document.getElementById('btc-mid');
        if (btcMid) btcMid.textContent = '$' + Number(state.latest_price).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
        const btcMark = document.getElementById('btc-mark');
        if (btcMark) btcMark.textContent = '$' + (Number(state.latest_price) + 1.5).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
      }}

      // Update Quantitative Sleeve Telemetry Box
      if (st) {{
        const fc = st.funding_carry;
        if (fc) {{
          const carryAprEl = document.getElementById('sleeve-carry-apr');
          if (carryAprEl) carryAprEl.textContent = (fc.annualized_apr_pct >= 0 ? '+' : '') + Number(fc.annualized_apr_pct).toFixed(1) + '% APR';
          const crossSpreadEl = document.getElementById('sleeve-cross-spread');
          if (crossSpreadEl) crossSpreadEl.textContent = (fc.cross_venue_spread_bps === null || fc.cross_venue_spread_bps === undefined) ? 'n/a (no second venue feed)' : `${{Number(fc.cross_venue_spread_bps) >= 0 ? '+' : ''}}${{Number(fc.cross_venue_spread_bps).toFixed(1)}} bps (${{fc.venue_pair || 'HL vs Bybit'}})`;
          const breakevenEl = document.getElementById('sleeve-breakeven');
          if (breakevenEl) breakevenEl.textContent = `${{Number(fc.breakeven_hours).toFixed(1)}} hours`;
        }}
        const da = st.derivatives_alpha;
        if (da) {{
          const skewEl = document.getElementById('sleeve-skew');
          if (skewEl) skewEl.textContent = `${{(da.deribit_25d_skew * 100).toFixed(1)}}% (${{da.deribit_25d_skew >= 0 ? 'Bullish Demand' : 'Bearish Demand'}})`;
          const termEl = document.getElementById('sleeve-term');
          if (termEl) termEl.textContent = `${{da.term_structure_label || 'n/a'}} (Slope: ${{Number(da.term_structure_slope) >= 0 ? '+' : ''}}${{Number(da.term_structure_slope).toFixed(3)}})`;
          const gexEl = document.getElementById('sleeve-gex');
          if (gexEl) gexEl.textContent = `${{Number(da.net_dealer_gex_usd) >= 0 ? '+' : '-'}}$${{Math.abs(Number(da.net_dealer_gex_usd) / 1e6).toFixed(1)}}M (${{da.gex_regime || 'n/a'}})`;
          const vaultEl = document.getElementById('sleeve-vault');
          if (vaultEl) vaultEl.textContent = `${{da.hlp_crowding || 'Balanced'}} (${{da.counterparty_vulnerability || 'Low'}} Vuln)`;
        }}
      }}

      // Update Live Feed Availability
      if (state.live_feeds) {{
        const box = document.getElementById('feed-availability');
        if (box) {{
          box.innerHTML = '';
          Object.entries(state.live_feeds).forEach(([name, ok]) => {{
            const pill = document.createElement('span');
            pill.className = 'pos-pill ' + (ok ? 'pos-long' : 'pos-short');
            pill.textContent = (ok ? '● ' : '○ ') + name.replace(/_/g, ' ') + (ok ? '' : ' (missing)');
            box.appendChild(pill);
          }});
        }}
        const sub = document.getElementById('feed-availability-sub');
        const mi = state.macro_inputs || {{}};
        if (sub) {{
          const depeg = (mi.stablecoin_depeg_bps === null || mi.stablecoin_depeg_bps === undefined) ? 'n/a' : Number(mi.stablecoin_depeg_bps).toFixed(1) + ' bps';
          const dvol = (mi.dvol_index === null || mi.dvol_index === undefined) ? 'n/a' : Number(mi.dvol_index).toFixed(1) + ' (z ' + Number(mi.dvol_zscore || 0).toFixed(2) + ')';
          const src = state.data_availability ? state.data_availability.feed_source : 'n/a';
          let streamTxt = '';
          if (state.streams && state.streams.enabled && state.streams.assets && state.streams.assets.BTC) {{
            const b = state.streams.assets.BTC;
            const lead = (b.lead_mid_change_bps_700ms === null || b.lead_mid_change_bps_700ms === undefined) ? 'n/a' : Number(b.lead_mid_change_bps_700ms).toFixed(2) + ' bps';
            const bi = b.boundary_imbalance && b.boundary_imbalance.normalized ? b.boundary_imbalance.normalized['30'] : null;
            streamTxt = ` • BTC lead move (700 ms): ${{lead}} • Boundary imbalance (30 s): ${{bi === null || bi === undefined ? 'n/a' : Number(bi).toFixed(2)}}`;
          }}
          let breakerTxt = '';
          const cbs = state.sleeve_telemetry ? state.sleeve_telemetry.circuit_breakers : null;
          if (cbs && cbs.portfolio_multiplier !== undefined) {{
            breakerTxt = ` • Breakers: size x${{Number(cbs.portfolio_multiplier).toFixed(2)}}${{cbs.lockdown ? ' LOCKDOWN' : ''}}${{(cbs.disabled_sleeves || []).length ? ' disabled: ' + cbs.disabled_sleeves.join(', ') : ''}}`;
          }}
          sub.textContent = `Bar feed: ${{src}} • Stablecoin max peg deviation: ${{depeg}} • Deribit DVOL: ${{dvol}} • Hourly return rows: ${{state.data_availability ? state.data_availability.hourly_return_rows : 'n/a'}}${{streamTxt}}${{breakerTxt}}`;
        }}
      }}

      // Update 3-Model MLX Risk Committee Telemetry Box
      if (rc) {{
        const consensusBadge = document.getElementById('risk-consensus-badge');
        if (consensusBadge) consensusBadge.textContent = 'Consensus M_regime: ' + Number(rc.regime_multiplier).toFixed(2);
        const sentinelScore = document.getElementById('risk-sentinel-score');
        if (sentinelScore) sentinelScore.textContent = Number(rc.macro_sentinel_score).toFixed(1) + ' / 100';
        const sentinelSub = document.getElementById('risk-sentinel-sub');
        if (sentinelSub && rc.macro_sentinel) {{
          sentinelSub.textContent = `Peg Dev: ${{Number(rc.macro_sentinel.stablecoin_depeg_bps).toFixed(1)}} bps • DVOL: ${{Number(rc.macro_sentinel.dvol_index).toFixed(1)}} (z ${{Number(rc.macro_sentinel.dvol_zscore).toFixed(2)}}) • ${{rc.macro_sentinel.regime_label}}`;
        }}
        const breadthScore = document.getElementById('risk-breadth-score');
        if (breadthScore) breadthScore.textContent = Number(rc.breadth_analyst_score).toFixed(1) + ' / 100';
        const criticScore = document.getElementById('risk-critic-score');
        if (criticScore) criticScore.textContent = Number(rc.red_team_critic_score).toFixed(1) + ' / 100';
        const consensusMult = document.getElementById('risk-consensus-multiplier');
        if (consensusMult) consensusMult.textContent = 'M_regime = ' + Number(rc.regime_multiplier).toFixed(2);
        const consensusSub = document.getElementById('risk-consensus-sub');
        if (consensusSub) consensusSub.textContent = `Unified ASRI: ${{Number(rc.consensus_risk_score).toFixed(0)}} • ${{(rc.regime_multiplier * 100).toFixed(0)}}% Risk Budget`;
        const rationaleEl = document.getElementById('risk-committee-rationale');
        if (rationaleEl && rc.committee_rationale) rationaleEl.innerHTML = '<strong>Committee Rationale:</strong> ' + rc.committee_rationale;
      }}

      // Update "Why Did It Trade?" Forensics & 10-Factor Feature Attribution
      const forensic = state.latest_forensic || (Array.isArray(state.forensics) && state.forensics.length > 0 ? state.forensics[state.forensics.length - 1] : null);
      if (forensic) {{
        const rationaleTextEl = document.getElementById('forensic-rationale-text');
        if (rationaleTextEl && forensic.why_did_it_trade) {{
          rationaleTextEl.textContent = forensic.why_did_it_trade;
        }}
        const badgeEl = document.getElementById('forensic-verdict-badge');
        if (badgeEl && forensic.action) {{
          badgeEl.textContent = `${{forensic.action}} | Conviction: ${{(forensic.p_meta * 100).toFixed(0)}}%`;
        }}
        if (forensic.feature_attribution) {{
          const fa = forensic.feature_attribution;
          const setFa = (id, key, fmt) => {{
            const el = document.getElementById(id);
            if (el && fa[key] !== undefined) el.textContent = fmt(fa[key]);
          }};
          setFa('fa-ffd-close', 'ffd_close_zscore', v => (v >= 0 ? '+' : '') + Number(v).toFixed(2));
          setFa('fa-ffd-oi', 'ffd_oi_zscore', v => (v >= 0 ? '+' : '') + Number(v).toFixed(2));
          setFa('fa-basis-z', 'oi_funding_basis_zscore', v => (v >= 0 ? '+' : '') + Number(v).toFixed(2));
          setFa('fa-basis-pct', 'basis_pct', v => (v >= 0 ? '+' : '') + (Number(v) * 100).toFixed(3) + '%');
          setFa('fa-funding-1h', 'funding_rate_1h', v => (v >= 0 ? '+' : '') + (Number(v) * 100).toFixed(4) + '%');
          setFa('fa-liq-z', 'liquidation_zscore', v => Number(v).toFixed(2));
          setFa('fa-edge-bps', 'regime_weight_bps', v => (v >= 0 ? '+' : '') + Number(v).toFixed(1) + ' bps');
          setFa('fa-kelly', 'kelly_conviction', v => (Number(v) * 100).toFixed(1) + '%');
          setFa('fa-m-regime', 'm_regime_multiplier', v => Number(v).toFixed(2));
          const diEl = document.getElementById('fa-di');
          if (diEl && fa['mahalanobis_gate'] !== undefined) {{
            diEl.textContent = `DI = ${{Number(fa['mahalanobis_gate']).toFixed(2)}} (${{fa['mahalanobis_gate'] <= 3.5 ? 'PASS' : 'BLOCKED'}})`;
          }}
        }}
      }}

      // Update Forensics Audit Trail Table
      const historyList = state.forensics || state.trade_history;
      if (Array.isArray(historyList) && historyList.length > 0) {{
        const auditRows = historyList.slice(-10).reverse().map(rec => {{
          const trId = rec.trade_id || 'tr_live';
          const time = rec.timestamp_utc ? rec.timestamp_utc.slice(11, 19) + ' UTC' : 'Live';
          const act = rec.action || 'HOLD';
          const actCls = act.includes('LONG') || act.includes('BUY') ? 'pos-long' : (act.includes('SHORT') || act.includes('SELL') ? 'pos-short' : 'pos-flat');
          const px = rec.fill_price ? '$' + Number(rec.fill_price).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) : '-';
          const pnlVal = rec.realized_pnl_usd !== undefined ? Number(rec.realized_pnl_usd) : 0;
          const pnlStr = (pnlVal >= 0 ? '+$' : '-$') + Math.abs(pnlVal).toFixed(2);
          const pnlColor = pnlVal >= 0 ? 'var(--green)' : 'var(--red)';
          const why = rec.why_did_it_trade || rec.reason || '-';
          return `<tr><td><code>${{trId}}</code></td><td>${{time}}</td><td><span class="pos-pill ${{actCls}}">${{act}}</span></td><td>${{px}}</td><td style="color: ${{pnlColor}}; font-weight: 600;">${{pnlStr}}</td><td>${{why}}</td></tr>`;
        }}).join('');
        const tbody = document.getElementById('forensics-audit-tbody');
        if (tbody) tbody.innerHTML = auditRows;
      }}

      // Update KPI Card 2 & Weights tables
      const weights = state.signed_hrp_cvar_weights || (state.portfolio_allocation && state.portfolio_allocation.weights) || {{}};
      const grossLev = state.gross_leverage !== undefined ? state.gross_leverage : 0;
      const activeCoins = Object.entries(weights).filter(([_, w]) => Math.abs(w) > 0.001);
      const actionValEl = document.getElementById('kpi-action-val');
      if (state.regime_mode === 'EMERGENCY_FLATTENED' || activeCoins.length === 0) {{
        actionValEl.textContent = '100% Cash (All Flat)';
        actionValEl.style.color = '#f87171';
      }} else if (state.regime_mode === 'PAUSED_ENTRIES') {{
        actionValEl.textContent = 'Paused (Holding Existing)';
        actionValEl.style.color = '#fbbf24';
      }} else if (state.regime_mode === 'CARRY_ONLY_FALLBACK') {{
        actionValEl.textContent = 'Hedged Carry Only';
        actionValEl.style.color = '#38bdf8';
      }} else {{
        actionValEl.textContent = activeCoins.map(([c, w]) => (w > 0 ? 'Long ' : 'Short ') + c.replace('USDT-PERP', '') + ' (' + (w * 100).toFixed(0) + '%)').join(', ');
        actionValEl.style.color = '#34d399';
      }}
      document.getElementById('kpi-action-sub').textContent = 'Total Leverage: ' + Number(grossLev).toFixed(2) + 'x (Safe Cap: 2.50x)';

      // Update compact positions table
      const compactRows = Object.entries(weights).map(([coin, w]) => {{
        const numW = Number(w);
        const cls = numW > 0 ? 'pos-long' : (numW < 0 ? 'pos-short' : 'pos-flat');
        const label = numW > 0 ? 'LONG (BUY)' : (numW < 0 ? 'SHORT (SELL)' : 'CASH (FLAT)');
        const pct = (numW >= 0 ? '+' : '') + (numW * 100).toFixed(1) + '%';
        return `<tr><td><strong>${{coin}}</strong></td><td><span class="pos-pill ${{cls}}">${{label}}</span></td><td><strong>${{pct}}</strong></td></tr>`;
      }}).join('');
      document.getElementById('compact-weights-tbody').innerHTML = compactRows;

      // Update activity log list
      if (Array.isArray(state.activity_log) && state.activity_log.length > 0) {{
        const logHtml = state.activity_log.map(item => `<li class="activity-item">${{item}}</li>`).join('');
        document.getElementById('activity-log-list').innerHTML = logHtml;
      }}

      // Update Plotly charts smoothly
      if (state.candlestick_figure) {{
        Plotly.react('candlestick-chart', state.candlestick_figure.data, state.candlestick_figure.layout, {{responsive: true}});
      }}
      if (state.weights_figure) {{
        Plotly.react('weights-chart', state.weights_figure.data, state.weights_figure.layout, {{responsive: true}});
      }}
    }}

    async function refreshFromServer() {{
      try {{
        const resp = await fetch('/api/state');
        const state = await resp.json();
        applyLiveStateToDOM(state, null);
      }} catch (_err) {{}}
    }}

    async function stepSingleBar() {{
      const resp = await fetch('/api/tick', {{ method: 'POST' }});
      const state = await resp.json();
      applyLiveStateToDOM(state, state.last_control_message);
    }}

    async function resetCapital() {{
      const inputEl = document.getElementById('starting-capital-input');
      const rawVal = inputEl ? parseFloat(inputEl.value) : 5000;
      const capUsd = (!isNaN(rawVal) && rawVal >= 100) ? rawVal : 5000;
      const resp = await fetch('/api/reset', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ starting_capital_usd: capUsd }})
      }});
      const data = await resp.json();
      if (data && data.live_state) {{
        applyLiveStateToDOM(data.live_state, data.plain_english_explanation);
      }}
    }}

    async function sendControl(action) {{
      highlightActiveButton(action);
      const resp = await fetch('/api/control/' + action, {{ method: 'POST' }});
      const data = await resp.json();
      if (data && data.live_state) {{
        applyLiveStateToDOM(data.live_state, data.plain_english_explanation);
      }}
    }}

    // Poll server state every 2.5s (server advances bars every 4.0s when Dry-Run is active)
    dryRunTimer = setInterval(refreshFromServer, 2500);
  </script>
</body>
</html>"""
