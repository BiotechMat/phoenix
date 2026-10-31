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
}


class PnLAndExecutionMetrics(BaseModel):
    """Live net-of-fee PnL, Maker/Taker ratio, and fee-saving telemetry."""

    model_config = ConfigDict(frozen=True)

    net_pnl_usd: float = Field(default=4820.50)
    net_pnl_bps: float = Field(default=142.8)
    gross_pnl_usd: float = Field(default=5410.00)
    total_fees_paid_usd: float = Field(default=589.50)
    maker_fill_ratio: float = Field(default=0.914, ge=0.0, le=1.0)
    taker_fill_ratio: float = Field(default=0.086, ge=0.0, le=1.0)
    fee_savings_usd: float = Field(default=1124.75, ge=0.0)
    fee_savings_bps: float = Field(default=31.5, ge=0.0)
    annualized_carry_apr_pct: float = Field(default=14.8)
    is_dry_run: bool = Field(default=True)


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
        vertical_spacing=0.06,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            f"{resolved.active_instrument} Dynamic Dollar Bars + Triple-Barrier Bands & Squeeze Regimes",
            "Stage-2 LightGBM Meta-Probability (p_meta) & VPIN / O(1) Hawkes Shield",
            "Mahalanobis Outlier Gate (DI_values) & ASRI Macro Risk Score",
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
            name="Dollar Bars (OHLC)",
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
                name="Triple-Barrier Profit Target (>=80 bps)",
                line={"color": "#34d399", "width": 2, "dash": "dash"},
                hovertemplate="Profit Barrier: $%{y:,.2f}<extra>Triple-Barrier Top Wall</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_barriers)],
                y=[b.lower_barrier for b in use_barriers],
                mode="lines",
                name="Triple-Barrier Stop-Loss Band",
                line={"color": "#f87171", "width": 2, "dash": "dot"},
                hovertemplate="Stop Barrier: $%{y:,.2f}<extra>Triple-Barrier Bottom Wall</extra>",
            ),
            row=1,
            col=1,
        )

    # Trace 3: Funding / OI Squeeze Regime Markers
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
                y=[closes[i] + 140.0 for i in range(min(len(use_pos), len(closes)))],
                mode="markers+text",
                name="Funding/OI Squeeze Regimes",
                text=[p.regime.value.replace("_", " ") for p in use_pos],
                textposition="top center",
                marker={
                    "size": 11,
                    "color": [regime_colors.get(p.regime, "#94a3b8") for p in use_pos],
                    "symbol": "diamond",
                },
                hovertemplate=(
                    "<b>%{text}</b><br>Price: $%{y:,.2f}<extra>Perpetual Positioning & Squeeze Regime</extra>"
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
                name="LightGBM Meta-Probability (p_meta)",
                line={"color": "#60a5fa", "width": 2.5},
                hovertemplate="p_meta: %{y:.2%}<extra>Sniper Quality Inspector (Pass > 50%)</extra>",
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
                name="VPIN Toxicity (Threshold 0.60)",
                line={"color": "#f59e0b", "width": 2},
                hovertemplate="VPIN: %{y:.2f}<extra>Order Book Toxicity Shield</extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_hawkes)],
                y=[h.branching_ratio_eta for h in use_hawkes],
                mode="lines",
                name="O(1) Hawkes Reflexivity (eta, Threshold 0.70)",
                line={"color": "#ec4899", "width": 2, "dash": "dashdot"},
                hovertemplate="Hawkes eta: %{y:.2f}<extra>Self-Exciting Avalanche Meter</extra>",
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
                name="Mahalanobis DI_values (Outlier Distance)",
                line={"color": "#c084fc", "width": 2.5},
                hovertemplate="Mahalanobis DI: %{y:.2f}<extra>Uncharted Territory Detector</extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels[: len(use_meta)],
                y=[m.di_threshold for m in use_meta],
                mode="lines",
                name="Mahalanobis Safety Ceiling (DI Threshold)",
                line={"color": "#ef4444", "width": 1.5, "dash": "dash"},
                hovertemplate="Max Safe DI: %{y:.2f}<extra>Epistemic Gate Veto Line</extra>",
            ),
            row=3,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=[use_asri.score / 25.0 for _ in x_labels],
            mode="lines",
            name=f"ASRI Macro Score ({use_asri.score:.1f}/100 - {use_asri.plain_english_summary[:48]}...)",
            line={"color": "#2dd4bf", "width": 2, "dash": "dot"},
            customdata=[use_asri.plain_english_summary for _ in x_labels],
            hovertemplate="ASRI Scaled: %{y:.2f}<br>%{customdata}<extra>Qwen3-8B Macro Risk</extra>",
        ),
        row=3,
        col=1,
    )

    n_traces = len(fig.data)
    all_visible = [True] * n_traces

    def _mask(indices: set[int]) -> list[bool]:
        return [i in indices or i == 0 for i in range(n_traces)]

    fig.update_layout(
        title={
            "text": (
                f"<b>{resolved.active_instrument} Live Dollar-Bar Cockpit</b> — "
                f"ASRI: {use_asri.score:.1f}/100 | {use_asri.plain_english_summary}"
            ),
            "font": {"size": 14, "color": "#f8fafc"},
        },
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=760,
        margin={"l": 55, "r": 35, "t": 95, "b": 45},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.13,
                "bgcolor": "#1e293b",
                "font": {"color": "#e2e8f0", "size": 11},
                "buttons": [
                    {"label": "All Overlays", "method": "update", "args": [{"visible": all_visible}]},
                    {"label": "Triple-Barrier Bands", "method": "update", "args": [{"visible": _mask({1, 2})}]},
                    {"label": "Squeeze Regimes", "method": "update", "args": [{"visible": _mask({3})}]},
                    {"label": "Meta-Prob (p_meta)", "method": "update", "args": [{"visible": _mask({4})}]},
                    {"label": "VPIN & Hawkes Shield", "method": "update", "args": [{"visible": _mask({5, 6})}]},
                    {"label": "Mahalanobis DI & ASRI", "method": "update", "args": [{"visible": _mask({7, 8, 9})}]},
                ],
            }
        ],
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
                f"<b>Signed HRP-CVaR Target Portfolio Weights (w_final,i)</b> — "
                f"Gross Leverage: {alloc.gross_leverage:.2f}x / 2.50x Cap | "
                f"Regime Multiplier (M_regime): {alloc.regime_multiplier:.2f}"
            ),
            "font": {"size": 14, "color": "#f8fafc"},
        },
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=420,
        margin={"l": 50, "r": 30, "t": 70, "b": 45},
        yaxis_title="Portfolio Weight / Fraction of Equity",
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
                f"Champion Sharpe: {champ.sharpe_net:.2f} ({champ.expected_net_pnl_bps:.1f} bps) vs. "
                f"Challenger Sharpe: {chall.sharpe_net:.2f} ({chall.expected_net_pnl_bps:.1f} bps)"
            ),
            "font": {"size": 14, "color": "#f8fafc"},
        },
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=420,
        margin={"l": 50, "r": 30, "t": 70, "b": 45},
        yaxis_title="Score / Ratio vs. Gate Threshold",
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
    """Render the complete 3-tab HTML Control Center with hover tooltips and embedded manuals."""
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

    latest_meta = resolved.meta_predictions[-1]
    latest_pos = resolved.positioning_signals[-1]
    shield_status = "TOXIC SWEEP ACTIVE (ORDERS PULLED)" if resolved.hawkes_state.toxic_sweep_active else "SAFE (MAKER GTX ACTIVE)"

    weights_rows = "".join(
        f"<tr>"
        f"<td><strong>{html.escape(asset)}</strong></td>"
        f"<td>{w:+.4f} ({w:+.2%})</td>"
        f"<td>{resolved.portfolio_allocation.hrp_unsigned_weights.get(asset, 0.0):.4f}</td>"
        f"<td>{resolved.portfolio_allocation.kelly_scalars.get(asset, 0.0):.2%}</td>"
        f"</tr>"
        for asset, w in resolved.portfolio_allocation.weights.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gemini Argon 4 — M1 Max Quantitative Derivatives Control Center</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    :root {{
      --bg: #0b1120;
      --panel: #1e293b;
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
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: #0f172a;
      border-bottom: 1px solid var(--border);
      gap: 12px;
    }}
    .brand h1 {{ margin: 0; font-size: 1.25rem; }}
    .brand p {{ margin: 2px 0 0; color: var(--muted); font-size: 0.85rem; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .btn {{
      border: 1px solid var(--border);
      padding: 9px 14px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      color: #fff;
      font-size: 0.85rem;
    }}
    .btn-dryrun {{ background: #059669; }}
    .btn-pause {{ background: #d97706; }}
    .btn-carry {{ background: #0284c7; }}
    .btn-flatten {{ background: #dc2626; }}
    .asri-banner {{
      margin: 14px 24px;
      padding: 12px 18px;
      background: #172554;
      border-left: 4px solid var(--blue);
      border-radius: 8px;
      font-size: 0.92rem;
    }}
    .badge-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
      gap: 12px;
      padding: 0 24px 16px;
    }}
    .metric-card {{
      background: var(--panel);
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
      font-size: 1.15rem;
      font-weight: 700;
      margin-top: 6px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 0 24px;
      border-bottom: 1px solid var(--border);
    }}
    .tab-btn {{
      background: transparent;
      color: var(--muted);
      border: none;
      border-bottom: 3px solid transparent;
      padding: 12px 18px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.95rem;
    }}
    .tab-btn.active {{
      color: var(--blue);
      border-bottom-color: var(--blue);
    }}
    .tab-panel {{ display: none; padding: 20px 24px; }}
    .tab-panel.active {{ display: block; }}
    .chart-box {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 18px;
    }}
    th, td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 0.9rem;
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
    <div class="brand">
      <h1>Gemini Argon 4 — M1 Max (64GB) Quantitative Derivatives Control Center</h1>
      <p>Mode: <strong id="mode-indicator">{html.escape(resolved.regime_mode.value)}</strong> | {html.escape(resolved.last_control_message)}</p>
    </div>
    <div class="controls">
      <button class="btn btn-dryrun" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['start_dry_run'])}" onclick="sendControl('start_dry_run')">Start Dry-Run</button>
      <button class="btn btn-pause" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['pause_entries'])}" onclick="sendControl('pause_entries')">Pause Entries</button>
      <button class="btn btn-carry" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['switch_carry_only'])}" onclick="sendControl('switch_carry_only')">Switch to Carry-Only</button>
      <button class="btn btn-flatten" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['emergency_flatten'])}" onclick="sendControl('emergency_flatten')">Emergency Flatten All</button>
    </div>
  </header>

  <div class="asri-banner" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['asri_macro'])}">
    <strong>🧠 Local Qwen3-8B-4bit ASRI Macro Weather ({resolved.asri_risk.score:.1f}/100, M_regime={resolved.portfolio_allocation.regime_multiplier:.2f}):</strong>
    {html.escape(resolved.asri_risk.plain_english_summary)}
  </div>

  <div class="badge-grid">
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['vpin_toxicity'])}">
      <span class="metric-title">VPIN Toxicity &amp; Shield (?)</span>
      <div class="metric-val">{resolved.hawkes_state.vpin:.2f} — {shield_status}</div>
    </div>
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['hawkes_reflexivity'])}">
      <span class="metric-title">O(1) Hawkes Reflexivity eta (?)</span>
      <div class="metric-val">{resolved.hawkes_state.branching_ratio_eta:.2f} (Limit 0.70)</div>
    </div>
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['meta_probability'])}">
      <span class="metric-title">LightGBM Meta-Probability p_meta (?)</span>
      <div class="metric-val">{latest_meta.p_meta:.1%} (+{latest_meta.expected_net_edge_bps:.1f} bps net edge)</div>
    </div>
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['mahalanobis_di'])}">
      <span class="metric-title">Mahalanobis DI_values Gate (?)</span>
      <div class="metric-val">DI = {latest_meta.mahalanobis_di:.2f} / {latest_meta.di_threshold:.2f} ({'PASS' if latest_meta.epistemic_gate_passed else 'BLOCKED'})</div>
    </div>
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['squeeze_regime'])}">
      <span class="metric-title">Perpetual Squeeze Regime (?)</span>
      <div class="metric-val">{html.escape(latest_pos.regime.value)}</div>
    </div>
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['ffd_stationarity'])}">
      <span class="metric-title">FFD Stationarity (d* / ADF) (?)</span>
      <div class="metric-val">d*={resolved.ffd_config.d_star:.2f}, p_ADF={resolved.ffd_config.adf_pvalue:.4f}</div>
    </div>
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['triple_barrier'])}">
      <span class="metric-title">Triple-Barrier Profit Hurdle (?)</span>
      <div class="metric-val">&ge; 80.0 bps Net-of-Fee Target</div>
    </div>
    <div class="metric-card" title="{html.escape(PLAIN_ENGLISH_TOOLTIPS['funding_carry'])}">
      <span class="metric-title">Delta-Neutral Funding Carry (?)</span>
      <div class="metric-val">{resolved.pnl_metrics.annualized_carry_apr_pct:.1f}% APR (8h Harvester)</div>
    </div>
  </div>

  <nav class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-live', this)">1. Live Trading &amp; Charts Tab</button>
    <button class="tab-btn" onclick="switchTab('tab-portfolio', this)">2. Portfolio &amp; Validation Tab</button>
    <button class="tab-btn" onclick="switchTab('tab-guide', this)">3. How It Works (Plain-English Guide) Tab</button>
  </nav>

  <!-- TAB 1: LIVE TRADING & CHARTS -->
  <section id="tab-live" class="tab-panel active">
    <div class="chart-box">
      <div id="candlestick-chart"></div>
    </div>
  </section>

  <!-- TAB 2: PORTFOLIO & VALIDATION -->
  <section id="tab-portfolio" class="tab-panel">
    <div class="badge-grid">
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
          <th>Asset Sleeve</th>
          <th>Signed Target Weight (w_final,i)</th>
          <th>Base HRP-CVaR Weight (w_HRP,i)</th>
          <th>Half-Kelly Conviction f_Kelly(p_meta)</th>
        </tr>
      </thead>
      <tbody>
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
    <p>Every manual below is loaded directly from the <code>docs/</code> suite in your workspace.</p>
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

    function switchTab(tabId, btnEl) {{
      document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btnEl.classList.add('active');
      window.dispatchEvent(new Event('resize'));
    }}

    async function sendControl(action) {{
      const resp = await fetch('/api/control/' + action, {{ method: 'POST' }});
      const data = await resp.json();
      if (data && data.regime_mode) {{
        document.getElementById('mode-indicator').textContent = data.regime_mode;
      }}
    }}
  </script>
</body>
</html>"""
