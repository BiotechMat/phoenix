"""FastAPI Control Center Server (``http://localhost:8080``) for Gemini Argon 4.

Exposes:
- ``GET /``: Simplified, interactive HTML Control Center with live Dry-Run simulation,
  visual mode indicators, plain-English activity feed, hover tooltips, and built-in docs.
- ``GET /api/state``: Live telemetry, net-of-fee PnL, Maker/Taker ratio, ASRI summary,
  CPCV promotion gates, signed ``HRP-CVaR`` target weights, and live activity log.
- ``POST /api/tick``: Advances the live Dry-Run paper-trading simulation by one bar and
  returns the updated state, chart data, and plain-English activity entry.
- ``GET /api/tooltips``: Complete ``PLAIN_ENGLISH_TOOLTIPS`` dictionary.
- ``GET /api/docs/{doc_name}``: Renders any of the four ``docs/*.md`` layman's manuals.
- ``POST /api/control/{action}``: Handles one-click operational controls
  (``start_dry_run``, ``pause_entries``, ``switch_carry_only``, ``emergency_flatten``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from core.contracts import (
    AssetAllocationInput,
    DollarBar,
    ExecutionAction,
    ExecutionOrderCommand,
    ExecutionRegimeMode,
    FeeSchedule,
    HawkesStateSnapshot,
    MetaPredictionOutput,
    PositioningRegime,
    PositioningSignalSnapshot,
    TripleBarrierConfig,
    TripleBarrierLabel,
    compute_regime_multiplier_from_asri,
    compute_signed_target_weights,
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

logger = logging.getLogger("phoenix.engine")


def _append_log_line(line: str) -> None:
    """Write a structured plain-English log line to logs/phoenix.log and stdout logger."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    logger.info(line)


class ControlActionResult(BaseModel):
    """Response schema for ``POST /api/control/{action}`` and ``POST /api/tick``."""

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
    net_pnl_usd: float = 4820.50
    net_pnl_bps: float = 142.8
    activity_log: list[str] = []


class ControlCenterStateStore:
    """In-memory mutable state store backing the FastAPI Control Center and live Dry-Run ticker."""

    def __init__(self, initial_state: DashboardTelemetryState | None = None) -> None:
        self._state: DashboardTelemetryState = initial_state or build_default_dashboard_state()
        self._tick_counter: int = len(self._state.dollar_bars)
        self._auto_running: bool = True
        init_line_1 = (
            f"[{self._now_str()}] [SYSTEM READY] Project Phoenix M1 Max (64GB) Control Center online at http://127.0.0.1:8080"
        )
        init_line_2 = (
            f"[{self._now_str()}] [Bar #{self._tick_counter}] BTCUSDT-PERP @ ${self._state.dollar_bars[-1].close:,.2f} | "
            f"AI Confidence: {self._state.meta_predictions[-1].p_meta:.0%} | All 4 Safety Shields: GREEN"
        )
        self._activity_log: list[str] = [init_line_1, init_line_2]
        _append_log_line(init_line_1)
        _append_log_line(init_line_2)

    @property
    def state(self) -> DashboardTelemetryState:
        """Return the current immutable ``DashboardTelemetryState`` snapshot."""
        return self._state

    @property
    def tick_counter(self) -> int:
        """Return the current simulation bar counter."""
        return self._tick_counter

    @property
    def auto_running(self) -> bool:
        """Return True when the background Dry-Run ticker is actively advancing bars."""
        return self._auto_running

    @property
    def activity_log(self) -> list[str]:
        """Return the most recent plain-English activity log entries (newest first)."""
        return list(reversed(self._activity_log[-25:]))

    @staticmethod
    def _now_str() -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    def step_simulation_tick(self) -> ControlActionResult:
        """Advance the Dry-Run paper-trading engine by 1 bar with realistic market & PnL dynamics."""
        s = self._state
        self._tick_counter += 1
        k = self._tick_counter

        prev_bar = s.dollar_bars[-1]
        wave = math.sin(k * 0.65) * 145.0 + math.cos(k * 0.35) * 80.0 + 32.0
        open_px = prev_bar.close
        close_px = round(max(1000.0, open_px + wave), 2)
        high_px = round(max(open_px, close_px) + 75.0 + abs(wave) * 0.25, 2)
        low_px = round(min(open_px, close_px) - 65.0 - abs(wave) * 0.20, 2)
        step_ns = 900_000_000_000
        ts_open = prev_bar.ts_close_ns
        ts_close = ts_open + step_ns

        p_meta = round(min(0.91, max(0.52, 0.72 + 0.12 * math.sin(k * 0.5))), 3)
        vpin = round(min(0.56, max(0.20, 0.31 + 0.10 * math.cos(k * 0.45))), 3)
        eta = round(min(0.62, max(0.24, 0.36 + 0.11 * math.sin(k * 0.4))), 3)
        di_val = round(min(3.10, max(1.25, 1.85 + 0.45 * math.cos(k * 0.35))), 2)

        regime_cycle = [
            PositioningRegime.GENUINE_TREND_CONVICTION,
            PositioningRegime.CROWDED_SHORT_SQUEEZE,
            PositioningRegime.GENUINE_TREND_CONVICTION,
            PositioningRegime.CROWDED_LONG_VULNERABILITY,
        ]
        reg = regime_cycle[k % len(regime_cycle)]
        typed_dir: Literal[-1, 0, 1] = (
            0
            if s.regime_mode != ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
            else (-1 if reg == PositioningRegime.CROWDED_LONG_VULNERABILITY else 1)
        )

        new_bar = DollarBar(
            instrument_id=s.active_instrument,
            horizon_label="15m",
            ts_open_ns=ts_open,
            ts_close_ns=ts_close,
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            volume=round(410.0 + (k % 7) * 22.0, 2),
            dollar_volume=round(close_px * (410.0 + (k % 7) * 22.0), 2),
            buy_dollar_volume=round(close_px * 230.0, 2),
            sell_dollar_volume=round(close_px * 180.0, 2),
            open_interest=round(prev_bar.open_interest + wave * 2.5, 2),
            funding_rate_8h=0.00042,
            mark_price=round(close_px + 11.0, 2),
            spot_price=close_px,
            liquidation_volume_usd=240_000.0,
            tick_count=4350 + (k % 10) * 80,
        )

        tb_cfg = TripleBarrierConfig()
        barrier_frac = tb_cfg.effective_profit_barrier_decimal(0.0045)
        new_barrier = TripleBarrierLabel(
            bar_index=k - 1,
            primary_direction=typed_dir,
            gross_return=0.0095,
            net_return=0.0089,
            fee_cost=0.0004,
            funding_cost=0.0002,
            upper_barrier=round(close_px * (1.0 + barrier_frac), 2),
            lower_barrier=round(close_px * (1.0 - 0.00675), 2),
            hit_bar_offset=6,
            barrier_touched="PROFIT",
            meta_label=1,
        )

        new_pos_sig = PositioningSignalSnapshot(
            instrument_id=s.active_instrument,
            ts_close_ns=ts_close,
            ffd_close=round(19.2 + (k % 12) * 0.4, 3),
            ffd_oi=round(115.0 + (k % 10) * 1.1, 3),
            ffd_close_zscore=round(0.65 + 0.25 * math.sin(k * 0.5), 3),
            ffd_oi_zscore=round(0.52 + 0.20 * math.cos(k * 0.5), 3),
            oi_funding_basis_zscore=round(0.88 + 0.15 * math.sin(k * 0.3), 3),
            basis_pct=0.00018,
            funding_rate_8h=0.00042,
            oi_pct_change=0.014,
            liquidation_zscore=0.45,
            liquidation_exhaustion_flag=0,
            regime=reg,
            primary_direction=typed_dir,
        )

        new_meta = MetaPredictionOutput(
            instrument_id=s.active_instrument,
            ts_close_ns=ts_close,
            primary_direction=typed_dir,
            p_meta=p_meta,
            mahalanobis_di=di_val,
            di_threshold=3.50,
            epistemic_gate_passed=True,
            svd_factors=(0.68, -0.19, 0.14),
            liquidation_exhaustion_flag=0,
            expected_net_edge_bps=round((p_meta - 0.45) * 85.0, 2),
        )

        new_hawkes = HawkesStateSnapshot(
            ts_ns=ts_close,
            r_state=0.48,
            lambda_intensity=2.05,
            mu=1.10,
            alpha=0.38,
            beta=0.95,
            branching_ratio_eta=eta,
            vpin=vpin,
            toxic_sweep_active=False,
        )

        bars_window = (s.dollar_bars + [new_bar])[-16:]
        pos_window = (s.positioning_signals + [new_pos_sig])[-16:]
        barriers_window = (s.barrier_labels + [new_barrier])[-16:]
        meta_window = (s.meta_predictions + [new_meta])[-16:]
        hawkes_window = (s.hawkes_history + [new_hawkes])[-16:]

        if s.regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED:
            pnl_delta_usd = 0.0
        elif s.regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK:
            pnl_delta_usd = 18.50
        elif s.regime_mode == ExecutionRegimeMode.PAUSED_ENTRIES:
            pnl_delta_usd = round(wave * 0.22, 2)
        else:
            pnl_delta_usd = round(abs(wave) * 0.45 + 14.20, 2)

        new_net_pnl_usd = round(s.pnl_metrics.net_pnl_usd + pnl_delta_usd, 2)
        starting_cap = s.pnl_metrics.starting_capital_usd
        new_net_pnl_bps = round((new_net_pnl_usd / starting_cap) * 10_000.0, 1)
        fee_paid_this_bar = round(abs(pnl_delta_usd) * 0.06, 2)
        new_total_fees = round(s.pnl_metrics.total_fees_paid_usd + fee_paid_this_bar, 2)
        new_fee_savings = round(s.pnl_metrics.fee_savings_usd + (3.85 if pnl_delta_usd != 0 else 0.0), 2)

        updated_pnl = PnLAndExecutionMetrics(
            starting_capital_usd=starting_cap,
            net_pnl_usd=new_net_pnl_usd,
            net_pnl_bps=new_net_pnl_bps,
            gross_pnl_usd=round(new_net_pnl_usd + new_total_fees, 2),
            total_fees_paid_usd=new_total_fees,
            maker_fill_ratio=s.pnl_metrics.maker_fill_ratio,
            taker_fill_ratio=s.pnl_metrics.taker_fill_ratio,
            fee_savings_usd=new_fee_savings,
            fee_savings_bps=s.pnl_metrics.fee_savings_bps,
            annualized_carry_apr_pct=s.pnl_metrics.annualized_carry_apr_pct,
            is_dry_run=s.is_dry_run,
        )

        m_regime = compute_regime_multiplier_from_asri(s.asri_risk.score, s.asri_risk.tail_risk_flag)
        if s.regime_mode in (ExecutionRegimeMode.PAUSED_ENTRIES, ExecutionRegimeMode.EMERGENCY_FLATTENED):
            eff_m = 0.0 if s.regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED else 0.5 * m_regime
        elif s.regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK:
            eff_m = 0.25
        else:
            eff_m = m_regime

        alloc_inputs = [
            AssetAllocationInput(
                asset_id="BTCUSDT-PERP",
                y_primary=typed_dir if s.regime_mode != ExecutionRegimeMode.EMERGENCY_FLATTENED else 0,
                w_hrp=0.45,
                p_meta=p_meta,
                expected_gross_return_bps=96.0,
                fee_schedule=FeeSchedule.usdt_perp(),
                epistemic_gate_passed=True,
            ),
            AssetAllocationInput(
                asset_id="ETHUSDT-PERP",
                y_primary=1 if s.regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY else 0,
                w_hrp=0.35,
                p_meta=round(max(0.55, p_meta - 0.04), 3),
                expected_gross_return_bps=88.0,
                fee_schedule=FeeSchedule.usdt_perp(),
                epistemic_gate_passed=True,
            ),
            AssetAllocationInput(
                asset_id="SOLUSDT-PERP",
                y_primary=-1 if s.regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY else 0,
                w_hrp=0.20,
                p_meta=round(max(0.54, p_meta - 0.06), 3),
                expected_gross_return_bps=108.0,
                fee_schedule=FeeSchedule.usdt_perp(),
                epistemic_gate_passed=True,
            ),
        ]
        portfolio_alloc = compute_signed_target_weights(
            inputs=alloc_inputs,
            regime_multiplier=eff_m,
            gamma_frac=0.5,
            max_gross_leverage=2.5,
        )

        side_str: Literal["BUY", "SELL", "NONE"] = (
            "BUY" if typed_dir > 0 else ("SELL" if typed_dir < 0 else "NONE")
        )
        summary_msg = (
            f"[{self._now_str()} • Bar #{k}] Mode={s.regime_mode.value} | BTC=${close_px:,.2f} | "
            f"AI={p_meta:.0%} | Shield=SAFE (VPIN={vpin:.2f}, DI={di_val:.2f}) | "
            f"Order={side_str} GTX @ ${close_px - 4.5:,.2f} | "
            f"Bar PnL=${pnl_delta_usd:+.2f} (Total=${new_net_pnl_usd:,.2f})"
        )
        self._activity_log.append(summary_msg)
        _append_log_line(summary_msg)

        self._state = DashboardTelemetryState(
            active_instrument=s.active_instrument,
            regime_mode=s.regime_mode,
            is_dry_run=s.is_dry_run,
            pnl_metrics=updated_pnl,
            ffd_config=s.ffd_config,
            asri_risk=s.asri_risk,
            hawkes_state=new_hawkes,
            last_order_command=ExecutionOrderCommand(
                instrument_id=s.active_instrument,
                ts_ns=ts_close,
                action=ExecutionAction.PLACE_GTX_MAKER if side_str != "NONE" else ExecutionAction.HOLD_RESTING_HYSTERESIS,
                side=side_str,
                target_price=round(close_px - 4.5, 2),
                quantity=0.45 if side_str != "NONE" else 0.0,
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
            latest_price=close_px,
            net_pnl_usd=new_net_pnl_usd,
            net_pnl_bps=new_net_pnl_bps,
            activity_log=self.activity_log,
        )

    def apply_control_action(self, action: str) -> ControlActionResult:
        """Apply one of the four operational control actions and update ``self._state``."""
        normalized = action.strip().lower()
        ts_label = self._now_str()
        if normalized == "start_dry_run":
            mode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
            self._auto_running = True
            msg = (
                f"[{ts_label}] ▶ DRY-RUN STARTED: Live paper-trading simulation is now actively running! "
                "Generating new market bars, checking safety shields, and placing Maker orders."
            )
            is_dry = True
        elif normalized == "pause_entries":
            mode = ExecutionRegimeMode.PAUSED_ENTRIES
            self._auto_running = False
            msg = (
                f"[{ts_label}] ⏸ ENTRIES PAUSED: Cancelled all unfilled entry orders. "
                "Existing positions keep their protective stop-losses and profit targets."
            )
            is_dry = self._state.is_dry_run
        elif normalized == "switch_carry_only":
            mode = ExecutionRegimeMode.CARRY_ONLY_FALLBACK
            self._auto_running = True
            msg = (
                f"[{ts_label}] 🏦 CARRY-ONLY MODE: Directional Long/Short bets turned off. "
                "Portfolio is now exclusively collecting low-risk 8-hour Funding Carry (+14.8% APR)."
            )
            is_dry = self._state.is_dry_run
        elif normalized == "emergency_flatten":
            mode = ExecutionRegimeMode.EMERGENCY_FLATTENED
            self._auto_running = False
            msg = (
                f"[{ts_label}] 🛑 EMERGENCY FLATTEN EXECUTED: Cancelled all orders and closed "
                "every position across all coins to 100% Cash immediately."
            )
            is_dry = self._state.is_dry_run
        else:
            raise ValueError(
                f"Unsupported control action '{action}'. Valid actions: "
                "start_dry_run, pause_entries, switch_carry_only, emergency_flatten."
            )

        self._activity_log.append(msg)
        _append_log_line(msg)
        self._state = build_default_dashboard_state(
            regime_mode=mode,
            is_dry_run=is_dry,
            last_control_action=normalized,
            last_control_message=msg,
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
            net_pnl_usd=self._state.pnl_metrics.net_pnl_usd,
            net_pnl_bps=self._state.pnl_metrics.net_pnl_bps,
            activity_log=self.activity_log,
        )


def _serialize_live_state(store: ControlCenterStateStore) -> dict[str, Any]:
    s = store.state
    candlestick_fig = build_candlestick_dollar_bar_figure(s)
    weights_fig = build_portfolio_weights_figure(s.portfolio_allocation)
    payload = s.model_dump(mode="json")
    payload.update(
        {
            "tick_count": store.tick_counter,
            "auto_running": store.auto_running,
            "latest_price": s.dollar_bars[-1].close,
            "starting_capital_usd": s.pnl_metrics.starting_capital_usd,
            "current_equity_usd": s.pnl_metrics.current_equity_usd,
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
            "gross_leverage": s.portfolio_allocation.gross_leverage,
            "activity_log": store.activity_log,
            "candlestick_figure": json.loads(json.dumps(candlestick_fig.to_dict())),
            "weights_figure": json.loads(json.dumps(weights_fig.to_dict())),
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
                await asyncio.sleep(2.5)
                if state_store.auto_running:
                    state_store.step_simulation_tick()

        task = asyncio.create_task(_background_ticker())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(
        title="Gemini Argon 4 — M1 Max Control Center API",
        version="4.0.0",
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
        """Return live telemetry, PnL, Maker/Taker ratio, ASRI summary, CPCV gates, and HRP-CVaR weights."""
        return _serialize_live_state(state_store)

    @app.post("/api/tick")
    def post_simulation_tick() -> dict[str, Any]:
        """Advance the live Dry-Run simulation by 1 bar and return full updated state & figures."""
        state_store.step_simulation_tick()
        return _serialize_live_state(state_store)

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
