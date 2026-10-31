"""FastAPI Control Center Server (``http://localhost:8080``) for Gemini Argon 4.

Exposes:
- ``GET /``: Interactive 3-tab HTML Control Center with hover tooltips and built-in docs.
- ``GET /api/state``: Live telemetry, net-of-fee PnL, Maker/Taker ratio, ASRI summary,
  CPCV promotion gates, and signed ``HRP-CVaR`` target weights.
- ``GET /api/tooltips``: Complete ``PLAIN_ENGLISH_TOOLTIPS`` dictionary.
- ``GET /api/docs/{doc_name}``: Renders any of the four ``docs/*.md`` layman's manuals.
- ``POST /api/control/{action}``: Handles one-click operational controls
  (``start_dry_run``, ``pause_entries``, ``switch_carry_only``, ``emergency_flatten``).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from core.contracts import ExecutionRegimeMode
from ui.dashboard_app import (
    DOC_FILENAMES,
    PLAIN_ENGLISH_TOOLTIPS,
    DashboardTelemetryState,
    build_default_dashboard_state,
    load_layman_document_markdown,
    render_full_dashboard_html,
)


class ControlActionResult(BaseModel):
    """Response schema for ``POST /api/control/{action}``."""

    model_config = ConfigDict(frozen=True)

    status: str
    action: str
    regime_mode: ExecutionRegimeMode
    is_dry_run: bool
    plain_english_explanation: str
    weights: dict[str, float]
    gross_leverage: float


class ControlCenterStateStore:
    """In-memory mutable state store backing the FastAPI Control Center and Telegram Bot."""

    def __init__(self, initial_state: DashboardTelemetryState | None = None) -> None:
        self._state: DashboardTelemetryState = initial_state or build_default_dashboard_state()

    @property
    def state(self) -> DashboardTelemetryState:
        """Return the current immutable ``DashboardTelemetryState`` snapshot."""
        return self._state

    def apply_control_action(self, action: str) -> ControlActionResult:
        """Apply one of the four operational control actions and update ``self._state``."""
        normalized = action.strip().lower()
        if normalized == "start_dry_run":
            mode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
            msg = (
                "Started Dry-Run (Paper Trading) in DIRECTIONAL_PLUS_CARRY mode: "
                "Full directional signals + Delta-Neutral Carry active on live market feeds."
            )
            is_dry = True
        elif normalized == "pause_entries":
            mode = ExecutionRegimeMode.PAUSED_ENTRIES
            msg = (
                "Paused New Entries (PAUSED_ENTRIES): Cancelled all unfilled entry orders. "
                "Existing positions keep their protective Triple-Barrier stop-losses and profit targets."
            )
            is_dry = self._state.is_dry_run
        elif normalized == "switch_carry_only":
            mode = ExecutionRegimeMode.CARRY_ONLY_FALLBACK
            msg = (
                "Switched to Carry-Only Mode (CARRY_ONLY_FALLBACK): Directional Long/Short bets disabled; "
                "capital restricted to Delta-Neutral Funding Carry and Cash."
            )
            is_dry = self._state.is_dry_run
        elif normalized == "emergency_flatten":
            mode = ExecutionRegimeMode.EMERGENCY_FLATTENED
            msg = (
                "EMERGENCY FLATTEN ALL EXECUTED (EMERGENCY_FLATTENED): All resting orders cancelled "
                "and all open positions closed immediately to 100% cash."
            )
            is_dry = self._state.is_dry_run
        else:
            raise ValueError(
                f"Unsupported control action '{action}'. Valid actions: "
                "start_dry_run, pause_entries, switch_carry_only, emergency_flatten."
            )

        self._state = build_default_dashboard_state(
            regime_mode=mode,
            is_dry_run=is_dry,
            last_control_action=normalized,
            last_control_message=msg,
        )
        return ControlActionResult(
            status="ok",
            action=normalized,
            regime_mode=self._state.regime_mode,
            is_dry_run=self._state.is_dry_run,
            plain_english_explanation=msg,
            weights=dict(self._state.portfolio_allocation.weights),
            gross_leverage=self._state.portfolio_allocation.gross_leverage,
        )


def create_app(store: ControlCenterStateStore | None = None) -> FastAPI:
    """Create and configure the FastAPI Control Center serving ``http://localhost:8080``."""
    state_store = store or ControlCenterStateStore()
    app = FastAPI(
        title="Gemini Argon 4 — M1 Max Control Center API",
        version="4.0.0",
        description="Interactive Web UI & REST Control Center for the M1 Max Quantitative Derivatives Engine.",
    )
    app.state.store = state_store

    @app.get("/", response_class=HTMLResponse)
    def get_dashboard_root() -> HTMLResponse:
        """Render the complete 3-tab interactive HTML Control Center."""
        html_doc = render_full_dashboard_html(state_store.state)
        return HTMLResponse(content=html_doc, status_code=200)

    @app.get("/api/state")
    def get_live_state() -> dict[str, Any]:
        """Return live telemetry, PnL, Maker/Taker ratio, ASRI summary, CPCV gates, and HRP-CVaR weights."""
        s = state_store.state
        payload = s.model_dump(mode="json")
        payload.update(
            {
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
            }
        )
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
        return result.model_dump(mode="json")

    return app


app: FastAPI = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)

