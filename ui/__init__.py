"""Interactive Web UI, FastAPI Control Center, and Plain-English Telegram Companion."""

from __future__ import annotations

from ui.api_server import ControlActionResult, ControlCenterStateStore, create_app
from ui.dashboard_app import (
    DOC_FILENAMES,
    DOCS_DIR,
    PLAIN_ENGLISH_TOOLTIPS,
    DashboardTelemetryState,
    PnLAndExecutionMetrics,
    build_candlestick_dollar_bar_figure,
    build_default_dashboard_state,
    build_portfolio_weights_figure,
    build_validation_gates_figure,
    load_layman_document_markdown,
    render_full_dashboard_html,
)
from ui.telegram_bot import (
    SUPPORTED_SLASH_COMMANDS,
    PlainEnglishTelegramBot,
    format_shield_cancel_notification,
    format_trade_opened_notification,
)

__all__ = [
    "ControlActionResult",
    "ControlCenterStateStore",
    "DOCS_DIR",
    "DOC_FILENAMES",
    "DashboardTelemetryState",
    "PLAIN_ENGLISH_TOOLTIPS",
    "PlainEnglishTelegramBot",
    "PnLAndExecutionMetrics",
    "SUPPORTED_SLASH_COMMANDS",
    "build_candlestick_dollar_bar_figure",
    "build_default_dashboard_state",
    "build_portfolio_weights_figure",
    "build_validation_gates_figure",
    "create_app",
    "format_shield_cancel_notification",
    "format_trade_opened_notification",
    "load_layman_document_markdown",
    "render_full_dashboard_html",
]
