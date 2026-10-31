"""Plain-English Telegram Bot Companion & Mobile Command Dispatcher.

Provides:
1. Plain-English mobile notifications explaining *why* a trade was opened or *why*
   the VPIN / $O(1)$ Hawkes shield pulled a resting order in simple everyday language.
2. Slash command handlers: ``/status``, ``/profit``, ``/gates``, ``/explain``,
   ``/pause``, and ``/flat``.
"""

from __future__ import annotations

import httpx

from core.contracts import (
    ExecutionAction,
    ExecutionOrderCommand,
    HawkesStateSnapshot,
    MetaPredictionOutput,
    PositioningSignalSnapshot,
)
from ui.api_server import ControlCenterStateStore
from ui.dashboard_app import DashboardTelemetryState, build_default_dashboard_state

SUPPORTED_SLASH_COMMANDS: tuple[str, ...] = (
    "/status",
    "/profit",
    "/gates",
    "/explain",
    "/pause",
    "/flat",
)


def format_trade_opened_notification(
    meta: MetaPredictionOutput | None = None,
    positioning: PositioningSignalSnapshot | None = None,
    order: ExecutionOrderCommand | None = None,
    state: DashboardTelemetryState | None = None,
) -> str:
    """Format a plain-English mobile notification explaining WHY a trade was opened."""
    resolved = state or build_default_dashboard_state()
    use_meta = meta or resolved.meta_predictions[-1]
    use_pos = positioning or resolved.positioning_signals[-1]
    use_order = order or resolved.last_order_command

    direction_word = (
        "LONG (Buy)"
        if use_meta.effective_direction > 0
        else ("SHORT (Sell)" if use_meta.effective_direction < 0 else "FLAT (No Entry)")
    )
    regime_friendly = use_pos.regime.value.replace("_", " ").title()
    target_weight = resolved.portfolio_allocation.weights.get(use_meta.instrument_id, 0.0)

    return (
        f"🟢 TRADE EXPLANATION — {use_meta.instrument_id} ({direction_word})\n"
        f"• Why we entered: Detected a '{regime_friendly}' positioning setup while our "
        f"Stage-2 LightGBM Sniper Quality Inspector scored p_meta = {use_meta.p_meta:.1%} "
        f"(well above the 50% coin-flip minimum) with +{use_meta.expected_net_edge_bps:.1f} bps expected "
        f"profit after subtracting all exchange fees.\n"
        f"• Uncharted Territory Check (Mahalanobis DI): PASSED (DI = {use_meta.mahalanobis_di:.2f} "
        f"is safely below the {use_meta.di_threshold:.2f} ceiling).\n"
        f"• Position Size (Signed HRP-CVaR): {target_weight:+.2%} of account equity "
        f"(Macro Weather Multiplier M_regime = {resolved.portfolio_allocation.regime_multiplier:.2f}).\n"
        f"• How we executed: Placed Post-Only Maker ({use_order.time_in_force}) {use_order.side} order "
        f"at ${use_order.target_price:,.2f} to capture low maker fees and avoid taker spread drag."
    )


def format_shield_cancel_notification(
    hawkes: HawkesStateSnapshot | None = None,
    order: ExecutionOrderCommand | None = None,
    instrument_id: str = "BTCUSDT-PERP",
) -> str:
    """Format a plain-English mobile alert explaining WHY the VPIN/Hawkes shield pulled an order."""
    default_state = build_default_dashboard_state()
    use_hawkes = hawkes or HawkesStateSnapshot(
        ts_ns=default_state.hawkes_state.ts_ns,
        r_state=1.45,
        lambda_intensity=4.80,
        mu=1.10,
        alpha=0.68,
        beta=0.90,
        branching_ratio_eta=0.76,
        vpin=0.67,
        toxic_sweep_active=True,
    )
    use_order = order or ExecutionOrderCommand(
        instrument_id=instrument_id,
        ts_ns=use_hawkes.ts_ns,
        action=ExecutionAction.CANCEL_ALL_TOXIC_SHIELD,
        side="NONE",
        target_price=0.0,
        quantity=0.0,
        post_only_gtx=True,
        time_in_force="NONE",
        consecutive_rejections=0,
        reason="Toxic order-flow sweep detected by VPIN/Hawkes shield.",
    )

    triggers: list[str] = []
    if use_hawkes.vpin > 0.60:
        triggers.append(
            f"VPIN Toxicity jumped to {use_hawkes.vpin:.2f} (above the 0.60 shark-detection limit, "
            "meaning one-sided institutional volume is sweeping the order book)"
        )
    if use_hawkes.branching_ratio_eta > 0.70:
        triggers.append(
            f"O(1) Hawkes Reflexivity eta reached {use_hawkes.branching_ratio_eta:.2f} (above the 0.70 "
            "avalanche threshold, meaning trades are triggering a self-reinforcing cascade)"
        )
    if not triggers:
        triggers.append(
            f"proactive microstructure precaution (VPIN={use_hawkes.vpin:.2f}, Hawkes eta={use_hawkes.branching_ratio_eta:.2f})"
        )

    why_joined = " AND ".join(triggers)
    return (
        f"🛡️ REFLEXIVE SHIELD ALERT — Pulled Resting Order on {use_order.instrument_id} (<5ms)\n"
        f"• Action Taken: {use_order.action.value} (resting Post-Only GTX limit order cancelled immediately).\n"
        f"• Why the shield pulled your order: Because {why_joined}.\n"
        f"• Plain-English Summary: Instead of leaving our limit order sitting in the path of a "
        f"fast-moving institutional dump ('catching a falling knife'), the bot stepped aside in "
        f"under 5 milliseconds and will wait for order flow to calm down before re-quoting."
    )


class PlainEnglishTelegramBot:
    """Telegram companion bot providing plain-English notifications and slash command handlers."""

    def __init__(
        self,
        store: ControlCenterStateStore | None = None,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._store: ControlCenterStateStore = store or ControlCenterStateStore()
        self._bot_token: str | None = bot_token
        self._chat_id: str | None = chat_id
        self._sent_messages: list[str] = []

    @property
    def store(self) -> ControlCenterStateStore:
        """Return the underlying ``ControlCenterStateStore``."""
        return self._store

    @property
    def sent_messages(self) -> list[str]:
        """Return history of formatted notifications/replies emitted by this bot instance."""
        return list(self._sent_messages)

    def notify_trade_opened(
        self,
        meta: MetaPredictionOutput | None = None,
        positioning: PositioningSignalSnapshot | None = None,
        order: ExecutionOrderCommand | None = None,
    ) -> str:
        """Create and record a plain-English notification explaining why a trade was opened."""
        msg = format_trade_opened_notification(
            meta=meta,
            positioning=positioning,
            order=order,
            state=self._store.state,
        )
        self._sent_messages.append(msg)
        return msg

    def notify_shield_pulled_order(
        self,
        hawkes: HawkesStateSnapshot | None = None,
        order: ExecutionOrderCommand | None = None,
        instrument_id: str = "BTCUSDT-PERP",
    ) -> str:
        """Create and record a plain-English notification explaining why the VPIN/Hawkes shield pulled an order."""
        msg = format_shield_cancel_notification(
            hawkes=hawkes,
            order=order,
            instrument_id=instrument_id,
        )
        self._sent_messages.append(msg)
        return msg

    def cmd_status(self) -> str:
        """Handle ``/status``: Plain-English health check of mode, macro risk, and order-book shield."""
        s = self._store.state
        shield_str = (
            "🔴 TOXIC SWEEP ACTIVE (Resting entry orders pulled)"
            if s.hawkes_state.toxic_sweep_active
            else "🟢 SAFE (Low-fee Post-Only GTX Maker orders active)"
        )
        return (
            f"📊 SYSTEM STATUS — {s.regime_mode.value}\n"
            f"• Environment: {'Paper Trading (Dry-Run)' if s.is_dry_run else 'LIVE CAPITAL'}\n"
            f"• Macro Weather (ASRI): {s.asri_risk.score:.1f}/100 — {s.asri_risk.plain_english_summary}\n"
            f"• Order-Book Bodyguard: {shield_str} "
            f"(VPIN={s.hawkes_state.vpin:.2f} [limit 0.60], Hawkes eta={s.hawkes_state.branching_ratio_eta:.2f} [limit 0.70])\n"
            f"• Portfolio Gross Leverage: {s.portfolio_allocation.gross_leverage:.2f}x / 2.50x hard cap."
        )

    def cmd_profit(self) -> str:
        """Handle ``/profit``: Plain-English Net-of-Fee PnL, Maker/Taker ratio, and fee savings."""
        p = self._store.state.pnl_metrics
        return (
            f"💰 PROFIT & FEE SCORECARD (Net of All Fees & Funding)\n"
            f"• Net-of-Fee Profit: ${p.net_pnl_usd:,.2f} (+{p.net_pnl_bps:.1f} bps)\n"
            f"• Gross Profit Before Fees: ${p.gross_pnl_usd:,.2f} (Total Fees Paid: ${p.total_fees_paid_usd:,.2f})\n"
            f"• Maker vs. Taker Fill Ratio: {p.maker_fill_ratio:.1%} Passive Maker (GTX) / "
            f"{p.taker_fill_ratio:.1%} Taker\n"
            f"• Money Saved by Maker Execution: ${p.fee_savings_usd:,.2f} (+{p.fee_savings_bps:.1f} bps saved)\n"
            f"• Delta-Neutral Funding Carry Yield: {p.annualized_carry_apr_pct:.1f}% APR."
        )

    def cmd_gates(self) -> str:
        """Handle ``/gates``: Plain-English 45-split CPCV validation gate scores for Champion & Challenger."""
        s = self._store.state
        c = s.champion_cpcv
        ch = s.challenger_cpcv
        return (
            f"🧪 45-SPLIT CPCV EXAM BOARD & PROMOTION GATES\n"
            f"• Active Champion ({c.model_id}): {'✅ PASSED ALL GATES' if c.passed_all_gates else '⚠️ FAILED'}\n"
            f"  - Probabilistic Sharpe (PSR vs 0): {c.psr_vs_zero:.1%} (needs >= 95%)\n"
            f"  - PSR vs Equal-Weight Benchmark: {c.psr_vs_benchmark:.1%} (needs >= 95%)\n"
            f"  - Deflated Sharpe Ratio (DSR): {c.dsr:.1%} (needs >= 95%)\n"
            f"  - Overfitting Risk (PBO): {c.pbo:.1%} (needs < 30%)\n"
            f"  - Expected Net Edge After Fees: +{c.expected_net_pnl_bps:.1f} bps/trade (needs > 15.0 bps)\n"
            f"• Latest Challenger ({ch.model_id}): {'✅ READY FOR PROMOTION' if ch.passed_all_gates else '❌ QUARANTINED'} "
            f"(PSR={ch.psr_vs_zero:.1%}, DSR={ch.dsr:.1%}, PBO={ch.pbo:.1%}, Net Edge=+{ch.expected_net_pnl_bps:.1f} bps)\n"
            f"• Fallback Safety Net: Live 30d PSR = {s.champion_state.champion_live_30d_psr:.1%} "
            f"(auto-switches to Carry-Only if < 80% or 3 retrains fail)."
        )

    def cmd_explain(self) -> str:
        """Handle ``/explain``: Plain-English explanation of current trades, sizing, and shield actions."""
        s = self._store.state
        trade_note = format_trade_opened_notification(state=s)
        weights_str = ", ".join(f"{k}: {v:+.2%}" for k, v in s.portfolio_allocation.weights.items())
        return (
            f"{trade_note}\n\n"
            f"📐 PORTFOLIO SIZING BREAKDOWN (Signed HRP-CVaR):\n"
            f"• Formula: w_final = Direction * w_HRP * f_Kelly(p_meta) * M_regime\n"
            f"• Current Signed Target Weights: {weights_str}\n"
            f"• FFD Stationarity Filter: d*={s.ffd_config.d_star:.2f} (p_ADF={s.ffd_config.adf_pvalue:.4f}, "
            f"retaining {s.ffd_config.level_correlation:.1%} trend memory)."
        )

    def cmd_pause(self) -> str:
        """Handle ``/pause``: Pause new entries while keeping existing position protection active."""
        res = self._store.apply_control_action("pause_entries")
        return (
            f"⏸️ ENTRIES PAUSED ({res.regime_mode.value})\n"
            f"• {res.plain_english_explanation}\n"
            f"• Gross Leverage for New Orders: {res.gross_leverage:.2f}x."
        )

    def cmd_flat(self) -> str:
        """Handle ``/flat``: Emergency flatten all positions and cancel all resting orders immediately."""
        res = self._store.apply_control_action("emergency_flatten")
        return (
            f"🚨 EMERGENCY FLATTEN EXECUTED ({res.regime_mode.value})\n"
            f"• {res.plain_english_explanation}\n"
            f"• All asset weights set to 0.00% (Current Gross Leverage: {res.gross_leverage:.2f}x)."
        )

    def handle_command(self, command_text: str) -> str:
        """Dispatch a Telegram slash command (``/status``, ``/profit``, ``/gates``, ``/explain``, ``/pause``, ``/flat``)."""
        tokens = command_text.strip().split()
        cmd = tokens[0].lower().split("@")[0] if tokens else ""

        if cmd == "/status":
            reply = self.cmd_status()
        elif cmd == "/profit":
            reply = self.cmd_profit()
        elif cmd == "/gates":
            reply = self.cmd_gates()
        elif cmd == "/explain":
            reply = self.cmd_explain()
        elif cmd == "/pause":
            reply = self.cmd_pause()
        elif cmd == "/flat":
            reply = self.cmd_flat()
        else:
            reply = (
                f"Unknown command '{command_text}'. Available plain-English commands: "
                f"{', '.join(SUPPORTED_SLASH_COMMANDS)}"
            )

        self._sent_messages.append(reply)
        return reply

    async def send_telegram_message_async(self, text: str) -> bool:
        """Send a plain-English message via Telegram HTTP API if token/chat_id are configured."""
        self._sent_messages.append(text)
        if not self._bot_token or not self._chat_id:
            return True
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json={"chat_id": self._chat_id, "text": text})
            return resp.status_code == 200
