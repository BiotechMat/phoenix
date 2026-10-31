"""Circuit breakers and drawdown governance (REINFORCEMENT_PLAN task P3.2).

Per sleeve:
- soft drawdown limit: a drawdown of ``soft_limit`` (5%) of the sleeve's capital allocation, measured from
  the sleeve's peak cumulative PnL (realised plus unrealised), halves its gross;
- hard drawdown limit: ``hard_limit`` (7.5%) disables the sleeve until it is re-promoted;
- daily loss budget: once the sleeve's PnL change within the UTC day falls below ``-daily_loss_budget``
  of its allocation, new entries are blocked until the next day.

Per (sleeve, asset) pair (the Freqtrade protections):
- StoplossGuard: 3 stop-outs within 48 bars lock the pair for 24 bars;
- CooldownPeriod: no re-entry for 4 bars after any exit;
- LowProfit: when the last 20 closed trades sum to a net loss, the pair is disabled until re-enabled.

Portfolio:
- smoothed Grossman-Zhou multiplier ``(E - a*M) / ((1 - a)*M)`` clipped to [0, 1], where ``M`` is a running
  maximum of equity that decays by 1% per day (so an old peak does not bind forever) and ``a = 1 - max_dd``,
  then smoothed with an EWMA.

Operational lockdown: a stale feed (older than 30 seconds), a reject storm, or an unknown position state
blocks every new entry and asks for resting orders to be cancelled.

Exits are never blocked: breakers only scale or refuse entries.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from core.contracts import ClosedTradeRecord, PositionExitReason

NS_PER_DAY: Final[int] = 86_400 * 1_000_000_000


class CircuitBreakerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sleeve_allocation_fraction: dict[str, float] = Field(
        default_factory=lambda: {"meta_labeller": 1 / 3, "derivatives_alpha": 1 / 3, "funding_carry": 1 / 3, "trend": 1 / 3},
        description="Share of equity treated as each sleeve's capital for its drawdown limits",
    )
    soft_limit: float = Field(default=0.05, gt=0.0)
    hard_limit: float = Field(default=0.075, gt=0.0)
    daily_loss_budget: float = Field(default=0.03, gt=0.0)
    stoploss_guard_count: int = Field(default=3, ge=1)
    stoploss_guard_window_bars: int = Field(default=48, ge=1)
    stoploss_guard_lock_bars: int = Field(default=24, ge=1)
    cooldown_bars: int = Field(default=4, ge=0)
    low_profit_trades: int = Field(default=20, ge=1)
    max_portfolio_drawdown: float = Field(default=0.20, gt=0.0, lt=1.0)
    running_max_decay_per_day: float = Field(default=0.01, ge=0.0, lt=1.0)
    gz_smoothing_bars: int = Field(default=4, ge=1)
    stale_feed_seconds: float = Field(default=30.0, gt=0.0)
    reject_storm_count: int = Field(default=5, ge=1)


class OperationalStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    feed_age_seconds: float = 0.0
    recent_rejects: int = 0
    position_state_known: bool = True


class EntryPermission(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    size_multiplier: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class BreakerSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    portfolio_multiplier: float
    portfolio_equity: float
    decayed_running_max: float
    lockdown: bool
    lockdown_reasons: list[str]
    disabled_sleeves: list[str]
    halved_sleeves: list[str]
    daily_blocked_sleeves: list[str]
    locked_pairs: dict[str, int]
    cooldown_pairs: dict[str, int]
    low_profit_disabled_pairs: list[str]
    sleeve_drawdowns: dict[str, float]
    events: list[str]


class _SleeveState:
    def __init__(self) -> None:
        self.cum_pnl: float = 0.0
        self.peak_pnl: float = 0.0
        self.disabled: bool = False
        self.day: int | None = None
        self.day_start_pnl: float = 0.0
        self.daily_blocked: bool = False


class _PairState:
    def __init__(self) -> None:
        self.stop_bars: deque[int] = deque()
        self.locked_until: int = -1
        self.cooldown_until: int = -1
        self.recent_net: deque[float] = deque()
        self.low_profit_disabled: bool = False


class CircuitBreakers:
    """Stateful breaker book. Feed it closed trades and per-bar marks; ask it before every entry."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config: CircuitBreakerConfig = config or CircuitBreakerConfig()
        self._sleeves: dict[str, _SleeveState] = {}
        self._pairs: dict[str, _PairState] = {}
        self._gz_raw: float = 1.0
        self._gz_smoothed: float = 1.0
        self._running_max: float | None = None
        self._last_ts_ns: int | None = None
        self._equity: float = 0.0
        self._lockdown_reasons: list[str] = []
        self._events: list[str] = []

    # ---------------------------------------------------------------- state updates

    def _sleeve(self, sleeve_id: str) -> _SleeveState:
        return self._sleeves.setdefault(sleeve_id, _SleeveState())

    def _pair(self, sleeve_id: str, asset: str) -> _PairState:
        return self._pairs.setdefault(f"{sleeve_id}:{asset}", _PairState())

    def _log(self, msg: str) -> None:
        self._events.append(msg)
        self._events = self._events[-50:]

    def record_closed_trade(self, trade: ClosedTradeRecord, bar_index: int) -> None:
        """Update pair guards from a closed trade (sleeve PnL comes from ``update_bar``)."""
        cfg = self.config
        key = f"{trade.sleeve_id}:{trade.asset}"
        pair = self._pair(trade.sleeve_id, trade.asset)
        pair.cooldown_until = max(pair.cooldown_until, bar_index + cfg.cooldown_bars)
        if trade.exit_reason == PositionExitReason.STOP_BARRIER:
            pair.stop_bars.append(bar_index)
        while pair.stop_bars and pair.stop_bars[0] < bar_index - cfg.stoploss_guard_window_bars:
            pair.stop_bars.popleft()
        if len(pair.stop_bars) >= cfg.stoploss_guard_count and pair.locked_until < bar_index + cfg.stoploss_guard_lock_bars:
            pair.locked_until = bar_index + cfg.stoploss_guard_lock_bars
            pair.stop_bars.clear()
            self._log(f"StoplossGuard: {key} locked until bar {pair.locked_until}")
        pair.recent_net.append(trade.net_pnl_usd)
        while len(pair.recent_net) > cfg.low_profit_trades:
            pair.recent_net.popleft()
        if len(pair.recent_net) == cfg.low_profit_trades and sum(pair.recent_net) < 0.0 and not pair.low_profit_disabled:
            pair.low_profit_disabled = True
            self._log(f"LowProfit: {key} disabled ({cfg.low_profit_trades}-trade net {sum(pair.recent_net):+.2f})")

    def update_bar(
        self,
        *,
        ts_ns: int,
        equity_usd: float,
        sleeve_cum_pnl: Mapping[str, float],
        operational: OperationalStatus | None = None,
    ) -> BreakerSnapshot:
        """Per-bar update: sleeve drawdowns and daily budgets, the portfolio multiplier, the lockdown state."""
        cfg = self.config
        day = ts_ns // NS_PER_DAY
        for sleeve_id, pnl in sleeve_cum_pnl.items():
            st = self._sleeve(sleeve_id)
            st.cum_pnl = pnl
            st.peak_pnl = max(st.peak_pnl, pnl)
            if st.day != day:
                st.day = day
                st.day_start_pnl = pnl
                if st.daily_blocked:
                    self._log(f"daily budget reset: {sleeve_id}")
                st.daily_blocked = False
            alloc = max(1e-9, equity_usd * cfg.sleeve_allocation_fraction.get(sleeve_id, 1.0))
            dd = (st.peak_pnl - pnl) / alloc
            if dd >= cfg.hard_limit and not st.disabled:
                st.disabled = True
                self._log(f"hard limit: {sleeve_id} disabled at drawdown {dd:.2%} of allocation")
            if (pnl - st.day_start_pnl) / alloc <= -cfg.daily_loss_budget and not st.daily_blocked:
                st.daily_blocked = True
                self._log(f"daily loss budget: {sleeve_id} entries blocked for the day")

        # Grossman-Zhou with a decaying running maximum.
        if self._running_max is None:
            self._running_max = equity_usd
        else:
            dt_days = max(0.0, (ts_ns - (self._last_ts_ns or ts_ns)) / NS_PER_DAY)
            decayed = self._running_max * (1.0 - cfg.running_max_decay_per_day) ** dt_days
            self._running_max = max(equity_usd, decayed)
        self._last_ts_ns = ts_ns
        self._equity = equity_usd
        floor = 1.0 - cfg.max_portfolio_drawdown
        m = self._running_max
        self._gz_raw = min(1.0, max(0.0, (equity_usd - floor * m) / ((1.0 - floor) * m))) if m > 0 else 0.0
        alpha = 2.0 / (cfg.gz_smoothing_bars + 1.0)
        self._gz_smoothed = (1.0 - alpha) * self._gz_smoothed + alpha * self._gz_raw

        reasons: list[str] = []
        op = operational or OperationalStatus()
        if op.feed_age_seconds > cfg.stale_feed_seconds:
            reasons.append(f"stale feed ({op.feed_age_seconds:.0f}s > {cfg.stale_feed_seconds:.0f}s)")
        if op.recent_rejects >= cfg.reject_storm_count:
            reasons.append(f"reject storm ({op.recent_rejects} rejects)")
        if not op.position_state_known:
            reasons.append("unknown position state")
        if reasons and not self._lockdown_reasons:
            self._log("operational lockdown: " + "; ".join(reasons) + " (cancel resting orders, block entries)")
        self._lockdown_reasons = reasons
        return self.snapshot(current_bar=None)

    # ---------------------------------------------------------------- queries

    def permission(self, sleeve_id: str, asset: str, bar_index: int) -> EntryPermission:
        """May this sleeve open a position in this asset now, and at what fraction of its target size?"""
        cfg = self.config
        reasons: list[str] = []
        if self._lockdown_reasons:
            reasons.append("operational lockdown: " + "; ".join(self._lockdown_reasons))
        st = self._sleeves.get(sleeve_id)
        multiplier = self._gz_smoothed
        if st is not None:
            if st.disabled:
                reasons.append(f"{sleeve_id} disabled by the hard drawdown limit")
            if st.daily_blocked:
                reasons.append(f"{sleeve_id} daily loss budget spent")
            alloc_dd_soft = st.peak_pnl - st.cum_pnl
            alloc = max(1e-9, self._equity * cfg.sleeve_allocation_fraction.get(sleeve_id, 1.0))
            if alloc_dd_soft / alloc >= cfg.soft_limit:
                multiplier *= 0.5
        pair = self._pairs.get(f"{sleeve_id}:{asset}")
        if pair is not None:
            if bar_index <= pair.locked_until:
                reasons.append(f"StoplossGuard lock until bar {pair.locked_until}")
            if bar_index <= pair.cooldown_until:
                reasons.append(f"cooldown until bar {pair.cooldown_until}")
            if pair.low_profit_disabled:
                reasons.append("LowProfit: pair disabled")
        if multiplier <= 0.0:
            reasons.append("portfolio drawdown multiplier at zero")
        return EntryPermission(allowed=not reasons, size_multiplier=float(min(1.0, max(0.0, multiplier))), reasons=reasons)

    def re_enable(self, sleeve_id: str | None = None, pair: str | None = None) -> None:
        """Re-promotion: clear a sleeve's hard-limit disable (resetting its peak) or a pair's LowProfit disable."""
        if sleeve_id is not None and sleeve_id in self._sleeves:
            st = self._sleeves[sleeve_id]
            st.disabled = False
            st.peak_pnl = st.cum_pnl
            self._log(f"re-enabled sleeve {sleeve_id}")
        if pair is not None and pair in self._pairs:
            self._pairs[pair].low_profit_disabled = False
            self._pairs[pair].recent_net.clear()
            self._log(f"re-enabled pair {pair}")

    @property
    def lockdown(self) -> bool:
        return bool(self._lockdown_reasons)

    @property
    def portfolio_multiplier(self) -> float:
        return self._gz_smoothed

    def snapshot(self, current_bar: int | None) -> BreakerSnapshot:
        cfg = self.config
        halved: list[str] = []
        drawdowns: dict[str, float] = {}
        for sid, st in self._sleeves.items():
            alloc = max(1e-9, self._equity * cfg.sleeve_allocation_fraction.get(sid, 1.0))
            dd = (st.peak_pnl - st.cum_pnl) / alloc
            drawdowns[sid] = dd
            if dd >= cfg.soft_limit and not st.disabled:
                halved.append(sid)
        bar = current_bar if current_bar is not None else -1
        return BreakerSnapshot(
            portfolio_multiplier=self._gz_smoothed,
            portfolio_equity=self._equity,
            decayed_running_max=self._running_max or 0.0,
            lockdown=self.lockdown,
            lockdown_reasons=list(self._lockdown_reasons),
            disabled_sleeves=sorted(s for s, st in self._sleeves.items() if st.disabled),
            halved_sleeves=sorted(halved),
            daily_blocked_sleeves=sorted(s for s, st in self._sleeves.items() if st.daily_blocked),
            locked_pairs={k: p.locked_until for k, p in self._pairs.items() if p.locked_until >= bar},
            cooldown_pairs={k: p.cooldown_until for k, p in self._pairs.items() if p.cooldown_until >= bar},
            low_profit_disabled_pairs=sorted(k for k, p in self._pairs.items() if p.low_profit_disabled),
            sleeve_drawdowns=drawdowns,
            events=list(self._events),
        )


__all__ = [
    "BreakerSnapshot",
    "CircuitBreakerConfig",
    "CircuitBreakers",
    "EntryPermission",
    "OperationalStatus",
]

