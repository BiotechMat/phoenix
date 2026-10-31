"""Defensive Maker-First Execution Algorithm with Re-Peg Hysteresis, GTX Backoff, and Toxic Sweep Shield.

Implements `DefensiveMakerExecutionAlgorithm` compatible with NautilusTrader's `ExecutionAlgorithm`:
1. Normal Regime (VPIN <= 0.60 and eta_Hawkes <= 0.70):
   Routes all entries and profit-taking exits as Post-Only (GTX) Limit Orders at best_bid (BUY)
   or best_ask (SELL).
2. Re-Peg Hysteresis & Cooldown Rule:
   Amends resting GTX orders (AMEND_GTX_MAKER) ONLY when BOTH:
   (a) |new_bbo - resting_price| / resting_price >= 1.5 bps (1.5e-4), AND
   (b) elapsed time since last modification >= 500 ms (500_000_000 ns).
   Otherwise emits HOLD_RESTING_HYSTERESIS.
3. GTX Post-Only Rejection Backoff:
   Caps consecutive top-of-book GTX rejection retries at 3. On the 4th attempt (after 3
   consecutive rejections), places the GTX order 1 tick deeper into the passive book
   (best_bid - tick_size for BUY, best_ask + tick_size for SELL) with BACKOFF_DEEPER_GTX
   and enforces a 1,000 ms (1_000_000_000 ns) cooldown.
4. Toxic Sweep Shield (VPIN > 0.60 or eta_Hawkes > 0.70):
   Immediately emits CANCEL_ALL_TOXIC_SHIELD in < 5 ms to cancel all resting entry orders.
   Restricts Taker orders (IOC/Market, EMERGENCY_TAKER_EXIT) exclusively to hard stop-loss
   breaches or emergency circuit-breaker exits.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.contracts import (
    EnvironmentConfig,
    ExecutionAction,
    ExecutionOrderCommand,
    HawkesStateSnapshot,
)
from core.execution.vpin_hawkes import (
    RecursiveHawkesMonitor,
    RollingBVCVPINMonitor,
)

ALO_ORDER_TYPE: Final[str] = "ALO_LIMIT"
ALO_ROUTING_TAG: Final[str] = "ALO_POST_ONLY"


class ALOOrderConstraint(BaseModel):
    """Add-Liquidity-Only (ALO) Post-Only Maker order constraints."""

    model_config = ConfigDict(frozen=True)

    is_alo: bool = True
    order_type: Literal["ALO_LIMIT"] = "ALO_LIMIT"
    routing_tag: str = "ALO_POST_ONLY"
    maker_fee_bps: float = 1.5
    taker_fee_bps: float = 4.5
    allow_taker_execution: bool = False

    @staticmethod
    def validate_passive_price(
        side: Literal["BUY", "SELL"], price: float, best_bid: float, best_ask: float
    ) -> bool:
        """Enforce that order strictly rests passively on the book and does not cross spread."""
        if side == "BUY":
            return price <= best_bid
        return price >= best_ask

    @staticmethod
    def verify_no_taker_fill(
        side: Literal["BUY", "SELL"], fill_price: float, best_bid: float, best_ask: float
    ) -> bool:
        """Reject any fill that would execute as taker."""
        if side == "BUY" and fill_price >= best_ask:
            return False
        if side == "SELL" and fill_price <= best_bid:
            return False
        return True


if TYPE_CHECKING:

    class _NautilusExecutionAlgorithmBase:
        """Static type stub for NautilusTrader ExecutionAlgorithm base class."""

        def __init__(self, config: Any | None = None) -> None:
            self._nautilus_config = config

else:
    try:
        from nautilus_trader.execution.algorithm import (
            ExecutionAlgorithm as _NautilusExecutionAlgorithmBase,
        )
    except ImportError:

        class _NautilusExecutionAlgorithmBase:
            """Runtime fallback base class when running outside the compiled NautilusTrader Cython binary."""

            def __init__(self, config: Any | None = None) -> None:
                self._nautilus_config = config


_MS_TO_NS: Final[int] = 1_000_000
_BPS_TO_DECIMAL: Final[float] = 1e-4
_FLOAT_TOL: Final[float] = 1e-12


class RestingMakerOrderState(BaseModel):
    """State of an active resting Post-Only (GTX) limit order managed by the execution algorithm."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    side: Literal["BUY", "SELL"]
    resting_price: float = Field(gt=0.0)
    quantity: float = Field(gt=0.0)
    last_mod_ts_ns: int
    cooldown_until_ts_ns: int
    is_entry: bool = True


class DefensiveMakerConfig(BaseModel):
    """Configuration parameters for DefensiveMakerExecutionAlgorithm."""

    model_config = ConfigDict(frozen=True)

    vpin_toxic_threshold: float = Field(default=0.60, gt=0.0, le=1.0)
    hawkes_eta_toxic_threshold: float = Field(default=0.70, gt=0.0)
    repeg_min_bps: float = Field(default=1.5, gt=0.0)
    repeg_cooldown_ms: int = Field(default=500, ge=1)
    max_gtx_rejections: int = Field(default=3, ge=1)
    rejection_backoff_ms: int = Field(default=1000, ge=1)
    vpin_bucket_volume: float = Field(default=100.0, gt=0.0)
    vpin_num_buckets: int = Field(default=50, ge=1)
    async_hawkes_refit: bool = True
    lead_pull_k: float = Field(default=0.5, gt=0.0, description="Pull when the lead mid moves more than k x the HL spread")
    lead_pull_window_ms: int = Field(default=700, ge=1)

    @classmethod
    def from_env_config(cls, env_config: EnvironmentConfig) -> DefensiveMakerConfig:
        """Construct DefensiveMakerConfig from global EnvironmentConfig."""
        return cls(
            vpin_toxic_threshold=env_config.vpin_toxic_threshold,
            hawkes_eta_toxic_threshold=env_config.hawkes_eta_toxic_threshold,
            repeg_min_bps=env_config.repeg_min_bps,
            repeg_cooldown_ms=env_config.repeg_cooldown_ms,
            max_gtx_rejections=env_config.max_gtx_rejections,
            rejection_backoff_ms=env_config.rejection_backoff_ms,
        )


def should_pull_for_lead_move(
    side: Literal["BUY", "SELL"],
    lead_mid_change_bps: float | None,
    hl_spread_bps: float,
    k: float = 0.5,
) -> bool:
    """True when the lead venue moved against a resting quote by more than ``k`` x the HL spread (REINFORCEMENT_PLAN P4.1).

    A resting bid is exposed when the lead mid falls; a resting ask when it rises. Spreads are floored at
    0.1 bp so a locked or one-tick book still needs a real move.
    """
    if lead_mid_change_bps is None:
        return False
    threshold = k * max(hl_spread_bps, 0.1)
    return lead_mid_change_bps < -threshold if side == "BUY" else lead_mid_change_bps > threshold


class DefensiveMakerExecutionAlgorithm(_NautilusExecutionAlgorithmBase):
    """Defensive Maker-First Execution Algorithm with O(1) Hawkes/VPIN shield and re-peg hysteresis."""

    def __init__(
        self,
        config: DefensiveMakerConfig | None = None,
        *,
        vpin_toxic_threshold: float = 0.60,
        hawkes_eta_toxic_threshold: float = 0.70,
        repeg_min_bps: float = 1.5,
        repeg_cooldown_ms: int = 500,
        max_gtx_rejections: int = 3,
        rejection_backoff_ms: int = 1000,
        vpin_bucket_volume: float = 100.0,
        vpin_num_buckets: int = 50,
        async_hawkes_refit: bool = True,
    ) -> None:
        resolved_cfg = config or DefensiveMakerConfig(
            vpin_toxic_threshold=vpin_toxic_threshold,
            hawkes_eta_toxic_threshold=hawkes_eta_toxic_threshold,
            repeg_min_bps=repeg_min_bps,
            repeg_cooldown_ms=repeg_cooldown_ms,
            max_gtx_rejections=max_gtx_rejections,
            rejection_backoff_ms=rejection_backoff_ms,
            vpin_bucket_volume=vpin_bucket_volume,
            vpin_num_buckets=vpin_num_buckets,
            async_hawkes_refit=async_hawkes_refit,
        )
        try:
            super().__init__(config=None)
        except TypeError:
            super().__init__()

        self._config: DefensiveMakerConfig = resolved_cfg
        self._repeg_min_ratio: float = resolved_cfg.repeg_min_bps * _BPS_TO_DECIMAL
        self._repeg_cooldown_ns: int = resolved_cfg.repeg_cooldown_ms * _MS_TO_NS
        self._rejection_backoff_ns: int = resolved_cfg.rejection_backoff_ms * _MS_TO_NS

        self._lead_pull_counts: dict[str, int] = {}
        self._resting_orders: dict[str, RestingMakerOrderState] = {}
        self._consecutive_rejections: dict[str, int] = {}
        self._backoff_cooldown_until_ns: dict[str, int] = {}

        self._vpin_monitors: dict[str, RollingBVCVPINMonitor] = {}
        self._hawkes_monitors: dict[str, RecursiveHawkesMonitor] = {}
        self._latest_snapshots: dict[str, HawkesStateSnapshot] = {}

    @property
    def config(self) -> DefensiveMakerConfig:
        """Active execution algorithm configuration."""
        return self._config

    @property
    def vpin_toxic_threshold(self) -> float:
        """VPIN threshold above which Toxic Sweep Shield triggers (default 0.60)."""
        return self._config.vpin_toxic_threshold

    @property
    def hawkes_eta_toxic_threshold(self) -> float:
        """Hawkes branching ratio threshold above which Toxic Sweep Shield triggers (default 0.70)."""
        return self._config.hawkes_eta_toxic_threshold

    @property
    def repeg_min_bps(self) -> float:
        """Minimum BBO shift in basis points required to re-peg a resting GTX order (default 1.5)."""
        return self._config.repeg_min_bps

    @property
    def repeg_cooldown_ns(self) -> int:
        """Minimum cooldown in nanoseconds between resting order modifications (default 500_000_000)."""
        return self._repeg_cooldown_ns

    @property
    def max_gtx_rejections(self) -> int:
        """Maximum consecutive top-of-book GTX rejections before 1-tick deeper backoff (default 3)."""
        return self._config.max_gtx_rejections

    @property
    def rejection_backoff_ns(self) -> int:
        """Cooldown in nanoseconds enforced after a 3-strike deeper GTX backoff (default 1_000_000_000)."""
        return self._rejection_backoff_ns

    def lead_pull_count(self, instrument_id: str) -> int:
        """Quotes pulled because the lead venue moved (kept apart from VPIN/Hawkes cancels)."""
        return self._lead_pull_counts.get(instrument_id, 0)

    def get_resting_order(self, instrument_id: str) -> RestingMakerOrderState | None:
        """Return the currently tracked resting GTX order state for instrument_id, if any."""
        return self._resting_orders.get(instrument_id)

    def get_consecutive_rejections(self, instrument_id: str) -> int:
        """Return the current consecutive GTX rejection count for instrument_id."""
        return self._consecutive_rejections.get(instrument_id, 0)

    def get_vpin_monitor(self, instrument_id: str) -> RollingBVCVPINMonitor:
        """Return or create the per-instrument RollingBVCVPINMonitor."""
        monitor = self._vpin_monitors.get(instrument_id)
        if monitor is None:
            monitor = RollingBVCVPINMonitor(
                bucket_volume=self._config.vpin_bucket_volume,
                num_buckets=self._config.vpin_num_buckets,
                toxic_threshold=self._config.vpin_toxic_threshold,
            )
            self._vpin_monitors[instrument_id] = monitor
        return monitor

    def get_hawkes_monitor(self, instrument_id: str) -> RecursiveHawkesMonitor:
        """Return or create the per-instrument RecursiveHawkesMonitor."""
        monitor = self._hawkes_monitors.get(instrument_id)
        if monitor is None:
            monitor = RecursiveHawkesMonitor(
                eta_toxic_threshold=self._config.hawkes_eta_toxic_threshold,
                vpin_toxic_threshold=self._config.vpin_toxic_threshold,
                async_refit=self._config.async_hawkes_refit,
            )
            self._hawkes_monitors[instrument_id] = monitor
        return monitor

    def on_trade_tick(
        self,
        instrument_id: str,
        price: float,
        volume: float,
        ts_ns: int,
    ) -> HawkesStateSnapshot:
        """Update per-instrument BVC VPIN and O(1) Hawkes state on an exchange trade tick."""
        vpin_mon = self.get_vpin_monitor(instrument_id)
        hawkes_mon = self.get_hawkes_monitor(instrument_id)
        vpin_val = vpin_mon.update(price=price, volume=volume, ts_ns=ts_ns)
        snapshot = hawkes_mon.update(ts_ns=ts_ns, vpin=vpin_val)
        self._latest_snapshots[instrument_id] = snapshot
        return snapshot

    def on_order_rejected(self, instrument_id: str, ts_ns: int | None = None) -> int:
        """Record a post-only GTX rejection for instrument_id and clear resting state.

        Returns:
            Updated consecutive GTX rejection count for instrument_id.
        """
        _ = ts_ns
        new_count = self._consecutive_rejections.get(instrument_id, 0) + 1
        self._consecutive_rejections[instrument_id] = new_count
        self._resting_orders.pop(instrument_id, None)
        return new_count

    def record_gtx_rejection(self, instrument_id: str, ts_ns: int | None = None) -> int:
        """Alias for on_order_rejected."""
        return self.on_order_rejected(instrument_id=instrument_id, ts_ns=ts_ns)

    def on_order_accepted(self, instrument_id: str) -> None:
        """Reset consecutive GTX rejection counter once an order rests on the book."""
        self._consecutive_rejections[instrument_id] = 0

    def on_order_filled(self, instrument_id: str) -> None:
        """Clear resting order state and rejection counters once the order is completely filled."""
        self._resting_orders.pop(instrument_id, None)
        self._consecutive_rejections[instrument_id] = 0
        self._backoff_cooldown_until_ns.pop(instrument_id, None)

    def is_toxic_regime(
        self,
        vpin: float = 0.0,
        hawkes_eta: float = 0.0,
        hawkes_snapshot: HawkesStateSnapshot | None = None,
    ) -> bool:
        """Check whether VPIN > 0.60 or eta_Hawkes > 0.70 triggers the Toxic Sweep Shield."""
        if hawkes_snapshot is not None:
            if hawkes_snapshot.toxic_sweep_active:
                return True
            vpin = max(vpin, hawkes_snapshot.vpin)
            hawkes_eta = max(hawkes_eta, hawkes_snapshot.branching_ratio_eta)
        return (vpin > self._config.vpin_toxic_threshold) or (
            hawkes_eta > self._config.hawkes_eta_toxic_threshold
        )

    def evaluate_order_intent(
        self,
        instrument_id: str,
        ts_ns: int,
        side: Literal["BUY", "SELL"],
        quantity: float,
        best_bid: float,
        best_ask: float,
        tick_size: float = 0.01,
        vpin: float | None = None,
        hawkes_eta: float | None = None,
        hawkes_snapshot: HawkesStateSnapshot | None = None,
        resting_price: float | None = None,
        last_mod_ts_ns: int | None = None,
        consecutive_rejections: int | None = None,
        hard_stop_breached: bool = False,
        circuit_breaker_active: bool = False,
        is_entry: bool = True,
    ) -> ExecutionOrderCommand:
        """Evaluate microstructure state, hysteresis, and rejection backoff to emit an ExecutionOrderCommand.

        Priority order:
        1. Hard Stop-Loss / Emergency Circuit-Breaker -> EMERGENCY_TAKER_EXIT (IOC/Market).
        2. Toxic Sweep Shield (VPIN > 0.60 or eta_Hawkes > 0.70) -> CANCEL_ALL_TOXIC_SHIELD.
        3. 3-Strike GTX Rejection Backoff (consecutive_rejections >= 3) -> BACKOFF_DEEPER_GTX
           at 1 tick deeper (best_bid - tick_size for BUY, best_ask + tick_size for SELL)
           with 1,000 ms cooldown.
        4. Resting Order Re-Peg Hysteresis:
           Amends (AMEND_GTX_MAKER) ONLY when |new_bbo - resting_price| / resting_price >= 1.5 bps
           AND elapsed >= 500 ms (and past any active 1,000 ms backoff cooldown).
           Otherwise returns HOLD_RESTING_HYSTERESIS.
        5. Initial Normal Placement -> PLACE_GTX_MAKER at best_bid (BUY) or best_ask (SELL).
        """
        if best_bid <= 0.0 or best_ask <= 0.0:
            raise ValueError(f"BBO prices must be > 0.0, got bid={best_bid}, ask={best_ask}")
        if tick_size <= 0.0:
            raise ValueError(f"tick_size must be > 0.0, got {tick_size}")
        if quantity < 0.0:
            raise ValueError(f"quantity must be >= 0.0, got {quantity}")

        # 1. Hard stop-loss breach or emergency circuit-breaker exit (exclusive path for Taker orders)
        if hard_stop_breached or circuit_breaker_active:
            self._resting_orders.pop(instrument_id, None)
            self._consecutive_rejections[instrument_id] = 0
            self._backoff_cooldown_until_ns.pop(instrument_id, None)
            taker_price = best_ask if side == "BUY" else best_bid
            reason_str = (
                "Hard stop-loss breached: immediate IOC taker exit."
                if hard_stop_breached
                else "Emergency circuit-breaker active: immediate IOC taker exit."
            )
            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=ts_ns,
                action=ExecutionAction.EMERGENCY_TAKER_EXIT,
                side=side,
                target_price=float(taker_price),
                quantity=float(quantity),
                post_only_gtx=False,
                time_in_force="IOC",
                consecutive_rejections=0,
                reason=reason_str,
            )

        # Resolve effective VPIN and Hawkes branching ratio eta
        snap = hawkes_snapshot or self._latest_snapshots.get(instrument_id)
        eff_vpin = (
            float(vpin)
            if vpin is not None
            else (snap.vpin if snap is not None else 0.0)
        )
        eff_eta = (
            float(hawkes_eta)
            if hawkes_eta is not None
            else (snap.branching_ratio_eta if snap is not None else 0.0)
        )

        # 2. Toxic Sweep Shield (VPIN > 0.60 or eta_Hawkes > 0.70)
        if self.is_toxic_regime(vpin=eff_vpin, hawkes_eta=eff_eta, hawkes_snapshot=snap):
            self._resting_orders.pop(instrument_id, None)
            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=ts_ns,
                action=ExecutionAction.CANCEL_ALL_TOXIC_SHIELD,
                side="NONE",
                target_price=0.0,
                quantity=0.0,
                post_only_gtx=False,
                time_in_force="NONE",
                consecutive_rejections=self._consecutive_rejections.get(instrument_id, 0),
                reason=(
                    f"TOXIC_SWEEP_SHIELD triggered (VPIN={eff_vpin:.4f} > {self._config.vpin_toxic_threshold:.2f} "
                    f"or eta_Hawkes={eff_eta:.4f} > {self._config.hawkes_eta_toxic_threshold:.2f}): "
                    "cancelling all resting entry limit orders."
                ),
            )

        # Resolve effective consecutive GTX rejections
        eff_rejections = (
            int(consecutive_rejections)
            if consecutive_rejections is not None
            else self._consecutive_rejections.get(instrument_id, 0)
        )

        # 3. GTX Post-Only 3-Strike Rejection Backoff (4th attempt placed 1 tick deeper + 1,000 ms cooldown)
        if eff_rejections >= self._config.max_gtx_rejections:
            deeper_price = (
                max(tick_size, best_bid - tick_size)
                if side == "BUY"
                else (best_ask + tick_size)
            )
            cooldown_until = ts_ns + self._rejection_backoff_ns
            self._backoff_cooldown_until_ns[instrument_id] = cooldown_until
            self._resting_orders[instrument_id] = RestingMakerOrderState(
                instrument_id=instrument_id,
                side=side,
                resting_price=float(deeper_price),
                quantity=float(quantity),
                last_mod_ts_ns=ts_ns,
                cooldown_until_ts_ns=cooldown_until,
                is_entry=is_entry,
            )
            self._consecutive_rejections[instrument_id] = 0
            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=ts_ns,
                action=ExecutionAction.BACKOFF_DEEPER_GTX,
                side=side,
                target_price=float(deeper_price),
                quantity=float(quantity),
                post_only_gtx=True,
                time_in_force="GTX",
                consecutive_rejections=eff_rejections,
                reason=(
                    f"GTX rejected {eff_rejections} consecutive times (cap={self._config.max_gtx_rejections}): "
                    f"placing 4th attempt 1 tick deeper at {deeper_price:.6f} with "
                    f"{self._config.rejection_backoff_ms} ms cooldown."
                ),
            )

        new_bbo = best_bid if side == "BUY" else best_ask
        tracked_resting = self._resting_orders.get(instrument_id)

        eff_resting_price: float | None = (
            float(resting_price)
            if resting_price is not None
            else (tracked_resting.resting_price if tracked_resting is not None and tracked_resting.side == side else None)
        )
        eff_last_mod_ts_ns: int | None = (
            int(last_mod_ts_ns)
            if last_mod_ts_ns is not None
            else (tracked_resting.last_mod_ts_ns if tracked_resting is not None and tracked_resting.side == side else None)
        )
        eff_cooldown_until_ns: int = (
            tracked_resting.cooldown_until_ts_ns
            if tracked_resting is not None and tracked_resting.side == side
            else self._backoff_cooldown_until_ns.get(instrument_id, 0)
        )

        # 4. Re-Peg Hysteresis & Cooldown Rule when a resting GTX order is already active
        if eff_resting_price is not None and eff_last_mod_ts_ns is not None:
            shift_ratio = abs(new_bbo - eff_resting_price) / eff_resting_price
            elapsed_ns = ts_ns - eff_last_mod_ts_ns

            price_shifted = shift_ratio >= (self._repeg_min_ratio - _FLOAT_TOL)
            cooldown_elapsed = (elapsed_ns >= self._repeg_cooldown_ns) and (
                ts_ns >= eff_cooldown_until_ns
            )

            if price_shifted and cooldown_elapsed:
                new_cooldown_until = ts_ns + self._repeg_cooldown_ns
                self._resting_orders[instrument_id] = RestingMakerOrderState(
                    instrument_id=instrument_id,
                    side=side,
                    resting_price=float(new_bbo),
                    quantity=float(quantity),
                    last_mod_ts_ns=ts_ns,
                    cooldown_until_ts_ns=new_cooldown_until,
                    is_entry=is_entry,
                )
                return ExecutionOrderCommand(
                    instrument_id=instrument_id,
                    ts_ns=ts_ns,
                    action=ExecutionAction.AMEND_GTX_MAKER,
                    side=side,
                    target_price=float(new_bbo),
                    quantity=float(quantity),
                    post_only_gtx=True,
                    time_in_force="GTX",
                    consecutive_rejections=eff_rejections,
                    reason=(
                        f"Re-peg hysteresis satisfied: BBO shift={shift_ratio * 1e4:.2f} bps >= "
                        f"{self._config.repeg_min_bps:.2f} bps and elapsed={elapsed_ns / 1e6:.1f} ms >= "
                        f"{self._config.repeg_cooldown_ms} ms."
                    ),
                )

            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=ts_ns,
                action=ExecutionAction.HOLD_RESTING_HYSTERESIS,
                side=side,
                target_price=float(eff_resting_price),
                quantity=float(quantity),
                post_only_gtx=True,
                time_in_force="GTX",
                consecutive_rejections=eff_rejections,
                reason=(
                    f"Holding resting GTX order at {eff_resting_price:.6f}: "
                    f"BBO shift={shift_ratio * 1e4:.2f} bps (min {self._config.repeg_min_bps:.2f} bps), "
                    f"elapsed={elapsed_ns / 1e6:.1f} ms (min {self._config.repeg_cooldown_ms} ms)."
                ),
            )

        # 5. Initial Post-Only (GTX) Maker Order Placement at Top-of-Book
        cooldown_until = ts_ns + self._repeg_cooldown_ns
        self._resting_orders[instrument_id] = RestingMakerOrderState(
            instrument_id=instrument_id,
            side=side,
            resting_price=float(new_bbo),
            quantity=float(quantity),
            last_mod_ts_ns=ts_ns,
            cooldown_until_ts_ns=cooldown_until,
            is_entry=is_entry,
        )
        return ExecutionOrderCommand(
            instrument_id=instrument_id,
            ts_ns=ts_ns,
            action=ExecutionAction.PLACE_GTX_MAKER,
            side=side,
            target_price=float(new_bbo),
            quantity=float(quantity),
            post_only_gtx=True,
            time_in_force="GTX",
            consecutive_rejections=eff_rejections,
            reason=f"Normal regime (VPIN={eff_vpin:.4f}, eta={eff_eta:.4f}): placing Post-Only GTX at {new_bbo:.6f}.",
        )

    def decide_execution_action(
        self,
        instrument_id: str,
        ts_ns: int,
        side: Literal["BUY", "SELL"],
        quantity: float,
        best_bid: float,
        best_ask: float,
        tick_size: float = 0.01,
        vpin: float | None = None,
        hawkes_eta: float | None = None,
        hawkes_snapshot: HawkesStateSnapshot | None = None,
        resting_price: float | None = None,
        last_mod_ts_ns: int | None = None,
        consecutive_rejections: int | None = None,
        hard_stop_breached: bool = False,
        circuit_breaker_active: bool = False,
        is_entry: bool = True,
    ) -> ExecutionOrderCommand:
        """Alias for evaluate_order_intent."""
        return self.evaluate_order_intent(
            instrument_id=instrument_id,
            ts_ns=ts_ns,
            side=side,
            quantity=quantity,
            best_bid=best_bid,
            best_ask=best_ask,
            tick_size=tick_size,
            vpin=vpin,
            hawkes_eta=hawkes_eta,
            hawkes_snapshot=hawkes_snapshot,
            resting_price=resting_price,
            last_mod_ts_ns=last_mod_ts_ns,
            consecutive_rejections=consecutive_rejections,
            hard_stop_breached=hard_stop_breached,
            circuit_breaker_active=circuit_breaker_active,
            is_entry=is_entry,
        )

    def cancel_all_toxic_shield(
        self,
        instrument_id: str,
        ts_ns: int,
        vpin: float = 0.0,
        hawkes_eta: float = 0.0,
    ) -> ExecutionOrderCommand:
        """Immediately cancel resting entry orders for instrument_id under Toxic Sweep Shield."""
        self._resting_orders.pop(instrument_id, None)
        return ExecutionOrderCommand(
            instrument_id=instrument_id,
            ts_ns=ts_ns,
            action=ExecutionAction.CANCEL_ALL_TOXIC_SHIELD,
            side="NONE",
            target_price=0.0,
            quantity=0.0,
            post_only_gtx=False,
            time_in_force="NONE",
            consecutive_rejections=self._consecutive_rejections.get(instrument_id, 0),
            reason=(
                f"TOXIC_SWEEP_SHIELD invoked (VPIN={vpin:.4f}, eta_Hawkes={hawkes_eta:.4f}): "
                "all resting entry orders cancelled."
            ),
        )

    def evaluate_order_action(
        self,
        instrument_id: str,
        desired_side: Literal["BUY", "SELL"],
        current_mid: float,
        best_bid: float,
        best_ask: float,
        resting_order_price: float | None = None,
        resting_order_side: Literal["BUY", "SELL"] | None = None,
        vpin: float = 0.0,
        hawkes_eta: float = 0.0,
        consecutive_rejections: int = 0,
        quantity: float = 1.0,
        tick_size: float = 0.01,
        hard_stop_breached: bool = False,
        circuit_breaker_active: bool = False,
        ts_ns: int | None = None,
        is_alo: bool = True,
        lead_mid_change_bps: float | None = None,
    ) -> ExecutionOrderCommand:
        """Evaluate order action with explicit Add-Liquidity-Only (ALO) constraints and routing.

        ``lead_mid_change_bps`` is the Binance/Bybit mid move over ``lead_pull_window_ms``; when it moves against
        a resting quote by more than ``lead_pull_k`` x the HL spread the quote is pulled (CANCEL_LEAD_VENUE_PULL),
        counted separately from the VPIN/Hawkes shield.
        """
        now_ns = ts_ns if ts_ns is not None else time.time_ns()

        # 1. Hard stop-loss or emergency circuit-breaker exit
        if hard_stop_breached or circuit_breaker_active:
            return self.evaluate_order_intent(
                instrument_id=instrument_id,
                ts_ns=now_ns,
                side=desired_side,
                quantity=quantity,
                best_bid=best_bid,
                best_ask=best_ask,
                tick_size=tick_size,
                hard_stop_breached=hard_stop_breached,
                circuit_breaker_active=circuit_breaker_active,
            )

        # 2. Toxic Sweep Shield (VPIN > 0.60 or eta_Hawkes > 0.70)
        if self.is_toxic_regime(vpin=vpin, hawkes_eta=hawkes_eta):
            self._resting_orders.pop(instrument_id, None)
            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=now_ns,
                action=ExecutionAction.CANCEL_ALL_TOXIC_SHIELD,
                side="NONE",
                target_price=0.0,
                quantity=0.0,
                post_only_gtx=False,
                time_in_force="NONE",
                consecutive_rejections=consecutive_rejections or self._consecutive_rejections.get(instrument_id, 0),
                reason=(
                    f"TOXIC_SWEEP_SHIELD triggered (VPIN={vpin:.4f} > {self._config.vpin_toxic_threshold:.2f} "
                    f"or eta_Hawkes={hawkes_eta:.4f} > {self._config.hawkes_eta_toxic_threshold:.2f}): "
                    "cancelling all resting entry limit orders."
                ),
            )

        # 2b. Lead-venue quote pull (P4.1)
        resting = self._resting_orders.get(instrument_id)
        pull_side: Literal["BUY", "SELL"] | None = resting_order_side or (resting.side if resting is not None else None)
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10_000.0 if mid > 0.0 else 0.0
        if pull_side is not None and should_pull_for_lead_move(pull_side, lead_mid_change_bps, spread_bps, self._config.lead_pull_k):
            self._resting_orders.pop(instrument_id, None)
            self._lead_pull_counts[instrument_id] = self._lead_pull_counts.get(instrument_id, 0) + 1
            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=now_ns,
                action=ExecutionAction.CANCEL_LEAD_VENUE_PULL,
                side="NONE",
                target_price=0.0,
                quantity=0.0,
                post_only_gtx=False,
                time_in_force="NONE",
                consecutive_rejections=self._consecutive_rejections.get(instrument_id, 0),
                reason=(
                    f"LEAD_VENUE_PULL: lead mid moved {lead_mid_change_bps:+.2f} bps in {self._config.lead_pull_window_ms} ms "
                    f"against the resting {pull_side} (threshold {self._config.lead_pull_k:.2f} x spread {spread_bps:.2f} bps)."
                ),
            )

        # 3. 3-Strike GTX Rejection Backoff (consecutive_rejections >= 3)
        eff_rejections = consecutive_rejections if consecutive_rejections > 0 else self._consecutive_rejections.get(instrument_id, 0)
        if eff_rejections >= self._config.max_gtx_rejections:
            deeper_price = (
                max(tick_size, best_bid - tick_size)
                if desired_side == "BUY"
                else (best_ask + tick_size)
            )
            cooldown_until = now_ns + self._rejection_backoff_ns
            self._backoff_cooldown_until_ns[instrument_id] = cooldown_until
            self._resting_orders[instrument_id] = RestingMakerOrderState(
                instrument_id=instrument_id,
                side=desired_side,
                resting_price=float(deeper_price),
                quantity=float(quantity),
                last_mod_ts_ns=now_ns,
                cooldown_until_ts_ns=cooldown_until,
                is_entry=True,
            )
            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=now_ns,
                action=ExecutionAction.BACKOFF_DEEPER_GTX,
                side=desired_side,
                target_price=float(deeper_price),
                quantity=float(quantity),
                post_only_gtx=True,
                time_in_force="GTX",
                consecutive_rejections=eff_rejections,
                reason=(
                    f"GTX rejected {eff_rejections} consecutive times: "
                    f"placing deeper passive ALO maker at {deeper_price:.6f}."
                ),
            )

        # 4. ALO Post-Only Maker Constraints:
        # Check if resting order is crossing the spread
        is_crossed = False
        if resting_order_price is not None:
            if desired_side == "BUY" and resting_order_price >= best_ask:
                is_crossed = True
            elif desired_side == "SELL" and resting_order_price <= best_bid:
                is_crossed = True

        if is_crossed:
            # Cannot hold crossed order in ALO. Must place or amend passive post-only maker at best BBO
            passive_price = best_bid if desired_side == "BUY" else best_ask
            self._resting_orders[instrument_id] = RestingMakerOrderState(
                instrument_id=instrument_id,
                side=desired_side,
                resting_price=float(passive_price),
                quantity=float(quantity),
                last_mod_ts_ns=now_ns,
                cooldown_until_ts_ns=now_ns + self._repeg_cooldown_ns,
                is_entry=True,
            )
            return ExecutionOrderCommand(
                instrument_id=instrument_id,
                ts_ns=now_ns,
                action=ExecutionAction.PLACE_GTX_MAKER,
                side=desired_side,
                target_price=float(passive_price),
                quantity=float(quantity),
                post_only_gtx=True,
                time_in_force="GTX",
                consecutive_rejections=0,
                reason=(
                    f"ALO Post-Only constraint: crossed resting price {resting_order_price:.2f} clamped "
                    f"to passive BBO {passive_price:.2f} to guarantee maker fee."
                ),
            )

        # If resting order exists and is not crossed, evaluate hysteresis:
        if resting_order_price is not None:
            target_bbo = best_bid if desired_side == "BUY" else best_ask
            shift_ratio = abs(target_bbo - resting_order_price) / max(resting_order_price, 1e-9)
            tracked = self._resting_orders.get(instrument_id)
            elapsed_ns = (
                (now_ns - tracked.last_mod_ts_ns)
                if (tracked is not None and tracked.side == desired_side)
                else self._repeg_cooldown_ns + 1
            )
            if shift_ratio >= (self._repeg_min_ratio - _FLOAT_TOL) and elapsed_ns >= self._repeg_cooldown_ns:
                self._resting_orders[instrument_id] = RestingMakerOrderState(
                    instrument_id=instrument_id,
                    side=desired_side,
                    resting_price=float(target_bbo),
                    quantity=float(quantity),
                    last_mod_ts_ns=now_ns,
                    cooldown_until_ts_ns=now_ns + self._repeg_cooldown_ns,
                    is_entry=True,
                )
                return ExecutionOrderCommand(
                    instrument_id=instrument_id,
                    ts_ns=now_ns,
                    action=ExecutionAction.AMEND_GTX_MAKER,
                    side=desired_side,
                    target_price=float(target_bbo),
                    quantity=float(quantity),
                    post_only_gtx=True,
                    time_in_force="GTX",
                    consecutive_rejections=0,
                    reason=f"ALO maker re-peg hysteresis satisfied: amending to {target_bbo:.6f}.",
                )
            else:
                return ExecutionOrderCommand(
                    instrument_id=instrument_id,
                    ts_ns=now_ns,
                    action=ExecutionAction.HOLD_RESTING_HYSTERESIS,
                    side=desired_side,
                    target_price=float(resting_order_price),
                    quantity=float(quantity),
                    post_only_gtx=True,
                    time_in_force="GTX",
                    consecutive_rejections=0,
                    reason=f"ALO maker holding resting order at {resting_order_price:.6f}.",
                )

        # 5. Initial normal placement at passive BBO
        target_price = best_bid if desired_side == "BUY" else best_ask
        self._resting_orders[instrument_id] = RestingMakerOrderState(
            instrument_id=instrument_id,
            side=desired_side,
            resting_price=float(target_price),
            quantity=float(quantity),
            last_mod_ts_ns=now_ns,
            cooldown_until_ts_ns=now_ns + self._repeg_cooldown_ns,
            is_entry=True,
        )
        return ExecutionOrderCommand(
            instrument_id=instrument_id,
            ts_ns=now_ns,
            action=ExecutionAction.PLACE_GTX_MAKER,
            side=desired_side,
            target_price=float(target_price),
            quantity=float(quantity),
            post_only_gtx=True,
            time_in_force="GTX",
            consecutive_rejections=0,
            reason=f"ALO Post-Only maker order placed at passive BBO {target_price:.6f}.",
        )

    def shutdown(self) -> None:
        """Shut down all background Hawkes MLE refit executors."""
        for mon in self._hawkes_monitors.values():
            mon.shutdown()


__all__ = [
    "should_pull_for_lead_move",
    "ALOOrderConstraint",
    "ALO_ORDER_TYPE",
    "ALO_ROUTING_TAG",
    "DefensiveMakerConfig",
    "DefensiveMakerExecutionAlgorithm",
    "RecursiveHawkesMonitor",
    "RestingMakerOrderState",
    "RollingBVCVPINMonitor",
]
