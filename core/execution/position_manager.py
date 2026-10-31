"""Position lifecycle for the paper-trading loop (REINFORCEMENT_PLAN tasks P0.5 and P0.6).

Replaces the per-bar mark-to-market "win/loss" accounting with explicit positions:

- A position opens when a sleeve emits a non-zero target weight for an asset while the
  (sleeve, asset) pair is flat.
- A directional position exits on a triple-barrier touch evaluated against the bar's real
  high and low, on vertical expiry, on a regime veto, or when the sleeve's target flips sign.
- A hedged carry position (spot long / perp short) has no price barriers; it exits when the
  sleeve's target clears to zero or flips, or on a regime veto.
- Funding accrues only on open perpetual legs, with sign: a long pays positive funding, a
  short receives it and pays when funding is negative.
- Fees are charged on the notional actually traded at entry and at exit (maker by default,
  taker for stop-outs and vetoes).
- PnL, fees and the win/loss counters are realised on close only, attributed per sleeve.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.contracts import (
    ClosedTradeRecord,
    DollarBar,
    FeeSchedule,
    Position,
    PositionExitReason,
    SleeveAttribution,
    TripleBarrierConfig,
)
from core.models.triple_barrier import compute_effective_barriers

logger = logging.getLogger("phoenix.positions")

NANOSECONDS_PER_HOUR: Final[float] = 3_600.0 * 1_000_000_000.0

# Exits that cannot rest passively in the book are charged the taker fee.
TAKER_EXIT_REASONS: Final[frozenset[PositionExitReason]] = frozenset(
    {PositionExitReason.STOP_BARRIER, PositionExitReason.REGIME_VETO}
)


def position_key(sleeve_id: str, asset: str) -> str:
    """Canonical dictionary key for one (sleeve, asset) book."""
    return f"{sleeve_id}:{asset}"


class SleeveTarget(BaseModel):
    """Signed target weight for one (sleeve, asset) pair emitted by the allocation step."""

    model_config = ConfigDict(frozen=True)

    sleeve_id: str
    asset: str
    weight: float = Field(description="Signed fraction of equity; for hedged carry the per-leg weight (> 0)")
    hedged: bool = False
    p_meta: float = Field(default=0.5, ge=0.0, le=1.0)
    expected_gross_return_bps: float = 0.0
    reason: str = ""

    @property
    def key(self) -> str:
        """Dictionary key shared with ``PositionManager``."""
        return position_key(self.sleeve_id, self.asset)

    @property
    def side(self) -> Literal[-1, 0, 1]:
        """Sign of the perpetual leg implied by the target (hedged carry is a perp short)."""
        if self.weight == 0.0:
            return 0
        if self.hedged:
            return -1
        return 1 if self.weight > 0.0 else -1


class BarMarks(BaseModel):
    """Per-asset market context for one bar used to manage open positions."""

    model_config = ConfigDict(frozen=True)

    asset: str
    bar: DollarBar | None = None
    mark_price: float = Field(gt=0.0)
    spot_price: float = Field(gt=0.0)
    funding_rate_1h: float = 0.0
    bar_hours: float = Field(default=0.25, gt=0.0)

    @property
    def high(self) -> float:
        """Bar high when a bar exists, otherwise the mark price."""
        return self.bar.high if self.bar is not None else self.mark_price

    @property
    def low(self) -> float:
        """Bar low when a bar exists, otherwise the mark price."""
        return self.bar.low if self.bar is not None else self.mark_price

    @property
    def close(self) -> float:
        """Bar close when a bar exists, otherwise the mark price."""
        return self.bar.close if self.bar is not None else self.mark_price


class PositionManager:
    """Tracks open positions, applies exits, and realises PnL per sleeve."""

    def __init__(
        self,
        fee_schedule: FeeSchedule | None = None,
        barrier_config: TripleBarrierConfig | None = None,
        spot_fee_schedule: FeeSchedule | None = None,
    ) -> None:
        self._fee = fee_schedule or FeeSchedule.hyperliquid()
        self._spot_fee = spot_fee_schedule or self._fee
        self._barrier_cfg = barrier_config or TripleBarrierConfig(fee_schedule=self._fee)
        self._open: dict[str, Position] = {}
        self._closed: list[ClosedTradeRecord] = []
        self._attribution: dict[str, SleeveAttribution] = {}
        self._trade_counter: int = 0

    # ------------------------------------------------------------------ properties

    @property
    def fee_schedule(self) -> FeeSchedule:
        """Fee schedule applied to the perpetual leg."""
        return self._fee

    @property
    def barrier_config(self) -> TripleBarrierConfig:
        """Triple-barrier configuration used to place exits."""
        return self._barrier_cfg

    @property
    def open_positions(self) -> dict[str, Position]:
        """Copy of the open book keyed by ``sleeve:asset``."""
        return dict(self._open)

    @property
    def closed_trades(self) -> list[ClosedTradeRecord]:
        """All closed trades in close order."""
        return list(self._closed)

    @property
    def wins(self) -> int:
        """Closed trades with positive net PnL."""
        return sum(1 for t in self._closed if t.is_win)

    @property
    def losses(self) -> int:
        """Closed trades with non-positive net PnL."""
        return sum(1 for t in self._closed if not t.is_win)

    @property
    def realized_net_pnl_usd(self) -> float:
        """Sum of net PnL over closed trades."""
        return float(sum(t.net_pnl_usd for t in self._closed))

    @property
    def total_fees_usd(self) -> float:
        """Fees charged on closed trades."""
        return float(sum(t.fee_usd for t in self._closed))

    @property
    def total_funding_pnl_usd(self) -> float:
        """Signed funding realised on closed trades."""
        return float(sum(t.funding_pnl_usd for t in self._closed))

    def is_flat(self, sleeve_id: str, asset: str) -> bool:
        """True when no position is open for the pair."""
        return position_key(sleeve_id, asset) not in self._open

    def get(self, sleeve_id: str, asset: str) -> Position | None:
        """Open position for the pair, if any."""
        return self._open.get(position_key(sleeve_id, asset))

    def gross_exposure_usd(self) -> float:
        """Sum of absolute leg notionals at entry (hedged positions count both legs)."""
        total = 0.0
        for p in self._open.values():
            total += p.notional_usd * (2.0 if p.hedged else 1.0)
        return total

    # ------------------------------------------------------------------ attribution

    def sleeve_attribution(self, marks: Mapping[str, BarMarks] | None = None) -> dict[str, SleeveAttribution]:
        """Per-sleeve realised attribution, plus open-position count and unrealised PnL."""
        result: dict[str, SleeveAttribution] = {}
        sleeves = set(self._attribution) | {p.sleeve_id for p in self._open.values()}
        for sleeve_id in sorted(sleeves):
            base = self._attribution.get(sleeve_id, SleeveAttribution(sleeve_id=sleeve_id))
            open_for_sleeve = [p for p in self._open.values() if p.sleeve_id == sleeve_id]
            unreal = 0.0
            if marks is not None:
                for p in open_for_sleeve:
                    m = marks.get(p.asset)
                    if m is not None:
                        unreal += self.unrealized_pnl_usd(p, m.mark_price, m.spot_price)
            result[sleeve_id] = base.model_copy(
                update={"open_positions": len(open_for_sleeve), "unrealized_pnl_usd": round(unreal, 4)}
            )
        return result

    def unrealized_pnl_usd(self, position: Position, mark_price: float, spot_price: float) -> float:
        """Gross mark-to-market of an open position plus funding accrued so far, before exit fees."""
        gross = self._gross_pnl(position, exit_price=mark_price, spot_exit_price=spot_price)
        return gross + position.funding_accrued_usd

    def total_unrealized_pnl_usd(self, marks: Mapping[str, BarMarks]) -> float:
        """Unrealised PnL across the open book."""
        total = 0.0
        for p in self._open.values():
            m = marks.get(p.asset)
            if m is not None:
                total += self.unrealized_pnl_usd(p, m.mark_price, m.spot_price)
        return total

    # ------------------------------------------------------------------ entries

    def open_position(
        self,
        target: SleeveTarget,
        *,
        equity_usd: float,
        price: float,
        spot_price: float,
        bar_index: int,
        ts_ns: int,
        sigma_bar: float,
    ) -> Position | None:
        """Open a position for ``target`` when the pair is flat and the target is non-zero."""
        if target.weight == 0.0 or not self.is_flat(target.sleeve_id, target.asset):
            return None
        if equity_usd <= 0.0 or price <= 0.0 or spot_price <= 0.0:
            return None

        notional = abs(target.weight) * equity_usd
        if notional <= 0.0:
            return None
        size = notional / price
        side: Literal[-1, 1] = -1 if target.hedged else (1 if target.weight > 0.0 else -1)

        upper: float | None = None
        lower: float | None = None
        vertical: int | None = None
        if not target.hedged:
            ub, lb = compute_effective_barriers(max(sigma_bar, 0.0), self._barrier_cfg)
            if side > 0:
                upper = price * (1.0 + ub)
                lower = price * (1.0 - lb)
            else:
                upper = price * (1.0 + lb)
                lower = price * (1.0 - ub)
            vertical = bar_index + self._barrier_cfg.vertical_barrier_bars

        entry_fee = size * price * self._fee.maker_rate
        if target.hedged:
            entry_fee += size * spot_price * self._spot_fee.maker_rate

        position = Position(
            position_id=f"{target.sleeve_id}_{target.asset}_{bar_index}",
            asset=target.asset,
            sleeve_id=target.sleeve_id,
            side=side,
            hedged=target.hedged,
            entry_price=price,
            spot_entry_price=spot_price if target.hedged else None,
            entry_bar_index=bar_index,
            entry_ts_ns=ts_ns,
            size=size,
            notional_usd=notional,
            target_weight=target.weight,
            upper_barrier=upper,
            lower_barrier=lower,
            vertical_bar_index=vertical,
            entry_fee_usd=entry_fee,
            sigma_bar_at_entry=max(sigma_bar, 0.0),
            entry_reason=target.reason,
        )
        self._open[target.key] = position
        logger.info(
            "OPEN %s %s side=%+d notional=$%.2f entry=%.2f upper=%s lower=%s vertical=%s",
            position.sleeve_id,
            position.asset,
            position.side,
            position.notional_usd,
            position.entry_price,
            f"{upper:.2f}" if upper is not None else "-",
            f"{lower:.2f}" if lower is not None else "-",
            vertical if vertical is not None else "-",
        )
        return position

    def restore(self, position: Position) -> None:
        """Re-attach a persisted open position (ledger reload after a restart)."""
        key = position_key(position.sleeve_id, position.asset)
        if key in self._open:
            logger.warning("Duplicate persisted position for %s ignored.", key)
            return
        self._open[key] = position

    # ------------------------------------------------------------------ bar processing

    def process_bar(
        self,
        *,
        bar_index: int,
        ts_ns: int,
        marks: Mapping[str, BarMarks],
        targets: Mapping[str, SleeveTarget],
        regime_veto: bool = False,
    ) -> list[ClosedTradeRecord]:
        """Advance every open position by one bar: accrue funding, then apply exits.

        Exit precedence per position: regime veto, barrier touch (same-bar double touch
        resolves to the stop, as in the labeller), vertical expiry, then target flip/clear.
        """
        closed: list[ClosedTradeRecord] = []
        for key in list(self._open.keys()):
            pos = self._open[key]
            m = marks.get(pos.asset)
            if m is None:
                # No market data for this asset on this bar: hold, nothing accrues.
                continue

            # Funding accrues on the perpetual leg for the bar held (pro-rata by bar duration).
            funding_rate_bar = float(m.funding_rate_1h) * float(m.bar_hours)
            perp_notional = pos.size * m.mark_price
            funding_pnl_bar = -float(pos.side) * funding_rate_bar * perp_notional
            pos = pos.model_copy(
                update={
                    "funding_accrued_usd": pos.funding_accrued_usd + funding_pnl_bar,
                    "bars_held": pos.bars_held + 1,
                }
            )
            self._open[key] = pos

            exit_reason: PositionExitReason | None = None
            exit_price: float = m.close

            if regime_veto:
                exit_reason = PositionExitReason.REGIME_VETO
                exit_price = m.close
            elif not pos.hedged:
                profit_px = pos.profit_barrier
                stop_px = pos.stop_barrier
                hit_profit = False
                hit_stop = False
                if profit_px is not None and stop_px is not None:
                    if pos.side > 0:
                        hit_profit = m.high >= profit_px
                        hit_stop = m.low <= stop_px
                    else:
                        hit_profit = m.low <= profit_px
                        hit_stop = m.high >= stop_px
                if hit_stop and stop_px is not None:
                    exit_reason = PositionExitReason.STOP_BARRIER
                    exit_price = stop_px
                elif hit_profit and profit_px is not None:
                    exit_reason = PositionExitReason.PROFIT_BARRIER
                    exit_price = profit_px
                elif pos.vertical_bar_index is not None and bar_index >= pos.vertical_bar_index:
                    exit_reason = PositionExitReason.VERTICAL_EXPIRY
                    exit_price = m.close
                else:
                    tgt = targets.get(key)
                    if tgt is not None and tgt.side != 0 and tgt.side != pos.side:
                        exit_reason = PositionExitReason.TARGET_FLIPPED
                        exit_price = m.close
            else:
                tgt = targets.get(key)
                if tgt is None or tgt.weight == 0.0:
                    exit_reason = PositionExitReason.TARGET_CLEARED
                elif not tgt.hedged:
                    exit_reason = PositionExitReason.TARGET_FLIPPED
                exit_price = m.close

            if exit_reason is not None:
                closed.append(
                    self._close(
                        key,
                        pos,
                        exit_reason=exit_reason,
                        exit_price=exit_price,
                        spot_exit_price=m.spot_price,
                        bar_index=bar_index,
                        ts_ns=ts_ns,
                    )
                )
        return closed

    def close_all(
        self,
        *,
        bar_index: int,
        ts_ns: int,
        marks: Mapping[str, BarMarks],
        reason: PositionExitReason = PositionExitReason.REGIME_VETO,
    ) -> list[ClosedTradeRecord]:
        """Close every open position at the current mark (used by emergency flatten)."""
        closed: list[ClosedTradeRecord] = []
        for key in list(self._open.keys()):
            pos = self._open[key]
            m = marks.get(pos.asset)
            if m is None:
                continue
            closed.append(
                self._close(
                    key,
                    pos,
                    exit_reason=reason,
                    exit_price=m.mark_price,
                    spot_exit_price=m.spot_price,
                    bar_index=bar_index,
                    ts_ns=ts_ns,
                )
            )
        return closed

    # ------------------------------------------------------------------ internals

    def _gross_pnl(self, pos: Position, exit_price: float, spot_exit_price: float) -> float:
        perp_leg = float(pos.side) * (exit_price - pos.entry_price) * pos.size
        if not pos.hedged or pos.spot_entry_price is None:
            return perp_leg
        spot_leg = (spot_exit_price - pos.spot_entry_price) * pos.size
        return perp_leg + spot_leg

    def _exit_fee(self, pos: Position, exit_reason: PositionExitReason, exit_price: float, spot_exit_price: float) -> float:
        rate = self._fee.taker_rate if exit_reason in TAKER_EXIT_REASONS else self._fee.maker_rate
        fee = pos.size * exit_price * rate
        if pos.hedged:
            spot_rate = self._spot_fee.taker_rate if exit_reason in TAKER_EXIT_REASONS else self._spot_fee.maker_rate
            fee += pos.size * spot_exit_price * spot_rate
        return fee

    def _close(
        self,
        key: str,
        pos: Position,
        *,
        exit_reason: PositionExitReason,
        exit_price: float,
        spot_exit_price: float,
        bar_index: int,
        ts_ns: int,
    ) -> ClosedTradeRecord:
        gross = self._gross_pnl(pos, exit_price=exit_price, spot_exit_price=spot_exit_price)
        exit_fee = self._exit_fee(pos, exit_reason, exit_price, spot_exit_price)
        fees = pos.entry_fee_usd + exit_fee
        net = gross + pos.funding_accrued_usd - fees
        leg_notional = pos.notional_usd * (2.0 if pos.hedged else 1.0)
        self._trade_counter += 1
        record = ClosedTradeRecord(
            trade_id=f"trade_{self._trade_counter:06d}_{pos.position_id}",
            position_id=pos.position_id,
            asset=pos.asset,
            sleeve_id=pos.sleeve_id,
            side=pos.side,
            hedged=pos.hedged,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            spot_entry_price=pos.spot_entry_price,
            spot_exit_price=spot_exit_price if pos.hedged else None,
            entry_bar_index=pos.entry_bar_index,
            exit_bar_index=bar_index,
            entry_ts_ns=pos.entry_ts_ns,
            exit_ts_ns=ts_ns,
            size=pos.size,
            notional_usd=pos.notional_usd,
            exit_reason=exit_reason,
            gross_pnl_usd=gross,
            funding_pnl_usd=pos.funding_accrued_usd,
            fee_usd=fees,
            net_pnl_usd=net,
            net_return_bps=(net / leg_notional) * 10_000.0 if leg_notional > 0.0 else 0.0,
            bars_held=pos.bars_held,
            is_win=net > 0.0,
            entry_reason=pos.entry_reason,
        )
        del self._open[key]
        self._closed.append(record)

        prev = self._attribution.get(pos.sleeve_id, SleeveAttribution(sleeve_id=pos.sleeve_id))
        self._attribution[pos.sleeve_id] = prev.model_copy(
            update={
                "closed_trades": prev.closed_trades + 1,
                "wins": prev.wins + (1 if record.is_win else 0),
                "losses": prev.losses + (0 if record.is_win else 1),
                "gross_pnl_usd": prev.gross_pnl_usd + gross,
                "funding_pnl_usd": prev.funding_pnl_usd + pos.funding_accrued_usd,
                "fees_usd": prev.fees_usd + fees,
                "net_pnl_usd": prev.net_pnl_usd + net,
            }
        )
        logger.info(
            "CLOSE %s %s reason=%s exit=%.2f gross=$%.2f funding=$%.2f fees=$%.2f net=$%.2f bars=%d",
            pos.sleeve_id,
            pos.asset,
            exit_reason.value,
            exit_price,
            gross,
            pos.funding_accrued_usd,
            fees,
            net,
            pos.bars_held,
        )
        return record


__all__ = [
    "BarMarks",
    "PositionManager",
    "SleeveTarget",
    "TAKER_EXIT_REASONS",
    "position_key",
]
