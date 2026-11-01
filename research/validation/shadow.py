"""Shadow mode and promotion workflow (REINFORCEMENT_PLAN task P5.1).

A sleeve that has passed CPCV but is not yet trusted with capital runs in shadow: its targets go to a separate
``PositionManager`` whose trades are tracked at full paper notional but never booked to the ledger. For each
shadow trade the modelled cost (round-trip fee plus estimated slippage) is reconciled against the realised
cost (fees plus the adverse part of the 1-second exit markout, when a markout is available).

Kill thresholds are pre-registered from the CPCV out-of-sample path distribution before the shadow run
starts: the shadow run is killed when its Sharpe falls below 80% of the median path Sharpe or its drawdown
exceeds the 5th-percentile path drawdown. After ``min_trades`` (200) shadow trades with no kill, the sleeve
is eligible for promotion; promotion itself is a manual step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import ClosedTradeRecord, FeeSchedule

HOURS_PER_YEAR = 8760.0


class PreRegisteredThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    sleeve_id: str
    median_path_sharpe: float
    min_live_sharpe: float = Field(description="80% of the median CPCV path Sharpe")
    max_drawdown: float = Field(ge=0.0, description="Drawdown of the 5th-percentile CPCV path (as a positive fraction)")
    n_paths: int = Field(ge=1)
    registered_from: str = ""
    basis: Literal["per_hour", "per_trade"] = Field(
        default="per_hour",
        description="per_hour: Sharpe annualised from hourly path returns; per_trade: Sharpe of per-trade returns (the shadow book's unit)",
    )


def _path_sharpe(r: NDArray[np.float64], periods_per_year: float) -> float:
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * math.sqrt(periods_per_year)) if sd > 1e-15 else 0.0


def _path_drawdown(r: NDArray[np.float64]) -> float:
    eq = np.cumsum(np.where(np.isfinite(r), r, 0.0))
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    return float(np.max(peak - eq)) if eq.size else 0.0


def register_thresholds(sleeve_id: str, oos_paths: NDArray[np.float64], periods_per_year: float = HOURS_PER_YEAR, note: str = "") -> PreRegisteredThresholds:
    """Thresholds from CPCV out-of-sample paths (shape ``(n_paths, T)``) as returned by ``run_validation``."""
    paths = np.atleast_2d(np.asarray(oos_paths, dtype=np.float64))
    sharpes = np.array([_path_sharpe(p, periods_per_year) for p in paths])
    dds = np.array([_path_drawdown(p) for p in paths])
    med = float(np.median(sharpes))
    return PreRegisteredThresholds(
        sleeve_id=sleeve_id,
        median_path_sharpe=med,
        min_live_sharpe=0.8 * med,
        max_drawdown=float(np.percentile(dds, 95)),  # the 5th-percentile path is the 95th-percentile drawdown
        n_paths=int(paths.shape[0]),
        registered_from=note,
    )


class ShadowReconciliation(BaseModel):
    model_config = ConfigDict(frozen=True)

    trade_id: str
    modelled_cost_bps: float
    realised_cost_bps: float | None
    difference_bps: float | None


class ShadowStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    sleeve_id: str
    trades: int
    min_trades: int
    net_pnl_usd: float
    sharpe: float | None
    drawdown_frac: float
    killed: bool
    kill_reason: str | None
    eligible_for_promotion: bool
    thresholds: PreRegisteredThresholds | None
    mean_cost_gap_bps: float | None
    state: Literal["SHADOW", "KILLED", "ELIGIBLE"]


class ShadowBook:
    """Per-sleeve shadow statistics fed with closed shadow trades."""

    def __init__(self, sleeve_id: str, thresholds: PreRegisteredThresholds | None = None, min_trades: int = 200, capital_usd: float = 5_000.0) -> None:
        self.sleeve_id = sleeve_id
        self.thresholds = thresholds
        self.min_trades = min_trades
        self.capital_usd = capital_usd
        self._returns: list[float] = []
        self._recon: list[ShadowReconciliation] = []
        self._net: float = 0.0
        self._killed: str | None = None

    def record(self, trade: ClosedTradeRecord, fee: FeeSchedule, exit_markout_1s_bps: float | None = None) -> ShadowReconciliation:
        leg = 2.0 if trade.hedged else 1.0
        modelled = fee.roundtrip_maker_plus_slippage_bps * leg
        realised: float | None = None
        notional = trade.notional_usd * leg
        if notional > 0:
            fee_bps = trade.fee_usd / notional * 1e4
            realised = fee_bps + (max(0.0, -exit_markout_1s_bps) if exit_markout_1s_bps is not None else fee.estimated_slippage_bps * 2.0 * leg)
        rec = ShadowReconciliation(trade_id=trade.trade_id, modelled_cost_bps=modelled, realised_cost_bps=realised, difference_bps=None if realised is None else realised - modelled)
        self._recon.append(rec)
        self._net += trade.net_pnl_usd
        self._returns.append(trade.net_pnl_usd / self.capital_usd)
        self._check_kill()
        return rec

    def _check_kill(self) -> None:
        if self._killed is not None or self.thresholds is None or len(self._returns) < 20:
            return
        r = np.asarray(self._returns)
        dd = _path_drawdown(r)
        if dd > self.thresholds.max_drawdown:
            self._killed = f"drawdown {dd:.2%} beyond the 5th-percentile CPCV path ({self.thresholds.max_drawdown:.2%})"
            return
        if len(self._returns) >= self.min_trades:
            sd = float(np.std(r, ddof=1))
            per_trade_sharpe = float(np.mean(r) / sd) if sd > 0 else 0.0
            path_per_trade = (
                self.thresholds.min_live_sharpe
                if self.thresholds.basis == "per_trade"
                else self.thresholds.min_live_sharpe / math.sqrt(HOURS_PER_YEAR)
            )
            if per_trade_sharpe < path_per_trade:
                self._killed = f"Sharpe below 80% of the median CPCV path after {len(self._returns)} trades"

    def status(self) -> ShadowStatus:
        r = np.asarray(self._returns, dtype=np.float64)
        sd = float(np.std(r, ddof=1)) if r.size > 2 else 0.0
        sharpe = float(np.mean(r) / sd) if sd > 0 else None
        gaps = [x.difference_bps for x in self._recon if x.difference_bps is not None]
        eligible = self._killed is None and len(self._returns) >= self.min_trades and self.thresholds is not None
        return ShadowStatus(
            sleeve_id=self.sleeve_id,
            trades=len(self._returns),
            min_trades=self.min_trades,
            net_pnl_usd=self._net,
            sharpe=sharpe,
            drawdown_frac=_path_drawdown(r) if r.size else 0.0,
            killed=self._killed is not None,
            kill_reason=self._killed,
            eligible_for_promotion=eligible,
            thresholds=self.thresholds,
            mean_cost_gap_bps=float(np.mean(gaps)) if gaps else None,
            state="KILLED" if self._killed else ("ELIGIBLE" if eligible else "SHADOW"),
        )


def save_thresholds(items: list[PreRegisteredThresholds], path: Path) -> None:
    """Merge thresholds into a JSON file keyed by sleeve id (the pre-registration record)."""
    existing: dict[str, object] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for th in items:
        existing[th.sleeve_id] = th.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_thresholds(path: Path) -> dict[str, PreRegisteredThresholds]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: PreRegisteredThresholds(**v) for k, v in raw.items()}


__all__ = ["PreRegisteredThresholds", "ShadowBook", "ShadowReconciliation", "ShadowStatus", "load_thresholds", "register_thresholds", "save_thresholds"]

