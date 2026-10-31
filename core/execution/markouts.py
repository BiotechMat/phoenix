"""Markout ladder (REINFORCEMENT_PLAN task P4.2).

Every fill is stored with its size bucket, minutes to the next hourly funding settlement, the lead/lag state
(the lead venue's recent move relative to the fill side) and the sleeve. Mid-to-mid markouts are resolved at
1 s, 5 s, 10 s, 30 s, 60 s and 5 min from a mid history; a horizon whose mid is unavailable stays ``None``.
A slow EWMA of the 30-second median markout tightens the VPIN toxicity threshold when fills are being
picked off.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
import bisect
import statistics
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MARKOUT_HORIZONS_MS: tuple[int, ...] = (1_000, 5_000, 10_000, 30_000, 60_000, 300_000)
SIZE_BUCKETS_USD: tuple[float, ...] = (1_000.0, 10_000.0, 100_000.0)


def size_bucket(notional_usd: float) -> str:
    for edge in SIZE_BUCKETS_USD:
        if notional_usd < edge:
            return f"<{int(edge):,}"
    return f">={int(SIZE_BUCKETS_USD[-1]):,}"


def minutes_to_hourly_funding(ts_ms: int) -> float:
    return ((3_600_000 - ts_ms % 3_600_000) % 3_600_000) / 60_000.0


class FillRecord(BaseModel):
    model_config = ConfigDict(frozen=False)

    fill_id: str
    coin: str
    side: Literal["BUY", "SELL"]
    price: float = Field(gt=0.0)
    ts_ms: int
    notional_usd: float = Field(ge=0.0)
    sleeve_id: str
    size_bucket: str
    minutes_to_funding: float
    lead_state: str
    markouts_bps: dict[int, float | None] = Field(default_factory=dict)
    resolved: dict[int, bool] = Field(default_factory=dict)


class MidHistory:
    """Thread-safe recent mids per coin for markout resolution."""

    def __init__(self, retention_ms: int = 10 * 60_000, max_staleness_ms: int = 2_000) -> None:
        self._lock = threading.Lock()
        self._ts: dict[str, list[int]] = {}
        self._mid: dict[str, list[float]] = {}
        self._retention = retention_ms
        self._staleness = max_staleness_ms

    def add(self, coin: str, ts_ms: int, mid: float) -> None:
        c = coin.upper()
        with self._lock:
            ts = self._ts.setdefault(c, [])
            mids = self._mid.setdefault(c, [])
            if ts and ts_ms < ts[-1]:
                return
            ts.append(ts_ms)
            mids.append(mid)
            cut = bisect.bisect_left(ts, ts_ms - self._retention)
            if cut > 0:
                del ts[:cut]
                del mids[:cut]

    def mid_at(self, coin: str, ts_ms: int) -> float | None:
        with self._lock:
            ts = self._ts.get(coin.upper(), [])
            if not ts or ts_ms > ts[-1]:
                return None
            i = bisect.bisect_right(ts, ts_ms) - 1
            if i < 0 or ts_ms - ts[i] > self._staleness:
                return None
            return self._mid[coin.upper()][i]

    def latest_ts(self, coin: str) -> int | None:
        with self._lock:
            ts = self._ts.get(coin.upper(), [])
            return ts[-1] if ts else None


class MarkoutTracker:
    def __init__(self, horizons_ms: Sequence[int] = MARKOUT_HORIZONS_MS, keep: int = 2_000, ewma_alpha: float = 0.05) -> None:
        self.horizons: tuple[int, ...] = tuple(horizons_ms)
        self._fills: deque[FillRecord] = deque(maxlen=keep)
        self._alpha = ewma_alpha
        self._ewma_30s: float | None = None

    def record_fill(
        self,
        *,
        fill_id: str,
        coin: str,
        side: Literal["BUY", "SELL"],
        price: float,
        ts_ms: int,
        notional_usd: float,
        sleeve_id: str,
        lead_move_bps: float | None = None,
    ) -> FillRecord:
        sign = 1.0 if side == "BUY" else -1.0
        if lead_move_bps is None:
            lead_state = "unknown"
        elif sign * lead_move_bps < -0.5:
            lead_state = "lead_against"
        elif sign * lead_move_bps > 0.5:
            lead_state = "lead_with"
        else:
            lead_state = "lead_flat"
        rec = FillRecord(
            fill_id=fill_id, coin=coin.upper(), side=side, price=price, ts_ms=ts_ms, notional_usd=notional_usd, sleeve_id=sleeve_id,
            size_bucket=size_bucket(notional_usd), minutes_to_funding=minutes_to_hourly_funding(ts_ms), lead_state=lead_state,
            markouts_bps={h: None for h in self.horizons}, resolved={h: False for h in self.horizons},
        )
        self._fills.append(rec)
        return rec

    def resolve(self, mid_at: Callable[[str, int], float | None], now_ms: int, give_up_after_ms: int = 60_000) -> int:
        """Fill in every horizon whose time has passed; returns how many values were resolved."""
        n = 0
        for f in self._fills:
            sign = 1.0 if f.side == "BUY" else -1.0
            for h in self.horizons:
                if f.resolved[h] or now_ms < f.ts_ms + h:
                    continue
                m = mid_at(f.coin, f.ts_ms + h)
                if m is None and now_ms < f.ts_ms + h + give_up_after_ms:
                    continue  # mid not recorded yet; try again on a later call
                f.markouts_bps[h] = None if m is None else sign * (m - f.price) / f.price * 10_000.0
                f.resolved[h] = True
                n += 1
                if h == 30_000 and f.markouts_bps[h] is not None:
                    v = float(f.markouts_bps[h] or 0.0)
                    self._ewma_30s = v if self._ewma_30s is None else (1 - self._alpha) * self._ewma_30s + self._alpha * v
        return n

    @property
    def fills(self) -> list[FillRecord]:
        return list(self._fills)

    @property
    def ewma_30s_bps(self) -> float | None:
        return self._ewma_30s

    def toxicity_threshold(self, base_vpin_threshold: float, floor: float = 0.30) -> float:
        """Tighten the VPIN threshold in proportion to adverse 30-second markouts (-10 bps halves it at most)."""
        if self._ewma_30s is None or self._ewma_30s >= 0.0:
            return base_vpin_threshold
        return max(floor, base_vpin_threshold * (1.0 + max(-0.5, self._ewma_30s / 20.0)))

    def summary(self, by: Literal["all", "sleeve_id", "size_bucket", "lead_state"] = "all") -> dict[str, dict[int, dict[str, float]]]:
        groups: dict[str, list[FillRecord]] = {}
        for f in self._fills:
            key = "all" if by == "all" else str(getattr(f, by))
            groups.setdefault(key, []).append(f)
        out: dict[str, dict[int, dict[str, float]]] = {}
        for key, fs in groups.items():
            out[key] = {}
            for h in self.horizons:
                vals = sorted(float(v) for f in fs if (v := f.markouts_bps.get(h)) is not None)
                if not vals:
                    continue
                q = statistics.quantiles(vals, n=10) if len(vals) >= 2 else [vals[0]] * 9
                out[key][h] = {"n": float(len(vals)), "median": float(statistics.median(vals)), "p10": float(q[0]), "p90": float(q[-1]), "mean": float(statistics.fmean(vals))}
        return out


def lookup_from_history(history: MidHistory) -> Callable[[str, int], float | None]:
    return history.mid_at


def mids_from_mapping(series: Mapping[str, Sequence[tuple[int, float]]]) -> MidHistory:
    h = MidHistory(retention_ms=10**12)
    for coin, pts in series.items():
        for t, m in pts:
            h.add(coin, t, m)
    return h


__all__ = [
    "FillRecord",
    "MARKOUT_HORIZONS_MS",
    "MarkoutTracker",
    "MidHistory",
    "lookup_from_history",
    "minutes_to_hourly_funding",
    "mids_from_mapping",
    "size_bucket",
]
