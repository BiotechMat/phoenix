"""Deribit option surface metrics from per-instrument marks (REINFORCEMENT_PLAN task P2.4).

Replaces the earlier approximations (average call IV minus average put IV across all strikes; "near"
and "far" taken from response order) with:

- Black-Scholes delta and gamma from each option's mark IV, strike, time to expiry (from the
  instrument's expiration timestamp) and the expiry's underlying (forward) price, with r = 0 because
  Deribit quotes against the expiry's future.
- 25-delta risk reversal at the 30-day and 90-day expiries: IV of the call nearest +0.25 delta minus
  IV of the put nearest -0.25 delta, in vol points.
- Term structure from ATM IV (call nearest 0.50 delta) per expiry: slope between the ~7-day and
  ~90-day expiries as a ratio, plus the per-expiry ATM curve.
- Dealer gamma exposure in USD per 1% move: sum over options of gamma * OI * S^2 * 0.01, calls
  positive and puts negative (the conventional dealer-long-calls, dealer-short-puts sign).

Inputs are the public ``get_book_summary_by_currency`` and ``get_instruments`` payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

MS_PER_YEAR: Final[float] = 365.0 * 86_400_000.0
SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def black_scholes_delta_gamma(forward: float, strike: float, years: float, vol: float, is_call: bool) -> tuple[float, float]:
    """Forward delta and gamma (per unit of underlying) with zero rates."""
    if forward <= 0.0 or strike <= 0.0 or years <= 0.0 or vol <= 0.0:
        return (1.0 if (is_call and forward > strike) else (-1.0 if (not is_call and forward < strike) else 0.0)), 0.0
    sd = vol * math.sqrt(years)
    d1 = (math.log(forward / strike) + 0.5 * sd * sd) / sd
    delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
    gamma = _norm_pdf(d1) / (forward * sd)
    return delta, gamma


class OptionPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    instrument_name: str
    expiry_ms: int
    strike: float
    is_call: bool
    mark_iv: float = Field(description="Vol points, e.g. 38.4")
    forward: float
    open_interest: float
    years: float
    delta: float
    gamma: float


class SurfaceMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    ts_ms: int
    n_options: int
    rr25_30d: float | None = Field(default=None, description="25-delta risk reversal at ~30 days, vol points")
    rr25_90d: float | None = None
    expiry_30d_ms: int | None = None
    expiry_90d_ms: int | None = None
    atm_iv_by_expiry: dict[int, float] = Field(default_factory=dict)
    atm_iv_near: float | None = None
    atm_iv_far: float | None = None
    term_structure_slope: float | None = Field(default=None, description="(ATM far - ATM near) / ATM near")
    net_gex_usd_per_1pct: float = 0.0
    call_gex_usd_per_1pct: float = 0.0
    put_gex_usd_per_1pct: float = 0.0


_MONTHS: Final[dict[str, int]] = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def instrument_specs_from_names(book_summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive ``get_instruments``-style specs from Deribit option names such as ``BTC-27NOV26-77000-C``.

    Deribit options expire at 08:00 UTC on the date in the name.
    """
    import datetime as _dt

    out: list[dict[str, Any]] = []
    for row in book_summary:
        name = str(row.get("instrument_name", ""))
        parts = name.split("-")
        if len(parts) != 4 or parts[3] not in ("C", "P"):
            continue
        date = parts[1]
        try:
            day = int(date[:-5])
            month = _MONTHS[date[-5:-2]]
            year = 2000 + int(date[-2:])
            strike = float(parts[2].replace("d", "."))
        except (KeyError, ValueError):
            continue
        expiry = _dt.datetime(year, month, day, 8, 0, tzinfo=_dt.timezone.utc)
        out.append(
            {
                "instrument_name": name,
                "expiration_timestamp": int(expiry.timestamp() * 1000),
                "strike": strike,
                "option_type": "call" if parts[3] == "C" else "put",
            }
        )
    return out


def build_option_points(
    book_summary: Sequence[Mapping[str, Any]],
    instruments: Sequence[Mapping[str, Any]],
    now_ms: int,
) -> list[OptionPoint]:
    """Join book-summary marks with instrument specs and compute greeks; drops expired or unpriced rows."""
    spec = {str(i["instrument_name"]): i for i in instruments if "instrument_name" in i}
    out: list[OptionPoint] = []
    for row in book_summary:
        name = str(row.get("instrument_name", ""))
        s = spec.get(name)
        if s is None:
            continue
        try:
            expiry = int(s["expiration_timestamp"])
            strike = float(s["strike"])
            is_call = str(s["option_type"]).lower() == "call"
            iv = float(row.get("mark_iv") or 0.0)
            fwd = float(row.get("underlying_price") or 0.0)
            oi = float(row.get("open_interest") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        years = (expiry - now_ms) / MS_PER_YEAR
        if years <= 0.0 or iv <= 0.0 or fwd <= 0.0:
            continue
        delta, gamma = black_scholes_delta_gamma(fwd, strike, years, iv / 100.0, is_call)
        out.append(
            OptionPoint(
                instrument_name=name, expiry_ms=expiry, strike=strike, is_call=is_call, mark_iv=iv,
                forward=fwd, open_interest=oi, years=years, delta=delta, gamma=gamma,
            )
        )
    return out


def _nearest_expiry(points: Sequence[OptionPoint], now_ms: int, target_days: float, tolerance_days: float) -> int | None:
    expiries = sorted({p.expiry_ms for p in points})
    if not expiries:
        return None
    best = min(expiries, key=lambda e: abs((e - now_ms) / 86_400_000.0 - target_days))
    return best if abs((best - now_ms) / 86_400_000.0 - target_days) <= tolerance_days else None


def _iv_at_delta(points: Sequence[OptionPoint], target_delta: float, is_call: bool, max_delta_error: float = 0.10) -> float | None:
    candidates = [p for p in points if p.is_call == is_call]
    if not candidates:
        return None
    best = min(candidates, key=lambda p: abs(p.delta - target_delta))
    return best.mark_iv if abs(best.delta - target_delta) <= max_delta_error else None


def compute_surface_metrics(
    book_summary: Sequence[Mapping[str, Any]],
    instruments: Sequence[Mapping[str, Any]],
    now_ms: int,
    currency: str = "BTC",
    near_days: float = 7.0,
    far_days: float = 90.0,
) -> SurfaceMetrics:
    points = build_option_points(book_summary, instruments, now_ms)
    by_expiry: dict[int, list[OptionPoint]] = {}
    for p in points:
        by_expiry.setdefault(p.expiry_ms, []).append(p)

    atm: dict[int, float] = {}
    for expiry, pts in by_expiry.items():
        iv = _iv_at_delta(pts, 0.50, True)
        if iv is not None:
            atm[expiry] = iv

    def rr_for(days: float, tol: float) -> tuple[float | None, int | None]:
        e = _nearest_expiry(points, now_ms, days, tol)
        if e is None:
            return None, None
        call_iv = _iv_at_delta(by_expiry[e], 0.25, True)
        put_iv = _iv_at_delta(by_expiry[e], -0.25, False)
        if call_iv is None or put_iv is None:
            return None, e
        return call_iv - put_iv, e

    rr30, e30 = rr_for(30.0, 10.0)
    rr90, e90 = rr_for(90.0, 30.0)
    near_e = _nearest_expiry([p for p in points if p.expiry_ms in atm], now_ms, near_days, near_days)
    far_e = _nearest_expiry([p for p in points if p.expiry_ms in atm], now_ms, far_days, 30.0)
    atm_near = atm.get(near_e) if near_e is not None else None
    atm_far = atm.get(far_e) if far_e is not None else None
    slope = (atm_far - atm_near) / atm_near if (atm_near and atm_far is not None and atm_near > 0.0) else None

    call_gex = sum(p.gamma * p.open_interest * p.forward * p.forward * 0.01 for p in points if p.is_call)
    put_gex = sum(p.gamma * p.open_interest * p.forward * p.forward * 0.01 for p in points if not p.is_call)
    return SurfaceMetrics(
        currency=currency.upper(),
        ts_ms=now_ms,
        n_options=len(points),
        rr25_30d=rr30,
        rr25_90d=rr90,
        expiry_30d_ms=e30,
        expiry_90d_ms=e90,
        atm_iv_by_expiry=dict(sorted(atm.items())),
        atm_iv_near=atm_near,
        atm_iv_far=atm_far,
        term_structure_slope=slope,
        net_gex_usd_per_1pct=call_gex - put_gex,
        call_gex_usd_per_1pct=call_gex,
        put_gex_usd_per_1pct=put_gex,
    )


__all__ = [
    "OptionPoint",
    "SurfaceMetrics",
    "black_scholes_delta_gamma",
    "build_option_points",
    "compute_surface_metrics",
    "instrument_specs_from_names",
]
