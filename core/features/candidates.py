"""Candidate meta-labeller features for REINFORCEMENT_PLAN task P2.3.

Every function is causal: the value at bar ``t`` uses only bars at or before ``t``. Values that cannot be
computed yet (warm-up, missing inputs) are NaN; the evaluation harness drops them from IC and fills the
model column with zero. None of these is in the live feature set unless it passes P1.8.

- ``basket_residual_return``: the asset's 1-hour log return minus its rolling beta times the equal-weight
  basket return (BTC, ETH, SOL), beta from the trailing 30 days.
- ``momentum``: 1-week log return.
- ``hour_of_day_sin`` / ``hour_of_day_cos`` and ``nyse_open_flag`` (bar opens in the first hour after the
  09:30 New York open on a weekday; exchange holidays are not excluded).
- ``mark_index_divergence`` and its 1-hour change.
- ``taker_imbalance_variance``: 48-bar variance of (buy - sell) / (buy + sell) taker dollar volume.
- ``amihud_illiquidity``: log of the trailing mean of |return| per dollar traded.
- ``funding_zscore`` over 30 days and ``funding_clamp_flag`` (funding equal to the venue's baseline
  interest rate, i.e. the premium sat inside the clamp band).
- ``first_minute_taker_imbalance``: signed taker volume in the first 60 seconds of the bar from the
  first one-minute kline (the boundary imbalance of P1.1, measured on archive klines).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from core.contracts import DollarBar

BARS_PER_HOUR = 4
BARS_PER_DAY = 96
HL_BASELINE_FUNDING_1H = 0.0000125  # 0.01% per 8 hours, paid hourly
_NY = ZoneInfo("America/New_York")


def _log_return(closes: NDArray[np.float64], lag: int) -> NDArray[np.float64]:
    out = np.full(closes.shape[0], np.nan)
    if closes.shape[0] > lag:
        out[lag:] = np.log(closes[lag:] / closes[:-lag])
    return out


def _rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Trailing mean over ``window`` values ending at t (NaN until full; NaN inputs propagate)."""
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] < window:
        return out
    c = np.cumsum(np.concatenate(([0.0], x)))
    out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def momentum(closes: NDArray[np.float64], bars: int = 7 * BARS_PER_DAY) -> NDArray[np.float64]:
    return _log_return(np.asarray(closes, dtype=np.float64), bars)


def basket_residual_return(
    closes_by_coin: Mapping[str, NDArray[np.float64]],
    coin: str,
    *,
    horizon_bars: int = BARS_PER_HOUR,
    beta_window: int = 30 * BARS_PER_DAY,
) -> NDArray[np.float64]:
    """Residual of the coin's ``horizon_bars`` log return on the equal-weight basket, beta from the trailing window.

    The closes must be aligned on the same bar timestamps.
    """
    rets = {c: _log_return(np.asarray(v, dtype=np.float64), horizon_bars) for c, v in closes_by_coin.items()}
    basket = np.mean(np.vstack(list(rets.values())), axis=0)
    r = rets[coin]
    one = _log_return(np.asarray(closes_by_coin[coin], dtype=np.float64), 1)
    one_b = np.mean(np.vstack([_log_return(np.asarray(v, dtype=np.float64), 1) for v in closes_by_coin.values()]), axis=0)
    # beta from non-overlapping 1-bar returns over the trailing window (known at t)
    xy = _rolling_mean(np.nan_to_num(one * one_b), beta_window) - _rolling_mean(np.nan_to_num(one), beta_window) * _rolling_mean(np.nan_to_num(one_b), beta_window)
    xx = _rolling_mean(np.nan_to_num(one_b * one_b), beta_window) - _rolling_mean(np.nan_to_num(one_b), beta_window) ** 2
    beta = np.where(xx > 1e-18, xy / np.where(xx > 1e-18, xx, 1.0), np.nan)
    return np.asarray(r - beta * basket, dtype=np.float64)


def hour_of_day(ts_close_ns: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(sin, cos) of the UTC hour at which the bar opens."""
    hours = ((np.asarray(ts_close_ns, dtype=np.int64) - 1) // 1_000_000_000 % 86_400) / 3_600.0
    hours = np.floor(hours * BARS_PER_HOUR) / BARS_PER_HOUR
    ang = 2.0 * np.pi * hours / 24.0
    return np.sin(ang), np.cos(ang)


def nyse_open_flag(ts_open_ns: NDArray[np.int64]) -> NDArray[np.float64]:
    """1 when the bar opens in [09:30, 10:30) New York time on a weekday (daylight saving handled)."""
    out = np.zeros(len(ts_open_ns), dtype=np.float64)
    for i, t in enumerate(np.asarray(ts_open_ns, dtype=np.int64)):
        ny = datetime.fromtimestamp(int(t) / 1e9, tz=timezone.utc).astimezone(_NY)
        minutes = ny.hour * 60 + ny.minute
        if ny.weekday() < 5 and 570 <= minutes < 630:
            out[i] = 1.0
    return out


def mark_index_divergence(mark: NDArray[np.float64], index: NDArray[np.float64]) -> NDArray[np.float64]:
    m = np.asarray(mark, dtype=np.float64)
    x = np.asarray(index, dtype=np.float64)
    return np.where(x > 0.0, (m - x) / np.where(x > 0.0, x, 1.0), np.nan)


def change(x: NDArray[np.float64], bars: int = BARS_PER_HOUR) -> NDArray[np.float64]:
    out = np.full(x.shape[0], np.nan)
    out[bars:] = x[bars:] - x[:-bars]
    return out


def taker_imbalance_variance(buy_usd: NDArray[np.float64], sell_usd: NDArray[np.float64], window: int = 48) -> NDArray[np.float64]:
    b = np.asarray(buy_usd, dtype=np.float64)
    s = np.asarray(sell_usd, dtype=np.float64)
    tot = b + s
    imb = np.where(tot > 0.0, (b - s) / np.where(tot > 0.0, tot, 1.0), 0.0)
    m = _rolling_mean(imb, window)
    m2 = _rolling_mean(imb * imb, window)
    return np.maximum(m2 - m * m, 0.0)


def amihud_illiquidity(closes: NDArray[np.float64], dollar_volume: NDArray[np.float64], window: int = BARS_PER_DAY) -> NDArray[np.float64]:
    r = np.abs(_log_return(np.asarray(closes, dtype=np.float64), 1))
    dv = np.asarray(dollar_volume, dtype=np.float64)
    ratio = np.where((dv > 0.0) & np.isfinite(r), r / np.where(dv > 0.0, dv, 1.0), 0.0)
    m = _rolling_mean(ratio, window)
    return np.where(m > 0.0, np.log(np.where(m > 0.0, m, 1.0) * 1e9), np.nan)


def funding_zscore(funding_1h: NDArray[np.float64], window: int = 30 * BARS_PER_DAY) -> NDArray[np.float64]:
    f = np.asarray(funding_1h, dtype=np.float64)
    m = _rolling_mean(f, window)
    sd = np.sqrt(np.maximum(_rolling_mean(f * f, window) - m * m, 0.0))
    return np.where(sd > 1e-12, (f - m) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def funding_clamp_flag(funding_1h: NDArray[np.float64], baseline: float = HL_BASELINE_FUNDING_1H) -> NDArray[np.float64]:
    return (np.abs(np.asarray(funding_1h, dtype=np.float64) - baseline) < 1e-9).astype(np.float64)


def first_minute_taker_imbalance(
    ts_open_ns: NDArray[np.int64], minute_open_ms: NDArray[np.int64], minute_volume: NDArray[np.float64], minute_taker_buy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """(2 x taker buy - volume) / volume of the one-minute kline that opens with each bar (NaN when absent).

    The value is known one minute after the bar opens, so it is causal for any decision taken at the bar close.
    """
    out = np.full(len(ts_open_ns), np.nan)
    pos = {int(t): i for i, t in enumerate(np.asarray(minute_open_ms, dtype=np.int64))}
    v = np.asarray(minute_volume, dtype=np.float64)
    tb = np.asarray(minute_taker_buy, dtype=np.float64)
    for i, t in enumerate(np.asarray(ts_open_ns, dtype=np.int64)):
        j = pos.get(int(t) // 1_000_000)
        if j is not None and v[j] > 0.0:
            out[i] = (2.0 * tb[j] - v[j]) / v[j]
    return out


def candidate_features(
    bars: Sequence[DollarBar],
    *,
    companions: Mapping[str, Sequence[DollarBar]] | None = None,
    coin: str = "BTC",
    taker_flow_real: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """All bar-computable candidates for ``bars``; ``companions`` (other coins, same timestamps) enable the basket residual."""
    closes = np.array([b.close for b in bars], dtype=np.float64)
    ts_close = np.array([b.ts_close_ns for b in bars], dtype=np.int64)
    ts_open = np.array([b.ts_open_ns for b in bars], dtype=np.int64)
    funding = np.array([b.funding_rate_1h for b in bars], dtype=np.float64)
    div = mark_index_divergence(np.array([b.mark_price for b in bars]), np.array([b.spot_price for b in bars]))
    sin_h, cos_h = hour_of_day(ts_close)
    out: dict[str, NDArray[np.float64]] = {
        "momentum_1w": momentum(closes),
        "hour_of_day_sin": sin_h,
        "hour_of_day_cos": cos_h,
        "nyse_open_flag": nyse_open_flag(ts_open),
        "mark_index_divergence": div,
        "mark_index_divergence_chg_1h": change(div),
        "amihud_illiquidity": amihud_illiquidity(closes, np.array([b.dollar_volume for b in bars], dtype=np.float64)),
        "funding_zscore_30d": funding_zscore(funding),
        "funding_clamp_flag": funding_clamp_flag(funding),
    }
    if taker_flow_real:
        out["taker_imbalance_var_48"] = taker_imbalance_variance(
            np.array([b.buy_dollar_volume for b in bars]), np.array([b.sell_dollar_volume for b in bars])
        )
    if companions:
        idx = {int(t): i for i, t in enumerate(ts_close)}
        aligned: dict[str, NDArray[np.float64]] = {coin: closes}
        for c, cb in companions.items():
            arr = np.full(len(bars), np.nan)
            for b in cb:
                j = idx.get(int(b.ts_close_ns))
                if j is not None:
                    arr[j] = b.close
            aligned[c] = arr
        if all(np.isfinite(v).mean() > 0.99 for v in aligned.values()):
            filled = {c: _ffill(v) for c, v in aligned.items()}
            out["basket_residual_return_1h"] = basket_residual_return(filled, coin)
    return out


BAR_ONLY_FEATURES: tuple[str, ...] = (
    "momentum_1w",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "nyse_open_flag",
    "mark_index_divergence",
    "mark_index_divergence_chg_1h",
    "amihud_illiquidity",
    "funding_zscore_30d",
    "funding_clamp_flag",
)


def bar_feature_matrix(bars: Sequence[DollarBar], names: Sequence[str]) -> NDArray[np.float64]:
    """Columns for ``names`` computed from the primary coin's own bars only, NaN replaced by 0 as in evaluation.

    A bar whose mark and spot both equal its close carries no venue mark/index (seeded candles), so its
    divergence is treated as unknown rather than zero.
    """
    unknown = [n for n in names if n not in BAR_ONLY_FEATURES]
    if unknown:
        raise ValueError(f"not computable from bars alone: {unknown}")
    if not names:
        return np.zeros((len(bars), 0), dtype=np.float64)
    marks = np.array([b.mark_price for b in bars], dtype=np.float64)
    spots = np.array([b.spot_price for b in bars], dtype=np.float64)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    no_venue = (marks == closes) & (spots == closes)
    marks = np.where(no_venue, np.nan, marks)
    feats = candidate_features(bars)
    div = mark_index_divergence(marks, spots)  # NaN where the bar has no venue mark
    feats["mark_index_divergence"] = div
    feats["mark_index_divergence_chg_1h"] = change(div)
    cols = [np.where(np.isfinite(feats[n]), feats[n], 0.0) for n in names]
    return np.column_stack(cols).astype(np.float64)


def _ffill(x: NDArray[np.float64]) -> NDArray[np.float64]:
    out = x.copy()
    last = np.nan
    for i in range(out.shape[0]):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out


def label_uniqueness_weights(hit_offsets: Sequence[int] | NDArray[np.int64], n_bars: int | None = None) -> NDArray[np.float64]:
    """Average uniqueness of each label over its span [i, i + hit_offset] (Lopez de Prado), normalised to mean 1."""
    h = np.maximum(np.asarray(hit_offsets, dtype=np.int64), 0)
    n = int(n_bars if n_bars is not None else h.shape[0])
    conc = np.zeros(n + int(h.max(initial=0)) + 1, dtype=np.float64)
    for i, k in enumerate(h):
        conc[i : i + k + 1] += 1.0
    w = np.array([np.mean(1.0 / conc[i : i + k + 1]) for i, k in enumerate(h)], dtype=np.float64)
    return w / w.mean() if w.size and w.mean() > 0 else w


def recency_weights(n: int, half_life_bars: float) -> NDArray[np.float64]:
    """Exponential weights with the newest row at 1 and the given half-life, normalised to mean 1."""
    age = np.arange(n - 1, -1, -1, dtype=np.float64)
    w = 0.5 ** (age / half_life_bars)
    return w / w.mean() if n else w


def trend_scanning_labels(closes: Sequence[float] | NDArray[np.float64], spans: Sequence[int] = (8, 16, 32, 48)) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Trend-scanning label (Lopez de Prado): sign of the forward OLS slope t-value at the span maximising |t|.

    Forward-looking by construction: a target, never a feature. Rows without the longest span ahead get t = 0.
    """
    y = np.log(np.asarray(closes, dtype=np.float64))
    n = y.shape[0]
    best_t = np.zeros(n)
    for i in range(n):
        top = 0.0
        for L in spans:
            if i + L > n:
                break
            seg = y[i : i + L]
            x = np.arange(L, dtype=np.float64)
            xm = x - x.mean()
            beta = float(np.dot(xm, seg - seg.mean()) / np.dot(xm, xm))
            resid = seg - seg.mean() - beta * xm
            se = float(np.sqrt(np.dot(resid, resid) / max(L - 2, 1) / np.dot(xm, xm)))
            t = beta / se if se > 0 else 0.0
            if abs(t) > abs(top):
                top = t
        best_t[i] = top
    return np.sign(best_t).astype(np.int64), best_t


__all__ = [
    "BAR_ONLY_FEATURES",
    "amihud_illiquidity",
    "bar_feature_matrix",
    "basket_residual_return",
    "candidate_features",
    "change",
    "first_minute_taker_imbalance",
    "funding_clamp_flag",
    "funding_zscore",
    "hour_of_day",
    "label_uniqueness_weights",
    "mark_index_divergence",
    "momentum",
    "nyse_open_flag",
    "recency_weights",
    "taker_imbalance_variance",
    "trend_scanning_labels",
]
