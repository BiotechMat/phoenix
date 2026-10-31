"""Walk-forward evaluation of the jump-model regime state (REINFORCEMENT_PLAN task P3.3 acceptance).

Hourly features (causal): average realised volatility of BTC/ETH/SOL over 24h, average 3-day mean funding,
average 24h mean premium (basis), BTC DVOL, and the 30-day average pairwise correlation. Open-interest
change is omitted: no history is available.

Walk-forward: an expanding training window (first 12 months), then quarterly test blocks. In each
training window the number of states (3 or 4) and the jump penalty are chosen by the validation Sharpe
of the state-conditioned sleeve on the last 3 months of the window, the model is refitted on the full
window, and test states come from the causal forward filter. A sleeve is conditioned by trading only in
states whose training-window mean return is positive. The acceptance asks for a mean dwell above 24
hours and an out-of-fold improvement for at least one sleeve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
import polars as pl
from pydantic import BaseModel, ConfigDict

from research.trial_registry import TrialRecord, TrialRegistry
from risk.regime_jump_model import JumpModel, fit_jump_model, mean_dwell

HOURS_PER_YEAR: Final[float] = 8760.0
FEATURE_NAMES: Final[tuple[str, ...]] = ("rv24", "funding_3d", "basis_24h", "dvol", "corr_30d")


class FoldResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_start_ms: int
    test_end_ms: int
    n_states: int
    jump_penalty: float
    allowed_states: list[int]
    test_dwell_hours: float
    unconditioned: float
    conditioned: float


class RegimeEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    sleeve: str
    folds: list[FoldResult]
    oof_unconditioned: float
    oof_conditioned: float
    oof_sharpe_unconditioned: float
    oof_sharpe_conditioned: float
    mean_dwell_hours: float
    improves: bool
    persistent: bool
    trial_id: str | None = None


def _rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    c = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(x), x, 0.0))))
    cnt = np.concatenate(([0.0], np.cumsum(np.isfinite(x).astype(np.float64))))
    out = np.full(x.shape[0], np.nan)
    for_idx = np.arange(x.shape[0])
    lo = np.maximum(0, for_idx - window + 1)
    n = cnt[for_idx + 1] - cnt[lo]
    out = np.where(n > 0, (c[for_idx + 1] - c[lo]) / np.maximum(n, 1.0), np.nan)
    return out


def _rolling_std(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    m = _rolling_mean(x, window)
    m2 = _rolling_mean(x * x, window)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def build_features(
    ts_ms: NDArray[np.int64],
    closes: Mapping[str, NDArray[np.float64]],
    funding_1h: Mapping[str, NDArray[np.float64]],
    premium: Mapping[str, NDArray[np.float64]],
    dvol: NDArray[np.float64],
) -> NDArray[np.float64]:
    """(T, 5) causal feature matrix aligned to ``ts_ms``."""
    assets = list(closes)
    rets = {a: np.concatenate(([0.0], np.diff(np.log(closes[a])))) for a in assets}
    rv = np.mean(np.vstack([_rolling_std(rets[a], 24) * math.sqrt(24.0) for a in assets]), axis=0)
    f3 = np.mean(np.vstack([_rolling_mean(funding_1h[a], 72) for a in assets]), axis=0)
    basis = np.mean(np.vstack([_rolling_mean(premium[a], 24) for a in assets]), axis=0)
    # 30-day average pairwise correlation of hourly returns
    from sleeves.sleeve_2_trend import rolling_average_pairwise_correlation

    corr = rolling_average_pairwise_correlation(np.column_stack([np.log(closes[a]) for a in assets]), 720)
    feats = np.column_stack([rv, f3, basis, dvol, corr])
    return feats


def _cond_sharpe(r: NDArray[np.float64]) -> float:
    sd = float(np.std(r, ddof=1)) if r.size > 2 else 0.0
    return float(np.mean(r) / sd * math.sqrt(HOURS_PER_YEAR)) if sd > 1e-15 else 0.0


def _allowed_states(states: NDArray[np.int64], sleeve_ret: NDArray[np.float64], k: int) -> list[int]:
    return [s for s in range(k) if np.any(states == s) and float(np.mean(sleeve_ret[states == s])) > 0.0]


def walk_forward(
    ts_ms: NDArray[np.int64],
    features: NDArray[np.float64],
    sleeve_returns: NDArray[np.float64],
    *,
    sleeve: str,
    initial_train_hours: int = 365 * 24,
    test_hours: int = 91 * 24,
    validation_hours: int = 91 * 24,
    state_grid: Sequence[int] = (3, 4),
    penalty_grid: Sequence[float] = (10.0, 50.0, 200.0),
    registry: TrialRegistry | None = None,
) -> RegimeEvalReport:
    ok = np.all(np.isfinite(features), axis=1) & np.isfinite(sleeve_returns)
    first = int(np.argmax(ok))
    ts, x, r = ts_ms[first:], features[first:], np.where(np.isfinite(sleeve_returns[first:]), sleeve_returns[first:], 0.0)
    x = np.where(np.isfinite(x), x, 0.0)
    folds: list[FoldResult] = []
    start = initial_train_hours
    all_uncond: list[NDArray[np.float64]] = []
    all_cond: list[NDArray[np.float64]] = []
    dwell_weighted: list[tuple[float, int]] = []
    while start + 24 < x.shape[0]:
        end = min(start + test_hours, x.shape[0])
        train_x, train_r = x[:start], r[:start]
        fit_x, val_x = train_x[: start - validation_hours], train_x[start - validation_hours :]
        fit_r, val_r = train_r[: start - validation_hours], train_r[start - validation_hours :]
        best: tuple[float, int, float] | None = None
        for k in state_grid:
            for lam in penalty_grid:
                m = fit_jump_model(fit_x, k, lam, n_init=4)
                s_fit = m.filter_states(fit_x)
                allowed = _allowed_states(s_fit, fit_r, k)
                s_val = m.filter_states(np.vstack([fit_x, val_x]))[-val_x.shape[0] :]
                sh = _cond_sharpe(val_r * np.isin(s_val, allowed))
                if best is None or sh > best[0]:
                    best = (sh, k, lam)
        assert best is not None
        _, k, lam = best
        model: JumpModel = fit_jump_model(train_x, k, lam, n_init=6)
        s_train = model.filter_states(train_x)
        allowed = _allowed_states(s_train, train_r, k)
        s_all = model.filter_states(x[:end])
        s_test = s_all[start:end]
        test_r = r[start:end]
        cond = test_r * np.isin(s_test, allowed)
        all_uncond.append(test_r)
        all_cond.append(cond)
        dwell_weighted.append((mean_dwell(s_test), s_test.size))
        folds.append(
            FoldResult(
                test_start_ms=int(ts[start]),
                test_end_ms=int(ts[end - 1]),
                n_states=k,
                jump_penalty=lam,
                allowed_states=allowed,
                test_dwell_hours=mean_dwell(s_test),
                unconditioned=float(np.sum(test_r)),
                conditioned=float(np.sum(cond)),
            )
        )
        start = end
    u = np.concatenate(all_uncond) if all_uncond else np.zeros(0)
    c = np.concatenate(all_cond) if all_cond else np.zeros(0)
    total_n = sum(n for _, n in dwell_weighted)
    dwell = sum(d * n for d, n in dwell_weighted) / total_n if total_n else 0.0
    report = RegimeEvalReport(
        sleeve=sleeve,
        folds=folds,
        oof_unconditioned=float(np.sum(u)),
        oof_conditioned=float(np.sum(c)),
        oof_sharpe_unconditioned=_cond_sharpe(u),
        oof_sharpe_conditioned=_cond_sharpe(c),
        mean_dwell_hours=dwell,
        improves=float(np.sum(c)) > float(np.sum(u)) and _cond_sharpe(c) > _cond_sharpe(u),
        persistent=dwell > 24.0,
    )
    if registry is not None:
        rec = registry.append(
            TrialRecord(
                trial_id=f"trial_regime_{sleeve}_{len(folds)}f_{int(ts[0])}",
                recorded_at_iso=datetime.now(timezone.utc).isoformat(),
                sleeve_family="regime",
                model_id=f"jump_model_conditioning_{sleeve}",
                features=FEATURE_NAMES,
                parameters={"state_grid": list(state_grid), "penalty_grid": list(penalty_grid), "test_hours": test_hours},
                window_start_iso=datetime.fromtimestamp(int(ts[0]) / 1000, tz=timezone.utc).isoformat(),
                window_end_iso=datetime.fromtimestamp(int(ts[-1]) / 1000, tz=timezone.utc).isoformat(),
                n_bars=int(x.shape[0]),
                total_trials_used=registry.next_total_trials("regime"),
                metrics={
                    "oof_unconditioned": report.oof_unconditioned,
                    "oof_conditioned": report.oof_conditioned,
                    "oof_sharpe_unconditioned": report.oof_sharpe_unconditioned,
                    "oof_sharpe_conditioned": report.oof_sharpe_conditioned,
                    "mean_dwell_hours": dwell,
                },
                passed_all_gates=report.improves and report.persistent,
            )
        )
        report = report.model_copy(update={"trial_id": rec.trial_id})
    return report


def load_aligned(data_dir: Path, carry_dir: Path, dvol_path: Path, coins: Sequence[str] = ("BTC", "ETH", "SOL")) -> tuple[NDArray[np.int64], dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]], NDArray[np.float64]]:
    """Hourly closes and funding (from the 1h Parquets), premium (from settlements) and DVOL, on common hours."""
    from research.trend_eval import load_hourly_closes

    ts, closes, funding = load_hourly_closes(data_dir, coins)
    hour = (ts // 3_600_000).astype(np.int64)
    premium: dict[str, NDArray[np.float64]] = {}
    for c in coins:
        st = pl.read_parquet(carry_dir / f"{c.lower()}_funding.parquet").sort("t_ms")
        h = (st.get_column("t_ms").to_numpy() // 3_600_000).astype(np.int64)
        p = st.get_column("premium").to_numpy().astype(np.float64)
        idx = np.searchsorted(h, hour, side="right") - 1  # last settlement at or before the hour (causal)
        premium[c] = np.where(idx >= 0, p[np.maximum(idx, 0)], np.nan)
    dv = pl.read_parquet(dvol_path).sort("t_ms")
    dh = (dv.get_column("t_ms").to_numpy() // 3_600_000).astype(np.int64)
    dvals = dv.get_column("dvol").to_numpy().astype(np.float64)
    idx = np.searchsorted(dh, hour, side="right") - 1
    dvol = np.where(idx >= 0, dvals[np.maximum(idx, 0)], np.nan)
    return ts, closes, funding, premium, dvol


__all__ = ["FEATURE_NAMES", "FoldResult", "RegimeEvalReport", "build_features", "load_aligned", "walk_forward"]
