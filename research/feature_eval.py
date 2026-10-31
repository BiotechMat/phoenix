"""Feature evaluation harness (REINFORCEMENT_PLAN task P1.8).

For a candidate feature aligned to a bar series this computes:

- IC by horizon: Spearman rank correlation with forward log returns at 15m, 1h, 4h, 12h and 24h, with
  a t-statistic on the overlap-adjusted sample size ``n / h``.
- IC half-life: the lag (in bars) at which the 1-bar-ahead IC of the lagged feature falls to half its
  lag-0 value.
- Out-of-fold calibrated log-loss delta: the meta-labeller (Stage-1 SVD, Mahalanobis gate, calibrated
  LightGBM) is fitted on each CPCV training split with and without the feature, using the same fee-
  adjusted triple-barrier labels as the live loop, and scored on the purged test split.
- Fold stability: permutation importance of the feature on every test split (log-loss increase when the
  column is shuffled) and the fraction of folds where it is positive; the sign consistency of the IC
  across test splits.

A feature passes when it improves out-of-fold log-loss on most folds with stable importance, or when its
IC is significant with a consistent sign across folds. Every evaluation is appended to the trial
registry under the ``feature_eval`` family.

The permutation shuffle uses a seeded generator. This is research code, never a live path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
from typing import Final
import uuid

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from core.contracts import DollarBar, FeeSchedule, TripleBarrierConfig
from core.features import calibrate_optimal_d, compute_positioning_snapshots
from core.models import FeeAdjustedTripleBarrierLabeler, LightGBMMetaLabelWorker, extract_continuous_and_binary_from_snapshots
from research.trial_registry import TrialRecord, TrialRegistry
from research.validation.cpcv_gatekeeper import CombinatorialPurgedCV

logger = logging.getLogger("phoenix.feature_eval")

DEFAULT_HORIZONS_BARS: Final[dict[str, int]] = {"15m": 1, "1h": 4, "4h": 16, "12h": 48, "24h": 96}
BASE_FEATURE_NAMES: Final[tuple[str, str, str]] = ("ffd_close_zscore", "ffd_oi_zscore", "oi_funding_basis_zscore")
LOGLOSS_EPS: Final[float] = 1e-6


class FeatureEvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizons_bars: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_HORIZONS_BARS))
    n_groups: int = Field(default=6, ge=3)
    k_test_groups: int = Field(default=2, ge=1)
    embargo_pct: float = Field(default=0.01, ge=0.0)
    half_life_max_lag: int = Field(default=48, ge=2)
    min_ic_tstat: float = 3.0
    min_ic_sign_consistency: float = 0.7
    min_folds_improved: float = 0.7
    min_importance_stability: float = 0.7
    max_logloss_delta_tstat: float = -2.0
    permutation_seed: int = 20261031
    fee_schedule: FeeSchedule = Field(default_factory=FeeSchedule.hyperliquid)


class FeatureEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    n_bars: int
    ic_by_horizon: dict[str, float | None]
    ic_tstat_by_horizon: dict[str, float | None]
    best_horizon: str | None
    ic_half_life_bars: float | None
    oof_logloss_baseline: float | None
    oof_logloss_candidate: float | None
    logloss_delta: float | None = Field(description="Candidate minus baseline; negative is better")
    folds_evaluated: int
    folds_improved_frac: float
    logloss_delta_tstat: float | None = Field(default=None, description="Mean fold delta over its standard error")
    importance_by_fold: list[float]
    importance_mean: float | None
    importance_stability: float
    ic_sign_consistency: float
    passed: bool
    reasons: list[str]
    trial_id: str | None = None


# ------------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------------


def forward_log_returns(closes: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    """``log(close[t+h] / close[t])``; NaN for the last ``h`` bars."""
    c = np.asarray(closes, dtype=np.float64)
    out = np.full(c.shape[0], np.nan, dtype=np.float64)
    if horizon < c.shape[0]:
        out[:-horizon] = np.log(c[horizon:] / c[:-horizon])
    return out


def spearman_ic(feature: NDArray[np.float64], target: NDArray[np.float64]) -> float | None:
    f = np.asarray(feature, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(t)
    if int(ok.sum()) < 10 or float(np.std(f[ok])) <= 1e-15 or float(np.std(t[ok])) <= 1e-15:
        return None
    rho = spearmanr(f[ok], t[ok]).statistic
    val = float(rho)
    return val if math.isfinite(val) else None


def ic_tstat(ic: float | None, n_obs: int, horizon: int) -> float | None:
    """t-statistic of an IC on the overlap-adjusted effective sample ``n / h``."""
    if ic is None:
        return None
    n_eff = n_obs / max(1, horizon)
    if n_eff <= 3 or abs(ic) >= 1.0:
        return None
    return float(ic * math.sqrt(n_eff - 2.0) / math.sqrt(1.0 - ic * ic))


def ic_half_life(feature: NDArray[np.float64], closes: NDArray[np.float64], max_lag: int) -> float | None:
    """Lag at which the 1-bar IC of the lagged feature first falls to half its lag-0 magnitude (interpolated)."""
    fwd = forward_log_returns(closes, 1)
    f = np.asarray(feature, dtype=np.float64)
    ic0 = spearman_ic(f, fwd)
    if ic0 is None or abs(ic0) < 1e-6:
        return None
    half = abs(ic0) / 2.0
    prev_lag, prev_val = 0, abs(ic0)
    for lag in range(1, max_lag + 1):
        lagged = np.full_like(f, np.nan)
        lagged[lag:] = f[:-lag]
        ic = spearman_ic(lagged, fwd)
        val = abs(ic) if ic is not None else 0.0
        if val <= half:
            if prev_val == val:
                return float(lag)
            return float(prev_lag + (prev_val - half) / (prev_val - val) * (lag - prev_lag))
        prev_lag, prev_val = lag, val
    return None


def _log_loss(y: NDArray[np.int64], p: NDArray[np.float64]) -> float:
    pc = np.clip(p, LOGLOSS_EPS, 1.0 - LOGLOSS_EPS)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1.0 - pc)))


# ------------------------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------------------------


class PreparedBars(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    closes: NDArray[np.float64]
    x_base: NDArray[np.float64]
    flags: NDArray[np.int64]
    labels: NDArray[np.int64]
    n_labelled: int


def prepare_bars(bars: Sequence[DollarBar], config: FeatureEvalConfig | None = None) -> PreparedBars:
    """Positioning features and fee-adjusted triple-barrier labels exactly as the live loop computes them."""
    cfg = config or FeatureEvalConfig()
    barrier = TripleBarrierConfig(fee_schedule=cfg.fee_schedule)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    ffd = calibrate_optimal_d(np.log(closes), split_id="feature_eval")
    snaps = compute_positioning_snapshots(bars, close_ffd_config=ffd)
    x_base, flags, dirs = extract_continuous_and_binary_from_snapshots(snaps)
    labels = np.array([lbl.meta_label for lbl in FeeAdjustedTripleBarrierLabeler(barrier).label_bars(bars, primary_directions=dirs)], dtype=np.int64)
    n_lab = max(0, len(bars) - barrier.vertical_barrier_bars)
    return PreparedBars(closes=closes, x_base=x_base, flags=flags, labels=labels, n_labelled=n_lab)


# ------------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------------


def _fit_predict(x_train: NDArray[np.float64], y_train: NDArray[np.int64], f_train: NDArray[np.int64], x_test: NDArray[np.float64], f_test: NDArray[np.int64], fee: FeeSchedule) -> NDArray[np.float64]:
    worker = LightGBMMetaLabelWorker(di_threshold=3.5, n_jobs=1, fee_schedule=fee)
    worker.fit(x_train, y_train, liquidation_exhaustion_flag_train=f_train)
    return np.asarray(worker.predict_proba(x_test, liquidation_exhaustion_flag=f_test), dtype=np.float64)


def evaluate_feature(
    name: str,
    feature: Sequence[float] | NDArray[np.float64],
    bars: Sequence[DollarBar] | None = None,
    *,
    prepared: PreparedBars | None = None,
    base_columns: Sequence[int] | None = None,
    config: FeatureEvalConfig | None = None,
    registry: TrialRegistry | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> FeatureEvalResult:
    """Evaluate one candidate feature aligned to ``bars`` (or to an already ``prepared`` series)."""
    cfg = config or FeatureEvalConfig()
    if prepared is None:
        if bars is None:
            raise ValueError("bars or prepared is required")
        prepared = prepare_bars(bars, cfg)
    f = np.asarray(feature, dtype=np.float64).reshape(-1)
    n = prepared.closes.shape[0]
    if f.shape[0] != n:
        raise ValueError(f"feature length {f.shape[0]} != bars {n}")
    cols = list(base_columns) if base_columns is not None else list(range(prepared.x_base.shape[1]))
    x_base = prepared.x_base[:, cols]
    f_model = np.where(np.isfinite(f), f, 0.0)
    x_cand = np.column_stack([x_base, f_model])

    # IC by horizon
    ic: dict[str, float | None] = {}
    tst: dict[str, float | None] = {}
    for label, h in cfg.horizons_bars.items():
        fwd = forward_log_returns(prepared.closes, h)
        ic[label] = spearman_ic(f, fwd)
        tst[label] = ic_tstat(ic[label], int(np.isfinite(fwd).sum()), h)
    scored = {k: abs(v) for k, v in tst.items() if v is not None}
    best = max(scored, key=lambda k: scored[k]) if scored else None
    half_life = ic_half_life(f, prepared.closes, cfg.half_life_max_lag)

    # CPCV out-of-fold log-loss, permutation importance, IC sign consistency
    n_lab = prepared.n_labelled
    y = prepared.labels[:n_lab]
    cv = CombinatorialPurgedCV(vertical_barrier_bars=TripleBarrierConfig().vertical_barrier_bars, n_groups=cfg.n_groups, k_test_groups=cfg.k_test_groups, embargo_pct=cfg.embargo_pct)
    rng = np.random.default_rng(cfg.permutation_seed)
    base_losses: list[float] = []
    cand_losses: list[float] = []
    importances: list[float] = []
    ic_signs: list[float] = []
    best_h = cfg.horizons_bars.get(best, 1) if best else 1
    fwd_best = forward_log_returns(prepared.closes, best_h)
    for split in cv.generate_splits(n_lab):
        tr = np.asarray(split.train_indices, dtype=np.int64)
        te = np.asarray(split.test_indices, dtype=np.int64)
        if tr.size < 50 or te.size < 10 or np.unique(y[tr]).size < 2:
            continue
        p_base = _fit_predict(x_base[tr], y[tr], prepared.flags[tr], x_base[te], prepared.flags[te], cfg.fee_schedule)
        p_cand = _fit_predict(x_cand[tr], y[tr], prepared.flags[tr], x_cand[te], prepared.flags[te], cfg.fee_schedule)
        lb = _log_loss(y[te], p_base)
        lc = _log_loss(y[te], p_cand)
        base_losses.append(lb)
        cand_losses.append(lc)
        # Permutation importance: refit-free, shuffle the candidate column in the test fold.
        worker = LightGBMMetaLabelWorker(di_threshold=3.5, n_jobs=1, fee_schedule=cfg.fee_schedule)
        worker.fit(x_cand[tr], y[tr], liquidation_exhaustion_flag_train=prepared.flags[tr])
        x_perm = x_cand[te].copy()
        x_perm[:, -1] = rng.permutation(x_perm[:, -1])
        p_perm = np.asarray(worker.predict_proba(x_perm, liquidation_exhaustion_flag=prepared.flags[te]), dtype=np.float64)
        p_ref = np.asarray(worker.predict_proba(x_cand[te], liquidation_exhaustion_flag=prepared.flags[te]), dtype=np.float64)
        importances.append(_log_loss(y[te], p_perm) - _log_loss(y[te], p_ref))
        fold_ic = spearman_ic(f[te], fwd_best[te])
        if fold_ic is not None:
            ic_signs.append(math.copysign(1.0, fold_ic))

    folds = len(base_losses)
    base_mean = float(np.mean(base_losses)) if folds else None
    cand_mean = float(np.mean(cand_losses)) if folds else None
    delta = (cand_mean - base_mean) if (cand_mean is not None and base_mean is not None) else None
    improved = float(np.mean([c < b for b, c in zip(base_losses, cand_losses)])) if folds else 0.0
    stability = float(np.mean([imp > 0.0 for imp in importances])) if importances else 0.0
    diffs = np.array(cand_losses, dtype=np.float64) - np.array(base_losses, dtype=np.float64)
    delta_t: float | None = None
    if diffs.size >= 3:
        sd = float(np.std(diffs, ddof=1))
        delta_t = float(np.mean(diffs) / (sd / math.sqrt(diffs.size))) if sd > 1e-15 else None
    best_ic = ic[best] if best is not None else None
    best_t = tst[best] if best is not None else None
    overall_sign = math.copysign(1.0, best_ic) if best_ic is not None else 0.0
    sign_consistency = float(np.mean([s == overall_sign for s in ic_signs])) if ic_signs and overall_sign != 0.0 else 0.0

    reasons: list[str] = []
    logloss_ok = (
        delta is not None
        and delta < 0.0
        and delta_t is not None
        and delta_t <= cfg.max_logloss_delta_tstat
        and improved >= cfg.min_folds_improved
        and stability >= cfg.min_importance_stability
    )
    ic_ok = best_t is not None and abs(best_t) >= cfg.min_ic_tstat and sign_consistency >= cfg.min_ic_sign_consistency
    if logloss_ok:
        reasons.append(
            f"log-loss improves by {-(delta or 0.0):.5f} (t={delta_t:.2f}) on {improved:.0%} of {folds} folds; importance positive on {stability:.0%}"
        )
    if ic_ok and best_ic is not None and best_t is not None:
        reasons.append(f"IC {best_ic:+.4f} at {best} (t={best_t:.2f}) with sign consistent on {sign_consistency:.0%} of folds")
    if not (logloss_ok or ic_ok):
        reasons.append(
            f"fails: log-loss delta {delta if delta is not None else float('nan'):+.5f} (t={delta_t if delta_t is not None else float('nan'):.2f}, "
            f"improved {improved:.0%} of {folds} folds, importance stability {stability:.0%}); "
            f"best |IC t| {abs(best_t or 0.0):.2f} (sign consistency {sign_consistency:.0%})"
        )
    result = FeatureEvalResult(
        name=name,
        n_bars=n,
        ic_by_horizon=ic,
        ic_tstat_by_horizon=tst,
        best_horizon=best,
        ic_half_life_bars=half_life,
        oof_logloss_baseline=base_mean,
        oof_logloss_candidate=cand_mean,
        logloss_delta=delta,
        folds_evaluated=folds,
        folds_improved_frac=improved,
        logloss_delta_tstat=delta_t,
        importance_by_fold=importances,
        importance_mean=float(np.mean(importances)) if importances else None,
        importance_stability=stability,
        ic_sign_consistency=sign_consistency,
        passed=bool(logloss_ok or ic_ok),
        reasons=reasons,
    )
    if registry is not None:
        metrics: dict[str, float] = {"folds": float(folds), "folds_improved_frac": improved, "importance_stability": stability, "ic_sign_consistency": sign_consistency}
        if delta is not None:
            metrics["logloss_delta"] = delta
        for k, v in ic.items():
            if v is not None:
                metrics[f"ic_{k}"] = v
        if half_life is not None:
            metrics["ic_half_life_bars"] = half_life
        rec = registry.append(
            TrialRecord(
                trial_id=f"trial_{uuid.uuid4().hex[:12]}",
                recorded_at_iso=datetime.now(timezone.utc).isoformat(),
                sleeve_family="feature_eval",
                model_id=f"feature:{name}",
                features=(name,),
                parameters={"base_columns": cols, "n_groups": cfg.n_groups, "k_test_groups": cfg.k_test_groups},
                window_start_iso=window_start.isoformat() if window_start else None,
                window_end_iso=window_end.isoformat() if window_end else None,
                n_bars=n,
                seed=cfg.permutation_seed,
                total_trials_used=registry.next_total_trials("feature_eval"),
                metrics=metrics,
                passed_all_gates=result.passed,
                notes="; ".join(reasons),
            )
        )
        result = result.model_copy(update={"trial_id": rec.trial_id})
    return result


def evaluate_existing_features(
    bars: Sequence[DollarBar],
    *,
    config: FeatureEvalConfig | None = None,
    registry: TrialRegistry | None = None,
) -> dict[str, FeatureEvalResult]:
    """Leave-one-out evaluation of the three live features: baseline is the other two, candidate adds the one."""
    cfg = config or FeatureEvalConfig()
    prepared = prepare_bars(bars, cfg)
    out: dict[str, FeatureEvalResult] = {}
    for idx, name in enumerate(BASE_FEATURE_NAMES):
        others = [i for i in range(prepared.x_base.shape[1]) if i != idx]
        out[name] = evaluate_feature(name, prepared.x_base[:, idx], prepared=prepared, base_columns=others, config=cfg, registry=registry)
    return out


def summarize_results(results: Mapping[str, FeatureEvalResult]) -> str:
    lines = [f"{'feature':<26}{'pass':>6}{'LL delta':>11}{'improved':>10}{'stable':>8}{'best IC':>10}{'t':>8}{'half-life':>11}"]
    for name, r in results.items():
        best_ic = r.ic_by_horizon.get(r.best_horizon) if r.best_horizon else None
        best_t = r.ic_tstat_by_horizon.get(r.best_horizon) if r.best_horizon else None
        lines.append(
            f"{name:<26}{'yes' if r.passed else 'no':>6}"
            f"{(f'{r.logloss_delta:+.5f}' if r.logloss_delta is not None else 'n/a'):>11}"
            f"{r.folds_improved_frac:>10.0%}{r.importance_stability:>8.0%}"
            f"{(f'{best_ic:+.4f}' if best_ic is not None else 'n/a'):>10}"
            f"{(f'{best_t:+.2f}' if best_t is not None else 'n/a'):>8}"
            f"{(f'{r.ic_half_life_bars:.1f}' if r.ic_half_life_bars is not None else 'n/a'):>11}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from research.replay import bars_from_parquet

    parser = argparse.ArgumentParser(description="Evaluate the live features (and a seeded noise control) on stored 15m bars.")
    parser.add_argument("--parquet", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "btcusdt_perp_bars.parquet")
    parser.add_argument("--last", type=int, default=3000, help="Use the last N bars")
    parser.add_argument("--registry", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    bars = bars_from_parquet(args.parquet)[-args.last :]
    if len(bars) < 300:
        print(f"Only {len(bars)} bars in {args.parquet}; backfill first.")
        return 1
    registry = TrialRegistry(args.registry) if args.registry else TrialRegistry()
    results = evaluate_existing_features(bars, registry=registry)
    noise = np.random.default_rng(12345).standard_normal(len(bars))
    results["noise_control"] = evaluate_feature("noise_control", noise, bars, registry=registry)
    print(summarize_results(results))
    return 0


__all__ = [
    "BASE_FEATURE_NAMES",
    "DEFAULT_HORIZONS_BARS",
    "FeatureEvalConfig",
    "FeatureEvalResult",
    "PreparedBars",
    "evaluate_existing_features",
    "evaluate_feature",
    "forward_log_returns",
    "ic_half_life",
    "ic_tstat",
    "prepare_bars",
    "spearman_ic",
    "summarize_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
