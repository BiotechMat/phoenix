"""Meta-labeller feature expansion evaluation (REINFORCEMENT_PLAN task P2.3).

Each candidate from ``core.features.candidates`` is added, one at a time, to the live baseline features and
run through the P1.8 harness (CPCV out-of-fold calibrated log-loss, permutation-importance stability, IC by
horizon with fold sign consistency). Every run is appended to the trial registry. The boundary imbalance uses
the first one-minute kline of each bar from the Binance archive.

Two training changes are evaluated the same way, against the unweighted live model on the same splits:
label-uniqueness sample weights and recency weights. The trend-scanning label is evaluated as an alternative
training target: a model trained on "trend-scan sign agrees with the primary direction" is scored by AUC
against the triple-barrier meta label it would replace.

Research only; nothing here runs in the live loop.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import roc_auc_score

from core.contracts import DollarBar, TripleBarrierConfig
from core.features import calibrate_optimal_d, compute_positioning_snapshots
from core.features.candidates import (
    candidate_features,
    first_minute_taker_imbalance,
    label_uniqueness_weights,
    trend_scanning_labels,
)
from core.models import FeeAdjustedTripleBarrierLabeler, LightGBMMetaLabelWorker, extract_continuous_and_binary_from_snapshots
from data.backfill_storage import fetch_binance_vision_klines, parse_binance_vision_klines
from research.feature_eval import FeatureEvalConfig, FeatureEvalResult, PreparedBars, _log_loss, evaluate_feature, summarize_results
from research.replay import bars_from_parquet
from research.trial_registry import TrialRegistry
from research.validation.cpcv_gatekeeper import CombinatorialPurgedCV

logger = logging.getLogger("phoenix.research.p23")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TrainingVariantResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    folds: int
    oof_logloss_base: float
    oof_logloss_variant: float
    folds_improved: float
    metric: str = "log-loss"
    passed: bool


def prepare_with_offsets(bars: Sequence[DollarBar], config: FeatureEvalConfig | None = None) -> tuple[PreparedBars, NDArray[np.int64], NDArray[np.int64]]:
    """``prepare_bars`` plus the label hit offsets and primary directions it computes internally."""
    cfg = config or FeatureEvalConfig()
    barrier = TripleBarrierConfig(fee_schedule=cfg.fee_schedule)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    ffd = calibrate_optimal_d(np.log(closes), split_id="feature_eval")
    snaps = compute_positioning_snapshots(bars, close_ffd_config=ffd)
    x_base, flags, dirs = extract_continuous_and_binary_from_snapshots(snaps)
    labels = FeeAdjustedTripleBarrierLabeler(barrier).label_bars(bars, primary_directions=dirs)
    y = np.array([lbl.meta_label for lbl in labels], dtype=np.int64)
    offsets = np.array([lbl.hit_bar_offset for lbl in labels], dtype=np.int64)
    n_lab = max(0, len(bars) - barrier.vertical_barrier_bars)
    prepared = PreparedBars(closes=closes, x_base=x_base, flags=flags, labels=y, n_labelled=n_lab)
    return prepared, offsets, np.asarray(dirs, dtype=np.int64)


def _splits(n: int, cfg: FeatureEvalConfig) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    cv = CombinatorialPurgedCV(vertical_barrier_bars=TripleBarrierConfig().vertical_barrier_bars, n_groups=cfg.n_groups, k_test_groups=cfg.k_test_groups, embargo_pct=cfg.embargo_pct)
    return [(np.asarray(s.train_indices, dtype=np.int64), np.asarray(s.test_indices, dtype=np.int64)) for s in cv.generate_splits(n)]


def _worker(cfg: FeatureEvalConfig) -> LightGBMMetaLabelWorker:
    return LightGBMMetaLabelWorker(di_threshold=3.5, n_jobs=1, fee_schedule=cfg.fee_schedule)


def evaluate_weighting(
    name: str,
    prepared: PreparedBars,
    weight_fn: Callable[[NDArray[np.int64]], NDArray[np.float64]],
    config: FeatureEvalConfig | None = None,
) -> TrainingVariantResult:
    """OOF log-loss of the live model trained with ``weight_fn(train_indices)`` against unweighted, same splits."""
    cfg = config or FeatureEvalConfig()
    n = prepared.n_labelled
    x, y, fl = prepared.x_base[:n], prepared.labels[:n], prepared.flags[:n]
    base: list[float] = []
    var: list[float] = []
    for tr, te in _splits(n, cfg):
        if tr.size < 50 or te.size < 10 or np.unique(y[tr]).size < 2:
            continue
        pb = _worker(cfg).fit(x[tr], y[tr], liquidation_exhaustion_flag_train=fl[tr]).predict_proba(x[te], liquidation_exhaustion_flag=fl[te])
        pw = _worker(cfg).fit(x[tr], y[tr], liquidation_exhaustion_flag_train=fl[tr], sample_weight=weight_fn(tr)).predict_proba(x[te], liquidation_exhaustion_flag=fl[te])
        base.append(_log_loss(y[te], np.asarray(pb)))
        var.append(_log_loss(y[te], np.asarray(pw)))
    b, v = float(np.mean(base)), float(np.mean(var))
    improved = float(np.mean([vv < bb for bb, vv in zip(base, var)]))
    return TrainingVariantResult(name=name, folds=len(base), oof_logloss_base=b, oof_logloss_variant=v, folds_improved=improved, passed=bool(v < b and improved >= cfg.min_folds_improved))


def evaluate_trend_scanning_target(prepared: PreparedBars, directions: NDArray[np.int64], config: FeatureEvalConfig | None = None) -> TrainingVariantResult:
    """AUC on the triple-barrier meta label: model trained on it versus model trained on the trend-scanning target."""
    cfg = config or FeatureEvalConfig()
    n = prepared.n_labelled
    x, y, fl = prepared.x_base[:n], prepared.labels[:n], prepared.flags[:n]
    ts_sign, _ = trend_scanning_labels(prepared.closes)
    y_ts = ((ts_sign[:n] != 0) & (ts_sign[:n] == directions[:n])).astype(np.int64)
    base: list[float] = []
    var: list[float] = []
    for tr, te in _splits(n, cfg):
        if tr.size < 50 or te.size < 10 or np.unique(y[tr]).size < 2 or np.unique(y_ts[tr]).size < 2 or np.unique(y[te]).size < 2:
            continue
        pb = _worker(cfg).fit(x[tr], y[tr], liquidation_exhaustion_flag_train=fl[tr]).predict_proba(x[te], liquidation_exhaustion_flag=fl[te])
        pt = _worker(cfg).fit(x[tr], y_ts[tr], liquidation_exhaustion_flag_train=fl[tr]).predict_proba(x[te], liquidation_exhaustion_flag=fl[te])
        base.append(float(roc_auc_score(y[te], np.asarray(pb))))
        var.append(float(roc_auc_score(y[te], np.asarray(pt))))
    b, v = float(np.mean(base)), float(np.mean(var))
    improved = float(np.mean([vv > bb for bb, vv in zip(base, var)]))
    return TrainingVariantResult(
        name="trend_scanning_target", folds=len(base), oof_logloss_base=b, oof_logloss_variant=v, folds_improved=improved,
        metric="AUC (higher is better)", passed=bool(v > b and improved >= cfg.min_folds_improved),
    )


def recency_by_index(train_idx: NDArray[np.int64], half_life_bars: float) -> NDArray[np.float64]:
    """Weights halving every ``half_life_bars`` of bar age before the newest training row, mean 1."""
    age = (int(train_idx.max()) - train_idx).astype(np.float64)
    w = 0.5 ** (age / half_life_bars)
    return np.asarray(w / w.mean(), dtype=np.float64)


def evaluate_feature_set(
    name: str,
    prepared: PreparedBars,
    columns: Mapping[str, NDArray[np.float64]],
    config: FeatureEvalConfig | None = None,
    noise_reference: bool = True,
) -> TrainingVariantResult:
    """OOF log-loss of the live model with all ``columns`` added at once, against a reference on the same splits.

    With ``noise_reference`` the reference is the live baseline plus as many seeded Gaussian noise columns
    as the set adds, so a set only passes when it beats a model that merely has more columns.
    """
    cfg = config or FeatureEvalConfig()
    n = prepared.n_labelled
    extra = np.column_stack([np.where(np.isfinite(v), v, 0.0) for v in columns.values()])[:n]
    xb, y, fl = prepared.x_base[:n], prepared.labels[:n], prepared.flags[:n]
    if noise_reference:
        xb = np.column_stack([xb, np.random.default_rng(54321).standard_normal((n, extra.shape[1]))])  # research reference only
    xc = np.column_stack([prepared.x_base[:n], extra])
    base: list[float] = []
    var: list[float] = []
    for tr, te in _splits(n, cfg):
        if tr.size < 50 or te.size < 10 or np.unique(y[tr]).size < 2:
            continue
        pb = _worker(cfg).fit(xb[tr], y[tr], liquidation_exhaustion_flag_train=fl[tr]).predict_proba(xb[te], liquidation_exhaustion_flag=fl[te])
        pc = _worker(cfg).fit(xc[tr], y[tr], liquidation_exhaustion_flag_train=fl[tr]).predict_proba(xc[te], liquidation_exhaustion_flag=fl[te])
        base.append(_log_loss(y[te], np.asarray(pb)))
        var.append(_log_loss(y[te], np.asarray(pc)))
    b, v = float(np.mean(base)), float(np.mean(var))
    improved = float(np.mean([vv < bb for bb, vv in zip(base, var)]))
    return TrainingVariantResult(name=name, folds=len(base), oof_logloss_base=b, oof_logloss_variant=v, folds_improved=improved, passed=bool(v < b and improved >= cfg.min_folds_improved))


def _aligned(bars_by_coin: Mapping[str, list[DollarBar]], coin: str) -> dict[str, list[DollarBar]]:
    return {c: b for c, b in bars_by_coin.items() if c != coin}


def first_minute_feature(bars: Sequence[DollarBar], symbol: str) -> NDArray[np.float64]:
    """Signed taker share of the first one-minute kline of each bar, in quote (dollar) volume."""
    start_ms = bars[0].ts_open_ns // 1_000_000
    end_ms = bars[-1].ts_open_ns // 1_000_000
    minute = [c for t in fetch_binance_vision_klines(symbol, "1m", start_ms, end_ms) for c in parse_binance_vision_klines(t)]
    return first_minute_taker_imbalance(
        np.array([b.ts_open_ns for b in bars], dtype=np.int64),
        np.array([c.t_ms for c in minute], dtype=np.int64),
        np.array([c.quote_volume for c in minute], dtype=np.float64),
        np.array([c.taker_buy_quote or 0.0 for c in minute], dtype=np.float64),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--last", type=int, default=35_040, help="Use the last N 15m bars (default: one year)")
    parser.add_argument("--skip-last", type=int, default=0, help="Drop the newest N bars first (an earlier, non-overlapping window)")
    parser.add_argument("--features", default="", help="Comma-separated subset of candidates to evaluate (default: all)")
    parser.add_argument("--set", action="append", default=[], help="Evaluate these candidates together as one set (repeatable)")
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the results as JSON")
    parser.add_argument("--skip-first-minute", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    coin = args.coin.upper()
    all_bars = {c: bars_from_parquet(args.data_dir / f"{c.lower()}usdt_perp_bars.parquet") for c in ("BTC", "ETH", "SOL")}
    series = all_bars[coin][: len(all_bars[coin]) - args.skip_last] if args.skip_last else all_bars[coin]
    bars = series[-args.last :]
    if len(bars) < 1_000:
        print(f"Only {len(bars)} {coin} bars; backfill first.")
        return 1
    t0, t1 = bars[0].ts_close_ns, bars[-1].ts_close_ns
    companions = {c: [b for b in v if t0 <= b.ts_close_ns <= t1] for c, v in _aligned(all_bars, coin).items()}
    window = (datetime.fromtimestamp(bars[0].ts_close_ns / 1e9, tz=timezone.utc), datetime.fromtimestamp(bars[-1].ts_close_ns / 1e9, tz=timezone.utc))
    logger.info("P2.3 on %d %s bars, %s to %s", len(bars), coin, window[0].date(), window[1].date())

    cfg = FeatureEvalConfig()
    registry = TrialRegistry(args.registry) if args.registry else TrialRegistry()
    prepared, offsets, directions = prepare_with_offsets(bars, cfg)
    feats = candidate_features(bars, companions=companions, coin=coin, taker_flow_real=True)
    wanted = [f.strip() for f in args.features.split(",") if f.strip()]
    sets = [[f.strip() for f in group.split(",") if f.strip()] for group in args.set]
    set_names = [f for g in sets for f in g]
    if not args.skip_first_minute and (not wanted or "first_minute_taker_imbalance" in wanted + set_names):
        feats["first_minute_taker_imbalance"] = first_minute_feature(bars, f"{coin}USDT")
    feats["noise_control"] = np.random.default_rng(12345).standard_normal(len(bars))  # seeded reference column, research only
    set_cols = [{k: feats[k] for k in g} for g in sets]
    if wanted:
        feats = {k: v for k, v in feats.items() if k in wanted}

    results: dict[str, FeatureEvalResult] = {}
    for name, values in feats.items():
        results[name] = evaluate_feature(name, values, prepared=prepared, config=cfg, registry=registry, window_start=window[0], window_end=window[1])
        logger.info("%s: %s", name, "; ".join(results[name].reasons))
    print(summarize_results(results))

    n = prepared.n_labelled
    uniq = label_uniqueness_weights(offsets[:n])
    variants: list[TrainingVariantResult] = []
    for cols in set_cols:
        variants.append(evaluate_feature_set("set:" + "+".join(cols), prepared, cols, cfg))
    if not args.skip_variants:
        variants += [
            evaluate_weighting("uniqueness_weights", prepared, lambda tr: uniq[tr] / max(float(np.mean(uniq[tr])), 1e-12), cfg),
            evaluate_weighting("recency_weights_30d", prepared, lambda tr: recency_by_index(tr, 30 * 96), cfg),
            evaluate_trend_scanning_target(prepared, directions, cfg),
        ]
    for v in variants:
        print(f"{v.name:<26} {v.metric}: base {v.oof_logloss_base:.5f} variant {v.oof_logloss_variant:.5f} improved {v.folds_improved:.0%} of {v.folds} folds -> {'pass' if v.passed else 'fail'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"coin": coin, "window": [w.isoformat() for w in window], "n_bars": len(bars),
                        "features": {k: r.model_dump(mode="json") for k, r in results.items()},
                        "variants": [v.model_dump(mode="json") for v in variants]}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TrainingVariantResult", "evaluate_feature_set", "evaluate_trend_scanning_target", "evaluate_weighting", "first_minute_feature", "prepare_with_offsets"]
