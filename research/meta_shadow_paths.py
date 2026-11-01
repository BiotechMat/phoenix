"""CPCV trade paths for the meta-labeller and its P2.3 challenger; pre-registered shadow kill thresholds (P5.1).

For each of the 45 CPCV splits (10 groups, 2 test groups) the model is fitted on the purged training rows exactly
as the store fits it and predicts every test row. The test-row predictions are stitched into the 9 complete
out-of-sample paths. Along each path, trades are replayed in time order with the store's rules: enter when the
Mahalanobis gate is open and the governing allocation equation gives a non-zero weight (equal HRP weight across
the three assets, regime multiplier 1), hold until the triple-barrier exit, no new entry while a trade is open.
A trade's return as a fraction of capital is ``|weight| x net label return`` (net of fees and funding), the unit
the shadow book records.

Thresholds are registered on a per-trade basis: 80% of the median path's per-trade Sharpe, and the
95th-percentile path drawdown. They are written to ``data/shadow_thresholds.json``, which the store loads at start.
Research only; the CPCV split and the model fit are deterministic.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import logging
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from core.contracts import AssetAllocationInput, DollarBar, FeeSchedule, TripleBarrierConfig, compute_signed_target_weights
from core.features import calibrate_optimal_d, compute_positioning_snapshots
from core.features.candidates import bar_feature_matrix
from core.models import FeeAdjustedTripleBarrierLabeler, LightGBMMetaLabelWorker, extract_continuous_and_binary_from_snapshots
from core.models.triple_barrier import compute_effective_barriers, compute_ewma_volatility
from research.replay import bars_from_parquet
from research.validation.cpcv_gatekeeper import CombinatorialPurgedCV
from research.validation.shadow import PreRegisteredThresholds, save_thresholds

logger = logging.getLogger("phoenix.research.meta_shadow_paths")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
THRESHOLDS_PATH = DATA_DIR / "shadow_thresholds.json"
MAX_GROSS_LEVERAGE = 2.5


class PathSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    n_paths: int
    trades_per_path: list[int]
    per_trade_sharpe_by_path: list[float]
    drawdown_by_path: list[float]
    net_return_by_path: list[float]
    median_per_trade_sharpe: float
    drawdown_p95: float
    median_net_return: float
    positive_paths: int


def per_trade_sharpe(r: NDArray[np.float64]) -> float:
    if r.size < 3:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd) if sd > 1e-15 else 0.0


def drawdown(r: NDArray[np.float64]) -> float:
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    return float(np.max(peak - eq)) if eq.size else 0.0


def replay_path(weights: NDArray[np.float64], net_returns: NDArray[np.float64], hold_bars: NDArray[np.int64]) -> NDArray[np.float64]:
    """Sequential trades along one path: enter on a non-zero weight, hold to the barrier exit, no overlap."""
    out: list[float] = []
    i = 0
    n = weights.shape[0]
    while i < n:
        w = float(weights[i])
        if w != 0.0:
            out.append(abs(w) * float(net_returns[i]))
            i += max(1, int(hold_bars[i]))
        else:
            i += 1
    return np.asarray(out, dtype=np.float64)


def simulate_paths(
    bars: Sequence[DollarBar],
    extra_features: Sequence[str],
    *,
    n_groups: int = 10,
    k_test: int = 2,
    fee: FeeSchedule | None = None,
    n_jobs: int = 4,
    payoff_kelly: bool = False,
) -> PathSummary:
    fee = fee or FeeSchedule.hyperliquid()
    cfg = TripleBarrierConfig(fee_schedule=fee)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    log_close = np.log(closes)
    ffd = calibrate_optimal_d(log_close, split_id="meta_shadow_paths")
    snaps = compute_positioning_snapshots(bars, close_ffd_config=ffd)
    x, flags, dirs = extract_continuous_and_binary_from_snapshots(snaps)
    if extra_features:
        x = np.column_stack([x, bar_feature_matrix(bars, extra_features)])
    labels = FeeAdjustedTripleBarrierLabeler(cfg).label_bars(bars, primary_directions=dirs)
    n = len(bars) - cfg.vertical_barrier_bars
    x, flags = x[:n], flags[:n]
    y = np.array([lbl.meta_label for lbl in labels], dtype=np.int64)[:n]
    net = np.array([lbl.net_return for lbl in labels], dtype=np.float64)[:n]
    hold = np.array([lbl.hit_bar_offset for lbl in labels], dtype=np.int64)[:n]
    prim = np.array([lbl.primary_direction for lbl in labels], dtype=np.int64)[:n]
    sigma = np.concatenate(([1e-4], compute_ewma_volatility(np.diff(log_close), span=96)))[:n]

    cv = CombinatorialPurgedCV(vertical_barrier_bars=cfg.vertical_barrier_bars, n_groups=n_groups, k_test_groups=k_test, embargo_pct=0.01)
    splits = cv.generate_splits(n)
    weights_by_split: dict[int, NDArray[np.float64]] = {}
    for sp in splits:
        tr = np.asarray(sp.train_indices, dtype=np.int64)
        te = np.asarray(sp.test_indices, dtype=np.int64)
        w_split = np.zeros(te.size, dtype=np.float64)
        if np.unique(y[tr]).size >= 2:
            worker = LightGBMMetaLabelWorker(di_threshold=3.5, n_jobs=n_jobs, fee_schedule=fee)
            worker.fit(x[tr], y[tr], liquidation_exhaustion_flag_train=flags[tr], sigma_bar_train=sigma[tr])
            p = np.asarray(worker.predict_proba(x[te], liquidation_exhaustion_flag=flags[te], sigma_bar=sigma[te]), dtype=np.float64)
            di = np.asarray(worker.outlier_gate.compute_di(worker.svd_orthogonalizer.transform(x[te])), dtype=np.float64)
            for j, idx in enumerate(te):
                d = int(prim[idx])
                if d == 0 or not (di[j] <= worker.outlier_gate.di_threshold):
                    continue
                ub, lb = compute_effective_barriers(max(float(sigma[idx]), 0.0), cfg)
                exp_bps = (p[j] * ub - (1.0 - p[j]) * lb) * 10_000.0
                alloc = compute_signed_target_weights(
                    [AssetAllocationInput(asset_id="BTC", y_primary=(1 if d > 0 else -1), w_hrp=1.0 / 3.0, p_meta=float(p[j]), expected_gross_return_bps=exp_bps, fee_schedule=fee, epistemic_gate_passed=True,
                                          payoff_ratio=(ub / lb) if (payoff_kelly and lb > 0) else None)],
                    regime_multiplier=1.0,
                    max_gross_leverage=MAX_GROSS_LEVERAGE,
                )
                w_split[j] = float(alloc.weights.get("BTC", 0.0))
        weights_by_split[sp.split_id] = w_split
        logger.info("split %d/%d: %d entries signalled", sp.split_id + 1, len(splits), int(np.count_nonzero(w_split)))

    weight_paths = cv.reconstruct_oos_paths(weights_by_split, n, splits)
    trade_paths = [replay_path(wp, net, hold) for wp in weight_paths]
    sharpes = [per_trade_sharpe(t) for t in trade_paths]
    dds = [drawdown(t) for t in trade_paths]
    nets = [float(np.sum(t)) for t in trade_paths]
    return PathSummary(
        model="+".join(["live", *extra_features]) + (" (payoff Kelly)" if payoff_kelly else " (even-odds Kelly)"),
        n_paths=len(trade_paths),
        trades_per_path=[int(t.size) for t in trade_paths],
        per_trade_sharpe_by_path=sharpes,
        drawdown_by_path=dds,
        net_return_by_path=nets,
        median_per_trade_sharpe=float(np.median(sharpes)),
        drawdown_p95=float(np.percentile(dds, 95)),
        median_net_return=float(np.median(nets)),
        positive_paths=int(sum(1 for v in nets if v > 0)),
    )


def thresholds_from_summary(sleeve_id: str, summary: PathSummary, note: str) -> PreRegisteredThresholds:
    med = summary.median_per_trade_sharpe
    return PreRegisteredThresholds(
        sleeve_id=sleeve_id,
        median_path_sharpe=med,
        min_live_sharpe=0.8 * med,
        max_drawdown=summary.drawdown_p95,
        n_paths=summary.n_paths,
        registered_from=note,
        basis="per_trade",
    )


def main(argv: Sequence[str] | None = None) -> int:
    from ui.api_server import P23_CHALLENGER_FEATURES, SLEEVE_META_CHALLENGER

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last", type=int, default=35_040)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--register", action="store_true", help="Write the challenger's thresholds to data/shadow_thresholds.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bars = [b for b in bars_from_parquet(args.data_dir / "btcusdt_perp_bars.parquet") if b.open_interest > 0][-args.last :]
    window = f"{datetime.fromtimestamp(bars[0].ts_close_ns / 1e9, tz=timezone.utc).date()} to {datetime.fromtimestamp(bars[-1].ts_close_ns / 1e9, tz=timezone.utc).date()}"
    results = {
        "live, even-odds Kelly": simulate_paths(bars, ()),
        "live, payoff Kelly": simulate_paths(bars, (), payoff_kelly=True),
        "challenger": simulate_paths(bars, P23_CHALLENGER_FEATURES, payoff_kelly=True),
    }
    for name, r in results.items():
        print(
            f"{name:<22} trades/path {int(np.median(r.trades_per_path)):>5}  per-trade Sharpe median {r.median_per_trade_sharpe:+.4f} "
            f"(annualised at {int(np.median(r.trades_per_path))} trades/yr: {r.median_per_trade_sharpe * math.sqrt(max(np.median(r.trades_per_path), 1)):+.2f})  "
            f"net median {r.median_net_return:+.2%}  positive paths {r.positive_paths}/{r.n_paths}  drawdown p95 {r.drawdown_p95:.2%}"
        )
    ch = results["challenger"]
    if args.register:
        if ch.median_per_trade_sharpe <= 0.0:
            print("Challenger median path loses; thresholds not registered (it should not be in shadow).")
            return 2
        th = thresholds_from_summary(SLEEVE_META_CHALLENGER, ch, f"CPCV 10x2 trade paths, BTC 15m {window}, {len(bars)} bars")
        save_thresholds([th], THRESHOLDS_PATH)
        print(f"Registered {th.model_dump_json()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["THRESHOLDS_PATH", "PathSummary", "replay_path", "simulate_paths", "thresholds_from_summary"]
