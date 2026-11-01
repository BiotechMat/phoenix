"""P2.3 tests: candidate features are causal and correct on constructed inputs; weights and the trend-scanning label."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from core.features.candidates import (
    amihud_illiquidity,
    basket_residual_return,
    candidate_features,
    first_minute_taker_imbalance,
    funding_clamp_flag,
    funding_zscore,
    hour_of_day,
    label_uniqueness_weights,
    mark_index_divergence,
    momentum,
    nyse_open_flag,
    recency_weights,
    taker_imbalance_variance,
    trend_scanning_labels,
)
from core.models import LightGBMMetaLabelWorker
from tests.phase0_helpers import sine_history


def _ns(y: int, m: int, d: int, hh: int, mm: int) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1e9)


def test_features_are_causal() -> None:
    bars = sine_history("BTC", 1200, 100.0)
    full = candidate_features(bars)
    cut = candidate_features(bars[:900])
    for name, values in cut.items():
        a, b = values, full[name][:900]
        assert np.allclose(np.nan_to_num(a, nan=-9.0), np.nan_to_num(b, nan=-9.0)), name  # appending later bars changes nothing earlier


def test_momentum_and_divergence() -> None:
    closes = np.exp(np.arange(700, dtype=np.float64) * 0.001)
    m = momentum(closes)
    assert np.isnan(m[671]) and m[672] == pytest.approx(0.672)
    assert mark_index_divergence(np.array([101.0]), np.array([100.0]))[0] == pytest.approx(0.01)


def test_basket_residual_removes_the_common_move() -> None:
    i = np.arange(4000, dtype=np.float64)
    common = 0.002 * np.sin(i / 13.0)
    own = 0.001 * np.cos(i / 7.0)
    btc = 100.0 * np.exp(np.cumsum(common + own))
    eth = 100.0 * np.exp(np.cumsum(common))
    sol = 100.0 * np.exp(np.cumsum(common))
    res = basket_residual_return({"BTC": btc, "ETH": eth, "SOL": sol}, "BTC", beta_window=960)
    raw = np.log(btc[4:] / btc[:-4])
    ok = np.isfinite(res)
    assert ok.sum() > 2000
    assert np.std(res[ok]) < 0.6 * np.std(raw[ok[4:]])


def test_calendar_features() -> None:
    s, c = hour_of_day(np.array([_ns(2026, 1, 5, 6, 15)], dtype=np.int64))  # bar opening 06:00 UTC
    assert s[0] == pytest.approx(1.0) and c[0] == pytest.approx(0.0, abs=1e-12)
    winter = _ns(2026, 1, 5, 14, 30)  # Monday 09:30 New York (EST)
    summer = _ns(2026, 7, 6, 13, 30)  # Monday 09:30 New York (EDT)
    saturday = _ns(2026, 1, 10, 14, 30)
    late = _ns(2026, 1, 5, 15, 30)  # 10:30 New York: outside the first hour
    assert nyse_open_flag(np.array([winter, summer, saturday, late], dtype=np.int64)).tolist() == [1.0, 1.0, 0.0, 0.0]


def test_flow_liquidity_and_funding_features() -> None:
    v = taker_imbalance_variance(np.array([3.0, 1.0] * 30), np.array([1.0, 3.0] * 30), window=48)
    assert np.isnan(v[46]) and v[47] == pytest.approx(0.25)
    a = amihud_illiquidity(np.exp(np.arange(200) * 0.001), np.full(200, 1e6), window=96)
    assert a[150] == pytest.approx(np.log(0.001 / 1e6 * 1e9), rel=1e-6)
    f = np.concatenate([np.full(100, 0.0000125), [0.0001]])
    assert funding_clamp_flag(f)[:100].all() and funding_clamp_flag(f)[100] == 0.0
    z = funding_zscore(np.concatenate([np.tile([1e-5, 2e-5], 50), [5e-5]]), window=101)
    assert z[-1] > 3.0


def test_first_minute_imbalance_matches_the_minute_that_opens_the_bar() -> None:
    bar_open = np.array([_ns(2026, 1, 5, 0, 0), _ns(2026, 1, 5, 0, 15)], dtype=np.int64)
    m_open = np.array([bar_open[0] // 1_000_000, bar_open[0] // 1_000_000 + 60_000], dtype=np.int64)
    out = first_minute_taker_imbalance(bar_open, m_open, np.array([100.0, 50.0]), np.array([75.0, 0.0]))
    assert out[0] == pytest.approx(0.5) and np.isnan(out[1])


def test_uniqueness_and_recency_weights() -> None:
    w = label_uniqueness_weights([0, 0, 3, 0])  # label 2 spans bars 2..5 overlapping label 3 at bar 3
    raw = np.array([1.0, 1.0, np.mean([1.0, 0.5, 1.0, 1.0]), 0.5])  # bar 3 is shared by two labels
    assert np.allclose(w, raw / raw.mean())
    r = recency_weights(3, 1.0)
    assert np.allclose(r / r[-1], [0.25, 0.5, 1.0])


def test_trend_scanning_labels_follow_the_forward_trend() -> None:
    up = np.exp(np.linspace(0, 0.05, 60) + 0.001 * np.sin(np.arange(60)))
    sign, t = trend_scanning_labels(up, spans=(8, 16))
    assert (sign[:44] == 1).all() and (t[:44] > 2.0).all()
    assert sign[-1] == 0  # no forward window left


def test_worker_accepts_sample_weights() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    y = (x[:, 0] > 0).astype(np.int64)
    w = LightGBMMetaLabelWorker(n_jobs=1).fit(x, y, sample_weight=np.linspace(0.1, 2.0, 300))
    p = np.asarray(w.predict_proba(x[:5]))
    assert p.shape == (5,) and np.all((p >= 0) & (p <= 1))
    with pytest.raises(ValueError):
        LightGBMMetaLabelWorker(n_jobs=1).fit(x, y, sample_weight=np.ones(10))


def test_bar_feature_matrix_treats_seeded_bars_without_venue_marks_as_unknown() -> None:
    from core.features.candidates import bar_feature_matrix

    bars = sine_history("BTC", 800, 100.0)
    bars = [b.model_copy(update={"mark_price": b.close, "spot_price": b.close}) for b in bars]  # no venue mark/index
    m = bar_feature_matrix(bars, ["mark_index_divergence", "mark_index_divergence_chg_1h", "momentum_1w"])
    assert m.shape == (800, 3) and np.all(m[:, :2] == 0.0) and np.all(np.isfinite(m))
    assert m[700, 2] != 0.0
    with pytest.raises(ValueError):
        bar_feature_matrix(bars, ["taker_imbalance_var_48"])


def test_store_trains_and_predicts_with_extra_features(tmp_path: object) -> None:
    from pathlib import Path

    from core.contracts import DataAvailability
    from tests.phase0_helpers import build_store, companion_flat_bars, make_snapshot, ohlc_path
    from ui.api_server import MIN_TRAIN_LABELLED_BARS

    history = sine_history("BTC", MIN_TRAIN_LABELLED_BARS + 800, 100.0)
    live = ohlc_path("BTC", [(100.0, 100.3, 99.7, 100.2)])
    store = build_store(
        Path(str(tmp_path)),
        live_bars={"BTC": live, **companion_flat_bars(1)},
        snapshots=[make_snapshot()],
        history={"BTC": history},
        availability=DataAvailability(candles=True, funding_history=True, open_interest_history=True, feed_source="SCRIPTED"),
        meta_extra_features=("momentum_1w", "mark_index_divergence_chg_1h"),
    )
    assert store.ml_worker is not None
    assert store.residual_telemetry["meta_features"][-2:] == ["momentum_1w", "mark_index_divergence_chg_1h"]
    store.step_simulation_tick()
    assert 0.0 < store.state.meta_predictions[-1].p_meta < 1.0


def test_challenger_is_fitted_and_trades_only_in_the_shadow_book(tmp_path: object) -> None:
    from pathlib import Path

    from core.contracts import DataAvailability
    from tests.phase0_helpers import build_store, companion_flat_bars, make_snapshot, ohlc_path
    from ui.api_server import MIN_TRAIN_LABELLED_BARS, P23_CHALLENGER_FEATURES, SLEEVE_META_CHALLENGER

    history = sine_history("BTC", MIN_TRAIN_LABELLED_BARS + 800, 100.0)
    live = ohlc_path("BTC", [(100.0, 100.3, 99.7, 100.2), (100.2, 100.5, 99.9, 100.1)])
    store = build_store(
        Path(str(tmp_path)),
        live_bars={"BTC": live, **companion_flat_bars(2)},
        snapshots=[make_snapshot()],
        history={"BTC": history},
        availability=DataAvailability(candles=True, funding_history=True, open_interest_history=True, feed_source="SCRIPTED"),
        meta_challenger_features=P23_CHALLENGER_FEATURES,
    )
    tele = store.residual_telemetry
    assert tele["meta_features"] == ["ffd_close_zscore", "ffd_oi_zscore", "oi_funding_basis_zscore"]  # live model unchanged
    assert tele["challenger_features"] == list(P23_CHALLENGER_FEATURES) and tele["challenger_fitted"] is True
    assert SLEEVE_META_CHALLENGER in store.shadow_sleeve_ids
    store.step_simulation_tick()
    store.step_simulation_tick()
    assert store.residual_telemetry["challenger_p_meta"] is not None
    assert all(p.sleeve_id != SLEEVE_META_CHALLENGER for p in store.position_manager.open_positions.values())
    assert all(t.sleeve_id != SLEEVE_META_CHALLENGER for t in store.position_manager.closed_trades)


def test_live_feed_without_oi_history_holds_the_model_neutral_for_the_warmup(tmp_path: object) -> None:
    from pathlib import Path

    from core.contracts import DataAvailability
    from tests.phase0_helpers import build_store, companion_flat_bars, make_snapshot, ohlc_path
    from ui.api_server import MIN_TRAIN_LABELLED_BARS, OI_WARMUP_BARS

    history = sine_history("BTC", MIN_TRAIN_LABELLED_BARS + 400, 100.0)
    live = ohlc_path("BTC", [(100.0, 100.3, 99.7, 100.2)] * 3)
    avail = DataAvailability(candles=True, funding_history=True, open_interest_history=False, feed_source="HYPERLIQUID-L1")
    store = build_store(
        Path(str(tmp_path)), live_bars={"BTC": live, **companion_flat_bars(3)}, snapshots=[make_snapshot()],
        history={"BTC": history}, availability=avail,
    )
    assert store.residual_telemetry["oi_warmup_remaining_bars"] == OI_WARMUP_BARS
    store.step_simulation_tick()
    meta = store.state.meta_predictions[-1]
    assert meta.p_meta == 0.5 and meta.epistemic_gate_passed is False
    assert store.residual_telemetry["oi_warmup_remaining_bars"] == OI_WARMUP_BARS - 1
    assert store.residual_telemetry["challenger_p_meta"] is None


def test_challenger_is_off_by_default(tmp_path: object) -> None:
    from pathlib import Path

    from tests.phase0_helpers import build_store, companion_flat_bars, make_snapshot, ohlc_path
    from ui.api_server import SLEEVE_META_CHALLENGER

    store = build_store(
        Path(str(tmp_path)), live_bars={"BTC": ohlc_path("BTC", [(100.0, 100.3, 99.7, 100.2)]), **companion_flat_bars(1)},
        snapshots=[make_snapshot()], history={"BTC": sine_history("BTC", 200, 100.0)},
    )
    assert SLEEVE_META_CHALLENGER not in store.shadow_sleeve_ids and store.residual_telemetry["challenger_features"] == []
