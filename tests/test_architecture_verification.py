"""Comprehensive Causality, Fee-Accounting, O(1) Hawkes Latency, HRP-CVaR & Fallback Verification Suite."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from core.contracts import (
    AssetAllocationInput,
    CPCVValidationResult,
    DollarBar,
    ExecutionAction,
    ExecutionRegimeMode,
    FeeSchedule,
    PositioningRegime,
    SystemicRiskOutput,
    TripleBarrierConfig,
    compute_fractional_kelly,
    compute_regime_multiplier_from_asri,
    compute_signed_target_weights,
)
from core.execution import (
    DefensiveMakerExecutionAlgorithm,
    RecursiveHawkesMonitor,
    RollingBVCVPINMonitor,
    hawkes_recursive_batch_numba,
    hawkes_recursive_step_numba,
)
from core.features import (
    FrozenFFDTransformer,
    apply_ffd_weights_numba,
    calibrate_optimal_d,
    classify_positioning_regime,
    compute_ffd_weights,
    compute_positioning_snapshots,
)
from core.hardware_bootstrap import (
    bootstrap_m1max_hardware,
    enforce_spawn_and_thread_caps,
    load_environment_config,
)
from core.models import (
    CONTINUOUS_SIGNAL_COLUMNS,
    EXCLUDED_BINARY_COLUMNS,
    ContinuousSVDOrthogonalizer,
    FeeAdjustedTripleBarrierLabeler,
    LightGBMMetaLabelWorker,
    MahalanobisOutlierGate,
    extract_continuous_and_binary_from_snapshots,
)
from research.validation import (
    ChampionChallengerGatekeeper,
    CombinatorialPurgedCV,
    compute_dsr,
    compute_net_of_fee_returns,
    compute_pbo,
    compute_psr,
)
from risk import (
    SignedHRPCVaRAllocator,
    compute_hrp_cvar_weights,
)
from sentiment import (
    PersistentMLXRiskDaemon,
    parse_and_validate_llm_risk_json,
)
from sleeves.derivatives.funding_carry import FundingCarrySleeve
from ui import (
    DOC_FILENAMES,
    PLAIN_ENGLISH_TOOLTIPS,
    PlainEnglishTelegramBot,
    create_app,
)


def _make_synthetic_dollar_bars(n_bars: int = 160, seed: int = 42) -> list[DollarBar]:
    """Generate deterministic medium-frequency DollarBar instances for testing."""
    rng = np.random.default_rng(seed)
    bars: list[DollarBar] = []
    price = 50_000.0
    oi = 100_000.0
    base_ts = 1_700_000_000_000_000_000
    step_ns = 15 * 60 * 1_000_000_000

    for i in range(n_bars):
        ret = float(rng.normal(0.0002, 0.006))
        open_p = price
        close_p = max(100.0, open_p * (1.0 + ret))
        high_p = max(open_p, close_p) * (1.0 + abs(float(rng.normal(0.0, 0.002))))
        low_p = min(open_p, close_p) * (1.0 - abs(float(rng.normal(0.0, 0.002))))
        vol = float(rng.uniform(50.0, 250.0))
        dvol = vol * close_p
        buy_frac = float(rng.uniform(0.35, 0.65))
        oi = max(10_000.0, oi * (1.0 + float(rng.normal(0.0005, 0.015))))
        funding = float(rng.normal(0.0001, 0.0004))
        basis_spread = float(rng.normal(0.0005, 0.001))
        mark_p = close_p * (1.0 + basis_spread)
        liq_usd = float(rng.exponential(25_000.0))
        if i % 35 == 0 and i > 20:
            liq_usd *= 25.0  # occasional liquidation spike

        bars.append(
            DollarBar(
                instrument_id="BTCUSDT-PERP",
                horizon_label="15m",
                ts_open_ns=base_ts + i * step_ns,
                ts_close_ns=base_ts + (i + 1) * step_ns,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=vol,
                dollar_volume=dvol,
                buy_dollar_volume=dvol * buy_frac,
                sell_dollar_volume=dvol * (1.0 - buy_frac),
                open_interest=oi,
                funding_rate_8h=funding,
                mark_price=mark_p,
                spot_price=close_p,
                liquidation_volume_usd=liq_usd,
                tick_count=200,
            )
        )
        price = close_p
    return bars


# ==============================================================================
# 1. ZERO LOOK-AHEAD BIAS & CAUSALITY INVARIANCE (1e-12 PRECISION)
# ==============================================================================


def test_zero_lookahead_bias_frozen_ffd_svd_vpin_hawkes_positioning() -> None:
    """Verify perturbing future bars t+1...T leaves FFD, Positioning, SVD, DI, VPIN, and Hawkes at bar t invariant to 1e-12."""
    bars_baseline = _make_synthetic_dollar_bars(n_bars=140, seed=101)
    train_cut = 70
    eval_t = 105  # strictly in out-of-sample region (train_cut < eval_t < n_bars - 1)

    # 1. Calibrate and freeze d* strictly on in-sample split [0 : train_cut]
    train_log_close = np.log(np.array([b.close for b in bars_baseline[:train_cut]], dtype=np.float64))
    train_log_oi = np.log(np.array([b.open_interest for b in bars_baseline[:train_cut]], dtype=np.float64))

    ffd_close_cfg = calibrate_optimal_d(train_log_close, split_id="cpcv_split_01")
    ffd_oi_cfg = calibrate_optimal_d(train_log_oi, split_id="cpcv_split_01_oi")
    assert 0.05 <= ffd_close_cfg.d_star <= 0.60

    frozen_close_ffd = FrozenFFDTransformer(ffd_close_cfg)

    # Construct perturbed future series where bars strictly after eval_t (t+1 ... T) are shocked by 50x
    bars_perturbed = list(bars_baseline)
    for k in range(eval_t + 1, len(bars_perturbed)):
        b_orig = bars_perturbed[k]
        bars_perturbed[k] = DollarBar(
            instrument_id=b_orig.instrument_id,
            horizon_label=b_orig.horizon_label,
            ts_open_ns=b_orig.ts_open_ns,
            ts_close_ns=b_orig.ts_close_ns,
            open=b_orig.open * 15.0,
            high=b_orig.high * 25.0,
            low=b_orig.low * 0.2,
            close=b_orig.close * 20.0,
            volume=b_orig.volume * 100.0,
            dollar_volume=b_orig.dollar_volume * 2000.0,
            buy_dollar_volume=b_orig.buy_dollar_volume * 1800.0,
            sell_dollar_volume=b_orig.sell_dollar_volume * 200.0,
            open_interest=b_orig.open_interest * 50.0,
            funding_rate_8h=0.025,
            mark_price=b_orig.mark_price * 22.0,
            spot_price=b_orig.spot_price * 20.0,
            liquidation_volume_usd=b_orig.liquidation_volume_usd * 5000.0 + 1e8,
            tick_count=10000,
        )

    # Verify Frozen FFD causality at index <= eval_t
    log_close_base = np.log(np.array([b.close for b in bars_baseline], dtype=np.float64))
    log_close_pert = np.log(np.array([b.close for b in bars_perturbed], dtype=np.float64))

    ffd_out_base = frozen_close_ffd.transform(log_close_base)
    ffd_out_pert = frozen_close_ffd.transform(log_close_pert)
    max_ffd_diff = float(np.max(np.abs(ffd_out_base[: eval_t + 1] - ffd_out_pert[: eval_t + 1])))
    assert max_ffd_diff <= 1e-12, f"FFD causality breach: {max_ffd_diff}"

    # Verify Positioning snapshots causality at index <= eval_t
    snaps_base = compute_positioning_snapshots(
        bars_baseline, close_ffd_config=ffd_close_cfg, oi_ffd_config=ffd_oi_cfg
    )
    snaps_pert = compute_positioning_snapshots(
        bars_perturbed, close_ffd_config=ffd_close_cfg, oi_ffd_config=ffd_oi_cfg
    )
    for idx in range(eval_t + 1):
        sb = snaps_base[idx]
        sp = snaps_pert[idx]
        assert abs(sb.ffd_close - sp.ffd_close) <= 1e-12
        assert abs(sb.ffd_oi - sp.ffd_oi) <= 1e-12
        assert abs(sb.ffd_close_zscore - sp.ffd_close_zscore) <= 1e-12
        assert abs(sb.ffd_oi_zscore - sp.ffd_oi_zscore) <= 1e-12
        assert abs(sb.oi_funding_basis_zscore - sp.oi_funding_basis_zscore) <= 1e-12
        assert abs(sb.liquidation_zscore - sp.liquidation_zscore) <= 1e-12
        assert sb.liquidation_exhaustion_flag == sp.liquidation_exhaustion_flag
        assert sb.regime == sp.regime
        assert sb.primary_direction == sp.primary_direction

    # Verify Stage 1 Continuous-Only SVD and Mahalanobis DI_values causality at index <= eval_t
    x_cont_base, bin_base, _ = extract_continuous_and_binary_from_snapshots(snaps_base)
    x_cont_pert, bin_pert, _ = extract_continuous_and_binary_from_snapshots(snaps_pert)

    svd_ortho = ContinuousSVDOrthogonalizer().fit(x_cont_base[:train_cut])
    maha_gate = MahalanobisOutlierGate().fit(x_cont_base[:train_cut])

    svd_factors_base = svd_ortho.transform(x_cont_base)
    svd_factors_pert = svd_ortho.transform(x_cont_pert)
    max_svd_diff = float(np.max(np.abs(svd_factors_base[: eval_t + 1] - svd_factors_pert[: eval_t + 1])))
    assert max_svd_diff <= 1e-12
    assert np.array_equal(bin_base[: eval_t + 1], bin_pert[: eval_t + 1])

    di_base = maha_gate.compute_di(x_cont_base)
    di_pert = maha_gate.compute_di(x_cont_pert)
    max_di_diff = float(np.max(np.abs(di_base[: eval_t + 1] - di_pert[: eval_t + 1])))
    assert max_di_diff <= 1e-12

    # Verify BVC VPIN and O(1) Hawkes causality at index <= eval_t
    vpin_mon_base = RollingBVCVPINMonitor(bucket_volume=500.0, num_buckets=50)
    vpin_mon_pert = RollingBVCVPINMonitor(bucket_volume=500.0, num_buckets=50)
    hawkes_mon_base = RecursiveHawkesMonitor(async_refit=False)
    hawkes_mon_pert = RecursiveHawkesMonitor(async_refit=False)

    vpin_hist_base: list[float] = []
    vpin_hist_pert: list[float] = []
    hawkes_hist_base: list[float] = []
    hawkes_hist_pert: list[float] = []

    for b in bars_baseline:
        v_b = vpin_mon_base.update(price=b.close, volume=b.volume, ts_ns=b.ts_close_ns)
        h_b = hawkes_mon_base.update(ts_ns=b.ts_close_ns, vpin=v_b)
        vpin_hist_base.append(v_b)
        hawkes_hist_base.append(h_b.lambda_intensity)

    for b in bars_perturbed:
        v_p = vpin_mon_pert.update(price=b.close, volume=b.volume, ts_ns=b.ts_close_ns)
        h_p = hawkes_mon_pert.update(ts_ns=b.ts_close_ns, vpin=v_p)
        vpin_hist_pert.append(v_p)
        hawkes_hist_pert.append(h_p.lambda_intensity)

    assert np.max(np.abs(np.array(vpin_hist_base[: eval_t + 1]) - np.array(vpin_hist_pert[: eval_t + 1]))) <= 1e-12
    assert np.max(np.abs(np.array(hawkes_hist_base[: eval_t + 1]) - np.array(hawkes_hist_pert[: eval_t + 1]))) <= 1e-12


@settings(max_examples=25, deadline=None)
@given(
    d_order=st.floats(min_value=0.05, max_value=0.60, allow_nan=False, allow_infinity=False),
    future_shock=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_ffd_causal_invariance(d_order: float, future_shock: float) -> None:
    """Property test: FFD with frozen weights at index t is strictly invariant (<= 1e-12) to future shocks."""
    rng = np.random.default_rng(12345)
    series_a = np.cumsum(rng.normal(0.0, 0.02, size=80)) + 10.0
    series_b = series_a.copy()
    t_idx = 45
    series_b[t_idx + 1 :] += future_shock

    weights = compute_ffd_weights(d=d_order, tau=1e-5)
    out_a = apply_ffd_weights_numba(series_a, weights)
    out_b = apply_ffd_weights_numba(series_b, weights)
    assert float(np.max(np.abs(out_a[: t_idx + 1] - out_b[: t_idx + 1]))) <= 1e-12


# ==============================================================================
# 2. O(1) HAWKES BENCHMARK (< 5ms FOR 10,000 UPDATES) & HYSTERESIS VERIFICATION
# ==============================================================================


def test_o1_hawkes_10000_updates_under_5ms_benchmark() -> None:
    """Prove that 10,000 sequential @numba.njit recursive Hawkes updates execute in under 5 ms total."""
    n_events = 10_000
    rng = np.random.default_rng(7)
    inter_arrivals = rng.exponential(scale=0.05, size=n_events)
    timestamps_sec = np.cumsum(inter_arrivals).astype(np.float64)

    # Warm-up JIT compilation
    _ = hawkes_recursive_step_numba(0.05, 0.0, 1.5, 0.8, 2.0)
    _ = hawkes_recursive_batch_numba(timestamps_sec[:10], 1.5, 0.8, 2.0)

    t0 = time.perf_counter()
    r_arr, lam_arr = hawkes_recursive_batch_numba(timestamps_sec, 1.5, 0.8, 2.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert r_arr.shape[0] == n_events
    assert lam_arr.shape[0] == n_events
    assert elapsed_ms < 5.0, f"10,000 sequential Hawkes updates took {elapsed_ms:.3f} ms (SLA < 5.0 ms)"


def test_defensive_maker_repeg_hysteresis_backoff_and_toxic_shield() -> None:
    """Verify 1.5 bps + 500ms dual hysteresis, 3-strike deeper GTX backoff, and <5ms Toxic Sweep Shield."""
    algo = DefensiveMakerExecutionAlgorithm(async_hawkes_refit=False)
    inst = "BTCUSDT-PERP"
    t0_ns = 1_700_000_000_000_000_000

    # 1. Initial normal placement: VPIN=0.30 <= 0.60, eta=0.40 <= 0.70 -> PLACE_GTX_MAKER at best_bid
    cmd1 = algo.evaluate_order_intent(
        instrument_id=inst,
        ts_ns=t0_ns,
        side="BUY",
        quantity=1.5,
        best_bid=10_000.0,
        best_ask=10_001.0,
        tick_size=0.5,
        vpin=0.30,
        hawkes_eta=0.40,
    )
    assert cmd1.action == ExecutionAction.PLACE_GTX_MAKER
    assert cmd1.post_only_gtx is True
    assert cmd1.target_price == 10_000.0

    # 2. BBO shift < 1.5 bps (e.g. +1.0 bps to 10_001.0) even after 600ms -> HOLD_RESTING_HYSTERESIS
    cmd2 = algo.evaluate_order_intent(
        instrument_id=inst,
        ts_ns=t0_ns + 600_000_000,  # 600 ms (> 500 ms)
        side="BUY",
        quantity=1.5,
        best_bid=10_001.0,  # +1.0 bps (< 1.5 bps threshold)
        best_ask=10_002.0,
        tick_size=0.5,
        vpin=0.30,
        hawkes_eta=0.40,
    )
    assert cmd2.action == ExecutionAction.HOLD_RESTING_HYSTERESIS

    # 3. BBO shift >= 1.5 bps (e.g. +2.0 bps to 10_002.0) but elapsed time = 300ms (< 500ms) -> HOLD_RESTING_HYSTERESIS
    cmd3 = algo.evaluate_order_intent(
        instrument_id=inst,
        ts_ns=t0_ns + 300_000_000,  # 300 ms (< 500 ms cooldown)
        side="BUY",
        quantity=1.5,
        best_bid=10_002.0,  # +2.0 bps
        best_ask=10_003.0,
        tick_size=0.5,
        vpin=0.30,
        hawkes_eta=0.40,
    )
    assert cmd3.action == ExecutionAction.HOLD_RESTING_HYSTERESIS

    # 4. BOTH BBO shift >= 1.5 bps (+2.0 bps) AND elapsed >= 500ms (600ms) -> AMEND_GTX_MAKER
    cmd4 = algo.evaluate_order_intent(
        instrument_id=inst,
        ts_ns=t0_ns + 600_000_000,
        side="BUY",
        quantity=1.5,
        best_bid=10_002.0,
        best_ask=10_003.0,
        tick_size=0.5,
        vpin=0.30,
        hawkes_eta=0.40,
    )
    assert cmd4.action == ExecutionAction.AMEND_GTX_MAKER
    assert cmd4.target_price == 10_002.0

    # 5. 3 consecutive GTX rejections -> 4th attempt placed 1 tick deeper (best_bid - tick_size) + 1000ms wait
    algo.on_order_rejected(inst)
    algo.on_order_rejected(inst)
    rej_count = algo.on_order_rejected(inst)
    assert rej_count == 3

    cmd5 = algo.evaluate_order_intent(
        instrument_id=inst,
        ts_ns=t0_ns + 700_000_000,
        side="BUY",
        quantity=1.5,
        best_bid=10_002.0,
        best_ask=10_003.0,
        tick_size=0.5,
        vpin=0.30,
        hawkes_eta=0.40,
    )
    assert cmd5.action == ExecutionAction.BACKOFF_DEEPER_GTX
    assert cmd5.target_price == 10_002.0 - 0.5

    # 6. Toxic Sweep Shield (VPIN > 0.60 or eta_Hawkes > 0.70) cancels resting orders in < 5ms
    t_cancel_start = time.perf_counter()
    cmd6 = algo.evaluate_order_intent(
        instrument_id=inst,
        ts_ns=t0_ns + 900_000_000,
        side="BUY",
        quantity=1.5,
        best_bid=10_002.0,
        best_ask=10_003.0,
        tick_size=0.5,
        vpin=0.68,  # > 0.60 toxic threshold
        hawkes_eta=0.50,
    )
    cancel_latency_ms = (time.perf_counter() - t_cancel_start) * 1000.0
    assert cmd6.action == ExecutionAction.CANCEL_ALL_TOXIC_SHIELD
    assert cancel_latency_ms < 5.0
    assert algo.get_resting_order(inst) is None

    # 7. Hard stop-loss breach triggers immediate IOC Taker exit
    cmd7 = algo.evaluate_order_intent(
        instrument_id=inst,
        ts_ns=t0_ns + 1_000_000_000,
        side="SELL",
        quantity=1.5,
        best_bid=9_900.0,
        best_ask=9_901.0,
        tick_size=0.5,
        hard_stop_breached=True,
    )
    assert cmd7.action == ExecutionAction.EMERGENCY_TAKER_EXIT
    assert cmd7.post_only_gtx is False
    assert cmd7.time_in_force == "IOC"


# ==============================================================================
# 3. SIGNED HRP-CVAR, FEE ACCOUNTING, TRIPLE-BARRIER & CONTINUOUS-ONLY SVD
# ==============================================================================


def test_signed_hrp_cvar_and_fee_accounting_rules() -> None:
    """Verify w_final preserves sign of y_primary, caps gross leverage <= 2.5, and rejects <= 2*Maker Fee."""
    rng = np.random.default_rng(99)
    returns_matrix = rng.normal(0.0005, 0.018, size=(200, 4))
    w_hrp = compute_hrp_cvar_weights(returns_matrix)
    assert w_hrp.shape == (4,)
    assert np.all(w_hrp > 0.0)
    assert abs(float(np.sum(w_hrp)) - 1.0) <= 1e-9

    usdt_fees = FeeSchedule.usdt_perp()  # 2.0 bps Maker -> 2 * Maker = 4.0 bps round-trip hurdle
    inputs = [
        AssetAllocationInput(
            asset_id="BTCUSDT-PERP",
            y_primary=1,
            w_hrp=float(w_hrp[0]),
            p_meta=0.85,
            expected_gross_return_bps=120.0,  # > 4.0 bps -> accepted
            fee_schedule=usdt_fees,
        ),
        AssetAllocationInput(
            asset_id="ETHUSDT-PERP",
            y_primary=-1,
            w_hrp=float(w_hrp[1]),
            p_meta=0.80,
            expected_gross_return_bps=95.0,  # > 4.0 bps -> accepted
            fee_schedule=usdt_fees,
        ),
        AssetAllocationInput(
            asset_id="SOLUSDT-PERP",
            y_primary=1,
            w_hrp=float(w_hrp[2]),
            p_meta=0.90,
            expected_gross_return_bps=3.5,  # <= 4.0 bps (2 * Maker Fee) -> MUST BE REJECTED
            fee_schedule=usdt_fees,
        ),
        AssetAllocationInput(
            asset_id="AVAXUSDT-PERP",
            y_primary=0,
            w_hrp=float(w_hrp[3]),
            p_meta=0.75,
            expected_gross_return_bps=110.0,
            fee_schedule=usdt_fees,
        ),
    ]

    alloc = compute_signed_target_weights(inputs, regime_multiplier=0.9)
    assert alloc.weights["BTCUSDT-PERP"] > 0.0  # preserves +1 sign
    assert alloc.weights["ETHUSDT-PERP"] < 0.0  # preserves -1 sign
    assert alloc.weights["SOLUSDT-PERP"] == 0.0  # rejected due to gross return <= 2 * Maker Fee
    assert "SOLUSDT-PERP" in alloc.rejected_for_fees
    assert alloc.weights["AVAXUSDT-PERP"] == 0.0
    assert alloc.gross_leverage <= 2.5


@settings(max_examples=40, deadline=None)
@given(
    y_signs=st.lists(st.sampled_from([-1, 0, 1]), min_size=5, max_size=8),
    p_metas=st.lists(st.floats(min_value=0.55, max_value=1.0), min_size=5, max_size=8),
    regime_mult=st.floats(min_value=0.1, max_value=1.0),
)
def test_hypothesis_signed_weights_leverage_and_sign_invariants(
    y_signs: list[Literal[-1, 0, 1]],
    p_metas: list[float],
    regime_mult: float,
) -> None:
    """Property test: Signed target weights never exceed 2.5x gross leverage and always preserve y_primary sign."""
    m = min(len(y_signs), len(p_metas))
    inputs = [
        AssetAllocationInput(
            asset_id=f"ASSET_{i}",
            y_primary=y_signs[i],
            w_hrp=0.95,  # intentionally large before leverage scaling so unscaled sum > 2.5
            p_meta=p_metas[i],
            expected_gross_return_bps=150.0,
        )
        for i in range(m)
    ]
    out = compute_signed_target_weights(inputs, regime_multiplier=regime_mult, gamma_frac=1.0)
    assert out.gross_leverage <= 2.5 + 1e-9
    for i in range(m):
        w = out.weights[f"ASSET_{i}"]
        if y_signs[i] == 0:
            assert w == 0.0
        elif y_signs[i] == 1:
            assert w > 0.0
        else:
            assert w < 0.0


def test_triple_barrier_80bps_hurdle_and_continuous_svd_exclusion() -> None:
    """Verify Triple-Barrier >= 80 bps minimum hurdle and Stage 1 SVD exclusion of sparse binary flag."""
    tb_cfg = TripleBarrierConfig()
    # Even when sigma_bar is tiny (e.g. 5 bps), effective profit barrier is >= 80 bps (0.0080)
    assert tb_cfg.effective_profit_barrier_decimal(0.0005) >= 0.0080

    bars = _make_synthetic_dollar_bars(n_bars=90, seed=55)
    snaps = compute_positioning_snapshots(bars)
    labeler = FeeAdjustedTripleBarrierLabeler(tb_cfg)
    primary_dirs = np.array([s.primary_direction for s in snaps], dtype=np.int64)
    labels = labeler.label_bars(bars, primary_directions=primary_dirs)
    assert len(labels) == len(bars)
    for lbl in labels:
        assert lbl.upper_barrier >= 0.0080 - 1e-12

    # Verify Stage 1 SVD strictly excludes liquidation_exhaustion_flag
    assert "liquidation_exhaustion_flag" in EXCLUDED_BINARY_COLUMNS
    assert "liquidation_exhaustion_flag" not in CONTINUOUS_SIGNAL_COLUMNS

    x_cont, bin_flags, _ = extract_continuous_and_binary_from_snapshots(snaps)
    assert x_cont.shape == (len(snaps), 3)
    svd = ContinuousSVDOrthogonalizer().fit(x_cont[:60])
    # Verify dominant loading of every right singular vector is strictly positive
    for col_idx in range(svd.right_singular_vectors_.shape[1]):
        col_vec = svd.right_singular_vectors_[:, col_idx]
        dom_idx = int(np.argmax(np.abs(col_vec)))
        assert col_vec[dom_idx] > 0.0

    # End-to-end Stage 2 LightGBMMetaLabelWorker training & Mahalanobis epistemic gate blocking
    meta_targets = np.array([lbl.meta_label for lbl in labels[:60]], dtype=np.int64)
    if np.unique(meta_targets).size < 2:
        meta_targets[:10] = 1
        meta_targets[10:20] = 0
    worker = LightGBMMetaLabelWorker(di_threshold=4.0, n_jobs=4, use_xgboost_ensemble=True)
    worker.fit(x_cont[:60], meta_targets, liquidation_exhaustion_flag_train=bin_flags[:60])

    normal_pred = worker.predict_single(
        instrument_id="BTCUSDT-PERP",
        ts_close_ns=bars[65].ts_close_ns,
        x_continuous_row=x_cont[65],
        liquidation_exhaustion_flag=int(bin_flags[65]),
        primary_direction=1,
    )
    assert 0.0 <= normal_pred.p_meta <= 1.0

    # Inject extreme out-of-distribution feature vector -> Mahalanobis gate MUST block entry
    extreme_outlier_row = x_cont[65] + 100.0
    blocked_pred = worker.predict_single(
        instrument_id="BTCUSDT-PERP",
        ts_close_ns=bars[65].ts_close_ns,
        x_continuous_row=extreme_outlier_row,
        liquidation_exhaustion_flag=0,
        primary_direction=1,
    )
    assert blocked_pred.mahalanobis_di > blocked_pred.di_threshold
    assert blocked_pred.epistemic_gate_passed is False
    assert blocked_pred.effective_direction == 0

    # Verify FundingCarrySleeve positive delta-neutral carry & negative funding ASRI gate
    carry_sleeve = FundingCarrySleeve()
    pos_carry = carry_sleeve.evaluate_opportunity(
        instrument_id="BTCUSDC-PERP",
        funding_rate_8h=0.0006,  # +6 bps / 8h
        basis_pct=0.001,
        fee_schedule=FeeSchedule.usdc_perp(),
        asri_score=25.0,
    )
    assert pos_carry.is_actionable is True
    assert pos_carry.net_delta_weight == 0.0
    assert pos_carry.spot_target_weight > 0.0
    assert pos_carry.perp_target_weight < 0.0

    # Negative funding blocked when ASRI > 40.0
    neg_carry_blocked = carry_sleeve.evaluate_opportunity(
        instrument_id="BTCUSDC-PERP",
        funding_rate_8h=-0.0008,
        fee_schedule=FeeSchedule.usdc_perp(),
        asri_score=65.0,  # > 40.0 threshold
    )
    assert neg_carry_blocked.is_actionable is False


# ==============================================================================
# 4. 45-SPLIT CPCV & CHAMPION/CHALLENGER FALLBACK STATE MACHINE
# ==============================================================================


def test_cpcv_45_splits_9_paths_and_challenger_fallback_state_machine() -> None:
    """Verify 45-split CPCV zero label overlap and prove Challenger PSR=0.82 (<0.95) triggers rejection/fallback."""
    cpcv = CombinatorialPurgedCV(vertical_barrier_bars=16, n_groups=10, k_test_groups=2, embargo_pct=0.01)
    assert cpcv.num_splits == 45
    assert cpcv.n_oos_paths == 9
    assert cpcv.purge_bars == 16

    splits = cpcv.generate_splits(n_samples=600)
    assert len(splits) == 45
    assert cpcv.verify_zero_label_overlap(splits, label_horizon_bars=16) is True

    # Initialize Gatekeeper with an active Champion
    gatekeeper = ChampionChallengerGatekeeper(
        asset_id="BTCUSDT-PERP",
        initial_champion_id="champ_v1",
        initial_live_30d_psr=0.97,
    )
    assert gatekeeper.state.regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY
    assert gatekeeper.state.champion_model_id == "champ_v1"

    # Inject a Challenger with PSR = 0.82 (< 0.95) -> must be rejected and preserve Champion
    bad_challenger = CPCVValidationResult(
        model_id="challenger_psr_082",
        purge_bars=16,
        psr_vs_zero=0.82,  # < 0.95 gate failure!
        psr_vs_benchmark=0.96,
        dsr=0.96,
        pbo=0.18,
        expected_net_pnl_bps=24.0,
        sharpe_net=1.4,
        skewness=0.1,
        kurtosis=3.2,
        total_trials=45,
    )
    assert bad_challenger.passed_all_gates is False

    state_after_1 = gatekeeper.evaluate_challenger(bad_challenger)
    assert state_after_1.champion_model_id == "champ_v1"
    assert state_after_1.consecutive_failed_retrains == 1
    assert state_after_1.regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY

    # 3 consecutive failed retrains -> automatically transitions to CARRY_ONLY_FALLBACK
    gatekeeper.evaluate_challenger(bad_challenger)
    state_after_3 = gatekeeper.evaluate_challenger(bad_challenger)
    assert state_after_3.consecutive_failed_retrains == 3
    assert state_after_3.regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK

    # Promoting a valid Challenger recovers DIRECTIONAL_PLUS_CARRY
    good_challenger = CPCVValidationResult(
        model_id="challenger_v2_strong",
        purge_bars=16,
        psr_vs_zero=0.98,
        psr_vs_benchmark=0.97,
        dsr=0.96,
        pbo=0.12,
        expected_net_pnl_bps=32.5,
        sharpe_net=2.4,
        skewness=0.2,
        kurtosis=3.1,
        total_trials=45,
    )
    assert good_challenger.passed_all_gates is True
    state_promoted = gatekeeper.evaluate_challenger(good_challenger)
    assert state_promoted.champion_model_id == "challenger_v2_strong"
    assert state_promoted.consecutive_failed_retrains == 0
    assert state_promoted.regime_mode == ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY

    # Live 30d PSR dropping below 0.80 transitions to CARRY_ONLY_FALLBACK
    state_degraded = gatekeeper.update_live_30d_psr(live_30d_psr=0.74)
    assert state_degraded.regime_mode == ExecutionRegimeMode.CARRY_ONLY_FALLBACK


# ==============================================================================
# 5. HARDWARE BOOTSTRAP, MLX RISK DAEMON, FUNDING CARRY & WEB UI / DOCS
# ==============================================================================


def test_hardware_bootstrap_and_mlx_risk_daemon_schema_fallback() -> None:
    """Verify M1 Max 64GB bootstrap, MLX singleton, Pydantic v2 validation, and ASRI multiplier."""
    prod_cfg = load_environment_config("prod_m1max_64gb")
    mock_cfg = load_environment_config("ci_mock")
    assert prod_cfg.wired_limit_mb == 40960
    assert mock_cfg.is_mock is True

    status = bootstrap_m1max_hardware(mock_cfg)
    assert status.start_method == "spawn"
    assert status.omp_threads == 4
    assert status.polars_threads == 4

    # Persistent MLX Risk Daemon singleton + malformed JSON neutral fallback
    daemon1 = PersistentMLXRiskDaemon.get_instance(mock_cfg)
    daemon2 = PersistentMLXRiskDaemon.get_instance(mock_cfg)
    assert daemon1 is daemon2

    fallback_out = parse_and_validate_llm_risk_json("{malformed_json: true, extra: 123")
    assert fallback_out.score == 0.0
    assert fallback_out.confidence == 0.0
    assert fallback_out.tail_risk_flag is False

    valid_out = parse_and_validate_llm_risk_json(
        '{"score": 82.0, "confidence": 0.91, "tail_risk_flag": false, "plain_english_summary": "Elevated regulatory headline risk."}'
    )
    assert valid_out.score == 82.0
    m_regime = compute_regime_multiplier_from_asri(valid_out.score, valid_out.tail_risk_flag)
    assert 0.0 < m_regime < 0.50  # scaled down because ASRI > 70
    assert compute_regime_multiplier_from_asri(50.0, tail_risk_flag=True) == 0.0


def test_web_ui_control_center_telegram_and_layman_docs() -> None:
    """Verify FastAPI Web UI endpoints, hover tooltips, Telegram commands, and all 4 layman's manuals."""
    app = create_app()
    client = TestClient(app)

    resp_home = client.get("/")
    assert resp_home.status_code == 200
    html = resp_home.text
    assert "Plain-English Guide" in html
    assert "Start Dry-Run" in html
    assert "Emergency Flatten All" in html

    resp_state = client.get("/api/state")
    assert resp_state.status_code == 200
    state_json = resp_state.json()
    assert "pnl_metrics" in state_json
    assert "regime_mode" in state_json

    resp_tips = client.get("/api/tooltips")
    assert resp_tips.status_code == 200
    assert "vpin_toxicity" in PLAIN_ENGLISH_TOOLTIPS

    for action in ("start_dry_run", "pause_entries", "switch_carry_only", "emergency_flatten"):
        resp_ctrl = client.post(f"/api/control/{action}")
        assert resp_ctrl.status_code == 200

    resp_reset = client.post("/api/reset", json={"starting_capital_usd": 25000.0})
    assert resp_reset.status_code == 200
    reset_json = resp_reset.json()
    assert reset_json["starting_capital_usd"] == 25000.0
    assert reset_json["net_pnl_usd"] == 0.0

    # Verify all 4 required layman documentation manuals exist and are populated
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    for doc_file in (
        "LAYMANS_ARCHITECTURE_GUIDE.md",
        "UI_AND_OPERATIONS_MANUAL.md",
        "PLAIN_ENGLISH_GLOSSARY.md",
        "TROUBLESHOOTING_AND_SAFETY.md",
    ):
        p = docs_dir / doc_file
        assert p.exists(), f"Missing documentation file: {doc_file}"
        assert len(p.read_text(encoding="utf-8")) > 500

    # Verify Telegram slash commands
    bot = PlainEnglishTelegramBot()
    for cmd in ("/status", "/profit", "/gates", "/explain", "/pause", "/flat"):
        reply = bot.handle_command(cmd)
        assert len(reply) > 20
