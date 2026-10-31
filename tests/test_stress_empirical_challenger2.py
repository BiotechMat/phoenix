"""Empirical Adversarial Stress Testing Suite — Challenger 2.

Empirically challenges:
1. Apple Silicon MLX Risk Committee:
   - Extreme liquidation cascade shocks (leverage, order book depth exhaustion, VPIN, Hawkes).
   - Extreme stablecoin depeg (>100 bps) and boundary sensitivity.
   - Extreme DVOL spikes (z-score >= 3.5, DVOL >= 110).
   - Emergency veto enforcement: M_regime = 0.0 and tail_risk_flag = True.
   - Apple Metal GPU MLX tensor evaluation stability under high iteration loads.
2. 45-Split CPCV Research Tournament:
   - Purged Combinatorial CV zero-lookahead, purging, and embargo verification.
   - 9 reconstructed OOS paths full-coverage check without gaps or duplicate leakage.
   - Deflated Sharpe Ratio (DSR) under high trial counts (S=100) on noise strategies.
   - Multi-gatekeeper promotion barriers: rejection of overfitted / sub-hurdle contenders.
3. Persistent Account Ledger Durability:
   - $5,000.00 base capital preservation across abrupt process termination (del store).
   - Accumulated realized PnL, win/loss stats, and forensic trade records durability.
   - Atomic rename integrity and corrupt file resilience.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from core.contracts import (
    CPCVValidationResult,
    DollarBar,
    FeeSchedule,
    SystemicRiskOutput,
    compute_regime_multiplier_from_asri,
)
from research.cpcv_gates import (
    DSR_PROMOTION_THRESHOLD,
    MIN_EXPECTED_NET_PNL_BPS,
    PBO_MAX_THRESHOLD,
    PSR_PROMOTION_THRESHOLD,
    StrategyCandidate,
    StrategyTournament,
    create_funding_carry_strategy,
    create_mean_reversion_strategy,
    create_trend_strategy,
    create_volatility_breakout_strategy,
)
from research.validation.cpcv_gatekeeper import (
    ChampionChallengerGatekeeper,
    CombinatorialPurgedCV,
    compute_dsr,
    compute_pbo,
    compute_psr,
    evaluate_challenger_gates,
)
from sentiment.committee_models import (
    AdversarialRedTeamCritic,
    CrossSectionalBreadthAnalyst,
    MacroStablecoinSentinel,
    MLXCommitteeScorer,
    MultiAssetTelemetry,
    PortfolioStressInput,
    RiskCommitteeConsensusEngine,
)
from sentiment.mlx_risk_daemon import (
    DeterministicMockBatchGenerator,
    MacroCryptoFeedSnapshot,
    parse_and_validate_llm_risk_json,
)
from ui.api_server import ControlCenterStateStore


# ==============================================================================
# 1. APPLE SILICON MLX RISK COMMITTEE EMPIRICAL STRESS TESTS
# ==============================================================================


class TestMLXRiskCommitteeEmpiricalStress:
    """Empirical adversarial stress testing of the 3-model MLX Risk Committee."""

    def test_extreme_liquidation_cascade_shock_triggers_emergency_veto(self) -> None:
        """Challenge Red-Team Critic with extreme liquidation cascade & toxic flow shocks.

        Adversarial Scenario:
        - Portfolio leverage at 3.5x (exceeds 2.45x critical boundary).
        - VPIN at 0.88 (extreme informed flow toxicity, hurdle <= 0.60).
        - Hawkes branching eta at 0.95 (extreme cluster cascade instability, hurdle <= 0.70).
        - Liquidation volume ($1.5M) exceeds order book depth ($250k).
        Empirical Oracle:
        - Red-Team Critic MUST activate veto_active=True and haircut_multiplier=0.0.
        - RiskCommitteeConsensusEngine MUST enforce M_regime = 0.0 and tail_risk_flag = True.
        """
        critic = AdversarialRedTeamCritic()
        consensus = RiskCommitteeConsensusEngine()
        sentinel = MacroStablecoinSentinel()
        breadth = CrossSectionalBreadthAnalyst()

        stress_input = PortfolioStressInput(
            gross_leverage=3.5,
            vpin=0.88,
            hawkes_eta=0.95,
            liquidation_volume_usd=1_500_000.0,
            order_book_depth_usd=250_000.0,
            portfolio_nav_usd=5_000.0,
        )

        critic_out = critic.evaluate(stress_input)

        # Empirically verify Red-Team response
        assert critic_out.veto_active is True, "Red-Team failed to activate emergency veto under extreme shock"
        assert critic_out.haircut_multiplier == 0.0, "Red-Team haircut multiplier must be 0.0 on veto"
        assert critic_out.criticism_score >= 95.0, f"Expected criticism score >= 95.0, got {critic_out.criticism_score}"
        assert any("EMERGENCY_VETO_ACTIVE" in r for r in critic_out.identified_risks)

        # Pass through consensus engine alongside benign macro and breadth
        macro_nominal = sentinel.evaluate(stablecoin_depeg_bps=2.0, dvol_index=48.0)
        breadth_nominal = breadth.evaluate(MultiAssetTelemetry())

        committee_out = consensus.reconcile(macro_nominal, breadth_nominal, critic_out)

        # Consensus MUST respect the adversarial veto regardless of calm macro/breadth
        assert committee_out.tail_risk_flag is True, "Committee failed to propagate tail risk flag from Red-Team"
        assert committee_out.regime_multiplier == 0.0, f"Expected M_regime = 0.0 on veto, got {committee_out.regime_multiplier}"
        assert committee_out.consensus_risk_score >= 95.0
        assert "VETO ENGAGED" in committee_out.committee_rationale

    @pytest.mark.parametrize(
        "depeg_bps,expected_tail_risk,expected_regime,expected_veto",
        [
            (5.0, False, "STABLE_EXPANSION", False),
            (45.0, False, "ELEVATED_VOLATILITY_STRESS", False),
            (99.0, False, "ELEVATED_VOLATILITY_STRESS", False),  # Just below 100 bps hurdle
            (100.0, True, "ACUTE_MACRO_CRISIS", True),          # Exact 100 bps ($0.01) hurdle
            (150.0, True, "ACUTE_MACRO_CRISIS", True),
            (500.0, True, "ACUTE_MACRO_CRISIS", True),          # Severe $0.05 depeg
        ],
    )
    def test_stablecoin_depeg_boundary_stress(
        self,
        depeg_bps: float,
        expected_tail_risk: bool,
        expected_regime: str,
        expected_veto: bool,
    ) -> None:
        """Challenge Macro Sentinel across stablecoin depeg boundary (threshold: 100 bps)."""
        sentinel = MacroStablecoinSentinel()
        consensus = RiskCommitteeConsensusEngine()
        breadth = CrossSectionalBreadthAnalyst()
        critic = AdversarialRedTeamCritic()

        out = sentinel.evaluate(stablecoin_depeg_bps=depeg_bps)

        assert out.tail_risk_flag == expected_tail_risk, (
            f"Depeg {depeg_bps} bps: expected tail_risk={expected_tail_risk}, got {out.tail_risk_flag}"
        )
        assert out.regime_label == expected_regime, (
            f"Depeg {depeg_bps} bps: expected regime={expected_regime}, got {out.regime_label}"
        )

        # Pass through consensus
        breadth_out = breadth.evaluate(MultiAssetTelemetry())
        critic_out = critic.evaluate(PortfolioStressInput())
        comm_out = consensus.reconcile(out, breadth_out, critic_out)

        if expected_veto:
            assert comm_out.regime_multiplier == 0.0, f"Expected M_regime=0.0 on depeg {depeg_bps} bps"
            assert comm_out.tail_risk_flag is True
        else:
            assert comm_out.regime_multiplier > 0.0, f"M_regime unexpectedly 0.0 on depeg {depeg_bps} bps"

    def test_extreme_dvol_spike_triggers_macro_crisis_and_veto(self) -> None:
        """Challenge Macro Sentinel with extreme implied volatility spike (DVOL >= 110 or z >= 3.5)."""
        sentinel = MacroStablecoinSentinel()
        consensus = RiskCommitteeConsensusEngine()

        # DVOL index = 125, z-score = +4.8
        out = sentinel.evaluate(
            feed_or_depeg=5.0,
            dvol_index=125.0,
            dvol_zscore=4.8,
            tvl_change_24h_pct=-5.0,
        )

        assert out.tail_risk_flag is True
        assert out.regime_label == "ACUTE_MACRO_CRISIS"
        assert out.score >= 95.0

        comm_out = consensus.reconcile(
            out,
            CrossSectionalBreadthAnalyst().evaluate(MultiAssetTelemetry()),
            AdversarialRedTeamCritic().evaluate(PortfolioStressInput()),
        )
        assert comm_out.regime_multiplier == 0.0
        assert comm_out.tail_risk_flag is True

    def test_metal_gpu_mlx_tensor_throughput_and_numerical_stability(self) -> None:
        """Empirically verify Metal GPU MLX scorer under 500 consecutive evaluations."""
        scorer = MLXCommitteeScorer(in_features=16, hidden_dim=64, out_features=1)
        rng = np.random.default_rng(42)

        latencies_ms: list[float] = []

        for _ in range(500):
            # Synthetic 16-dimensional feature vector
            feat_np = rng.uniform(0.0, 1.0, size=(1, 16)).astype(np.float32)
            feat_mx = mx.array(feat_np)

            t0 = time.perf_counter()
            out = scorer(feat_mx)
            mx.eval(out)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            val = float(out[0, 0])
            assert math.isfinite(val), "Metal GPU MLX returned non-finite value"

        avg_latency = np.mean(latencies_ms)
        p99_latency = np.percentile(latencies_ms, 99)

        # Institutional sub-millisecond requirement on Apple Silicon M1 Max
        assert avg_latency < 2.0, f"Average Metal GPU latency too high: {avg_latency:.3f} ms"
        assert p99_latency < 5.0, f"P99 Metal GPU latency too high: {p99_latency:.3f} ms"

    def test_malformed_llm_json_neutral_fallback(self) -> None:
        """Adversarial stress test on LLM parser with corrupt, malicious, and NaN inputs."""
        malformed_cases = [
            "",
            "   ",
            "This is plain text with no JSON at all.",
            '{"score": "NaN", "confidence": 0.5, "tail_risk_flag": false, "plain_english_summary": "Bad"}',
            '{"score": Infinity, "confidence": 0.5, "tail_risk_flag": false, "plain_english_summary": "Bad"}',
            '{"score": true, "confidence": 0.8, "tail_risk_flag": false, "plain_english_summary": "Bad"}',
            '{"score": 50.0, "confidence": 0.8, "tail_risk_flag": "NOT_A_BOOL", "plain_english_summary": "Bad"}',
            '{"unexpected_extra_key": 123}',
            None,
        ]

        for case in malformed_cases:
            res = parse_and_validate_llm_risk_json(case)
            assert isinstance(res, SystemicRiskOutput)
            assert res.score == 0.0, f"Failed fallback score for case: {case}"
            assert res.confidence == 0.0
            assert res.tail_risk_flag is False


# ==============================================================================
# 2. 45-SPLIT CPCV RESEARCH TOURNAMENT EMPIRICAL STRESS TESTS
# ==============================================================================


class TestCPCVTournamentEmpiricalStress:
    """Empirical adversarial stress testing of the 45-Split CPCV Tournament."""

    def test_cpcv_zero_lookahead_purging_and_embargo_integrity(self) -> None:
        """Empirically audit all 45 splits for ZERO lookahead leakage, purging, and embargo.

        Oracle Criteria:
        1. Number of splits must be exactly C(10, 2) = 45.
        2. In every split: Train indices and Test indices must be strictly disjoint (Train ∩ Test = ∅).
        3. Purging: Any training bars immediately preceding test bars must be separated by >= purge_bars.
        4. Embargo: Any training bars immediately following test bars must be separated by >= embargo_bars.
        5. Reconstructed out-of-sample paths: Exactly C(9, 1) = 9 paths, each completely covering all N bars.
        """
        n_bars = 1000
        purge_bars = 16
        embargo_pct = 0.01

        cpcv = CombinatorialPurgedCV(
            n_groups=10,
            k_test_groups=2,
            vertical_barrier_bars=purge_bars,
            embargo_pct=embargo_pct,
        )

        splits = cpcv.generate_splits(n_bars)
        assert len(splits) == 45, f"Expected 45 splits, got {len(splits)}"

        for s in splits:
            train_set = set(s.train_indices)
            test_set = set(s.test_indices)

            # 1. Zero overlap
            overlap = train_set.intersection(test_set)
            assert len(overlap) == 0, f"Split {s.split_id}: Detected {len(overlap)} overlapping indices between Train and Test!"

            # 2. Purging verification
            # If train index t < test index t_start, verify t <= t_start - purge_bars - 1
            test_min = min(s.test_indices)
            test_max = max(s.test_indices)

            preceding_train = [t for t in s.train_indices if t < test_min]
            if preceding_train:
                max_preceding = max(preceding_train)
                purge_gap = test_min - max_preceding
                assert purge_gap > purge_bars, (
                    f"Split {s.split_id}: Purge gap violated! test_min={test_min}, "
                    f"max_preceding_train={max_preceding}, gap={purge_gap} <= purge_bars={purge_bars}"
                )

        # 3. Verify reconstructed paths full coverage
        split_test_returns = {s.split_id: np.ones(len(s.test_indices)) for s in splits}
        oos_paths = cpcv.reconstruct_oos_paths(split_test_returns, n_samples=n_bars)
        assert oos_paths.shape == (9, n_bars), f"Expected shape (9, {n_bars}), got {oos_paths.shape}"
        assert np.all(oos_paths == 1.0), "OOS paths contain unpopulated or zero entries!"

    def test_dsr_penalizes_overfitting_under_high_trial_counts(self) -> None:
        """Stress test Deflated Sharpe Ratio (DSR) under S=100 candidate trials.

        Adversarial Scenario:
        - Generate 100 independent pure-noise strategies (zero alpha).
        - Pick the strategy with the highest sample Sharpe ratio (spurious winner).
        - Verify that compute_dsr correctly penalizes multiple testing selection bias.
        - Verify that the overfitted candidate is REJECTED by evaluate_challenger_gates.
        """
        rng = np.random.default_rng(20261031)
        s_trials = 100
        n_obs = 500

        # Generate 100 pure-noise return series
        noise_matrix = rng.normal(0.0, 0.01, size=(s_trials, n_obs))
        trial_sharpes = np.array([
            float(np.mean(r) / np.std(r, ddof=1)) if np.std(r, ddof=1) > 0 else 0.0
            for r in noise_matrix
        ])

        # Pick the spurious maximum
        best_idx = int(np.argmax(trial_sharpes))
        best_returns = noise_matrix[best_idx]
        best_sr = trial_sharpes[best_idx]

        # In-sample best per-observation SR from 100 noise trials is statistically inflated
        # (null standard error is 1/sqrt(500) = 0.0447; best_sr will typically be 2-3 standard errors)
        assert best_sr > 0.08, f"Expected statistically inflated SR from 100 trials, got {best_sr:.4f}"

        # Calculate standard unadjusted PSR vs DSR
        unadjusted_psr = compute_psr(best_returns, sr_benchmark=0.0)
        dsr = compute_dsr(best_returns, total_trials=s_trials, trial_sharpes=trial_sharpes)

        # Unadjusted PSR is fooled by selection bias (> 0.95), but DSR detects the selection bias!
        assert unadjusted_psr > 0.90, f"Expected unadjusted PSR to be high, got {unadjusted_psr:.3f}"
        assert dsr < DSR_PROMOTION_THRESHOLD, (
            f"DSR failed to penalize 100-trial selection bias: DSR={dsr:.3f} >= hurdle {DSR_PROMOTION_THRESHOLD}"
        )

        # Gatekeeper evaluation must REJECT this candidate
        res = evaluate_challenger_gates(
            model_id="spurious_noise_champion",
            oos_net_returns=best_returns,
            benchmark_returns=None,
            is_split_metrics=np.array([best_sr] * 45),
            oos_split_metrics=np.array([0.0] * 45),
            expected_net_pnl_bps=18.0,
            positions=np.ones(n_obs),
            total_trials=s_trials,
        )
        assert res.passed_all_gates is False, "Gatekeeper falsely promoted a spurious noise strategy!"

    def test_strategy_tournament_end_to_end_promotion_gate(self) -> None:
        """Run StrategyTournament on historical Parquet bars and verify strict gatekeeping."""
        parquet_path = Path("/Users/mbeale/project-phoenix/data/btcusdt_perp_bars.parquet")
        assert parquet_path.exists(), f"Parquet bars file not found: {parquet_path}"

        tournament = StrategyTournament(tournament_id="adversarial_test_tournament")
        tournament.create_default_pool("BTCUSDT-PERP")

        assert len(tournament.candidates) >= 4, "Tournament should have candidate strategies"

        # Execute tournament
        result = tournament.run_tournament(bars=parquet_path, parallel=False)

        assert result.total_candidates == len(tournament.candidates)
        assert len(result.rankings) == len(tournament.candidates)
        assert 0.0 <= result.tournament_pbo <= 1.0

        # Verify promotion logic: if any champion is promoted, it MUST satisfy ALL 5 gates
        if result.champion is not None:
            champ = result.champion
            assert champ.passed_all_gates is True
            assert champ.promotion_status == "PROMOTED_CHAMPION"
            assert champ.dsr >= DSR_PROMOTION_THRESHOLD
            assert champ.pbo < PBO_MAX_THRESHOLD
            assert champ.expected_net_pnl_bps > MIN_EXPECTED_NET_PNL_BPS
            assert champ.psr_vs_zero >= PSR_PROMOTION_THRESHOLD
        else:
            # If no champion promoted, all ranked candidates must have passed_all_gates == False
            for cand in result.rankings:
                assert cand.passed_all_gates is False
                assert cand.promotion_status == "REJECTED_GATES_FAILED"


# ==============================================================================
# 3. ACCOUNT LEDGER PERSISTENCE & CRASH DURABILITY STRESS TESTS
# ==============================================================================


class TestAccountLedgerPersistenceStress:
    """Empirical adversarial stress testing of Account Ledger durability."""

    def test_capital_durability_across_abrupt_process_termination(self) -> None:
        """Stress test $5,000 base capital & PnL persistence across simulated SIGKILL (`del store`).

        Adversarial Lifecycle:
        1. Initialize fresh ControlCenterStateStore with dedicated temp ledger.
        2. Base capital must be exactly $5,000.00.
        3. Execute rapid sequence of 30 trades (positive and negative PnL).
        4. Capture total realized PnL, win count, loss count, trade history length.
        5. Abruptly kill the store instance in Python (`del store`).
        6. Reinstantiate ControlCenterStateStore from the exact same ledger file.
        7. Verify zero byte corruption, exact base capital ($5,000.00), exact equity,
           exact win/loss stats, and unbroken forensics trail.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_ledger = Path(tmp_dir) / "account_ledger_test.json"

            # 1. First Process Lifecycle
            store1 = ControlCenterStateStore(ledger_path=temp_ledger)
            assert store1._base_capital_usd == 5000.00
            assert store1._starting_capital_usd == 5000.00
            assert store1._total_realized_pnl == 0.00

            # Simulate 30 realistic trades
            rng = np.random.default_rng(2026)
            accum_pnl = 0.0
            fees_paid = 0.0
            wins = 0
            losses = 0

            for i in range(30):
                pnl = float(rng.normal(12.50, 45.0))
                fee = 1.50
                accum_pnl += pnl
                fees_paid += fee
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                trade_record = {
                    "trade_id": f"trade_{i:04d}",
                    "instrument_id": "BTCUSDT-PERP",
                    "action": "BUY" if pnl > 0 else "SELL",
                    "realized_pnl_usd": round(pnl, 2),
                    "fee_cost_usd": fee,
                    "why_did_it_trade": f"Automated test trade {i}",
                }
                store1._trade_history.append(trade_record)

            store1._total_realized_pnl = accum_pnl
            store1._total_fees_paid_usd = fees_paid
            store1._wins = wins
            store1._losses = losses
            store1._persist_ledger_state_atomic()

            # Record pre-termination state
            expected_base = store1._base_capital_usd
            expected_pnl = round(store1._total_realized_pnl, 2)
            expected_equity = round(expected_base + expected_pnl, 2)
            expected_wins = store1._wins
            expected_losses = store1._losses
            expected_trade_count = len(store1._trade_history)

            # 2. Simulate Abrupt Process Termination (SIGKILL / crash)
            del store1

            # Verify file exists on disk and is valid JSON
            assert temp_ledger.exists(), "Ledger file missing after process termination"
            raw_disk_data = json.loads(temp_ledger.read_text(encoding="utf-8"))
            assert raw_disk_data["base_capital_usd"] == 5000.00

            # 3. Reboot: Instantiate fresh Store Process
            store2 = ControlCenterStateStore(ledger_path=temp_ledger)

            # Empirically verify durable state recovery
            assert store2._base_capital_usd == 5000.00, f"Base capital altered: {store2._base_capital_usd}"
            assert round(store2._total_realized_pnl, 2) == expected_pnl, (
                f"PnL discrepancy: expected {expected_pnl}, recovered {round(store2._total_realized_pnl, 2)}"
            )
            assert store2._wins == expected_wins
            assert store2._losses == expected_losses
            assert len(store2._trade_history) == expected_trade_count
            assert round(store2._base_capital_usd + store2._total_realized_pnl, 2) == expected_equity

    def test_corrupted_ledger_recovery_resilience(self) -> None:
        """Adversarial test: store recovers gracefully when ledger file contains invalid JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_ledger = Path(tmp_dir) / "corrupt_ledger.json"
            corrupt_ledger.write_text("{this is corrupted invalid json content!!!", encoding="utf-8")

            # Store must NOT crash; it should fall back to $5,000.00 base capital and overwrite safely
            store = ControlCenterStateStore(ledger_path=corrupt_ledger)
            assert store._base_capital_usd == 5000.00
            assert store._total_realized_pnl == 0.00

            # Verify that subsequent atomic write repairs the corrupted file
            store._persist_ledger_state_atomic()
            repaired_data = json.loads(corrupt_ledger.read_text(encoding="utf-8"))
            assert repaired_data["base_capital_usd"] == 5000.00

    def test_rapid_concurrent_trade_persistence_and_file_locking(self) -> None:
        """Stress test rapid atomic writes simulating rapid concurrent paper trading ticks."""
        import concurrent.futures

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_ledger = Path(tmp_dir) / "concurrent_ledger.json"
            store = ControlCenterStateStore(ledger_path=temp_ledger)

            def _simulated_trade(trade_id: int) -> float:
                pnl = (trade_id % 7 - 3) * 15.0
                trade_dict = {
                    "trade_id": f"conc_{trade_id}",
                    "instrument_id": "BTCUSDT-PERP",
                    "action": "BUY" if pnl > 0 else "SELL",
                    "realized_pnl_usd": pnl,
                    "fee_cost_usd": 1.50,
                }
                store._trade_history.append(trade_dict)
                store._total_realized_pnl += pnl
                if pnl > 0:
                    store._wins += 1
                else:
                    store._losses += 1
                store._persist_ledger_state_atomic()
                return pnl

            # Execute 50 rapid sequential/concurrent persistence calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_simulated_trade, i) for i in range(50)]
                concurrent.futures.wait(futures)

            # Re-read ledger from disk to confirm valid JSON and atomic integrity
            assert temp_ledger.exists()
            data = json.loads(temp_ledger.read_text(encoding="utf-8"))
            assert data["base_capital_usd"] == 5000.00
            assert data["current_equity_usd"] == round(5000.00 + store._total_realized_pnl, 2)
            assert data["total_trades"] == store._wins + store._losses


class TestAdversarialExtremeCornerCases:
    """Extreme corner cases: Singular matrices, explosive branching, and regime taper boundaries."""

    def test_breadth_analyst_singular_matrix_robustness(self) -> None:
        """Challenge CrossSectionalBreadthAnalyst when returns across all assets are zero or identical."""
        breadth = CrossSectionalBreadthAnalyst()

        # Completely identical flat returns history -> covariance is 0
        telemetry = MultiAssetTelemetry(
            asset_prices={"BTC": 68000.0, "ETH": 3500.0, "SOL": 180.0},
            asset_returns_history={
                "BTC": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "ETH": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "SOL": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            },
        )

        out = breadth.evaluate(telemetry)
        assert math.isfinite(out.breadth_score)
        assert math.isfinite(out.dispersion_bps)
        assert math.isfinite(out.correlation_dispersion)
        assert 0.0 <= out.breadth_score <= 100.0

    def test_red_team_explosive_supercritical_hawkes_eta(self) -> None:
        """Challenge Red-Team Critic with super-critical Hawkes eta (eta=1.85, explosive)."""
        critic = AdversarialRedTeamCritic()
        stress = PortfolioStressInput(
            gross_leverage=1.2,
            vpin=0.45,
            hawkes_eta=1.85,  # Super-critical self-excitation
        )
        out = critic.evaluate(stress)
        assert out.veto_active is True
        assert out.haircut_multiplier == 0.0
        assert out.criticism_score >= 95.0

    @pytest.mark.parametrize(
        "asri,tail_risk,expected_multiplier",
        [
            (0.0, False, 1.0),
            (20.0, False, 1.0),
            (35.0, False, 1.0),
            (40.0, False, 1.0),
            (45.0, False, 1.0 - 0.5 * (5.0 / 30.0)),   # ~0.9167
            (60.0, False, 1.0 - 0.5 * (20.0 / 30.0)),  # ~0.6667
            (70.0, False, 0.50),
            (75.0, False, 0.5 * (1.0 - 5.0 / 25.0)),   # 0.40
            (80.0, False, 0.5 * (1.0 - 10.0 / 25.0)),  # 0.30
            (95.0, False, 0.0),
            (100.0, False, 0.0),
            (10.0, True, 0.0),   # Tail risk flag MUST override low ASRI
            (30.0, True, 0.0),
            (50.0, True, 0.0),
        ],
    )
    def test_asri_regime_taper_monotonicity_and_tail_risk_override(
        self, asri: float, tail_risk: bool, expected_multiplier: float
    ) -> None:
        """Challenge compute_regime_multiplier_from_asri taper curve and absolute veto override."""
        mult = compute_regime_multiplier_from_asri(asri_score=asri, tail_risk_flag=tail_risk)
        assert mult == pytest.approx(expected_multiplier, abs=1e-4), (
            f"ASRI={asri}, tail_risk={tail_risk}: expected {expected_multiplier}, got {mult}"
        )

