"""45-Split Combinatorial Purged Cross-Validation (CPCV) Gates and Automated Strategy Tournament.

Provides:
- 45-split CPCV gatekeeper with 9 reconstructed out-of-sample paths and zero label overlap.
- Automated StrategyTournament evaluating candidate trading strategies across all 45 splits.
- Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO) enforcement.
- Integration with Champion/Challenger fallback state machine.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import stats

from core.contracts import (
    CPCVValidationResult,
    ChampionChallengerState,
    DollarBar,
    ExecutionRegimeMode,
    FeeSchedule,
)
from research.validation.cpcv_gatekeeper import (
    DEFAULT_EMBARGO_PCT,
    DEFAULT_K_TEST_GROUPS,
    DEFAULT_N_GROUPS,
    DSR_PROMOTION_THRESHOLD,
    LIVE_30D_PSR_FALLBACK_THRESHOLD,
    MAX_CONSECUTIVE_FAILED_RETRAINS,
    MAX_CPCV_WORKERS,
    MIN_EXPECTED_NET_PNL_BPS,
    OMP_THREADS_PER_WORKER,
    PBO_MAX_THRESHOLD,
    PSR_PROMOTION_THRESHOLD,
    CPCVSplit,
    ChampionChallengerGatekeeper,
    CombinatorialPurgedCV,
    NetReturnBreakdown,
    compute_dsr,
    compute_expected_max_sharpe,
    compute_net_of_fee_returns,
    compute_pbo,
    compute_pbo_logits,
    compute_psr,
    evaluate_challenger_gates,
)

logger = logging.getLogger("phoenix.research.cpcv_gates")

__all__ = [
    # Re-exported CPCV primitives
    "CPCVSplit",
    "ChampionChallengerGatekeeper",
    "CombinatorialPurgedCV",
    "NetReturnBreakdown",
    "compute_dsr",
    "compute_expected_max_sharpe",
    "compute_net_of_fee_returns",
    "compute_pbo",
    "compute_pbo_logits",
    "compute_psr",
    "evaluate_challenger_gates",
    # Constants
    "DEFAULT_EMBARGO_PCT",
    "DEFAULT_K_TEST_GROUPS",
    "DEFAULT_N_GROUPS",
    "DSR_PROMOTION_THRESHOLD",
    "LIVE_30D_PSR_FALLBACK_THRESHOLD",
    "MAX_CONSECUTIVE_FAILED_RETRAINS",
    "MAX_CPCV_WORKERS",
    "MIN_EXPECTED_NET_PNL_BPS",
    "OMP_THREADS_PER_WORKER",
    "PBO_MAX_THRESHOLD",
    "PSR_PROMOTION_THRESHOLD",
    # Strategy Tournament primitives
    "StrategyCandidate",
    "TournamentCandidateResult",
    "TournamentResult",
    "StrategyTournament",
    "create_trend_strategy",
    "create_mean_reversion_strategy",
    "create_funding_carry_strategy",
    "create_volatility_breakout_strategy",
]


@dataclass
class StrategyCandidate:
    """Specification and execution logic for a tournament trading strategy contender."""

    strategy_id: str
    strategy_type: str
    description: str
    parameters: dict[str, Any]
    position_generator: Callable[[Sequence[DollarBar]], npt.NDArray[np.float64]]

    def generate_positions(self, bars: Sequence[DollarBar]) -> npt.NDArray[np.float64]:
        """Generate signed target positions in [-1.0, 1.0] across all input bars."""
        pos = self.position_generator(bars)
        arr = np.asarray(pos, dtype=np.float64).ravel()
        if arr.shape[0] != len(bars):
            raise ValueError(
                f"Position array length ({arr.shape[0]}) does not match bars count ({len(bars)}) for '{self.strategy_id}'"
            )
        return np.clip(arr, -1.0, 1.0)


@dataclass(frozen=True)
class TournamentCandidateResult:
    """Comprehensive performance and CPCV evaluation metrics for a single tournament contender."""

    strategy_id: str
    strategy_type: str
    rank: int
    cpcv_result: CPCVValidationResult
    sharpe_net: float
    dsr: float
    pbo: float
    psr_vs_zero: float
    psr_vs_benchmark: float
    expected_net_pnl_bps: float
    is_split_sharpes: list[float]
    oos_split_sharpes: list[float]
    passed_all_gates: bool
    promotion_status: Literal["PROMOTED_CHAMPION", "QUALIFIED_CONTENDER", "REJECTED_GATES_FAILED"]


@dataclass(frozen=True)
class TournamentResult:
    """Synthesized outcome of an automated multi-strategy CPCV backtest tournament."""

    tournament_id: str
    total_candidates: int
    champion: TournamentCandidateResult | None
    rankings: list[TournamentCandidateResult]
    pool_sharpe_std: float
    tournament_pbo: float
    passed_candidates_count: int
    evaluation_timestamp_ns: int
    summary: str


def _bars_to_dollar_bars(
    data: Sequence[DollarBar] | pl.DataFrame | Path | str,
) -> list[DollarBar]:
    """Convert input data into a concrete list of DollarBar instances."""
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Parquet bars file not found: {path}")
        df = pl.read_parquet(path)
        return [DollarBar(**row) for row in df.iter_rows(named=True)]
    elif isinstance(data, pl.DataFrame):
        return [DollarBar(**row) for row in data.iter_rows(named=True)]
    else:
        bar_list = list(data)
        if len(bar_list) > 0 and not isinstance(bar_list[0], DollarBar):
            raise TypeError(f"Expected sequence of DollarBar, got sequence of {type(bar_list[0])}")
        return bar_list


# ==============================================================================
# Standard Trading Strategy Candidate Archetype Builders
# ==============================================================================


def create_trend_strategy(
    strategy_id: str = "trend_momentum",
    fast_window: int = 8,
    slow_window: int = 32,
    threshold_pct: float = 0.0005,
) -> StrategyCandidate:
    """Construct an exponential moving average trend-following strategy contender."""

    def _generator(bars: Sequence[DollarBar]) -> npt.NDArray[np.float64]:
        closes = np.array([b.close for b in bars], dtype=np.float64)
        n = len(closes)
        pos = np.zeros(n, dtype=np.float64)
        if n < slow_window:
            return pos

        # Strictly causal exponential moving averages
        alpha_fast = 2.0 / (fast_window + 1.0)
        alpha_slow = 2.0 / (slow_window + 1.0)

        ema_fast = closes[0]
        ema_slow = closes[0]

        for t in range(n):
            ema_fast = alpha_fast * closes[t] + (1.0 - alpha_fast) * ema_fast
            ema_slow = alpha_slow * closes[t] + (1.0 - alpha_slow) * ema_slow

            if t >= slow_window:
                spread = (ema_fast - ema_slow) / ema_slow
                if spread > threshold_pct:
                    pos[t] = 1.0
                elif spread < -threshold_pct:
                    pos[t] = -1.0
                else:
                    pos[t] = 0.0
        return pos

    return StrategyCandidate(
        strategy_id=strategy_id,
        strategy_type="trend_following",
        description=f"EMA Momentum Crossover (fast={fast_window}, slow={slow_window}, thresh={threshold_pct*1e4:.1f}bps)",
        parameters={
            "fast_window": fast_window,
            "slow_window": slow_window,
            "threshold_pct": threshold_pct,
        },
        position_generator=_generator,
    )


def create_mean_reversion_strategy(
    strategy_id: str = "mean_reversion_zscore",
    window: int = 20,
    z_entry: float = 1.5,
    z_exit: float = 0.25,
) -> StrategyCandidate:
    """Construct a rolling z-score mean-reversion strategy contender."""

    def _generator(bars: Sequence[DollarBar]) -> npt.NDArray[np.float64]:
        closes = np.array([b.close for b in bars], dtype=np.float64)
        n = len(closes)
        pos = np.zeros(n, dtype=np.float64)
        if n < window:
            return pos

        current_pos = 0.0
        for t in range(window, n):
            window_slice = closes[t - window + 1 : t + 1]
            mean_px = float(np.mean(window_slice))
            std_px = float(np.std(window_slice, ddof=1))
            if std_px < 1e-8:
                pos[t] = current_pos
                continue

            z = (closes[t] - mean_px) / std_px
            if z <= -z_entry:
                current_pos = 1.0  # Oversold -> Long
            elif z >= z_entry:
                current_pos = -1.0  # Overbought -> Short
            elif abs(z) <= z_exit:
                current_pos = 0.0  # Mean reached -> Flat

            pos[t] = current_pos
        return pos

    return StrategyCandidate(
        strategy_id=strategy_id,
        strategy_type="mean_reversion",
        description=f"Rolling Z-Score Mean Reversion (window={window}, z_entry={z_entry}, z_exit={z_exit})",
        parameters={"window": window, "z_entry": z_entry, "z_exit": z_exit},
        position_generator=_generator,
    )


def create_funding_carry_strategy(
    strategy_id: str = "funding_carry_arbitrage",
    entry_threshold_bps: float = 0.8,
    exit_threshold_bps: float = 0.1,
) -> StrategyCandidate:
    """Construct a perpetual funding rate carry arbitrage strategy contender."""

    def _generator(bars: Sequence[DollarBar]) -> npt.NDArray[np.float64]:
        funding_8h = np.array([b.funding_rate_8h for b in bars], dtype=np.float64)
        n = len(funding_8h)
        pos = np.zeros(n, dtype=np.float64)
        current_pos = 0.0
        entry_dec = entry_threshold_bps * 1e-4
        exit_dec = exit_threshold_bps * 1e-4

        for t in range(n):
            f = funding_8h[t]
            # When funding is positive, longs pay shorts -> short perp to collect carry
            if f >= entry_dec:
                current_pos = -1.0
            # When funding is negative, shorts pay longs -> long perp to collect carry
            elif f <= -entry_dec:
                current_pos = 1.0
            # When funding neutralizes, exit position
            elif abs(f) <= exit_dec:
                current_pos = 0.0

            pos[t] = current_pos
        return pos

    return StrategyCandidate(
        strategy_id=strategy_id,
        strategy_type="funding_carry",
        description=f"Funding Carry Arbitrage (entry={entry_threshold_bps}bps, exit={exit_threshold_bps}bps)",
        parameters={
            "entry_threshold_bps": entry_threshold_bps,
            "exit_threshold_bps": exit_threshold_bps,
        },
        position_generator=_generator,
    )


def create_volatility_breakout_strategy(
    strategy_id: str = "volatility_breakout",
    window: int = 24,
    channel_mult: float = 1.25,
) -> StrategyCandidate:
    """Construct an adaptive Donchian / volatility breakout strategy contender."""

    def _generator(bars: Sequence[DollarBar]) -> npt.NDArray[np.float64]:
        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        n = len(closes)
        pos = np.zeros(n, dtype=np.float64)
        if n < window:
            return pos

        current_pos = 0.0
        for t in range(window, n):
            # Rolling window strictly causal up to t-1
            w_highs = highs[t - window : t]
            w_lows = lows[t - window : t]
            channel_high = float(np.max(w_highs))
            channel_low = float(np.min(w_lows))
            mid = 0.5 * (channel_high + channel_low)
            half_width = 0.5 * (channel_high - channel_low)

            upper_band = mid + channel_mult * half_width
            lower_band = mid - channel_mult * half_width

            if closes[t] > upper_band:
                current_pos = 1.0
            elif closes[t] < lower_band:
                current_pos = -1.0
            elif (current_pos == 1.0 and closes[t] < mid) or (current_pos == -1.0 and closes[t] > mid):
                current_pos = 0.0

            pos[t] = current_pos
        return pos

    return StrategyCandidate(
        strategy_id=strategy_id,
        strategy_type="volatility_breakout",
        description=f"Adaptive Volatility Breakout (window={window}, channel_mult={channel_mult})",
        parameters={"window": window, "channel_mult": channel_mult},
        position_generator=_generator,
    )


# ==============================================================================
# Strategy Tournament Engine
# ==============================================================================


class StrategyTournament:
    """Automated multi-strategy CPCV backtest tournament and challenger evaluation engine.

    Evaluates competing trading strategies across the 45-split Combinatorial Purged Cross-Validation
    matrix, computes multi-trial Sharpe dispersion across the entire candidate pool, calculates
    the Deflated Sharpe Ratio (DSR) penalizing selection bias across all S trials, determines the
    overall Probability of Backtest Overfitting (PBO), and enforces hard promotion gates
    prior to promoting weights or models.
    """

    def __init__(
        self,
        tournament_id: str = "phoenix_alpha_tournament",
        cpcv_engine: CombinatorialPurgedCV | None = None,
        candidates: Sequence[StrategyCandidate] | None = None,
        fee_schedule: FeeSchedule | None = None,
        vertical_barrier_bars: int = 16,
    ) -> None:
        self.tournament_id = tournament_id
        self.cpcv_engine = cpcv_engine or CombinatorialPurgedCV(
            vertical_barrier_bars=vertical_barrier_bars,
            n_groups=DEFAULT_N_GROUPS,
            k_test_groups=DEFAULT_K_TEST_GROUPS,
            embargo_pct=DEFAULT_EMBARGO_PCT,
            max_workers=MAX_CPCV_WORKERS,
        )
        self.fee_schedule = fee_schedule or FeeSchedule.hyperliquid()
        self._candidates: list[StrategyCandidate] = list(candidates) if candidates else []

    @property
    def candidates(self) -> list[StrategyCandidate]:
        """Current list of registered tournament contenders."""
        return list(self._candidates)

    def add_candidate(self, candidate: StrategyCandidate) -> None:
        """Register a new strategy candidate to compete in the tournament."""
        if any(c.strategy_id == candidate.strategy_id for c in self._candidates):
            raise ValueError(f"Candidate with strategy_id '{candidate.strategy_id}' is already registered")
        self._candidates.append(candidate)

    def create_default_pool(self, asset_id: str = "BTCUSDT-PERP") -> list[StrategyCandidate]:
        """Populate a balanced, diverse pool of candidate trading strategies."""
        default_pool = [
            create_trend_strategy(
                strategy_id=f"{asset_id}_trend_fast",
                fast_window=6,
                slow_window=20,
                threshold_pct=0.0003,
            ),
            create_trend_strategy(
                strategy_id=f"{asset_id}_trend_medium",
                fast_window=10,
                slow_window=36,
                threshold_pct=0.0006,
            ),
            create_mean_reversion_strategy(
                strategy_id=f"{asset_id}_mean_reversion_conservative",
                window=24,
                z_entry=1.75,
                z_exit=0.30,
            ),
            create_mean_reversion_strategy(
                strategy_id=f"{asset_id}_mean_reversion_aggressive",
                window=16,
                z_entry=1.25,
                z_exit=0.20,
            ),
            create_funding_carry_strategy(
                strategy_id=f"{asset_id}_funding_carry_standard",
                entry_threshold_bps=0.75,
                exit_threshold_bps=0.10,
            ),
            create_volatility_breakout_strategy(
                strategy_id=f"{asset_id}_volatility_breakout_standard",
                window=24,
                channel_mult=1.20,
            ),
        ]
        for c in default_pool:
            if not any(existing.strategy_id == c.strategy_id for existing in self._candidates):
                self._candidates.append(c)
        return self._candidates

    def run_tournament(
        self,
        bars: Sequence[DollarBar] | pl.DataFrame | Path | str,
        benchmark_returns: npt.NDArray[np.float64] | None = None,
        parallel: bool = False,
        now_ns: int | None = None,
    ) -> TournamentResult:
        """Execute full 45-split CPCV tournament across all registered contenders.

        Process:
        1. Evaluates each candidate strategy on the bars across all 45 CPCV splits.
        2. Reconstructs all 9 complete out-of-sample paths per candidate.
        3. Measures in-sample and out-of-sample Sharpe ratios across all 45 splits.
        4. Calculates the empirical cross-trial Sharpe ratio dispersion sigma_SR across all S candidates.
        5. Computes Deflated Sharpe Ratio (DSR) using total_trials=S and the candidate pool distribution.
        6. Constructs the (45, S) IS and OOS performance matrices to compute multi-strategy tournament PBO.
        7. Enforces all 4 promotion gates (PSR_0 >= 0.95, PSR_bench >= 0.95, DSR >= 0.95, PBO < 0.30, NetPnL > 15bps).
        8. Ranks all candidates and identifies the winning Champion.
        """
        bar_list = _bars_to_dollar_bars(bars)
        n_bars = len(bar_list)
        if n_bars < self.cpcv_engine.n_groups * (self.cpcv_engine.purge_bars + 4):
            raise ValueError(
                f"Insufficient bars ({n_bars}) for {self.cpcv_engine.n_groups} groups with "
                f"purge_bars={self.cpcv_engine.purge_bars}"
            )

        if not self._candidates:
            self.create_default_pool(bar_list[0].instrument_id)

        s_candidates = len(self._candidates)
        closes = np.array([b.close for b in bar_list], dtype=np.float64)
        gross_asset_returns = np.zeros(n_bars, dtype=np.float64)
        if n_bars > 1:
            gross_asset_returns[1:] = np.diff(closes) / closes[:-1]

        funding_rates = np.array([b.funding_rate_8h for b in bar_list], dtype=np.float64)

        # Collect raw evaluation metrics per candidate
        # Matrix shapes: is_sharpe_matrix (45, S), oos_sharpe_matrix (45, S)
        n_splits = self.cpcv_engine.num_splits
        is_matrix = np.empty((n_splits, s_candidates), dtype=np.float64)
        oos_matrix = np.empty((n_splits, s_candidates), dtype=np.float64)
        candidate_oos_paths: list[npt.NDArray[np.float64]] = []
        candidate_positions: list[npt.NDArray[np.float64]] = []
        candidate_pnl_bps: list[float] = []

        logger.info(
            "Starting CPCV Strategy Tournament '%s' with %d contenders across %d splits",
            self.tournament_id,
            s_candidates,
            n_splits,
        )

        for col_idx, cand in enumerate(self._candidates):
            pos = cand.generate_positions(bar_list)
            candidate_positions.append(pos)

            cpcv_res, oos_paths, _ = self.cpcv_engine.run_validation(
                model_id=cand.strategy_id,
                gross_asset_returns=gross_asset_returns,
                positions=pos,
                funding_rates_8h=funding_rates,
                benchmark_returns=benchmark_returns,
                fee_schedule=self.fee_schedule,
                total_trials=s_candidates,
                parallel=parallel,
            )

            candidate_oos_paths.append(oos_paths)
            candidate_pnl_bps.append(cpcv_res.expected_net_pnl_bps)

            # Extract per-split IS and OOS sharpes
            # Generate splits to measure split Sharpe ratios exactly
            splits = self.cpcv_engine.generate_splits(n_bars)
            breakdown = compute_net_of_fee_returns(
                gross_asset_returns=gross_asset_returns,
                positions=pos,
                funding_rates_8h=funding_rates,
                fee_schedule=self.fee_schedule,
            )
            net_rets = breakdown.net_returns

            for s in splits:
                train_slice = net_rets[s.train_indices]
                test_slice = net_rets[s.test_indices]
                std_tr = float(np.std(train_slice, ddof=1)) if train_slice.size > 2 else 0.0
                std_te = float(np.std(test_slice, ddof=1)) if test_slice.size > 2 else 0.0
                is_matrix[s.split_id, col_idx] = float(np.mean(train_slice) / std_tr) if std_tr > 1e-12 else 0.0
                oos_matrix[s.split_id, col_idx] = float(np.mean(test_slice) / std_te) if std_te > 1e-12 else 0.0

        # Multi-Trial Cross-Sectional Analysis
        # Flat OOS sharpe for each candidate across reconstructed paths
        flat_sharpes = np.empty(s_candidates, dtype=np.float64)
        for i, oos_paths in enumerate(candidate_oos_paths):
            flat_oos = np.mean(oos_paths, axis=0) if oos_paths.ndim == 2 else oos_paths.ravel()
            std_oos = float(np.std(flat_oos, ddof=1)) if flat_oos.size > 2 else 0.0
            flat_sharpes[i] = float(np.mean(flat_oos) / std_oos) if std_oos > 1e-12 else 0.0

        pool_sharpe_std = float(np.std(flat_sharpes, ddof=1)) if s_candidates > 1 else 0.0
        tournament_pbo = compute_pbo(is_matrix, oos_matrix)

        # Final gate evaluations incorporating cross-trial DSR and multi-strategy PBO
        candidate_results: list[TournamentCandidateResult] = []

        for idx, cand in enumerate(self._candidates):
            oos_paths = candidate_oos_paths[idx]
            flat_oos = np.mean(oos_paths, axis=0) if oos_paths.ndim == 2 else oos_paths.ravel()

            cpcv_res = evaluate_challenger_gates(
                model_id=cand.strategy_id,
                oos_net_returns=flat_oos,
                benchmark_returns=benchmark_returns,
                is_split_metrics=is_matrix[:, idx],
                oos_split_metrics=oos_matrix[:, idx],
                expected_net_pnl_bps=candidate_pnl_bps[idx],
                positions=candidate_positions[idx],
                purge_bars=self.cpcv_engine.purge_bars,
                embargo_pct=self.cpcv_engine.embargo_pct,
                total_trials=s_candidates,
                n_groups=self.cpcv_engine.n_groups,
                k_test_groups=self.cpcv_engine.k_test_groups,
            )

            # Re-evaluate DSR with cross-trial distribution
            adjusted_dsr = compute_dsr(
                flat_oos,
                total_trials=s_candidates,
                trial_sharpes=flat_sharpes,
            )

            # Evaluate pass criteria
            passed = (
                cpcv_res.psr_vs_zero >= PSR_PROMOTION_THRESHOLD
                and cpcv_res.psr_vs_benchmark >= PSR_PROMOTION_THRESHOLD
                and adjusted_dsr >= DSR_PROMOTION_THRESHOLD
                and tournament_pbo < PBO_MAX_THRESHOLD
                and cpcv_res.expected_net_pnl_bps > MIN_EXPECTED_NET_PNL_BPS
            )

            res = TournamentCandidateResult(
                strategy_id=cand.strategy_id,
                strategy_type=cand.strategy_type,
                rank=0,  # assigned after sorting
                cpcv_result=cpcv_res,
                sharpe_net=cpcv_res.sharpe_net,
                dsr=adjusted_dsr,
                pbo=tournament_pbo,
                psr_vs_zero=cpcv_res.psr_vs_zero,
                psr_vs_benchmark=cpcv_res.psr_vs_benchmark,
                expected_net_pnl_bps=cpcv_res.expected_net_pnl_bps,
                is_split_sharpes=is_matrix[:, idx].tolist(),
                oos_split_sharpes=oos_matrix[:, idx].tolist(),
                passed_all_gates=passed,
                promotion_status="QUALIFIED_CONTENDER" if passed else "REJECTED_GATES_FAILED",
            )
            candidate_results.append(res)

        # Ranking: qualifying contenders first, then highest DSR, then highest Sharpe
        candidate_results.sort(
            key=lambda r: (r.passed_all_gates, r.dsr, r.sharpe_net),
            reverse=True,
        )

        ranked: list[TournamentCandidateResult] = []
        champion: TournamentCandidateResult | None = None

        for rank_num, item in enumerate(candidate_results, start=1):
            is_champion = (rank_num == 1 and item.passed_all_gates)
            status: Literal["PROMOTED_CHAMPION", "QUALIFIED_CONTENDER", "REJECTED_GATES_FAILED"] = (
                "PROMOTED_CHAMPION"
                if is_champion
                else item.promotion_status
            )
            updated = TournamentCandidateResult(
                strategy_id=item.strategy_id,
                strategy_type=item.strategy_type,
                rank=rank_num,
                cpcv_result=item.cpcv_result,
                sharpe_net=item.sharpe_net,
                dsr=item.dsr,
                pbo=item.pbo,
                psr_vs_zero=item.psr_vs_zero,
                psr_vs_benchmark=item.psr_vs_benchmark,
                expected_net_pnl_bps=item.expected_net_pnl_bps,
                is_split_sharpes=item.is_split_sharpes,
                oos_split_sharpes=item.oos_split_sharpes,
                passed_all_gates=item.passed_all_gates,
                promotion_status=status,
            )
            if is_champion:
                champion = updated
            ranked.append(updated)

        passed_count = sum(1 for r in ranked if r.passed_all_gates)
        timestamp_ns = now_ns or bar_list[-1].ts_close_ns

        summary_msg = (
            f"Tournament '{self.tournament_id}' completed: {s_candidates} candidates evaluated across {n_splits} splits. "
            f"Tournament PBO={tournament_pbo:.3f}, Cross-Trial Sharpe Std={pool_sharpe_std:.3f}. "
            f"Passing candidates={passed_count}/{s_candidates}. "
            f"Champion: '{champion.strategy_id}' (DSR={champion.dsr:.3f}, SR={champion.sharpe_net:.3f})"
            if champion
            else f"Tournament '{self.tournament_id}' completed: 0/{s_candidates} candidates cleared all 4 CPCV gates. No Champion promoted."
        )

        logger.info(summary_msg)

        return TournamentResult(
            tournament_id=self.tournament_id,
            total_candidates=s_candidates,
            champion=champion,
            rankings=ranked,
            pool_sharpe_std=pool_sharpe_std,
            tournament_pbo=tournament_pbo,
            passed_candidates_count=passed_count,
            evaluation_timestamp_ns=timestamp_ns,
            summary=summary_msg,
        )

    def promote_to_gatekeeper(
        self,
        gatekeeper: ChampionChallengerGatekeeper,
        tournament_result: TournamentResult,
    ) -> ChampionChallengerState:
        """Promote the tournament Champion to the asset sleeve Gatekeeper."""
        if tournament_result.champion is None:
            # No candidate qualified -> evaluate a dummy failing result to trigger or advance fallback
            logger.warning("No tournament Champion qualified; triggering gatekeeper rejection fallback")
            failing_res = tournament_result.rankings[0].cpcv_result if tournament_result.rankings else CPCVValidationResult(
                model_id="tournament_none_qualified",
                purge_bars=self.cpcv_engine.purge_bars,
                psr_vs_zero=0.0,
                psr_vs_benchmark=0.0,
                dsr=0.0,
                pbo=1.0,
                expected_net_pnl_bps=0.0,
                sharpe_net=0.0,
                skewness=0.0,
                kurtosis=3.0,
                total_trials=max(1, tournament_result.total_candidates),
            )
            return gatekeeper.evaluate_challenger(failing_res)

        champ_res = tournament_result.champion.cpcv_result
        return gatekeeper.evaluate_challenger(
            champ_res,
            live_psr_on_promotion=tournament_result.champion.psr_vs_zero,
        )
