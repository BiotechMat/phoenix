"""P5.4 tests: the report renders for the baseline pool with shuffled drawdowns, MAE/MFE, TC, turnover and impact."""

from __future__ import annotations

import numpy as np

from research.cpcv_gates import StrategyTournament
from research.tournament_report import candidate_report_extras, render_report, tournament_report
from tests.phase0_helpers import sine_history


def test_report_renders_for_the_baseline_pool() -> None:
    bars = sine_history("BTC", 600, 100.0, amplitude_pct=0.03, period_bars=48)
    pool = StrategyTournament(candidates=[]).create_default_pool("BTCUSDT-PERP")
    extras = tournament_report(pool, bars, n_shuffles=200)
    assert set(extras) == {c.strategy_id for c in pool}
    text = render_report(extras)
    assert "MC DD p5/p50/p95" in text and all(c.strategy_id in text for c in pool)
    for e in extras.values():
        if e.shuffled_max_drawdown_p50 is not None:
            assert e.shuffled_max_drawdown_p5 <= e.shuffled_max_drawdown_p50 <= e.shuffled_max_drawdown_p95  # type: ignore[operator]
        assert e.turnover >= 0.0 and e.impact_cost >= 0.0
        assert e.transfer_coefficient is None or abs(e.transfer_coefficient - 1.0) < 1e-9


def test_turnover_counts_position_changes() -> None:
    bars = sine_history("BTC", 300, 100.0)
    pool = StrategyTournament(candidates=[]).create_default_pool("BTCUSDT-PERP")
    cand = pool[0]
    e = candidate_report_extras(cand, bars, n_shuffles=50)
    pos = cand.generate_positions(bars)
    assert e.turnover == np.sum(np.abs(np.diff(np.concatenate(([0.0], pos)))))
    assert sum(int(v["trades"]) for v in e.mae_mfe_by_regime.values()) == e.n_trades
