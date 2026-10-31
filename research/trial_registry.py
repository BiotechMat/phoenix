"""Trial registry and minimum backtest length (REINFORCEMENT_PLAN task P1.6).

Every CPCV run appends one JSON line: sleeve family, model id, features, parameters, window,
seed and metrics. The registry's count per sleeve family is the number of trials the Deflated
Sharpe Ratio must be corrected for, so running the same configuration twice lowers the DSR of
the second run. ``compute_min_backtest_length_years`` implements the MinBTL bound of Bailey,
Borwein, Lopez de Prado and Zhu (2014): with ``N`` trials and a target annualised Sharpe ratio
``SR``, a backtest shorter than ``MinBTL`` years cannot distinguish skill from selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Final
import uuid

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm

from core.contracts import CPCVValidationResult, FeeSchedule

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH: Final[Path] = PROJECT_ROOT / "data" / "trial_registry.jsonl"
BARS_PER_YEAR_15M: Final[float] = 365.0 * 96.0
BARS_PER_YEAR_1H: Final[float] = 365.0 * 24.0
EULER_GAMMA: Final[float] = float(np.euler_gamma)


class TrialRecord(BaseModel):
    """One logged validation run."""

    model_config = ConfigDict(frozen=True)

    trial_id: str
    recorded_at_iso: str
    sleeve_family: str
    model_id: str
    features: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    window_start_iso: str | None = None
    window_end_iso: str | None = None
    n_bars: int = Field(default=0, ge=0)
    seed: int | None = None
    total_trials_used: int = Field(default=1, ge=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    passed_all_gates: bool = False
    notes: str = ""

    @classmethod
    def from_result(
        cls,
        sleeve_family: str,
        result: CPCVValidationResult,
        *,
        features: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        n_bars: int = 0,
        seed: int | None = None,
        notes: str = "",
    ) -> TrialRecord:
        return cls(
            trial_id=f"trial_{uuid.uuid4().hex[:12]}",
            recorded_at_iso=datetime.now(timezone.utc).isoformat(),
            sleeve_family=sleeve_family,
            model_id=result.model_id,
            features=tuple(features),
            parameters=dict(parameters or {}),
            window_start_iso=window_start.isoformat() if window_start else None,
            window_end_iso=window_end.isoformat() if window_end else None,
            n_bars=n_bars,
            seed=seed,
            total_trials_used=result.total_trials,
            metrics={
                "psr_vs_zero": result.psr_vs_zero,
                "psr_vs_benchmark": result.psr_vs_benchmark,
                "dsr": result.dsr,
                "pbo": result.pbo,
                "expected_net_pnl_bps": result.expected_net_pnl_bps,
                "sharpe_net": result.sharpe_net,
                "skewness": result.skewness,
                "kurtosis": result.kurtosis,
            },
            passed_all_gates=result.passed_all_gates,
            notes=notes,
        )


class TrialRegistry:
    """Append-only JSON-lines registry of validation trials."""

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def records(self, sleeve_family: str | None = None) -> list[TrialRecord]:
        if not self._path.exists():
            return []
        out: list[TrialRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = TrialRecord(**json.loads(line))
            except Exception:
                continue
            if sleeve_family is None or rec.sleeve_family == sleeve_family:
                out.append(rec)
        return out

    def count(self, sleeve_family: str) -> int:
        return len(self.records(sleeve_family))

    def next_total_trials(self, sleeve_family: str) -> int:
        """Trial count to pass to the DSR for the run about to be made (prior trials plus this one)."""
        return self.count(sleeve_family) + 1

    def append(self, record: TrialRecord) -> TrialRecord:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record


# ------------------------------------------------------------------------------------------------
# Minimum backtest length
# ------------------------------------------------------------------------------------------------


def expected_max_sharpe_of_null_trials(n_trials: int) -> float:
    """E[max of N standard-normal Sharpe draws] ≈ (1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))."""
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    return float((1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * math.e)))


def compute_min_backtest_length_years(n_trials: int, target_sharpe_annual: float) -> float:
    """MinBTL in years: the backtest must be at least ``(E[max SR_N] / SR_target)²`` years long."""
    if target_sharpe_annual <= 0.0:
        return math.inf
    z = expected_max_sharpe_of_null_trials(n_trials)
    return float((z / target_sharpe_annual) ** 2)


def passes_min_backtest_length(
    n_bars: int,
    n_trials: int,
    target_sharpe_annual: float,
    bars_per_year: float = BARS_PER_YEAR_15M,
) -> tuple[bool, float, float]:
    """``(ok, sample_years, min_btl_years)`` for a sample of ``n_bars`` bars."""
    sample_years = float(n_bars) / float(bars_per_year) if bars_per_year > 0.0 else 0.0
    min_years = compute_min_backtest_length_years(n_trials, target_sharpe_annual)
    return sample_years >= min_years, sample_years, min_years


# ------------------------------------------------------------------------------------------------
# Registered validation
# ------------------------------------------------------------------------------------------------


def run_registered_validation(
    registry: TrialRegistry,
    sleeve_family: str,
    model_id: str,
    gross_asset_returns: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    cpcv: Any,
    funding_rates_8h: NDArray[np.float64] | float = 0.0,
    fee_schedule: FeeSchedule | None = None,
    features: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    seed: int | None = None,
    notes: str = "",
) -> tuple[CPCVValidationResult, TrialRecord]:
    """Run ``cpcv.run_validation`` with the registry's trial count for this family and log the trial.

    ``cpcv`` is a ``CombinatorialPurgedCV`` (passed in to avoid an import cycle with the gatekeeper).
    """
    total_trials = registry.next_total_trials(sleeve_family)
    result, _, _ = cpcv.run_validation(
        model_id=model_id,
        gross_asset_returns=gross_asset_returns,
        positions=positions,
        funding_rates_8h=funding_rates_8h,
        fee_schedule=fee_schedule,
        total_trials=total_trials,
    )
    record = registry.append(
        TrialRecord.from_result(
            sleeve_family,
            result,
            features=features,
            parameters=parameters,
            window_start=window_start,
            window_end=window_end,
            n_bars=int(np.asarray(gross_asset_returns).shape[0]),
            seed=seed,
            notes=notes,
        )
    )
    return result, record


__all__ = [
    "BARS_PER_YEAR_15M",
    "BARS_PER_YEAR_1H",
    "DEFAULT_REGISTRY_PATH",
    "TrialRecord",
    "TrialRegistry",
    "compute_min_backtest_length_years",
    "expected_max_sharpe_of_null_trials",
    "passes_min_backtest_length",
    "run_registered_validation",
]
