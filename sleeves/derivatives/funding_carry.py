"""Delta-Neutral Perpetual Funding Carry & Basis Harvesting Sleeve.

Implements a low-turnover structural carry sleeve that harvests extreme positive 8h funding
rates via delta-neutral Spot-Long / Perpetual-Short positions, and optionally captures deeply
negative 8h funding rates via Perpetual-Longs strictly when ASRI systemic risk is low
(asri_score <= 40.0). Accounts for USDT (2.0 bps Maker) vs. USDC (0.0 bps Maker) fee tiers
and enforces minimum breakeven holding period hurdles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import math

from pydantic import BaseModel, ConfigDict, Field

from core.contracts import (
    DollarBar,
    ExecutionRegimeMode,
    FeeSchedule,
    SettlementCurrency,
     compute_regime_multiplier_from_asri,
)


class CarryDirection(str, Enum):
    """Structural funding carry position configuration."""

    SPOT_LONG_PERP_SHORT = "SPOT_LONG_PERP_SHORT"
    PERP_LONG_NEGATIVE_FUNDING = "PERP_LONG_NEGATIVE_FUNDING"
    FLAT = "FLAT"


class FundingCarryConfig(BaseModel):
    """Configuration for FundingCarrySleeve breakeven and risk gates."""

    model_config = ConfigDict(frozen=True)

    min_positive_funding_8h: float = Field(
        default=0.0003,
        gt=0.0,
        description="Minimum 8h positive funding rate to enter delta-neutral carry (e.g. 0.0003 = 3 bps/8h)",
    )
    min_negative_funding_8h: float = Field(
        default=0.0005,
        gt=0.0,
        description="Minimum magnitude of negative 8h funding to enter perp-long carry when ASRI <= 40",
    )
    exit_funding_threshold_8h: float = Field(
        default=0.00005,
        ge=0.0,
        description="Hysteresis exit threshold when funding mean-reverts toward zero",
    )
    expected_holding_epochs: int = Field(
        default=9,
        ge=1,
        description="Expected holding horizon in 8-hour funding epochs (9 epochs = 72 hours)",
    )
    max_breakeven_epochs: float = Field(
        default=6.0,
        gt=0.0,
        description="Maximum allowable 8h funding epochs required to amortize round-trip fees",
    )
    min_net_carry_edge_bps: float = Field(
        default=8.0,
        ge=0.0,
        description="Minimum net-of-fee expected carry edge in bps over the holding horizon",
    )
    max_asri_for_negative_funding: float = Field(
        default=40.0,
        ge=0.0,
        le=100.0,
        description="Hard ASRI ceiling for capturing deeply negative funding via perpetual longs",
    )
    max_asri_for_positive_carry: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Hard ASRI ceiling for delta-neutral spot-long / perp-short carry",
    )
    max_carry_gross_leverage: float = Field(
        default=1.0,
        gt=0.0,
        le=2.5,
        description="Maximum gross leverage allocated to the carry sleeve",
    )


class CarryOpportunityEvaluation(BaseModel):
    """Net-of-fee evaluation and target leg weights for a single carry instrument."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    settlement: SettlementCurrency
    funding_rate_8h: float
    annualized_funding_apr: float
    basis_pct: float
    direction: CarryDirection
    spot_target_weight: float
    perp_target_weight: float
    net_delta_weight: float
    gross_leverage_used: float
    roundtrip_cost_bps: float = Field(ge=0.0)
    breakeven_epochs_8h: float = Field(ge=0.0)
    breakeven_hours: float = Field(ge=0.0)
    expected_gross_carry_bps: float
    expected_net_carry_bps: float
    asri_score: float = Field(ge=0.0, le=100.0)
    is_actionable: bool
    reason: str


class FundingCarrySleeve:
    """Low-turnover delta-neutral funding carry and reverse-funding harvester."""

    def __init__(self, config: FundingCarryConfig | None = None) -> None:
        self.config: FundingCarryConfig = config or FundingCarryConfig()
        self._active_positions: dict[str, CarryDirection] = {}

    @property
    def active_positions(self) -> dict[str, CarryDirection]:
        """Return a copy of currently active carry positions by instrument_id."""
        return dict(self._active_positions)

    @staticmethod
    def compute_roundtrip_carry_cost_bps(
        fee_schedule: FeeSchedule,
        include_spot_leg: bool = True,
        spot_maker_bps: float | None = None,
    ) -> float:
        """Compute total round-trip execution cost in bps across perpetual and optional spot leg.

        For USDC settlement (0.0 bps Maker), the perpetual maker fee is 0.0 bps.
        For USDT settlement (2.0 bps Maker), the perpetual round-trip maker fee is 4.0 bps.
        Includes estimated maker/queue slippage per leg.
        """
        perp_rt_bps = fee_schedule.roundtrip_maker_plus_slippage_bps
        if not include_spot_leg:
            return perp_rt_bps

        if spot_maker_bps is None:
            resolved_spot_bps = (
                0.0 if fee_schedule.settlement == SettlementCurrency.USDC else fee_schedule.maker_bps
            )
        else:
            resolved_spot_bps = max(0.0, float(spot_maker_bps))

        spot_rt_bps = 2.0 * (resolved_spot_bps + fee_schedule.estimated_slippage_bps * 0.5)
        return perp_rt_bps + spot_rt_bps

    @classmethod
    def compute_breakeven_epochs(
        cls,
        funding_rate_8h: float,
        fee_schedule: FeeSchedule,
        include_spot_leg: bool = True,
        spot_maker_bps: float | None = None,
    ) -> float:
        """Compute number of 8-hour funding intervals required to break even on round-trip fees."""
        funding_bps_per_epoch = abs(funding_rate_8h) * 1e4
        if funding_bps_per_epoch <= 1e-9:
            return float("inf")
        rt_cost_bps = cls.compute_roundtrip_carry_cost_bps(
            fee_schedule=fee_schedule,
            include_spot_leg=include_spot_leg,
            spot_maker_bps=spot_maker_bps,
        )
        return rt_cost_bps / funding_bps_per_epoch

    def evaluate_opportunity(
        self,
        instrument_id: str,
        funding_rate_8h: float,
        basis_pct: float = 0.0,
        fee_schedule: FeeSchedule | None = None,
        asri_score: float = 20.0,
        tail_risk_flag: bool = False,
        regime_mode: ExecutionRegimeMode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY,
        target_notional_weight: float = 0.25,
    ) -> CarryOpportunityEvaluation:
        """Evaluate net-of-fee carry profitability, breakeven horizon, and leg weights."""
        sched = fee_schedule or FeeSchedule.usdt_perp()
        annualized_apr = funding_rate_8h * 3.0 * 365.0
        capped_weight = min(max(target_notional_weight, 0.0), self.config.max_carry_gross_leverage * 0.5)
        current_dir = self._active_positions.get(instrument_id, CarryDirection.FLAT)

        # Emergency or paused regime checks
        if (
            tail_risk_flag
            or regime_mode == ExecutionRegimeMode.EMERGENCY_FLATTENED
            or asri_score >= self.config.max_asri_for_positive_carry
        ):
            self._active_positions[instrument_id] = CarryDirection.FLAT
            return CarryOpportunityEvaluation(
                instrument_id=instrument_id,
                settlement=sched.settlement,
                funding_rate_8h=funding_rate_8h,
                annualized_funding_apr=annualized_apr,
                basis_pct=basis_pct,
                direction=CarryDirection.FLAT,
                spot_target_weight=0.0,
                perp_target_weight=0.0,
                net_delta_weight=0.0,
                gross_leverage_used=0.0,
                roundtrip_cost_bps=self.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=True),
                breakeven_epochs_8h=999.0,
                breakeven_hours=7992.0,
                expected_gross_carry_bps=0.0,
                expected_net_carry_bps=0.0,
                asri_score=asri_score,
                is_actionable=False,
                reason="Carry halted due to elevated ASRI systemic risk or emergency flatten mode.",
            )

        # Case 1: Positive Funding -> Delta-Neutral Spot-Long / Perp-Short Carry
        pos_threshold = (
            self.config.exit_funding_threshold_8h
            if current_dir == CarryDirection.SPOT_LONG_PERP_SHORT
            else self.config.min_positive_funding_8h
        )
        if funding_rate_8h >= pos_threshold and funding_rate_8h > 0.0:
            rt_cost_bps = self.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=True)
            funding_bps_8h = funding_rate_8h * 1e4
            be_epochs = rt_cost_bps / max(funding_bps_8h, 1e-9)
            be_hours = 8.0 * be_epochs
            # Positive basis convergence bonus when shorting perp at a premium to spot
            basis_capture_bps = max(basis_pct * 1e4 * 0.25, 0.0)
            gross_carry_bps = float(self.config.expected_holding_epochs) * funding_bps_8h + basis_capture_bps
            net_carry_bps = gross_carry_bps - rt_cost_bps

            is_continuing = current_dir == CarryDirection.SPOT_LONG_PERP_SHORT
            if is_continuing or (
                be_epochs <= self.config.max_breakeven_epochs
                and net_carry_bps >= self.config.min_net_carry_edge_bps
                and regime_mode != ExecutionRegimeMode.PAUSED_ENTRIES
            ):
                self._active_positions[instrument_id] = CarryDirection.SPOT_LONG_PERP_SHORT
                return CarryOpportunityEvaluation(
                    instrument_id=instrument_id,
                    settlement=sched.settlement,
                    funding_rate_8h=funding_rate_8h,
                    annualized_funding_apr=annualized_apr,
                    basis_pct=basis_pct,
                    direction=CarryDirection.SPOT_LONG_PERP_SHORT,
                    spot_target_weight=capped_weight,
                    perp_target_weight=-capped_weight,
                    net_delta_weight=0.0,
                    gross_leverage_used=2.0 * capped_weight,
                    roundtrip_cost_bps=rt_cost_bps,
                    breakeven_epochs_8h=be_epochs,
                    breakeven_hours=be_hours,
                    expected_gross_carry_bps=gross_carry_bps,
                    expected_net_carry_bps=net_carry_bps,
                    asri_score=asri_score,
                    is_actionable=True,
                    reason=(
                        f"Delta-neutral Spot-Long/Perp-Short carry active: "
                        f"funding={funding_bps_8h:.2f} bps/8h, breakeven={be_hours:.1f}h, "
                        f"net_edge={net_carry_bps:.1f} bps."
                    ),
                )

            self._active_positions[instrument_id] = CarryDirection.FLAT
            return CarryOpportunityEvaluation(
                instrument_id=instrument_id,
                settlement=sched.settlement,
                funding_rate_8h=funding_rate_8h,
                annualized_funding_apr=annualized_apr,
                basis_pct=basis_pct,
                direction=CarryDirection.FLAT,
                spot_target_weight=0.0,
                perp_target_weight=0.0,
                net_delta_weight=0.0,
                gross_leverage_used=0.0,
                roundtrip_cost_bps=rt_cost_bps,
                breakeven_epochs_8h=be_epochs,
                breakeven_hours=be_hours,
                expected_gross_carry_bps=gross_carry_bps,
                expected_net_carry_bps=net_carry_bps,
                asri_score=asri_score,
                is_actionable=False,
                reason=(
                    f"Positive funding {funding_bps_8h:.2f} bps/8h insufficient to clear "
                    f"breakeven ({be_epochs:.2f} > {self.config.max_breakeven_epochs} epochs) or fee hurdle."
                ),
            )

        # Case 2: Deeply Negative Funding -> Perpetual-Long Reverse Carry (ONLY when ASRI <= 40.0)
        if funding_rate_8h <= -self.config.min_negative_funding_8h:
            rt_cost_bps = self.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=False)
            funding_bps_8h = abs(funding_rate_8h) * 1e4
            be_epochs = rt_cost_bps / max(funding_bps_8h, 1e-9)
            be_hours = 8.0 * be_epochs
            gross_carry_bps = float(self.config.expected_holding_epochs) * funding_bps_8h
            net_carry_bps = gross_carry_bps - rt_cost_bps

            if asri_score > self.config.max_asri_for_negative_funding:
                self._active_positions[instrument_id] = CarryDirection.FLAT
                return CarryOpportunityEvaluation(
                    instrument_id=instrument_id,
                    settlement=sched.settlement,
                    funding_rate_8h=funding_rate_8h,
                    annualized_funding_apr=annualized_apr,
                    basis_pct=basis_pct,
                    direction=CarryDirection.FLAT,
                    spot_target_weight=0.0,
                    perp_target_weight=0.0,
                    net_delta_weight=0.0,
                    gross_leverage_used=0.0,
                    roundtrip_cost_bps=rt_cost_bps,
                    breakeven_epochs_8h=be_epochs,
                    breakeven_hours=be_hours,
                    expected_gross_carry_bps=gross_carry_bps,
                    expected_net_carry_bps=net_carry_bps,
                    asri_score=asri_score,
                    is_actionable=False,
                    reason=(
                        f"Negative funding perp-long blocked: ASRI={asri_score:.1f} exceeds "
                        f"low-risk threshold ({self.config.max_asri_for_negative_funding:.1f})."
                    ),
                )

            if (
                be_epochs <= self.config.max_breakeven_epochs
                and net_carry_bps >= self.config.min_net_carry_edge_bps
                and regime_mode not in (ExecutionRegimeMode.PAUSED_ENTRIES, ExecutionRegimeMode.CARRY_ONLY_FALLBACK)
            ):
                m_reg = compute_regime_multiplier_from_asri(asri_score, tail_risk_flag=tail_risk_flag)
                eff_w = capped_weight * m_reg
                self._active_positions[instrument_id] = CarryDirection.PERP_LONG_NEGATIVE_FUNDING
                return CarryOpportunityEvaluation(
                    instrument_id=instrument_id,
                    settlement=sched.settlement,
                    funding_rate_8h=funding_rate_8h,
                    annualized_funding_apr=annualized_apr,
                    basis_pct=basis_pct,
                    direction=CarryDirection.PERP_LONG_NEGATIVE_FUNDING,
                    spot_target_weight=0.0,
                    perp_target_weight=eff_w,
                    net_delta_weight=eff_w,
                    gross_leverage_used=eff_w,
                    roundtrip_cost_bps=rt_cost_bps,
                    breakeven_epochs_8h=be_epochs,
                    breakeven_hours=be_hours,
                    expected_gross_carry_bps=gross_carry_bps,
                    expected_net_carry_bps=net_carry_bps,
                    asri_score=asri_score,
                    is_actionable=True,
                    reason=(
                        f"Negative-funding Perp-Long active (low ASRI={asri_score:.1f}): "
                        f"funding={funding_rate_8h * 1e4:.2f} bps/8h, breakeven={be_hours:.1f}h."
                    ),
                )

        # Default: Flat
        rt_default = self.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=True)
        be_default = (
            rt_default / (abs(funding_rate_8h) * 1e4) if abs(funding_rate_8h) > 1e-9 else 999.0
        )
        self._active_positions[instrument_id] = CarryDirection.FLAT
        return CarryOpportunityEvaluation(
            instrument_id=instrument_id,
            settlement=sched.settlement,
            funding_rate_8h=funding_rate_8h,
            annualized_funding_apr=annualized_apr,
            basis_pct=basis_pct,
            direction=CarryDirection.FLAT,
            spot_target_weight=0.0,
            perp_target_weight=0.0,
            net_delta_weight=0.0,
            gross_leverage_used=0.0,
            roundtrip_cost_bps=rt_default,
            breakeven_epochs_8h=min(be_default, 999.0),
            breakeven_hours=min(be_default * 8.0, 7992.0),
            expected_gross_carry_bps=float(self.config.expected_holding_epochs) * abs(funding_rate_8h) * 1e4,
            expected_net_carry_bps=float(self.config.expected_holding_epochs) * abs(funding_rate_8h) * 1e4 - rt_default,
            asri_score=asri_score,
            is_actionable=False,
            reason="Funding rate within neutral band; no structural carry edge.",
        )

    def evaluate_from_bar(
        self,
        bar: DollarBar,
        fee_schedule: FeeSchedule | None = None,
        asri_score: float = 20.0,
        tail_risk_flag: bool = False,
        regime_mode: ExecutionRegimeMode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY,
        target_notional_weight: float = 0.25,
    ) -> CarryOpportunityEvaluation:
        """Evaluate carry opportunity directly from a latest DollarBar."""
        if fee_schedule is None:
            if "USDC" in bar.instrument_id.upper():
                resolved_sched = FeeSchedule.usdc_perp()
            else:
                resolved_sched = FeeSchedule.usdt_perp()
        else:
            resolved_sched = fee_schedule

        return self.evaluate_opportunity(
            instrument_id=bar.instrument_id,
            funding_rate_8h=bar.funding_rate_8h,
            basis_pct=bar.basis,
            fee_schedule=resolved_sched,
            asri_score=asri_score,
            tail_risk_flag=tail_risk_flag,
            regime_mode=regime_mode,
            target_notional_weight=target_notional_weight,
        )

    def allocate_portfolio(
        self,
        bars: Sequence[DollarBar],
        fee_schedules: Mapping[str, FeeSchedule] | None = None,
        asri_score: float = 20.0,
        tail_risk_flag: bool = False,
        regime_mode: ExecutionRegimeMode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY,
    ) -> dict[str, CarryOpportunityEvaluation]:
        """Evaluate and allocate carry capital across a universe of instruments respecting leverage limits."""
        if not bars:
            return {}

        n_assets = len(bars)
        per_asset_notional = min(0.25, self.config.max_carry_gross_leverage / float(2 * max(n_assets, 1)))
        results: dict[str, CarryOpportunityEvaluation] = {}
        for bar in bars:
            sched = fee_schedules.get(bar.instrument_id) if fee_schedules is not None else None
            eval_res = self.evaluate_from_bar(
                bar=bar,
                fee_schedule=sched,
                asri_score=asri_score,
                tail_risk_flag=tail_risk_flag,
                regime_mode=regime_mode,
                target_notional_weight=per_asset_notional,
            )
            results[bar.instrument_id] = eval_res

        total_gross = sum(r.gross_leverage_used for r in results.values() if r.is_actionable)
        if total_gross > self.config.max_carry_gross_leverage and total_gross > 0.0:
            scale = self.config.max_carry_gross_leverage / total_gross
            scaled_results: dict[str, CarryOpportunityEvaluation] = {}
            for inst_id, r in results.items():
                if not r.is_actionable:
                    scaled_results[inst_id] = r
                else:
                    scaled_results[inst_id] = CarryOpportunityEvaluation(
                        instrument_id=r.instrument_id,
                        settlement=r.settlement,
                        funding_rate_8h=r.funding_rate_8h,
                        annualized_funding_apr=r.annualized_funding_apr,
                        basis_pct=r.basis_pct,
                        direction=r.direction,
                        spot_target_weight=r.spot_target_weight * scale,
                        perp_target_weight=r.perp_target_weight * scale,
                        net_delta_weight=r.net_delta_weight * scale,
                        gross_leverage_used=r.gross_leverage_used * scale,
                        roundtrip_cost_bps=r.roundtrip_cost_bps,
                        breakeven_epochs_8h=r.breakeven_epochs_8h,
                        breakeven_hours=r.breakeven_hours,
                        expected_gross_carry_bps=r.expected_gross_carry_bps,
                        expected_net_carry_bps=r.expected_net_carry_bps,
                        asri_score=r.asri_score,
                        is_actionable=r.is_actionable,
                        reason=r.reason,
                    )
            return scaled_results

        return results
