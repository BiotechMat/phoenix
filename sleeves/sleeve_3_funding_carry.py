"""Sleeve 3: Hyperliquid 1-Hour Delta-Neutral Funding Carry & Cross-Venue Arbitrage.

Captures:
1. Delta-neutral Hyperliquid Spot-Long / Perp-Short on positive 1h funding settlements.
2. Delta-neutral Hyperliquid vs. Bybit/OKX cross-venue funding rate spreads.
3. Enforces Add-Liquidity-Only (ALO) Post-Only Maker routing and Hyperliquid fee tier (1.5 bps maker / 4.5 bps taker).
4. Evaluates breakeven horizons on 1-hour settlement intervals (24 settlements/day, 8760/yr).
5. Generates standardized SleeveSignalOutput compatible with SignedHRPCVaRAllocator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import math
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.contracts import (
    AssetMarketSnapshot,
    DollarBar,
    ExecutionRegimeMode,
    FeeSchedule,
    MultiAssetMarketSnapshot,
    SettlementCurrency,
    compute_regime_multiplier_from_asri,
)

ALO_ROUTING_TAG: Final[str] = "ALO_POST_ONLY"


class CarryDirection(str, Enum):
    """Structural funding carry position configuration."""

    SPOT_LONG_PERP_SHORT = "SPOT_LONG_PERP_SHORT"
    PERP_LONG_NEGATIVE_FUNDING = "PERP_LONG_NEGATIVE_FUNDING"
    CROSS_VENUE_SPREAD_ARBITRAGE = "CROSS_VENUE_SPREAD_ARBITRAGE"
    FLAT = "FLAT"


class SleeveSignalOutput(BaseModel):
    """Standardized quantitative sleeve signal contract for portfolio allocation."""

    model_config = ConfigDict(frozen=True)

    sleeve_id: str = "funding_carry"
    target_asset: str = "BTC"
    direction: int = Field(ge=-1, le=1)  # -1, 0, 1
    meta_probability: float = Field(ge=0.0, le=1.0)
    raw_expected_return: float
    recommended_leverage: float
    routing_tag: str = ALO_ROUTING_TAG
    forensic_metadata: dict[str, Any] = Field(default_factory=dict)


class FundingCarryConfig(BaseModel):
    """Configuration for FundingCarrySleeve breakeven and risk gates."""

    model_config = ConfigDict(frozen=True)

    min_positive_funding_8h: float = Field(
        default=0.0003,
        gt=0.0,
        description="Minimum 8h positive funding rate to enter delta-neutral carry (e.g. 0.0003 = 3 bps/8h)",
    )
    min_positive_funding_1h: float = Field(
        default=0.0003 / 8.0,
        gt=0.0,
        description="Minimum 1h positive funding rate (e.g. 0.0000375 = 0.375 bps/1h)",
    )
    min_negative_funding_8h: float = Field(
        default=0.0005,
        gt=0.0,
        description="Minimum magnitude of negative 8h funding to enter perp-long carry when ASRI <= 40",
    )
    min_negative_funding_1h: float = Field(
        default=0.0005 / 8.0,
        gt=0.0,
        description="Minimum magnitude of negative 1h funding",
    )
    exit_funding_threshold_8h: float = Field(
        default=0.00005,
        ge=0.0,
        description="Hysteresis exit threshold when funding mean-reverts toward zero",
    )
    exit_funding_threshold_1h: float = Field(
        default=0.00005 / 8.0,
        ge=0.0,
        description="1h hysteresis exit threshold",
    )
    expected_holding_epochs: int = Field(
        default=9,
        ge=1,
        description="Expected holding horizon in 8-hour funding epochs (9 epochs = 72 hours)",
    )
    expected_holding_hours: int = Field(
        default=72,
        ge=1,
        description="Expected holding horizon in hours (72 hours)",
    )
    max_breakeven_epochs: float = Field(
        default=6.0,
        gt=0.0,
        description="Maximum allowable 8h funding epochs required to amortize round-trip fees",
    )
    max_breakeven_hours: float = Field(
        default=48.0,
        gt=0.0,
        description="Maximum allowable hours required to amortize round-trip fees (48h)",
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
    cross_venue_min_spread_bps: float = Field(
        default=7.0,
        ge=0.0,
        description="Minimum funding spread in bps between venues to trigger cross-venue arbitrage",
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_1h_and_8h_thresholds(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            if "min_positive_funding_8h" in d and "min_positive_funding_1h" not in d:
                d["min_positive_funding_1h"] = float(d["min_positive_funding_8h"]) / 8.0
            elif "min_positive_funding_1h" in d and "min_positive_funding_8h" not in d:
                d["min_positive_funding_8h"] = float(d["min_positive_funding_1h"]) * 8.0

            if "min_negative_funding_8h" in d and "min_negative_funding_1h" not in d:
                d["min_negative_funding_1h"] = float(d["min_negative_funding_8h"]) / 8.0
            elif "min_negative_funding_1h" in d and "min_negative_funding_8h" not in d:
                d["min_negative_funding_8h"] = float(d["min_negative_funding_1h"]) * 8.0

            if "expected_holding_epochs" in d and "expected_holding_hours" not in d:
                d["expected_holding_hours"] = int(d["expected_holding_epochs"]) * 8
            elif "expected_holding_hours" in d and "expected_holding_epochs" not in d:
                d["expected_holding_epochs"] = max(1, int(d["expected_holding_hours"]) // 8)
            return d
        return data


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
    routing_tag: str = ALO_ROUTING_TAG
    funding_rate_1h: float = 0.0


class CrossVenueCarryEvaluation(BaseModel):
    """Evaluation of cross-venue funding rate spread arbitrage (Hyperliquid vs Bybit/OKX)."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    hl_funding_rate_8h: float
    bybit_funding_rate_8h: float
    spread_8h: float
    spread_bps_8h: float
    expected_spread_gross_bps: float
    roundtrip_hurdle_bps: float
    expected_net_spread_bps: float
    is_actionable: bool
    hl_side: Literal["SELL", "BUY", "NONE"]
    bybit_side: Literal["BUY", "SELL", "NONE"]
    hl_target_weight: float
    bybit_target_weight: float
    net_delta: float = 0.0
    routing_tag: str = ALO_ROUTING_TAG
    reason: str


class FundingCarrySleeve:
    """Hyperliquid 1-Hour Delta-Neutral Funding Carry & Cross-Venue Arbitrage Sleeve."""

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

        For Hyperliquid L1 (1.5 bps Maker / 4.5 bps Taker), ALO maker execution incurs 3.0 bps roundtrip
        on perpetual, plus estimated maker/queue slippage.
        """
        perp_rt_bps = fee_schedule.roundtrip_maker_plus_slippage_bps
        if not include_spot_leg:
            return perp_rt_bps

        if spot_maker_bps is None:
            if fee_schedule.settlement == SettlementCurrency.USDC:
                resolved_spot_bps = fee_schedule.maker_bps
            else:
                resolved_spot_bps = fee_schedule.maker_bps
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

    @classmethod
    def compute_breakeven_hours(
        cls,
        funding_rate_1h: float,
        fee_schedule: FeeSchedule,
        include_spot_leg: bool = True,
        spot_maker_bps: float | None = None,
    ) -> float:
        """Compute number of 1-hour funding intervals required to break even on round-trip fees."""
        funding_bps_1h = abs(funding_rate_1h) * 1e4
        if funding_bps_1h <= 1e-9:
            return float("inf")
        rt_cost_bps = cls.compute_roundtrip_carry_cost_bps(
            fee_schedule=fee_schedule,
            include_spot_leg=include_spot_leg,
            spot_maker_bps=spot_maker_bps,
        )
        return rt_cost_bps / funding_bps_1h

    def evaluate_opportunity(
        self,
        instrument_id: str,
        funding_rate_8h: float | None = None,
        basis_pct: float = 0.0,
        fee_schedule: FeeSchedule | None = None,
        asri_score: float = 20.0,
        tail_risk_flag: bool = False,
        regime_mode: ExecutionRegimeMode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY,
        target_notional_weight: float = 0.25,
        funding_rate_1h: float | None = None,
    ) -> CarryOpportunityEvaluation:
        """Evaluate net-of-fee carry profitability, 1h settlement APR, and leg weights."""
        sched = fee_schedule or FeeSchedule.hyperliquid()

        # Resolve 1h and 8h funding rates
        if funding_rate_1h is not None and funding_rate_8h is None:
            f_1h = float(funding_rate_1h)
            f_8h = f_1h * 8.0
        elif funding_rate_8h is not None and funding_rate_1h is None:
            f_8h = float(funding_rate_8h)
            f_1h = f_8h / 8.0
        elif funding_rate_8h is not None and funding_rate_1h is not None:
            f_8h = float(funding_rate_8h)
            f_1h = float(funding_rate_1h)
        else:
            f_8h = 0.0
            f_1h = 0.0

        # Annualized APR = rate_1h * 24 * 365 = rate_8h * 3 * 365
        annualized_apr = f_8h * 3.0 * 365.0
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
                funding_rate_8h=f_8h,
                funding_rate_1h=f_1h,
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
                routing_tag=ALO_ROUTING_TAG,
            )

        # Case 1: Positive Funding -> Delta-Neutral Spot-Long / Perp-Short Carry
        pos_threshold_8h = (
            self.config.exit_funding_threshold_8h
            if current_dir == CarryDirection.SPOT_LONG_PERP_SHORT
            else self.config.min_positive_funding_8h
        )
        if f_8h >= pos_threshold_8h and f_8h > 0.0:
            rt_cost_bps = self.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=True)
            funding_bps_8h = f_8h * 1e4
            funding_bps_1h = f_1h * 1e4
            be_epochs = rt_cost_bps / max(funding_bps_8h, 1e-9)
            be_hours = be_epochs * 8.0
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
                    funding_rate_8h=f_8h,
                    funding_rate_1h=f_1h,
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
                        f"funding_1h={funding_bps_1h:.3f} bps/h ({funding_bps_8h:.2f} bps/8h), "
                        f"breakeven={be_hours:.1f}h, net_edge={net_carry_bps:.1f} bps."
                    ),
                    routing_tag=ALO_ROUTING_TAG,
                )

            self._active_positions[instrument_id] = CarryDirection.FLAT
            return CarryOpportunityEvaluation(
                instrument_id=instrument_id,
                settlement=sched.settlement,
                funding_rate_8h=f_8h,
                funding_rate_1h=f_1h,
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
                routing_tag=ALO_ROUTING_TAG,
            )

        # Case 2: Deeply Negative Funding -> Perpetual-Long Reverse Carry (ONLY when ASRI <= 40.0)
        if f_8h <= -self.config.min_negative_funding_8h:
            rt_cost_bps = self.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=False)
            funding_bps_8h = abs(f_8h) * 1e4
            be_epochs = rt_cost_bps / max(funding_bps_8h, 1e-9)
            be_hours = be_epochs * 8.0
            gross_carry_bps = float(self.config.expected_holding_epochs) * funding_bps_8h
            net_carry_bps = gross_carry_bps - rt_cost_bps

            if asri_score > self.config.max_asri_for_negative_funding:
                self._active_positions[instrument_id] = CarryDirection.FLAT
                return CarryOpportunityEvaluation(
                    instrument_id=instrument_id,
                    settlement=sched.settlement,
                    funding_rate_8h=f_8h,
                    funding_rate_1h=f_1h,
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
                    routing_tag=ALO_ROUTING_TAG,
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
                    funding_rate_8h=f_8h,
                    funding_rate_1h=f_1h,
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
                        f"funding={f_8h * 1e4:.2f} bps/8h, breakeven={be_hours:.1f}h."
                    ),
                    routing_tag=ALO_ROUTING_TAG,
                )

        # Default: Flat
        rt_default = self.compute_roundtrip_carry_cost_bps(sched, include_spot_leg=True)
        be_default = (
            rt_default / (abs(f_8h) * 1e4) if abs(f_8h) > 1e-9 else 999.0
        )
        self._active_positions[instrument_id] = CarryDirection.FLAT
        return CarryOpportunityEvaluation(
            instrument_id=instrument_id,
            settlement=sched.settlement,
            funding_rate_8h=f_8h,
            funding_rate_1h=f_1h,
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
            expected_gross_carry_bps=float(self.config.expected_holding_epochs) * abs(f_8h) * 1e4,
            expected_net_carry_bps=float(self.config.expected_holding_epochs) * abs(f_8h) * 1e4 - rt_default,
            asri_score=asri_score,
            is_actionable=False,
            reason="Funding rate within neutral band; no structural carry edge.",
            routing_tag=ALO_ROUTING_TAG,
        )

    def evaluate_cross_venue_spread(
        self,
        instrument_id: str,
        hl_funding_rate_8h: float,
        bybit_funding_rate_8h: float,
        hl_fee: FeeSchedule | None = None,
        bybit_fee: FeeSchedule | None = None,
        target_notional_weight: float = 0.25,
        expected_epochs: int = 9,
    ) -> CrossVenueCarryEvaluation:
        """Evaluate delta-neutral funding rate spread arbitrage between Hyperliquid and Bybit/OKX."""
        fee_hl = hl_fee or FeeSchedule.hyperliquid()
        fee_bybit = bybit_fee or FeeSchedule.usdt_perp()

        spread_8h = hl_funding_rate_8h - bybit_funding_rate_8h
        spread_bps = spread_8h * 1e4

        roundtrip_hurdle_bps = fee_hl.roundtrip_maker_bps + fee_bybit.roundtrip_maker_bps
        expected_spread_gross_bps = spread_bps * float(expected_epochs)
        expected_net_bps = expected_spread_gross_bps - roundtrip_hurdle_bps

        capped_weight = min(max(target_notional_weight, 0.0), self.config.max_carry_gross_leverage * 0.5)

        if spread_bps >= self.config.cross_venue_min_spread_bps and expected_net_bps > 0.0:
            # Hyperliquid rate higher -> Short HL Perp, Long Bybit Perp
            return CrossVenueCarryEvaluation(
                instrument_id=instrument_id,
                hl_funding_rate_8h=hl_funding_rate_8h,
                bybit_funding_rate_8h=bybit_funding_rate_8h,
                spread_8h=spread_8h,
                spread_bps_8h=spread_bps,
                expected_spread_gross_bps=expected_spread_gross_bps,
                roundtrip_hurdle_bps=roundtrip_hurdle_bps,
                expected_net_spread_bps=expected_net_bps,
                is_actionable=True,
                hl_side="SELL",
                bybit_side="BUY",
                hl_target_weight=-capped_weight,
                bybit_target_weight=capped_weight,
                net_delta=0.0,
                routing_tag=ALO_ROUTING_TAG,
                reason=(
                    f"Cross-venue carry active: HL funding ({hl_funding_rate_8h*1e4:.1f} bps) > "
                    f"Bybit ({bybit_funding_rate_8h*1e4:.1f} bps), spread={spread_bps:.1f} bps/8h, "
                    f"net_edge={expected_net_bps:.1f} bps."
                ),
            )
        elif spread_bps <= -self.config.cross_venue_min_spread_bps and (abs(expected_spread_gross_bps) - roundtrip_hurdle_bps) > 0.0:
            # Bybit rate higher -> Long HL Perp, Short Bybit Perp
            net_rev_bps = abs(expected_spread_gross_bps) - roundtrip_hurdle_bps
            return CrossVenueCarryEvaluation(
                instrument_id=instrument_id,
                hl_funding_rate_8h=hl_funding_rate_8h,
                bybit_funding_rate_8h=bybit_funding_rate_8h,
                spread_8h=spread_8h,
                spread_bps_8h=spread_bps,
                expected_spread_gross_bps=abs(expected_spread_gross_bps),
                roundtrip_hurdle_bps=roundtrip_hurdle_bps,
                expected_net_spread_bps=net_rev_bps,
                is_actionable=True,
                hl_side="BUY",
                bybit_side="SELL",
                hl_target_weight=capped_weight,
                bybit_target_weight=-capped_weight,
                net_delta=0.0,
                routing_tag=ALO_ROUTING_TAG,
                reason=(
                    f"Cross-venue reverse carry active: Bybit funding ({bybit_funding_rate_8h*1e4:.1f} bps) > "
                    f"HL ({hl_funding_rate_8h*1e4:.1f} bps), spread={abs(spread_bps):.1f} bps/8h."
                ),
            )
        else:
            return CrossVenueCarryEvaluation(
                instrument_id=instrument_id,
                hl_funding_rate_8h=hl_funding_rate_8h,
                bybit_funding_rate_8h=bybit_funding_rate_8h,
                spread_8h=spread_8h,
                spread_bps_8h=spread_bps,
                expected_spread_gross_bps=expected_spread_gross_bps,
                roundtrip_hurdle_bps=roundtrip_hurdle_bps,
                expected_net_spread_bps=expected_net_bps,
                is_actionable=False,
                hl_side="NONE",
                bybit_side="NONE",
                hl_target_weight=0.0,
                bybit_target_weight=0.0,
                net_delta=0.0,
                routing_tag=ALO_ROUTING_TAG,
                reason=f"Cross-venue spread {abs(spread_bps):.2f} bps/8h insufficient to clear {roundtrip_hurdle_bps:.1f} bps hurdle.",
            )

    def evaluate_from_snapshot(
        self,
        snapshot: AssetMarketSnapshot,
        fee_schedule: FeeSchedule | None = None,
        asri_score: float = 20.0,
        tail_risk_flag: bool = False,
        regime_mode: ExecutionRegimeMode = ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY,
        target_notional_weight: float = 0.25,
    ) -> CarryOpportunityEvaluation:
        """Evaluate carry opportunity directly from a live AssetMarketSnapshot."""
        return self.evaluate_opportunity(
            instrument_id=snapshot.instrument_id,
            funding_rate_8h=snapshot.funding_rate_8h_equiv,
            funding_rate_1h=snapshot.funding_rate_1h,
            basis_pct=snapshot.basis_bps * 1e-4,
            fee_schedule=fee_schedule or FeeSchedule.hyperliquid(),
            asri_score=asri_score,
            tail_risk_flag=tail_risk_flag,
            regime_mode=regime_mode,
            target_notional_weight=target_notional_weight,
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
            if "USDC" in bar.instrument_id.upper() or "HYPER" in bar.instrument_id.upper():
                resolved_sched = FeeSchedule.hyperliquid()
            else:
                resolved_sched = FeeSchedule.usdt_perp()
        else:
            resolved_sched = fee_schedule

        return self.evaluate_opportunity(
            instrument_id=bar.instrument_id,
            funding_rate_8h=bar.funding_rate_8h,
            funding_rate_1h=bar.funding_rate_1h,
            basis_pct=bar.basis,
            fee_schedule=resolved_sched,
            asri_score=asri_score,
            tail_risk_flag=tail_risk_flag,
            regime_mode=regime_mode,
            target_notional_weight=target_notional_weight,
        )

    def generate_sleeve_signal(
        self,
        evaluation: CarryOpportunityEvaluation,
    ) -> SleeveSignalOutput:
        """Convert CarryOpportunityEvaluation into a standardized SleeveSignalOutput for the allocator."""
        if not evaluation.is_actionable:
            direction = 0
            meta_prob = 0.50
            raw_ret = 0.0
            rec_lev = 0.0
        elif evaluation.direction == CarryDirection.SPOT_LONG_PERP_SHORT:
            direction = 1  # Long spot / short perp combined carry allocation
            meta_prob = min(0.99, max(0.70, 0.70 + (evaluation.expected_net_carry_bps / 100.0) * 0.25))
            raw_ret = evaluation.expected_net_carry_bps
            rec_lev = evaluation.gross_leverage_used
        elif evaluation.direction == CarryDirection.PERP_LONG_NEGATIVE_FUNDING:
            direction = 1
            meta_prob = min(0.95, max(0.65, 0.65 + (evaluation.expected_net_carry_bps / 100.0) * 0.20))
            raw_ret = evaluation.expected_net_carry_bps
            rec_lev = evaluation.gross_leverage_used
        else:
            direction = 0
            meta_prob = 0.50
            raw_ret = 0.0
            rec_lev = 0.0

        return SleeveSignalOutput(
            sleeve_id="funding_carry",
            target_asset=evaluation.instrument_id,
            direction=direction,
            meta_probability=meta_prob,
            raw_expected_return=raw_ret,
            recommended_leverage=rec_lev,
            routing_tag=ALO_ROUTING_TAG,
            forensic_metadata={
                "annualized_apr": evaluation.annualized_funding_apr,
                "breakeven_hours": evaluation.breakeven_hours,
                "roundtrip_cost_bps": evaluation.roundtrip_cost_bps,
                "direction": evaluation.direction.value,
                "reason": evaluation.reason,
            },
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
                        funding_rate_1h=r.funding_rate_1h,
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
                        routing_tag=r.routing_tag,
                    )
            return scaled_results

        return results


__all__ = [
    "ALO_ROUTING_TAG",
    "CarryDirection",
    "CarryOpportunityEvaluation",
    "CrossVenueCarryEvaluation",
    "FundingCarryConfig",
    "FundingCarrySleeve",
    "SleeveSignalOutput",
]
