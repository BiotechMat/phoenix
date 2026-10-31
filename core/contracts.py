"""Shared Pydantic v2 schemas, fee schedules, and governing portfolio equations."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SettlementCurrency(str, Enum):
    """Perpetual futures settlement currency determining base-tier fee tiers."""

    USDT = "USDT"
    USDC = "USDC"


class FeeSchedule(BaseModel):
    """Exchange fee schedule in basis points and fractional decimals."""

    model_config = ConfigDict(frozen=True)

    settlement: SettlementCurrency = SettlementCurrency.USDT
    maker_bps: float = Field(default=2.0, ge=0.0, description="Maker fee in bps")
    taker_bps: float = Field(default=5.5, ge=0.0, description="Taker fee in bps")
    estimated_slippage_bps: float = Field(
        default=1.0, ge=0.0, description="Estimated queue/spread slippage per leg in bps"
    )

    @classmethod
    def usdt_perp(cls) -> FeeSchedule:
        """Standard base-tier USDT perpetual fee schedule (2.0 bps Maker / 5.5 bps Taker)."""
        return cls(
            settlement=SettlementCurrency.USDT,
            maker_bps=2.0,
            taker_bps=5.5,
            estimated_slippage_bps=1.0,
        )

    @classmethod
    def usdc_perp(cls) -> FeeSchedule:
        """Standard base-tier USDC perpetual fee schedule (0.0 bps Maker / 4.0 bps Taker)."""
        return cls(
            settlement=SettlementCurrency.USDC,
            maker_bps=0.0,
            taker_bps=4.0,
            estimated_slippage_bps=1.0,
        )

    @property
    def maker_rate(self) -> float:
        """Maker fee as decimal fraction."""
        return self.maker_bps * 1e-4

    @property
    def taker_rate(self) -> float:
        """Taker fee as decimal fraction."""
        return self.taker_bps * 1e-4

    @property
    def roundtrip_maker_bps(self) -> float:
        """Round-trip maker fee in basis points."""
        return 2.0 * self.maker_bps

    @property
    def roundtrip_maker_plus_slippage_bps(self) -> float:
        """Round-trip maker fee plus estimated slippage in basis points."""
        return 2.0 * (self.maker_bps + self.estimated_slippage_bps)


class DollarBar(BaseModel):
    """Medium-frequency Dynamic Dollar Bar aggregated from exchange trade and book streams."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    horizon_label: Literal["15m", "1h", "4h"] = "15m"
    ts_open_ns: int
    ts_close_ns: int
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    dollar_volume: float = Field(ge=0.0)
    buy_dollar_volume: float = Field(ge=0.0)
    sell_dollar_volume: float = Field(ge=0.0)
    open_interest: float = Field(ge=0.0)
    funding_rate_8h: float = Field(description="Current 8-hour funding rate as decimal (e.g. 0.0005 = 5 bps)")
    mark_price: float = Field(gt=0.0)
    spot_price: float = Field(gt=0.0)
    liquidation_volume_usd: float = Field(default=0.0, ge=0.0)
    tick_count: int = Field(default=1, ge=1)

    @property
    def basis(self) -> float:
        """Mark-to-spot relative basis: (Mark - Spot) / Spot."""
        return (self.mark_price - self.spot_price) / self.spot_price


class FFDWeightsConfig(BaseModel):
    """Frozen per-split Fractional Differentiation (FFD) parameters enforcing strict causality."""

    model_config = ConfigDict(frozen=True)

    split_id: str
    d_star: float = Field(ge=0.05, le=0.60)
    threshold_tau: float = Field(default=1e-5, gt=0.0)
    weights: tuple[float, ...]
    adf_pvalue: float = Field(ge=0.0, le=1.0)
    level_correlation: float = Field(ge=-1.0, le=1.0)

    @property
    def window_length(self) -> int:
        """Truncated memory length L* + 1."""
        return len(self.weights)


class PositioningRegime(str, Enum):
    """Four structural perpetual futures positioning & squeeze regimes plus neutral."""

    CROWDED_LONG_VULNERABILITY = "CROWDED_LONG_VULNERABILITY"
    CROWDED_SHORT_SQUEEZE = "CROWDED_SHORT_SQUEEZE"
    GENUINE_TREND_CONVICTION = "GENUINE_TREND_CONVICTION"
    CASCADE_EXHAUSTION_REVERSAL = "CASCADE_EXHAUSTION_REVERSAL"
    NEUTRAL = "NEUTRAL"


class PositioningSignalSnapshot(BaseModel):
    """Stationary feature vector and positioning regime classification for a single dollar bar."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    ts_close_ns: int
    ffd_close: float
    ffd_oi: float
    ffd_close_zscore: float
    ffd_oi_zscore: float
    oi_funding_basis_zscore: float
    basis_pct: float
    funding_rate_8h: float
    oi_pct_change: float
    liquidation_zscore: float
    liquidation_exhaustion_flag: int = Field(ge=0, le=1, description="Sparse 99th-pct binary flag (excluded from SVD)")
    regime: PositioningRegime
    primary_direction: Literal[-1, 0, 1]


class TripleBarrierConfig(BaseModel):
    """Fee-adjusted Triple-Barrier labelling parameters."""

    model_config = ConfigDict(frozen=True)

    pt_sigma_mult: float = Field(default=2.0, gt=0.0)
    sl_sigma_mult: float = Field(default=1.5, gt=0.0)
    vertical_barrier_bars: int = Field(default=16, ge=1)
    min_barrier_bps: float = Field(default=80.0, ge=80.0, description="Hard minimum profit hurdle >= 80 bps")
    fee_schedule: FeeSchedule = Field(default_factory=FeeSchedule.usdt_perp)

    def effective_profit_barrier_decimal(self, sigma_bar: float) -> float:
        """Compute Barrier_min = max(k_pt * sigma_bar, 6 * (Fee_roundtrip + Slippage_est), 80 bps)."""
        fee_plus_slip_decimal = self.fee_schedule.roundtrip_maker_plus_slippage_bps * 1e-4
        min_hurdle_decimal = max(6.0 * fee_plus_slip_decimal, self.min_barrier_bps * 1e-4)
        return max(self.pt_sigma_mult * sigma_bar, min_hurdle_decimal)


class TripleBarrierLabel(BaseModel):
    """Net-of-fee Triple-Barrier label outcome for a single bar."""

    model_config = ConfigDict(frozen=True)

    bar_index: int
    primary_direction: Literal[-1, 0, 1]
    gross_return: float
    net_return: float
    fee_cost: float
    funding_cost: float
    upper_barrier: float
    lower_barrier: float
    hit_bar_offset: int
    barrier_touched: Literal["PROFIT", "STOP", "VERTICAL", "NONE"]
    meta_label: Literal[0, 1]


class MetaPredictionOutput(BaseModel):
    """Two-stage Meta-Model inference result with Mahalanobis epistemic outlier gate."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    ts_close_ns: int
    primary_direction: Literal[-1, 0, 1]
    p_meta: float = Field(ge=0.0, le=1.0)
    mahalanobis_di: float = Field(ge=0.0)
    di_threshold: float = Field(gt=0.0)
    epistemic_gate_passed: bool
    svd_factors: tuple[float, ...]
    liquidation_exhaustion_flag: int = Field(ge=0, le=1)
    expected_net_edge_bps: float

    @property
    def effective_direction(self) -> Literal[-1, 0, 1]:
        """Returns 0 if blocked by Mahalanobis epistemic gate or net-of-fee hurdle."""
        if not self.epistemic_gate_passed:
            return 0
        return self.primary_direction


class SystemicRiskOutput(BaseModel):
    """Strict Pydantic v2 JSON schema for the persistent mlx-lm Qwen3-8B-4bit risk daemon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0.0, le=100.0, description="Aggregated Systemic Risk Index (ASRI) in [0, 100]")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in [0.0, 1.0]")
    tail_risk_flag: bool = Field(description="True if acute regulatory, exchange, or stablecoin depeg crisis detected")
    plain_english_summary: str = Field(min_length=1, max_length=1000)

    @classmethod
    def neutral_fallback(cls, reason: str = "Fallback to deterministic neutral baseline.") -> SystemicRiskOutput:
        """Deterministic fallback when LLM output is malformed or times out."""
        return cls(
            score=0.0,
            confidence=0.0,
            tail_risk_flag=False,
            plain_english_summary=f"Macro risk monitor in neutral fallback mode: {reason}",
        )


class HawkesStateSnapshot(BaseModel):
    """O(1) recursive Markovian Hawkes process state and BVC VPIN snapshot."""

    model_config = ConfigDict(frozen=True)

    ts_ns: int
    r_state: float = Field(ge=0.0)
    lambda_intensity: float = Field(ge=0.0)
    mu: float = Field(gt=0.0)
    alpha: float = Field(ge=0.0)
    beta: float = Field(gt=0.0)
    branching_ratio_eta: float = Field(ge=0.0)
    vpin: float = Field(ge=0.0, le=1.0)
    toxic_sweep_active: bool


class ExecutionAction(str, Enum):
    """Execution decision emitted by DefensiveMakerExecutionAlgorithm."""

    PLACE_GTX_MAKER = "PLACE_GTX_MAKER"
    AMEND_GTX_MAKER = "AMEND_GTX_MAKER"
    HOLD_RESTING_HYSTERESIS = "HOLD_RESTING_HYSTERESIS"
    BACKOFF_DEEPER_GTX = "BACKOFF_DEEPER_GTX"
    CANCEL_ALL_TOXIC_SHIELD = "CANCEL_ALL_TOXIC_SHIELD"
    EMERGENCY_TAKER_EXIT = "EMERGENCY_TAKER_EXIT"


class ExecutionOrderCommand(BaseModel):
    """Concrete order action emitted by DefensiveMakerExecutionAlgorithm."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    ts_ns: int
    action: ExecutionAction
    side: Literal["BUY", "SELL", "NONE"]
    target_price: float = Field(ge=0.0)
    quantity: float = Field(ge=0.0)
    post_only_gtx: bool = True
    time_in_force: Literal["GTX", "IOC", "NONE"] = "GTX"
    consecutive_rejections: int = Field(default=0, ge=0)
    reason: str


class CPCVValidationResult(BaseModel):
    """Net-of-fee 45-split Combinatorial Purged Cross-Validation metrics and gate outcomes."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    n_groups: int = 10
    k_test_groups: int = 2
    num_splits: int = 45
    n_oos_paths: int = 9
    purge_bars: int = Field(ge=1)
    embargo_pct: float = Field(default=0.01, ge=0.0)
    psr_vs_zero: float = Field(ge=0.0, le=1.0)
    psr_vs_benchmark: float = Field(ge=0.0, le=1.0)
    dsr: float = Field(ge=0.0, le=1.0)
    pbo: float = Field(ge=0.0, le=1.0)
    expected_net_pnl_bps: float
    sharpe_net: float
    skewness: float
    kurtosis: float
    total_trials: int = Field(ge=1)

    @property
    def passed_all_gates(self) -> bool:
        """All 4 hard net-of-fee promotion gates required for Challenger promotion."""
        return (
            self.psr_vs_zero >= 0.95
            and self.psr_vs_benchmark >= 0.95
            and self.dsr >= 0.95
            and self.pbo < 0.30
            and self.expected_net_pnl_bps > 15.0
        )


class ExecutionRegimeMode(str, Enum):
    """Portfolio operational state controlled by Champion/Challenger gatekeeper or UI."""

    DIRECTIONAL_PLUS_CARRY = "DIRECTIONAL_PLUS_CARRY"
    CARRY_ONLY_FALLBACK = "CARRY_ONLY_FALLBACK"
    PAUSED_ENTRIES = "PAUSED_ENTRIES"
    EMERGENCY_FLATTENED = "EMERGENCY_FLATTENED"


class ChampionChallengerState(BaseModel):
    """State of the active Champion model and fallback state machine for an asset sleeve."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    champion_model_id: str | None
    champion_live_30d_psr: float = Field(ge=0.0, le=1.0)
    consecutive_failed_retrains: int = Field(ge=0)
    regime_mode: ExecutionRegimeMode
    last_promotion_reason: str


class AssetAllocationInput(BaseModel):
    """Per-asset inputs for the Signed HRP-CVaR & Fractional Kelly allocation equation."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    y_primary: Literal[-1, 0, 1]
    w_hrp: float = Field(gt=0.0, le=1.0)
    p_meta: float = Field(ge=0.0, le=1.0)
    expected_gross_return_bps: float
    fee_schedule: FeeSchedule = Field(default_factory=FeeSchedule.usdt_perp)
    epistemic_gate_passed: bool = True

    @field_validator("y_primary")
    @classmethod
    def validate_primary(cls, v: int) -> Literal[-1, 0, 1]:
        if v not in (-1, 0, 1):
            raise ValueError("y_primary must be in {-1, 0, +1}")
        return v  # type: ignore[return-value]


class PortfolioAllocationOutput(BaseModel):
    """Signed portfolio target weights respecting the 2.5x gross leverage cap."""

    model_config = ConfigDict(frozen=True)

    weights: dict[str, float]
    raw_unscaled_weights: dict[str, float]
    hrp_unsigned_weights: dict[str, float]
    kelly_scalars: dict[str, float]
    regime_multiplier: float = Field(ge=0.0, le=1.0)
    gross_leverage: float = Field(ge=0.0, le=2.500000001)
    leverage_scale_factor: float = Field(gt=0.0, le=1.0)
    rejected_for_fees: list[str]


def compute_fractional_kelly(p_meta: float, gamma_frac: float = 0.5) -> float:
    """Compute f_Kelly(p_meta) = clip(gamma_frac * (2 * p_meta - 1), 0.0, 1.0)."""
    if p_meta <= 0.5:
        return 0.0
    raw = gamma_frac * (2.0 * p_meta - 1.0)
    return float(min(max(raw, 0.0), 1.0))


def compute_regime_multiplier_from_asri(
    asri_score: float, tail_risk_flag: bool = False
) -> float:
    """Map ASRI in [0, 100] to top-down portfolio risk multiplier M_regime in [0.0, 1.0].

    - If tail_risk_flag is True or ASRI >= 95: M_regime = 0.0 (full directional halt).
    - If ASRI <= 40: M_regime = 1.0 (full risk budget).
    - If 40 < ASRI <= 70: linear taper from 1.0 down to 0.5.
    - If 70 < ASRI < 95: steep crisis taper from 0.5 down to 0.0.
    """
    if tail_risk_flag or asri_score >= 95.0:
        return 0.0
    if asri_score <= 40.0:
        return 1.0
    if asri_score <= 70.0:
        return float(1.0 - 0.5 * ((asri_score - 40.0) / 30.0))
    return float(max(0.0, 0.5 * (1.0 - (asri_score - 70.0) / 25.0)))


def compute_signed_target_weights(
    inputs: list[AssetAllocationInput],
    regime_multiplier: float,
    gamma_frac: float = 0.5,
    max_gross_leverage: float = 2.5,
) -> PortfolioAllocationOutput:
    """Compute signed target portfolio weights via the exact governing equation:

    w_final,i = y_primary,i * w_HRP,i * f_Kelly(p_meta,i) * M_regime

    Subject to:
    1. Net-of-fee minimum gross return filter: rejects any trade where
       expected_gross_return_bps < 2 * Maker_Fee_bps (round-trip maker fee hurdle).
    2. Epistemic Mahalanobis gate check.
    3. Global gross leverage cap: sum_i |w_final,i| <= max_gross_leverage (2.5x).
    """
    if not (0.0 <= regime_multiplier <= 1.0):
        raise ValueError(f"regime_multiplier must be in [0.0, 1.0], got {regime_multiplier}")

    raw_weights: dict[str, float] = {}
    hrp_weights: dict[str, float] = {}
    kelly_map: dict[str, float] = {}
    rejected_for_fees: list[str] = []

    for item in inputs:
        hrp_weights[item.asset_id] = item.w_hrp
        f_kelly = compute_fractional_kelly(item.p_meta, gamma_frac=gamma_frac)
        kelly_map[item.asset_id] = f_kelly

        min_fee_hurdle_bps = item.fee_schedule.roundtrip_maker_bps
        if item.y_primary == 0 or not item.epistemic_gate_passed:
            raw_weights[item.asset_id] = 0.0
            continue

        if item.expected_gross_return_bps <= min_fee_hurdle_bps:
            raw_weights[item.asset_id] = 0.0
            rejected_for_fees.append(item.asset_id)
            continue

        w_i = float(item.y_primary) * item.w_hrp * f_kelly * regime_multiplier
        raw_weights[item.asset_id] = w_i

    gross_unscaled = sum(abs(w) for w in raw_weights.values())
    if gross_unscaled > max_gross_leverage and gross_unscaled > 0.0:
        scale_factor = max_gross_leverage / gross_unscaled
    else:
        scale_factor = 1.0

    final_weights = {k: v * scale_factor for k, v in raw_weights.items()}
    final_gross = sum(abs(w) for w in final_weights.values())

    return PortfolioAllocationOutput(
        weights=final_weights,
        raw_unscaled_weights=raw_weights,
        hrp_unsigned_weights=hrp_weights,
        kelly_scalars=kelly_map,
        regime_multiplier=regime_multiplier,
        gross_leverage=final_gross,
        leverage_scale_factor=scale_factor,
        rejected_for_fees=rejected_for_fees,
    )


class EnvironmentConfig(BaseModel):
    """System-wide runtime configuration loaded from config_m1max_64gb.json or config_ci_mock.json."""

    model_config = ConfigDict(frozen=True)

    profile_name: Literal["prod_m1max_64gb", "ci_mock"]
    is_mock: bool
    wired_limit_mb: int = Field(ge=1024, le=40960)
    os_ipc_budget_gb: float = 10.0
    nautilus_rust_budget_gb: float = 8.0
    lgbm_polars_budget_gb: float = 12.0
    pytorch_mps_budget_gb: float = 10.0
    mlx_llm_budget_gb: float = 10.0
    omp_num_threads: int = 4
    polars_max_threads: int = 4
    lgbm_n_jobs: int = 4
    cpcv_max_workers: int = 2
    mlx_model_id: str = "mlx-community/Qwen3-8B-4bit"
    redis_url: str = "redis://localhost:6379/0"
    ui_host: str = "127.0.0.1"
    ui_port: int = 8080
    max_gross_leverage: float = 2.5
    vpin_toxic_threshold: float = 0.60
    hawkes_eta_toxic_threshold: float = 0.70
    repeg_min_bps: float = 1.5
    repeg_cooldown_ms: int = 500
    max_gtx_rejections: int = 3
    rejection_backoff_ms: int = 1000

    @model_validator(mode="after")
    def verify_total_memory_budget(self) -> EnvironmentConfig:
        total_active_gb = (
            self.os_ipc_budget_gb
            + self.nautilus_rust_budget_gb
            + self.lgbm_polars_budget_gb
            + self.pytorch_mps_budget_gb
            + self.mlx_llm_budget_gb
        )
        if total_active_gb > 50.0:
            raise ValueError(f"Active memory budget {total_active_gb} GB exceeds 50.0 GB ceiling (64GB M1 Max)")
        return self
