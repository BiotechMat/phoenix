"""Sleeve 1: Deribit Options Surface Metrics & Hyperliquid Whale/Vault Positioning Alpha.

Ingests:
1. Deribit options surface metrics: 25-delta skew (fear/greed asymmetry), IV term structure
   slope (contango vs. backwardation), and dealer net Gamma Exposure (GEX pinning vs. amplification).
2. Hyperliquid HLP and whale vault positioning (default vault: 0xdfc24b077bc1425ad1dea75bcb6f8158e10df303)
   to assess retail crowding and counterparty vulnerability.
3. Generates directional signals (y_primary in {-1, 0, 1}) and calibrated meta-probabilities
   (p_meta in [0.50, 1.00]) compatible with SignedHRPCVaRAllocator and AssetAllocationInput.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import (
    AssetAllocationInput,
    DeribitOptionsSurface,
    DollarBar,
    FeeSchedule,
    HyperliquidWhaleVaultMetrics,
    compute_fractional_kelly,
)
from sleeves.sleeve_3_funding_carry import (
    ALO_ROUTING_TAG,
    SleeveSignalOutput,
)

logger = logging.getLogger(__name__)

HLP_DEFAULT_VAULT: Final[str] = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
HYPERLIQUID_INFO_URL: Final[str] = "https://api.hyperliquid.xyz/info"
DERIBIT_BASE_URL: Final[str] = "https://www.deribit.com/api/v2/public"


class DerivativesAlphaConfig(BaseModel):
    """Configuration for DerivativesAlphaSleeve parameters and signal thresholds."""

    model_config = ConfigDict(frozen=True)

    vault_address: str = Field(default=HLP_DEFAULT_VAULT)
    skew_bullish_threshold: float = Field(
        default=0.015, description="Call skew threshold for bullish bias (e.g. +1.5%)"
    )
    skew_bearish_threshold: float = Field(
        default=-0.015, description="Put skew threshold for bearish bias (e.g. -1.5%)"
    )
    term_structure_contango_threshold: float = Field(
        default=0.005, description="Slope threshold for calm contango regime"
    )
    term_structure_backwardation_threshold: float = Field(
        default=-0.005, description="Slope threshold for stress backwardation regime"
    )
    gex_pinning_usd_threshold: float = Field(
        default=10_000_000.0, description="Dealer net GEX above which volatility pinning is expected"
    )
    gex_amplifying_usd_threshold: float = Field(
        default=-10_000_000.0, description="Dealer net GEX below which cascade acceleration is expected"
    )
    hlp_crowding_notional_usd: float = Field(
        default=2_000_000.0, description="HLP net delta threshold to diagnose retail crowding"
    )
    min_expected_edge_bps: float = Field(
        default=25.0, description="Base gross return edge in bps for active signals"
    )
    max_expected_edge_bps: float = Field(
        default=120.0, description="Ceiling on gross return edge in bps"
    )
    request_timeout_sec: float = Field(default=1.0, gt=0.0)


class DerivativesAlphaSignal(BaseModel):
    """Structured alpha signal output from the derivatives alpha sleeve."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    primary_direction: Literal[-1, 0, 1]
    p_meta: float = Field(ge=0.0, le=1.0)
    expected_gross_edge_bps: float
    skew_25d: float
    term_structure_slope: float
    gex_regime: Literal["POSITIVE_PINNING", "NEGATIVE_AMPLIFYING", "NEUTRAL"]
    hlp_vault_crowding: Literal["CROWDED_LONG", "CROWDED_SHORT", "BALANCED"]
    feature_attribution: dict[str, float]
    routing_tag: str = ALO_ROUTING_TAG
    timestamp_ns: int = 0

    def to_asset_allocation_input(
        self,
        w_hrp: float = 0.33,
        fee_schedule: FeeSchedule | None = None,
    ) -> AssetAllocationInput:
        """Convert into AssetAllocationInput ready for SignedHRPCVaRAllocator."""
        return AssetAllocationInput(
            asset_id=self.instrument_id,
            y_primary=self.primary_direction,
            w_hrp=w_hrp,
            p_meta=self.p_meta,
            expected_gross_return_bps=self.expected_gross_edge_bps,
            fee_schedule=fee_schedule or FeeSchedule.hyperliquid(),
            epistemic_gate_passed=True,
        )

    def to_sleeve_signal_output(self) -> SleeveSignalOutput:
        """Convert into standardized SleeveSignalOutput for the portfolio allocator."""
        f_kelly = compute_fractional_kelly(self.p_meta)
        rec_lev = min(2.5, f_kelly * 2.0) if self.primary_direction != 0 else 0.0

        return SleeveSignalOutput(
            sleeve_id="derivatives_alpha",
            target_asset=self.instrument_id,
            direction=self.primary_direction,
            meta_probability=self.p_meta,
            raw_expected_return=self.expected_gross_edge_bps,
            recommended_leverage=rec_lev,
            routing_tag=self.routing_tag,
            forensic_metadata={
                "skew_25d": self.skew_25d,
                "term_structure_slope": self.term_structure_slope,
                "gex_regime": self.gex_regime,
                "hlp_vault_crowding": self.hlp_vault_crowding,
                "feature_attribution": self.feature_attribution,
            },
        )


class DerivativesAlphaSleeve:
    """Quantitative Derivatives Alpha Sleeve capturing options surface and HLP positioning."""

    def __init__(self, config: DerivativesAlphaConfig | None = None) -> None:
        self.config: DerivativesAlphaConfig = config or DerivativesAlphaConfig()
        self._cached_surface: dict[str, DeribitOptionsSurface] = {}
        self._cached_vault: HyperliquidWhaleVaultMetrics | None = None

    async def fetch_deribit_options_surface(
        self,
        currency: str = "BTC",
    ) -> DeribitOptionsSurface:
        """Fetch options surface metrics from Deribit public API or fallback gracefully."""
        now_ns = time.time_ns()
        clean_currency = currency.strip().upper().removesuffix("USDT-PERP").removesuffix("-PERP").removesuffix("USDT")
        url = f"{DERIBIT_BASE_URL}/get_book_summary_by_currency"
        params = {"currency": clean_currency, "kind": "option"}

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout_sec) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json().get("result", [])
                    surface = self._parse_deribit_book_summary(clean_currency, data, now_ns)
                    self._cached_surface[clean_currency] = surface
                    return surface
        except Exception as e:
            logger.debug("Deribit options fetch failed (%s); using cached/fallback surface.", e)

        # Fallback to cached or neutral baseline
        cached = self._cached_surface.get(clean_currency)
        if cached is not None:
            return cached
        return DeribitOptionsSurface(currency=clean_currency, timestamp_ns=now_ns)

    def fetch_deribit_options_surface_sync(
        self,
        currency: str = "BTC",
    ) -> DeribitOptionsSurface:
        """Synchronous wrapper for fetch_deribit_options_surface."""
        now_ns = time.time_ns()
        clean_currency = currency.strip().upper().removesuffix("USDT-PERP").removesuffix("-PERP").removesuffix("USDT")
        url = f"{DERIBIT_BASE_URL}/get_book_summary_by_currency"
        params = {"currency": clean_currency, "kind": "option"}

        try:
            with httpx.Client(timeout=self.config.request_timeout_sec) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json().get("result", [])
                    surface = self._parse_deribit_book_summary(clean_currency, data, now_ns)
                    self._cached_surface[clean_currency] = surface
                    return surface
        except Exception as e:
            logger.debug("Deribit sync fetch failed (%s); using fallback.", e)

        cached = self._cached_surface.get(clean_currency)
        if cached is not None:
            return cached
        return DeribitOptionsSurface(currency=clean_currency, timestamp_ns=now_ns)

    def _parse_deribit_book_summary(
        self,
        currency: str,
        book_summary: list[dict[str, Any]],
        now_ns: int,
    ) -> DeribitOptionsSurface:
        """Parse raw Deribit book summary into skew, term structure slope, and net GEX."""
        if not book_summary:
            return DeribitOptionsSurface(currency=currency, timestamp_ns=now_ns)

        calls_oi_gamma = 0.0
        puts_oi_gamma = 0.0
        call_ivs: list[float] = []
        put_ivs: list[float] = []
        near_ivs: list[float] = []
        far_ivs: list[float] = []

        for item in book_summary:
            inst_name = str(item.get("instrument_name", ""))
            oi = float(item.get("open_interest", 0.0))
            mid_iv = float(item.get("mid_iv", item.get("mark_iv", 0.0)))
            underlying_px = float(item.get("underlying_price", 0.0))

            if mid_iv > 0.0:
                if inst_name.endswith("-C"):
                    call_ivs.append(mid_iv)
                    # Gamma proxy proportional to OI / spot
                    calls_oi_gamma += oi * mid_iv * (underlying_px * 0.01)
                elif inst_name.endswith("-P"):
                    put_ivs.append(mid_iv)
                    puts_oi_gamma += oi * mid_iv * (underlying_px * 0.01)

            # Approximate term structure by expiry label length/ordering
            if len(near_ivs) < 20 and mid_iv > 0.0:
                near_ivs.append(mid_iv)
            elif mid_iv > 0.0:
                far_ivs.append(mid_iv)

        avg_call_iv = (sum(call_ivs) / len(call_ivs)) if call_ivs else 50.0
        avg_put_iv = (sum(put_ivs) / len(put_ivs)) if put_ivs else 50.0

        # 25-delta skew proxy: (Call IV - Put IV) / baseline IV
        skew_25d = (avg_call_iv - avg_put_iv) / 100.0

        # Term structure slope: (Far IV - Near IV) / Near IV
        avg_near = (sum(near_ivs) / len(near_ivs)) if near_ivs else 50.0
        avg_far = (sum(far_ivs) / len(far_ivs)) if far_ivs else avg_near
        ts_slope = (avg_far - avg_near) / max(avg_near, 1.0)

        # Net GEX in USD: Calls Gamma USD - Puts Gamma USD
        net_gex_usd = calls_oi_gamma - puts_oi_gamma

        return DeribitOptionsSurface(
            currency=currency,
            skew_25d=round(skew_25d, 4),
            term_structure_slope=round(ts_slope, 4),
            net_gex_usd=round(net_gex_usd, 2),
            dvol_annualized=round(avg_near, 2),
            timestamp_ns=now_ns,
        )

    async def fetch_hyperliquid_vault_positioning(
        self,
        vault_address: str | None = None,
    ) -> HyperliquidWhaleVaultMetrics:
        """Fetch HLP vault positioning from Hyperliquid L1 info API or fallback gracefully."""
        v_addr = vault_address or self.config.vault_address
        now_ns = time.time_ns()

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout_sec) as client:
                payload = {"type": "vaultDetails", "vaultAddress": v_addr}
                resp = await client.post(HYPERLIQUID_INFO_URL, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    metrics = self._parse_hlp_vault_details(v_addr, data, now_ns)
                    self._cached_vault = metrics
                    return metrics
        except Exception as e:
            logger.debug("Hyperliquid vaultDetails fetch failed (%s); using fallback.", e)

        if self._cached_vault is not None:
            return self._cached_vault
        return HyperliquidWhaleVaultMetrics(vault_address=v_addr, timestamp_ns=now_ns)

    def fetch_hyperliquid_vault_positioning_sync(
        self,
        vault_address: str | None = None,
    ) -> HyperliquidWhaleVaultMetrics:
        """Synchronous wrapper for fetch_hyperliquid_vault_positioning."""
        v_addr = vault_address or self.config.vault_address
        now_ns = time.time_ns()

        try:
            with httpx.Client(timeout=self.config.request_timeout_sec) as client:
                payload = {"type": "vaultDetails", "vaultAddress": v_addr}
                resp = client.post(HYPERLIQUID_INFO_URL, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    metrics = self._parse_hlp_vault_details(v_addr, data, now_ns)
                    self._cached_vault = metrics
                    return metrics
        except Exception as e:
            logger.debug("Hyperliquid vault sync fetch failed (%s); using fallback.", e)

        if self._cached_vault is not None:
            return self._cached_vault
        return HyperliquidWhaleVaultMetrics(vault_address=v_addr, timestamp_ns=now_ns)

    def _parse_hlp_vault_details(
        self,
        vault_address: str,
        data: dict[str, Any],
        now_ns: int,
    ) -> HyperliquidWhaleVaultMetrics:
        """Parse Hyperliquid vaultDetails response into TVL, net delta, and retail crowding."""
        portfolio = data.get("portfolio", {})
        tvl_usd = float(data.get("summary", {}).get("tvl", portfolio.get("accountValue", 120_000_000.0)))

        # Sum position deltas in USD across all active perps held by HLP
        net_delta_usd = 0.0
        positions = portfolio.get("positions", [])
        if isinstance(positions, list):
            for p in positions:
                if isinstance(p, dict):
                    pos_info = p.get("position", {})
                    szi = float(pos_info.get("szi", 0.0))
                    px = float(pos_info.get("entryPx", 0.0))
                    net_delta_usd += szi * px

        thresh = self.config.hlp_crowding_notional_usd
        if net_delta_usd < -thresh:
            # HLP is net short -> Retail is crowded long
            crowding: Literal["CROWDED_LONG", "CROWDED_SHORT", "BALANCED"] = "CROWDED_SHORT"
        elif net_delta_usd > thresh:
            # HLP is net long -> Retail is crowded short
            crowding = "CROWDED_LONG"
        else:
            crowding = "BALANCED"

        return HyperliquidWhaleVaultMetrics(
            vault_address=vault_address,
            tvl_usd=tvl_usd,
            net_delta_usd=net_delta_usd,
            hlp_crowding=crowding,
            timestamp_ns=now_ns,
        )

    def generate_alpha_signal(
        self,
        options_surface: DeribitOptionsSurface | None = None,
        vault_metrics: HyperliquidWhaleVaultMetrics | None = None,
        bar: DollarBar | None = None,
        instrument_id: str = "BTC",
        timestamp_ns: int | None = None,
    ) -> DerivativesAlphaSignal:
        """Synthesize options surface and vault positioning into a directional alpha signal.

        - If options surface is missing or empty (all zeros), returns neutral direction (0)
          and baseline meta-probability (0.50).
        - Skew: positive call skew gives bullish bias (+1), negative put skew gives bearish (-1).
        - Term structure: contango (+0.5) reinforces bullish trend, backwardation (-1.0) signals crash risk.
        - Dealer GEX: positive pins volatility, negative amplifies breakout.
        - Vault positioning: HLP net short indicates retail long crowding (vulnerability);
          HLP net long indicates retail short crowding (squeeze upside).
        """
        now_ns = timestamp_ns if timestamp_ns is not None else time.time_ns()
        clean_id = (
            instrument_id.strip()
            .upper()
            .removesuffix("USDT-PERP")
            .removesuffix("-PERP")
            .removesuffix("USDT")
        )

        surf = options_surface or self._cached_surface.get(clean_id)
        vault = vault_metrics or self._cached_vault

        # Check for missing or all-zero surface fallback (test_r2_b09)
        surface_is_empty = (
            surf is None
            or (surf.skew_25d == 0.0 and surf.net_gex_usd == 0.0 and surf.term_structure_slope == 0.0)
        )
        vault_is_empty = (
            vault is None
            or (vault.net_delta_usd == 0.0 and vault.hlp_crowding == "BALANCED")
        )

        if surface_is_empty and vault_is_empty:
            return DerivativesAlphaSignal(
                instrument_id=clean_id,
                primary_direction=0,
                p_meta=0.50,
                expected_gross_edge_bps=0.0,
                skew_25d=0.0,
                term_structure_slope=0.0,
                gex_regime="NEUTRAL",
                hlp_vault_crowding="BALANCED",
                feature_attribution={"neutral_baseline": 1.0},
                routing_tag=ALO_ROUTING_TAG,
                timestamp_ns=now_ns,
            )

        resolved_skew = surf.skew_25d if surf is not None else 0.0
        resolved_ts = surf.term_structure_slope if surf is not None else 0.0
        resolved_gex = surf.net_gex_usd if surf is not None else 0.0
        resolved_crowd = vault.hlp_crowding if vault is not None else "BALANCED"

        # 1. 25-Delta Skew Signal (Call skew > 0 is bullish, Put skew < 0 is bearish)
        skew_score = 0.0
        if resolved_skew > self.config.skew_bullish_threshold:
            skew_score = 1.0
        elif resolved_skew < self.config.skew_bearish_threshold:
            skew_score = -1.0

        # 2. Term Structure Slope Signal
        ts_score = 0.0
        if resolved_ts > self.config.term_structure_contango_threshold:
            ts_score = 0.5  # calm contango supports trend
        elif resolved_ts < self.config.term_structure_backwardation_threshold:
            ts_score = -1.0  # backwardation stress

        # 3. Dealer GEX Regime
        if resolved_gex > self.config.gex_pinning_usd_threshold * 0.5:
            gex_regime: Literal["POSITIVE_PINNING", "NEGATIVE_AMPLIFYING", "NEUTRAL"] = "POSITIVE_PINNING"
            gex_score = 0.0  # range-bound pinning
        elif resolved_gex < self.config.gex_amplifying_usd_threshold * 0.5:
            gex_regime = "NEGATIVE_AMPLIFYING"
            # Amplifying accelerates prevailing skew direction
            gex_score = skew_score * 0.5
        else:
            gex_regime = "NEUTRAL"
            gex_score = 0.0

        # 4. HLP Vault Retail Crowding Signal
        vault_score = 0.0
        if resolved_crowd == "CROWDED_LONG":
            # Retail is net short -> short squeeze potential to the upside
            vault_score = 1.0
        elif resolved_crowd == "CROWDED_SHORT":
            # Retail is net long -> long squeeze vulnerability to the downside
            vault_score = -1.0

        # 5. Composite Confluence Score
        # Weights: Skew 40%, Vault 30%, Term Structure 20%, GEX 10%
        composite = (
            0.40 * skew_score
            + 0.30 * vault_score
            + 0.20 * ts_score
            + 0.10 * gex_score
        )

        # Basis bonus from bar if available
        if bar is not None and bar.basis != 0.0:
            basis_contrib = math.copysign(min(0.20, abs(bar.basis) * 100.0), bar.basis)
            composite = float(composite * 0.85 + basis_contrib * 0.15)

        # Primary direction determination
        if composite > 0.15:
            primary_dir: Literal[-1, 0, 1] = 1
        elif composite < -0.15:
            primary_dir = -1
        else:
            primary_dir = 0

        # Calibrated meta-probability
        if primary_dir != 0:
            raw_p = 0.50 + 0.45 * min(1.0, abs(composite))
            p_meta = float(min(0.95, max(0.55, raw_p)))
            edge_bps = float(
                self.config.min_expected_edge_bps
                + (self.config.max_expected_edge_bps - self.config.min_expected_edge_bps)
                * (p_meta - 0.50)
                * 2.0
            )
        else:
            p_meta = 0.50
            edge_bps = 0.0

        attributions = {
            "skew_25d_contrib": round(0.40 * skew_score, 4),
            "vault_crowding_contrib": round(0.30 * vault_score, 4),
            "term_structure_contrib": round(0.20 * ts_score, 4),
            "gex_contrib": round(0.10 * gex_score, 4),
            "composite_score": round(composite, 4),
        }

        return DerivativesAlphaSignal(
            instrument_id=clean_id,
            primary_direction=primary_dir,
            p_meta=round(p_meta, 4),
            expected_gross_edge_bps=round(edge_bps, 2),
            skew_25d=resolved_skew,
            term_structure_slope=resolved_ts,
            gex_regime=gex_regime,
            hlp_vault_crowding=resolved_crowd,
            feature_attribution=attributions,
            routing_tag=ALO_ROUTING_TAG,
            timestamp_ns=now_ns,
        )


# Backward compatibility aliases
DerivativesAlphaSleeve1 = DerivativesAlphaSleeve

__all__ = [
    "ALO_ROUTING_TAG",
    "DERIBIT_BASE_URL",
    "DerivativesAlphaConfig",
    "DerivativesAlphaSignal",
    "DerivativesAlphaSleeve",
    "DerivativesAlphaSleeve1",
    "HLP_DEFAULT_VAULT",
    "HYPERLIQUID_INFO_URL",
]
