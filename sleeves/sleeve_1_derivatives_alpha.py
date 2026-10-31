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
from core.options_surface import SurfaceMetrics, compute_surface_metrics, instrument_specs_from_names
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
    allocate: bool = Field(
        default=False,
        description=(
            "Whether the live loop may size positions from this sleeve. Off until the composite passes the "
            "P2.4 validation (component ICs and an out-of-fold win over the best single component); the sleeve "
            "still evaluates and reports while off."
        ),
    )


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
        self._last_surface_metrics: dict[str, SurfaceMetrics] = {}

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
        """Surface metrics from per-instrument marks (REINFORCEMENT_PLAN P2.4).

        ``skew_25d`` is the 30-day 25-delta risk reversal in decimal vol (call IV minus put IV / 100),
        ``term_structure_slope`` is (ATM IV at ~90d - ATM IV at ~7d) / ATM IV at ~7d, ``net_gex_usd`` is the
        dealer gamma exposure per 1% move (calls minus puts), and ``dvol_annualized`` carries the near ATM IV.
        """
        if not book_summary:
            return DeribitOptionsSurface(currency=currency, timestamp_ns=now_ns)
        now_ms = now_ns // 1_000_000
        metrics = compute_surface_metrics(book_summary, instrument_specs_from_names(book_summary), now_ms, currency=currency)
        self._last_surface_metrics[currency] = metrics
        return DeribitOptionsSurface(
            currency=currency,
            skew_25d=round((metrics.rr25_30d or 0.0) / 100.0, 6),
            term_structure_slope=round(metrics.term_structure_slope or 0.0, 6),
            net_gex_usd=round(metrics.net_gex_usd_per_1pct, 2),
            dvol_annualized=round(metrics.atm_iv_near or 0.0, 4),
            timestamp_ns=now_ns,
        )

    @property
    def last_surface_metrics(self) -> dict[str, SurfaceMetrics]:
        """Full surface metrics from the latest fetch per currency."""
        return dict(self._last_surface_metrics)

    async def fetch_hyperliquid_vault_positioning(
        self,
        vault_address: str | None = None,
    ) -> HyperliquidWhaleVaultMetrics:
        """Async wrapper around ``fetch_hyperliquid_vault_positioning_sync``."""
        import asyncio

        return await asyncio.to_thread(self.fetch_hyperliquid_vault_positioning_sync, vault_address)

    def fetch_hyperliquid_vault_positioning_sync(
        self,
        vault_address: str | None = None,
    ) -> HyperliquidWhaleVaultMetrics:
        """Read HLP positions: the vault's ``clearinghouseState`` plus every child account's.

        HLP is a parent vault whose positions sit in child sub-accounts listed under
        ``vaultDetails.relationship.data.childAddresses``; the parent's own state holds none.
        """
        v_addr = vault_address or self.config.vault_address
        now_ns = time.time_ns()
        try:
            with httpx.Client(timeout=self.config.request_timeout_sec * 3) as client:
                details = client.post(HYPERLIQUID_INFO_URL, json={"type": "vaultDetails", "vaultAddress": v_addr}).json()
                accounts = [v_addr] + list(((details or {}).get("relationship") or {}).get("data", {}).get("childAddresses", []) or [])
                states: dict[str, dict[str, Any]] = {}
                for acct in accounts:
                    resp = client.post(HYPERLIQUID_INFO_URL, json={"type": "clearinghouseState", "user": acct})
                    if resp.status_code == 200 and isinstance(resp.json(), dict):
                        states[acct] = resp.json()
            metrics = self._parse_hlp_vault_details(v_addr, {"details": details, "states": states}, now_ns)
            self._cached_vault = metrics
            return metrics
        except Exception as e:
            logger.debug("Hyperliquid vault positions fetch failed (%s); using cached/empty.", e)

        if self._cached_vault is not None:
            return self._cached_vault
        return HyperliquidWhaleVaultMetrics(vault_address=v_addr, timestamp_ns=now_ns)

    def _parse_hlp_vault_details(
        self,
        vault_address: str,
        data: dict[str, Any],
        now_ns: int,
    ) -> HyperliquidWhaleVaultMetrics:
        """Sum signed position value at mark per coin across the vault and its child accounts."""
        states: dict[str, Any] = data.get("states", {}) or {}
        by_coin: dict[str, float] = {}
        tvl = 0.0
        for state in states.values():
            if not isinstance(state, dict):
                continue
            try:
                tvl += float((state.get("marginSummary") or {}).get("accountValue", 0.0))
            except (TypeError, ValueError):
                pass
            for ap in state.get("assetPositions", []) or []:
                pos = ap.get("position", {}) if isinstance(ap, dict) else {}
                try:
                    szi = float(pos.get("szi", 0.0))
                    value = float(pos.get("positionValue", 0.0))
                except (TypeError, ValueError):
                    continue
                coin = str(pos.get("coin", "")).upper()
                if not coin or szi == 0.0:
                    continue
                by_coin[coin] = by_coin.get(coin, 0.0) + math.copysign(value, szi)
        net_delta = sum(by_coin.values())
        thresh = self.config.hlp_crowding_notional_usd
        # HLP is the counterparty: HLP net short means retail is crowded long, and vice versa.
        crowding: Literal["CROWDED_LONG", "CROWDED_SHORT", "BALANCED"] = (
            "CROWDED_SHORT" if net_delta < -thresh else ("CROWDED_LONG" if net_delta > thresh else "BALANCED")
        )
        return HyperliquidWhaleVaultMetrics(
            vault_address=vault_address,
            tvl_usd=tvl,
            net_delta_usd=net_delta,
            hlp_crowding=crowding,
            timestamp_ns=now_ns,
            net_delta_by_coin_usd=by_coin,
            accounts_read=len(states),
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
            or (vault.net_delta_usd == 0.0 and vault.hlp_crowding == "BALANCED" and not vault.net_delta_by_coin_usd)
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
        resolved_crowd: Literal["CROWDED_LONG", "CROWDED_SHORT", "BALANCED"] = vault.hlp_crowding if vault is not None else "BALANCED"
        if vault is not None and clean_id in vault.net_delta_by_coin_usd:
            coin_delta = vault.net_delta_by_coin_usd[clean_id]
            thresh = self.config.hlp_crowding_notional_usd
            resolved_crowd = "CROWDED_SHORT" if coin_delta < -thresh else ("CROWDED_LONG" if coin_delta > thresh else "BALANCED")

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
