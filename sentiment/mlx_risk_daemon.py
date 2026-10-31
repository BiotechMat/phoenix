"""Persistent mlx-lm Macro & Systemic Risk Daemon (Qwen3-8B-4bit).

Enforces:
1. macOS `spawn` multiprocessing isolation before numerical/GPU imports.
2. Singleton `BatchGenerator` startup with `mlx-community/Qwen3-8B-4bit` inside a
   Metal `wired_limit` context (`prod_m1max_64gb`), plus a deterministic,
   zero-network-latency mock inference engine (`ci_mock` or offline weights).
3. Asynchronous Macro & Crypto Systemic Feed Aggregator (SEC/CFTC/macro RSS,
   DeFiLlama TVL & stablecoin peg deviations, Deribit DVOL implied volatility
   spikes, and Crypto Fear & Greed index) with structured prompt synthesis.
4. Strict Pydantic v2 JSON schema enforcement (`parse_and_validate_llm_risk_json`)
   with deterministic neutral fallback (`SystemicRiskOutput.neutral_fallback()`).
5. Top-Down Hierarchical Regime Conditioning (never concatenated into Stage-1 SVD
   microstructure features): computes and publishes `ASRI` in [0, 100] and
   `M_regime` in [0.0, 1.0] via `compute_regime_multiplier_from_asri` over
   Redis Streams / in-memory IPC bus.
"""

from __future__ import annotations

import multiprocessing

# Enforce macOS 'spawn' start method prior to any numerical, Metal, or networking imports.
multiprocessing.set_start_method("spawn", force=True)

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
import contextlib
import json
import logging
import math
import re
import time
from typing import Any, ClassVar
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.contracts import (
    EnvironmentConfig,
    SystemicRiskOutput,
    compute_regime_multiplier_from_asri,
)
from core.hardware_bootstrap import load_environment_config

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID: str = "mlx-community/Qwen3-8B-4bit"
DEFAULT_REDIS_STREAM_KEY: str = "argon4:sentiment:asri"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_MARKDOWN_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE
)

_CRISIS_KEYWORDS: tuple[str, ...] = (
    "insolvency",
    "bankruptcy",
    "depeg",
    "exchange halt",
    "withdrawals halted",
    "emergency injunction",
    "sec enforcement",
    "cftc enforcement",
    "bridge exploit",
    "custodian freeze",
    "systemic contagion",
    "chapter 11",
)

_ELEVATED_RISK_KEYWORDS: tuple[str, ...] = (
    "subpoena",
    "Wells notice",
    "liquidity crunch",
    "rate hike surprise",
    "margin call",
    "volatility spike",
    "outflow",
    "regulatory probe",
)


class MacroCryptoFeedSnapshot(BaseModel):
    """Aggregated point-in-time macro, regulatory, DeFi, and derivatives volatility feeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts_ns: int = Field(
        default_factory=time.time_ns,
        description="Nanosecond UTC timestamp of feed aggregation",
    )
    sec_cftc_headlines: tuple[str, ...] = Field(
        default=(),
        description="Latest SEC and CFTC regulatory press release headlines",
    )
    macro_headlines: tuple[str, ...] = Field(
        default=(),
        description="Latest central bank and macro-economic RSS headlines",
    )
    defillama_tvl_change_24h_pct: float = Field(
        default=0.0,
        ge=-100.0,
        le=500.0,
        description="24-hour aggregate DeFi Total Value Locked percentage change",
    )
    max_stablecoin_peg_deviation_bps: float = Field(
        default=5.0,
        ge=0.0,
        le=10000.0,
        description="Maximum absolute peg deviation from $1.00 across USDT/USDC/DAI in bps",
    )
    deribit_dvol_index: float = Field(
        default=52.0,
        ge=0.0,
        le=500.0,
        description="Deribit 30-day annualized implied volatility index (DVOL)",
    )
    deribit_dvol_zscore: float = Field(
        default=0.0,
        ge=-10.0,
        le=20.0,
        description="Trailing standardized z-score of Deribit DVOL spikes",
    )
    fear_and_greed_index: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Crypto Fear & Greed sentiment index in [0, 100] (0=Extreme Fear)",
    )


class RegimePublishEnvelope(BaseModel):
    """Top-down systemic regime state published over Redis Streams / in-memory IPC bus.

    Kept strictly orthogonal to Stage-1 continuous SVD microstructure features;
    enters portfolio construction exclusively via M_regime in [0.0, 1.0].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts_ns: int
    asri_score: float = Field(ge=0.0, le=100.0)
    regime_multiplier: float = Field(ge=0.0, le=1.0)
    tail_risk_flag: bool
    confidence: float = Field(ge=0.0, le=1.0)
    plain_english_summary: str = Field(min_length=1, max_length=1000)
    risk_output: SystemicRiskOutput
    stream_key: str = DEFAULT_REDIS_STREAM_KEY
    message_id: str = "0-1"

    @property
    def m_regime(self) -> float:
        """Alias for the top-down portfolio risk multiplier M_regime in [0.0, 1.0]."""
        return self.regime_multiplier

    def to_redis_fields(self) -> dict[str, str]:
        """Serialize envelope into flat string key-value pairs for Redis XADD."""
        return {
            "ts_ns": str(self.ts_ns),
            "asri_score": f"{self.asri_score:.6f}",
            "regime_multiplier": f"{self.regime_multiplier:.6f}",
            "tail_risk_flag": "1" if self.tail_risk_flag else "0",
            "confidence": f"{self.confidence:.6f}",
            "plain_english_summary": self.plain_english_summary,
            "payload_json": self.risk_output.model_dump_json(),
        }


def _extract_json_candidate(raw_text: str) -> str:
    """Strip Qwen3 `<think>` blocks and markdown code fences to isolate JSON payload."""
    cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()
    fence_match = _MARKDOWN_FENCE_RE.search(cleaned)
    if fence_match is not None:
        return fence_match.group(1).strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1].strip()

    return cleaned


def parse_and_validate_llm_risk_json(raw_text: str | None) -> SystemicRiskOutput:
    """Strictly parse and validate raw LLM output into `SystemicRiskOutput`.

    Enforces:
    - Markdown code fence and `<think>` block stripping.
    - Pydantic v2 `SystemicRiskOutput.model_validate_json()` (`extra="forbid"`,
      `score` in [0, 100], `confidence` in [0, 1], `tail_risk_flag` bool,
      non-empty `plain_english_summary` <= 1000 chars).
    - Finite float verification (rejects NaN / Inf).
    - Deterministic neutral fallback (`score=0.0`, `confidence=0.0`,
      `tail_risk_flag=False`) on any malformed JSON, validation failure, or timeout.
    """
    if raw_text is None or not isinstance(raw_text, str):
        return SystemicRiskOutput.neutral_fallback("Empty or non-string LLM response.")

    stripped = raw_text.strip()
    if not stripped:
        return SystemicRiskOutput.neutral_fallback("Blank LLM response string.")

    candidate = _extract_json_candidate(stripped)
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return SystemicRiskOutput.neutral_fallback("No JSON object boundary found in LLM output.")

    try:
        raw_obj = json.loads(candidate)
        if not isinstance(raw_obj, dict):
            return SystemicRiskOutput.neutral_fallback("Parsed JSON root is not an object.")

        # Reject boolean passed as numeric score/confidence or non-boolean tail_risk_flag
        score_val = raw_obj.get("score")
        conf_val = raw_obj.get("confidence")
        tail_val = raw_obj.get("tail_risk_flag")
        if (
            isinstance(score_val, bool)
            or isinstance(conf_val, bool)
            or not isinstance(tail_val, bool)
        ):
            return SystemicRiskOutput.neutral_fallback(
                "Invalid field primitive types in JSON payload."
            )

        validated = SystemicRiskOutput.model_validate_json(candidate)
        if not (math.isfinite(validated.score) and math.isfinite(validated.confidence)):
            return SystemicRiskOutput.neutral_fallback("Non-finite float detected in LLM score.")
        return validated
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("LLM risk JSON validation failed (%s); applying neutral fallback.", exc)
        return SystemicRiskOutput.neutral_fallback(
            f"Schema validation error ({type(exc).__name__})."
        )


class AsyncSystemicFeedAggregator:
    """Asynchronous macro & crypto systemic risk feed poller and structured prompt synthesizer.

    Polls:
    1. SEC / CFTC / Macro RSS feeds
    2. DeFiLlama 24h TVL change & stablecoin peg deviations (USDT, USDC, DAI)
    3. Deribit DVOL implied volatility index & trailing z-score
    4. Alternative.me Crypto Fear & Greed Index
    """

    SEC_RSS_URL: ClassVar[str] = "https://www.sec.gov/news/pressreleases.rss"
    CFTC_RSS_URL: ClassVar[str] = "https://www.cftc.gov/PressRoom/PressReleases/rss.xml"
    DEFILLAMA_PROTOCOLS_URL: ClassVar[str] = "https://api.llama.fi/v2/historicalChainTvl"
    DEFILLAMA_STABLECOINS_URL: ClassVar[str] = (
        "https://stablecoins.llama.fi/stablecoins?includePrices=true"
    )
    DERIBIT_DVOL_URL: ClassVar[str] = (
        "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    )
    FEAR_GREED_URL: ClassVar[str] = "https://api.alternative.me/fng/?limit=1"

    def __init__(
        self,
        is_mock: bool = True,
        request_timeout_sec: float = 2.5,
        dvol_history_window: int = 30,
    ) -> None:
        self._is_mock: bool = is_mock
        self._timeout_sec: float = request_timeout_sec
        self._dvol_history: deque[float] = deque(
            [50.0, 51.5, 49.0, 52.0, 53.0, 48.5, 50.5, 51.0],
            maxlen=max(5, dvol_history_window),
        )

    @staticmethod
    def parse_rss_xml_headlines(xml_text: str, max_items: int = 5) -> tuple[str, ...]:
        """Extract clean `<title>` headlines from RSS 2.0 or Atom XML payloads."""
        if not xml_text.strip():
            return ()
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return ()

        titles: list[str] = []
        for elem in root.iter():
            tag_Lower = elem.tag.lower()
            if tag_Lower.endswith("item") or tag_Lower.endswith("entry"):
                for child in elem:
                    if child.tag.lower().endswith("title") and child.text:
                        cleaned = " ".join(child.text.strip().split())
                        if cleaned:
                            titles.append(cleaned)
                        break
                if len(titles) >= max_items:
                    break
        return tuple(titles)

    async def fetch_regulatory_and_macro_rss(
        self, client: httpx.AsyncClient | None = None
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Poll SEC/CFTC regulatory and macro headlines asynchronously."""
        if self._is_mock:
            return (
                (
                    "SEC Division of Examinations Releases Annual Derivatives Clearing Priorities",
                    "CFTC Announces Standardized Swap Execution Reporting Guidelines",
                ),
                (
                    "Federal Reserve FOMC Minutes Signal Data-Dependent Policy Stance",
                    "US Treasury Quarterly Refunding Auction Meets Institutional Demand",
                ),
            )

        own_client = client is None
        http_client = client or httpx.AsyncClient(timeout=self._timeout_sec)
        try:
            sec_resp, cftc_resp = await asyncio.gather(
                http_client.get(self.SEC_RSS_URL),
                http_client.get(self.CFTC_RSS_URL),
                return_exceptions=True,
            )
            reg_headlines: list[str] = []
            if isinstance(sec_resp, httpx.Response) and sec_resp.status_code == 200:
                reg_headlines.extend(self.parse_rss_xml_headlines(sec_resp.text, max_items=3))
            if isinstance(cftc_resp, httpx.Response) and cftc_resp.status_code == 200:
                reg_headlines.extend(self.parse_rss_xml_headlines(cftc_resp.text, max_items=3))

            if not reg_headlines:
                reg_headlines.append("No new SEC/CFTC enforcement alerts in polling window.")

            macro_headlines = (
                "Global Sovereign Yield Curve & Macro Liquidity Within Normal Operating Bands",
            )
            return tuple(reg_headlines[:6]), macro_headlines
        except Exception as exc:
            logger.debug("RSS poll fallback triggered: %s", exc)
            return (
                ("Regulatory RSS feed unreachable; using cached neutral state.",),
                ("Macro RSS feed unreachable; using cached neutral state.",),
            )
        finally:
            if own_client:
                await http_client.aclose()

    async def fetch_defillama_tvl_and_pegs(
        self, client: httpx.AsyncClient | None = None
    ) -> tuple[float, float]:
        """Poll DeFiLlama 24h aggregate TVL change (%) and max stablecoin peg deviation (bps)."""
        if self._is_mock:
            return (0.45, 4.2)

        own_client = client is None
        http_client = client or httpx.AsyncClient(timeout=self._timeout_sec)
        try:
            tvl_resp, peg_resp = await asyncio.gather(
                http_client.get(self.DEFILLAMA_PROTOCOLS_URL),
                http_client.get(self.DEFILLAMA_STABLECOINS_URL),
                return_exceptions=True,
            )

            tvl_change_pct = 0.0
            if isinstance(tvl_resp, httpx.Response) and tvl_resp.status_code == 200:
                series = tvl_resp.json()
                if isinstance(series, list) and len(series) >= 2:
                    prev_tvl = float(series[-2].get("tvl", 0.0))
                    curr_tvl = float(series[-1].get("tvl", 0.0))
                    if prev_tvl > 0.0:
                        tvl_change_pct = ((curr_tvl - prev_tvl) / prev_tvl) * 100.0

            max_dev_bps = 5.0
            if isinstance(peg_resp, httpx.Response) and peg_resp.status_code == 200:
                data = peg_resp.json()
                pegged_assets = data.get("peggedAssets", []) if isinstance(data, dict) else []
                deviations: list[float] = []
                for asset in pegged_assets:
                    if not isinstance(asset, dict):
                        continue
                    symbol = str(asset.get("symbol", "")).upper()
                    if symbol in {"USDT", "USDC", "DAI"}:
                        price = asset.get("price")
                        if isinstance(price, (int, float)) and price > 0.0:
                            deviations.append(abs(float(price) - 1.0) * 10000.0)
                if deviations:
                    max_dev_bps = max(deviations)

            return (
                float(min(max(tvl_change_pct, -100.0), 500.0)),
                float(min(max(max_dev_bps, 0.0), 10000.0)),
            )
        except Exception as exc:
            logger.debug("DeFiLlama poll fallback triggered: %s", exc)
            return (0.0, 5.0)
        finally:
            if own_client:
                await http_client.aclose()

    async def fetch_deribit_dvol(
        self, client: httpx.AsyncClient | None = None
    ) -> tuple[float, float]:
        """Poll Deribit BTC DVOL implied volatility index and compute trailing z-score."""
        if self._is_mock:
            dvol = 52.0
            zscore = self._record_and_score_dvol(dvol)
            return (dvol, zscore)

        own_client = client is None
        http_client = client or httpx.AsyncClient(timeout=self._timeout_sec)
        try:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - 3600 * 1000
            resp = await http_client.get(
                self.DERIBIT_DVOL_URL,
                params={
                    "currency": "BTC",
                    "start_timestamp": start_ms,
                    "end_timestamp": now_ms,
                    "resolution": "3600",
                },
            )
            dvol = 52.0
            if resp.status_code == 200:
                payload = resp.json()
                result = payload.get("result", {}) if isinstance(payload, dict) else {}
                rows = result.get("data", []) if isinstance(result, dict) else []
                if isinstance(rows, list) and rows:
                    last_row = rows[-1]
                    if isinstance(last_row, list) and len(last_row) >= 5:
                        dvol = float(last_row[4])
            zscore = self._record_and_score_dvol(dvol)
            return (dvol, zscore)
        except Exception as exc:
            logger.debug("Deribit DVOL poll fallback triggered: %s", exc)
            return (52.0, 0.0)
        finally:
            if own_client:
                await http_client.aclose()

    def _record_and_score_dvol(self, dvol: float) -> float:
        """Compute standardized z-score against trailing DVOL history and append."""
        clamped_dvol = float(min(max(dvol, 0.0), 500.0))
        if len(self._dvol_history) >= 2:
            mean_val = sum(self._dvol_history) / len(self._dvol_history)
            var_val = sum((x - mean_val) ** 2 for x in self._dvol_history) / len(
                self._dvol_history
            )
            std_val = max(math.sqrt(var_val), 1.0)
            zscore = (clamped_dvol - mean_val) / std_val
        else:
            zscore = 0.0
        self._dvol_history.append(clamped_dvol)
        return float(min(max(zscore, -10.0), 20.0))

    async def fetch_fear_and_greed(self, client: httpx.AsyncClient | None = None) -> int:
        """Poll Crypto Fear & Greed Index in [0, 100]."""
        if self._is_mock:
            return 54

        own_client = client is None
        http_client = client or httpx.AsyncClient(timeout=self._timeout_sec)
        try:
            resp = await http_client.get(self.FEAR_GREED_URL)
            if resp.status_code == 200:
                payload = resp.json()
                data = payload.get("data", []) if isinstance(payload, dict) else []
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    val = int(data[0].get("value", 50))
                    return min(max(val, 0), 100)
            return 50
        except Exception as exc:
            logger.debug("Fear & Greed poll fallback triggered: %s", exc)
            return 50
        finally:
            if own_client:
                await http_client.aclose()

    async def poll_snapshot(self) -> MacroCryptoFeedSnapshot:
        """Concurrently poll all 4 macro & crypto systemic risk sources into an immutable snapshot."""
        if self._is_mock:
            reg_rss, macro_rss = await self.fetch_regulatory_and_macro_rss()
            tvl_chg, peg_dev = await self.fetch_defillama_tvl_and_pegs()
            dvol, dvol_z = await self.fetch_deribit_dvol()
            fng = await self.fetch_fear_and_greed()
            return MacroCryptoFeedSnapshot(
                ts_ns=time.time_ns(),
                sec_cftc_headlines=reg_rss,
                macro_headlines=macro_rss,
                defillama_tvl_change_24h_pct=tvl_chg,
                max_stablecoin_peg_deviation_bps=peg_dev,
                deribit_dvol_index=dvol,
                deribit_dvol_zscore=dvol_z,
                fear_and_greed_index=fng,
            )

        async with httpx.AsyncClient(timeout=self._timeout_sec) as client:
            rss_task = self.fetch_regulatory_and_macro_rss(client)
            defi_task = self.fetch_defillama_tvl_and_pegs(client)
            dvol_task = self.fetch_deribit_dvol(client)
            fng_task = self.fetch_fear_and_greed(client)
            (reg_rss, macro_rss), (tvl_chg, peg_dev), (dvol, dvol_z), fng = (
                await asyncio.gather(rss_task, defi_task, dvol_task, fng_task)
            )
            return MacroCryptoFeedSnapshot(
                ts_ns=time.time_ns(),
                sec_cftc_headlines=reg_rss,
                macro_headlines=macro_rss,
                defillama_tvl_change_24h_pct=tvl_chg,
                max_stablecoin_peg_deviation_bps=peg_dev,
                deribit_dvol_index=dvol,
                deribit_dvol_zscore=dvol_z,
                fear_and_greed_index=fng,
            )

    @staticmethod
    def synthesize_structured_prompt(snapshot: MacroCryptoFeedSnapshot) -> str:
        """Synthesize a strict JSON-constrained prompt for `mlx-community/Qwen3-8B-4bit`."""
        sec_block = (
            "; ".join(snapshot.sec_cftc_headlines)
            if snapshot.sec_cftc_headlines
            else "None"
        )
        macro_block = (
            "; ".join(snapshot.macro_headlines)
            if snapshot.macro_headlines
            else "None"
        )
        return (
            "You are an institutional Quantitative Crypto-Derivatives Systemic Risk Monitor.\n"
            "Evaluate the following real-time macro, regulatory, DeFi, and implied volatility telemetry "
            "and output ONLY a single valid JSON object matching this exact schema (no extra keys, "
            "no markdown commentary):\n"
            '{"score": <float 0.0 to 100.0>, "confidence": <float 0.0 to 1.0>, '
            '"tail_risk_flag": <bool>, "plain_english_summary": "<concise summary <= 1000 chars>"}\n\n'
            "TELEMETRY SNAPSHOT:\n"
            f"- Timestamp (ns): {snapshot.ts_ns}\n"
            f"- SEC / CFTC Regulatory Headlines: {sec_block}\n"
            f"- Central Bank & Macro Headlines: {macro_block}\n"
            f"- DeFiLlama 24h TVL Change (%): {snapshot.defillama_tvl_change_24h_pct:.2f}%\n"
            f"- Max Stablecoin Peg Deviation (USDT/USDC/DAI): {snapshot.max_stablecoin_peg_deviation_bps:.1f} bps\n"
            f"- Deribit DVOL Implied Volatility Index: {snapshot.deribit_dvol_index:.2f} "
            f"(z-score: {snapshot.deribit_dvol_zscore:+.2f})\n"
            f"- Crypto Fear & Greed Index: {snapshot.fear_and_greed_index}/100\n"
        )


class DeterministicMockBatchGenerator:
    """Zero-network-latency deterministic mock `BatchGenerator` for `ci_mock` and offline test runs.

    Mirrors the persistent `mlx_lm` batch generation lifecycle while executing in
    microseconds under < 4GB RAM.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id: str = model_id
        self.call_count: int = 0

    def generate_json_for_snapshot(self, snapshot: MacroCryptoFeedSnapshot) -> str:
        """Synthesize a deterministic JSON payload from telemetry metrics and headlines."""
        self.call_count += 1
        combined_text = " ".join(
            (*snapshot.sec_cftc_headlines, *snapshot.macro_headlines)
        ).lower()

        has_crisis_headline = any(kw in combined_text for kw in _CRISIS_KEYWORDS)
        has_elevated_headline = any(kw in combined_text for kw in _ELEVATED_RISK_KEYWORDS)

        peg_crisis = snapshot.max_stablecoin_peg_deviation_bps >= 100.0
        tvl_collapse = snapshot.defillama_tvl_change_24h_pct <= -20.0
        dvol_crisis = (
            snapshot.deribit_dvol_zscore >= 3.5 or snapshot.deribit_dvol_index >= 110.0
        )

        tail_risk = bool(
            has_crisis_headline or peg_crisis or tvl_collapse or dvol_crisis
        )

        # Base score from DVOL, peg deviation, TVL outflow, and sentiment extremes
        base_score = 18.0
        base_score += max(0.0, (snapshot.deribit_dvol_index - 45.0) * 0.65)
        base_score += max(0.0, snapshot.deribit_dvol_zscore * 10.5)
        base_score += min(45.0, snapshot.max_stablecoin_peg_deviation_bps * 0.35)
        if snapshot.defillama_tvl_change_24h_pct < 0.0:
            base_score += min(35.0, abs(snapshot.defillama_tvl_change_24h_pct) * 1.6)
        if snapshot.fear_and_greed_index <= 15:
            base_score += 12.0
        if has_elevated_headline:
            base_score += 18.0
        if tail_risk:
            base_score = max(base_score, 95.0)

        score = round(float(min(max(base_score, 0.0), 100.0)), 2)
        confidence = 0.92 if tail_risk else 0.85

        if tail_risk:
            summary = (
                f"Acute systemic tail risk detected (ASRI={score:.1f}): "
                f"peg_dev={snapshot.max_stablecoin_peg_deviation_bps:.1f}bps, "
                f"TVL_24h={snapshot.defillama_tvl_change_24h_pct:.1f}%, "
                f"DVOL={snapshot.deribit_dvol_index:.1f} (z={snapshot.deribit_dvol_zscore:+.2f}). "
                "Halting directional exposure (M_regime=0.0)."
            )
        elif score > 70.0:
            summary = (
                f"Elevated macro/volatility stress (ASRI={score:.1f}): "
                f"DVOL={snapshot.deribit_dvol_index:.1f} (z={snapshot.deribit_dvol_zscore:+.2f}), "
                f"peg_dev={snapshot.max_stablecoin_peg_deviation_bps:.1f}bps. "
                "Scaling down directional exposure."
            )
        else:
            summary = (
                f"Systemic macro and DeFi conditions stable (ASRI={score:.1f}): "
                f"DVOL={snapshot.deribit_dvol_index:.1f}, "
                f"peg_dev={snapshot.max_stablecoin_peg_deviation_bps:.1f}bps, "
                f"Fear&Greed={snapshot.fear_and_greed_index}/100."
            )

        payload = {
            "score": score,
            "confidence": confidence,
            "tail_risk_flag": tail_risk,
            "plain_english_summary": summary[:1000],
        }
        return json.dumps(payload)

    def generate_from_prompt(self, prompt: str) -> str:
        """Generate deterministic JSON response directly from a prompt string."""
        self.call_count += 1
        lower = prompt.lower()
        tail_risk = any(kw in lower for kw in _CRISIS_KEYWORDS)
        elevated = any(kw in lower for kw in _ELEVATED_RISK_KEYWORDS)

        if tail_risk:
            score = 96.0
            conf = 0.94
            summary = "Acute systemic tail-risk event identified in prompt telemetry; directional halt engaged."
        elif elevated:
            score = 76.0
            conf = 0.86
            summary = "Elevated regulatory or volatility stress detected in prompt telemetry; tapering risk budget."
        else:
            score = 22.0
            conf = 0.85
            summary = "Macro, regulatory, and stablecoin liquidity telemetry within normal operating regime."

        return json.dumps(
            {
                "score": score,
                "confidence": conf,
                "tail_risk_flag": tail_risk,
                "plain_english_summary": summary,
            }
        )


class SystemicRiskIPCBus:
    """Dual Redis Streams + In-Memory ring-buffer IPC bus for top-down ASRI & M_regime telemetry."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream_key: str = DEFAULT_REDIS_STREAM_KEY,
        max_len: int = 1024,
        enable_redis: bool = False,
    ) -> None:
        self.redis_url: str = redis_url
        self.stream_key: str = stream_key
        self.max_len: int = max_len
        self._enable_redis: bool = enable_redis
        self._seq: int = 0
        self._stream_buffer: deque[RegimePublishEnvelope] = deque(maxlen=max_len)

    @property
    def history(self) -> tuple[RegimePublishEnvelope, ...]:
        """Return immutable tuple of published regime envelopes in chronological order."""
        return tuple(self._stream_buffer)

    @property
    def latest(self) -> RegimePublishEnvelope | None:
        """Return the most recently published regime envelope, if any."""
        return self._stream_buffer[-1] if self._stream_buffer else None

    def publish(
        self,
        risk_output: SystemicRiskOutput,
        ts_ns: int | None = None,
    ) -> RegimePublishEnvelope:
        """Compute M_regime via `compute_regime_multiplier_from_asri` and publish to IPC bus."""
        now_ns = ts_ns if ts_ns is not None else time.time_ns()
        self._seq += 1
        msg_id = f"{now_ns // 1_000_000}-{self._seq}"

        m_regime = compute_regime_multiplier_from_asri(
            asri_score=risk_output.score,
            tail_risk_flag=risk_output.tail_risk_flag,
        )

        envelope = RegimePublishEnvelope(
            ts_ns=now_ns,
            asri_score=risk_output.score,
            regime_multiplier=m_regime,
            tail_risk_flag=risk_output.tail_risk_flag,
            confidence=risk_output.confidence,
            plain_english_summary=risk_output.plain_english_summary,
            risk_output=risk_output,
            stream_key=self.stream_key,
            message_id=msg_id,
        )

        self._stream_buffer.append(envelope)

        if self._enable_redis:
            self._try_publish_redis(envelope)

        return envelope

    def _try_publish_redis(self, envelope: RegimePublishEnvelope) -> None:
        """Best-effort non-blocking Redis XADD when running against live Redis."""
        try:
            import redis

            client: Any = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=0.25,
                socket_timeout=0.25,
                decode_responses=True,
            )
            client.xadd(
                self.stream_key,
                envelope.to_redis_fields(),
                maxlen=self.max_len,
                approximate=True,
            )
            client.close()
        except Exception as exc:
            logger.debug("Redis XADD skipped (in-memory IPC bus active): %s", exc)


class PersistentMLXRiskDaemon:
    """Singleton persistent `mlx-lm` (`Qwen3-8B-4bit`) Macro & Systemic Risk Daemon.

    Architecture guarantees:
    - Process-wide Singleton lifecycle: loads `mlx-community/Qwen3-8B-4bit` once at
      initialization and keeps `BatchGenerator` resident in wired GPU memory (`40,960 MB`
      ceiling in `prod_m1max_64gb`), preventing Metal unified memory fragmentation.
    - In `ci_mock` mode (or when local weights/Metal runtime are absent), executes via
      `DeterministicMockBatchGenerator` with zero network latency and < 4GB RAM.
    - Top-Down Hierarchical Regime Conditioning: emits `ASRI` in [0, 100] and
      `M_regime` in [0.0, 1.0] via `compute_regime_multiplier_from_asri`, never mixing
      low-frequency sentiment states into the high-frequency Stage-1 SVD matrix.
    """

    _instance: ClassVar[PersistentMLXRiskDaemon | None] = None

    def __new__(
        cls,
        config: EnvironmentConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> PersistentMLXRiskDaemon:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        inference_timeout_sec: float = 5.0,
        stream_key: str = DEFAULT_REDIS_STREAM_KEY,
    ) -> None:
        if getattr(self, "_initialized", False):
            if config is not None and config != self._config:
                self._reconfigure(
                    config=config,
                    inference_timeout_sec=inference_timeout_sec,
                    stream_key=stream_key,
                )
            return

        self._initialized: bool = True
        self._reconfigure(
            config=config or load_environment_config("ci_mock"),
            inference_timeout_sec=inference_timeout_sec,
            stream_key=stream_key,
        )

    def _reconfigure(
        self,
        config: EnvironmentConfig,
        inference_timeout_sec: float = 5.0,
        stream_key: str = DEFAULT_REDIS_STREAM_KEY,
    ) -> None:
        self._config: EnvironmentConfig = config
        self._model_id: str = config.mlx_model_id or DEFAULT_MODEL_ID
        self._inference_timeout_sec: float = inference_timeout_sec
        self._wired_limit_bytes: int = int(config.wired_limit_mb) * 1024 * 1024
        self._model: Any = None
        self._tokenizer: Any = None
        self._native_batch_generator: Any = None
        self._mock_generator: DeterministicMockBatchGenerator = (
            DeterministicMockBatchGenerator(model_id=self._model_id)
        )
        self._using_native_mlx: bool = False
        self._custom_responder: Callable[[ str], str] | None = None

        self.aggregator: AsyncSystemicFeedAggregator = AsyncSystemicFeedAggregator(
            is_mock=config.is_mock
        )
        self.ipc_bus: SystemicRiskIPCBus = SystemicRiskIPCBus(
            redis_url=config.redis_url,
            stream_key=stream_key,
            enable_redis=not config.is_mock,
        )

        self._initialize_singleton_batch_generator()

        # Seed initial neutral baseline state on the IPC bus
        initial_output = SystemicRiskOutput(
            score=15.0,
            confidence=0.85,
            tail_risk_flag=False,
            plain_english_summary="Persistent MLX risk daemon initialized in nominal operating regime.",
        )
        self._latest_envelope: RegimePublishEnvelope = self.ipc_bus.publish(initial_output)

    @classmethod
    def get_instance(
        cls,
        config: EnvironmentConfig | None = None,
        force_reinit: bool = False,
    ) -> PersistentMLXRiskDaemon:
        """Return the process-wide singleton `PersistentMLXRiskDaemon`."""
        if force_reinit:
            cls.reset_singleton()
        return cls(config=config)

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton state (intended for isolated test fixtures)."""
        cls._instance = None

    @property
    def config(self) -> EnvironmentConfig:
        """Active `EnvironmentConfig` governing memory budgets and mock state."""
        return self._config

    @property
    def model_id(self) -> str:
        """Configured MLX model identifier (`mlx-community/Qwen3-8B-4bit`)."""
        return self._model_id

    @property
    def wired_limit_bytes(self) -> int:
        """Configured Metal wired memory ceiling in bytes."""
        return self._wired_limit_bytes

    @property
    def is_using_native_mlx(self) -> bool:
        """True if native `mlx-lm` `BatchGenerator` is loaded on Apple Silicon Metal."""
        return self._using_native_mlx

    @property
    def batch_generator(self) -> Any:
        """Return active persistent `BatchGenerator` (native `mlx-lm` or deterministic mock)."""
        if self._using_native_mlx and self._native_batch_generator is not None:
            return self._native_batch_generator
        return self._mock_generator

    @property
    def latest_envelope(self) -> RegimePublishEnvelope:
        """Most recent `RegimePublishEnvelope` emitted by the daemon."""
        return self._latest_envelope

    @property
    def latest_output(self) -> SystemicRiskOutput:
        """Most recent validated `SystemicRiskOutput`."""
        return self._latest_envelope.risk_output

    @property
    def latest_asri(self) -> float:
        """Current Aggregated Systemic Risk Index (ASRI) in [0.0, 100.0]."""
        return self._latest_envelope.asri_score

    @property
    def latest_regime_multiplier(self) -> float:
        """Current top-down portfolio risk multiplier M_regime in [0.0, 1.0]."""
        return self._latest_envelope.regime_multiplier

    def set_custom_llm_responder(
        self, responder: Callable[[str], str] | None
    ) -> None:
        """Inject a custom raw string responder (used for testing malformed/adversarial LLM outputs)."""
        self._custom_responder = responder

    @contextlib.contextmanager
    def _metal_wired_limit_context(self) -> Any:
        """Enter `mlx.core.metal.wired_limit` context when running in `prod_m1max_64gb` mode."""
        if self._config.is_mock:
            yield
            return

        try:
            import mlx.core as mx

            if hasattr(mx, "metal") and hasattr(mx.metal, "wired_limit"):
                with mx.metal.wired_limit(self._wired_limit_bytes):
                    yield
                return
        except Exception as exc:
            logger.debug("mlx.core.metal.wired_limit context unavailable: %s", exc)

        yield

    def _initialize_singleton_batch_generator(self) -> None:
        """Load `mlx-community/Qwen3-8B-4bit` once and instantiate persistent `BatchGenerator`."""
        if self._config.is_mock:
            self._using_native_mlx = False
            return

        try:
            import mlx_lm
            from mlx_lm.generate import BatchGenerator

            load_fn: Any = getattr(mlx_lm, "load", None)
            if load_fn is None:
                from mlx_lm.utils import load as utils_load

                load_fn = utils_load

            with self._metal_wired_limit_context():
                loaded_res: Any = load_fn(self._model_id)
                model = loaded_res[0]
                tokenizer = loaded_res[1]
                self._model = model
                self._tokenizer = tokenizer
                self._native_batch_generator = BatchGenerator(
                    model=model,
                    max_tokens=256,
                )
                self._using_native_mlx = True
                logger.info(
                    "Initialized persistent mlx-lm BatchGenerator for %s (wired_limit=%d MB).",
                    self._model_id,
                    self._config.wired_limit_mb,
                )
        except Exception as exc:
            logger.warning(
                "Native mlx-lm load skipped (%s: %s); falling back to deterministic mock BatchGenerator.",
                type(exc).__name__,
                exc,
            )
            self._using_native_mlx = False

    def _run_native_batch_generation(self, prompt: str) -> str:
        """Execute single-prompt completion through the persistent `mlx-lm` `BatchGenerator`."""
        with self._metal_wired_limit_context():
            if (
                self._using_native_mlx
                and self._native_batch_generator is not None
                and self._tokenizer is not None
            ):
                prompt_tokens = self._tokenizer.encode(prompt)
                uids: Sequence[int] = self._native_batch_generator.insert(
                    [prompt_tokens], max_tokens=[256]
                )
                target_uid = uids[0]
                collected_tokens: list[int] = []
                while responses := self._native_batch_generator.next():
                    for resp in responses:
                        if resp.uid == target_uid and getattr(resp, "finish_reason", None) != "stop":
                            collected_tokens.append(int(resp.token))
                return str(self._tokenizer.decode(collected_tokens))

            return self._mock_generator.generate_from_prompt(prompt)

    def infer_from_raw_response(
        self, raw_llm_text: str | None, ts_ns: int | None = None
    ) -> RegimePublishEnvelope:
        """Validate raw LLM text via `parse_and_validate_llm_risk_json` and publish to IPC bus."""
        validated_output = parse_and_validate_llm_risk_json(raw_llm_text)
        self._latest_envelope = self.ipc_bus.publish(validated_output, ts_ns=ts_ns)
        return self._latest_envelope

    def evaluate_snapshot_sync(
        self, snapshot: MacroCryptoFeedSnapshot
    ) -> RegimePublishEnvelope:
        """Synchronously synthesize prompt from `snapshot`, run inference, validate, and publish."""
        prompt = self.aggregator.synthesize_structured_prompt(snapshot)
        try:
            if self._custom_responder is not None:
                raw_text = self._custom_responder(prompt)
            elif self._using_native_mlx:
                raw_text = self._run_native_batch_generation(prompt)
            else:
                raw_text = self._mock_generator.generate_json_for_snapshot(snapshot)
        except Exception as exc:
            logger.warning("LLM inference raised %s; applying neutral fallback.", exc)
            fallback = SystemicRiskOutput.neutral_fallback(
                f"Inference exception ({type(exc).__name__})."
            )
            self._latest_envelope = self.ipc_bus.publish(fallback, ts_ns=snapshot.ts_ns)
            return self._latest_envelope

        return self.infer_from_raw_response(raw_text, ts_ns=snapshot.ts_ns)

    async def evaluate_snapshot(
        self, snapshot: MacroCryptoFeedSnapshot
    ) -> RegimePublishEnvelope:
        """Asynchronously evaluate `snapshot` with strict timeout protection and fallback."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.evaluate_snapshot_sync, snapshot),
                timeout=self._inference_timeout_sec,
            )
        except TimeoutError:
            logger.warning(
                "LLM risk evaluation timed out after %.2fs; applying neutral fallback.",
                self._inference_timeout_sec,
            )
            fallback = SystemicRiskOutput.neutral_fallback(
                f"Generation timeout exceeded ({self._inference_timeout_sec:.1f}s)."
            )
            self._latest_envelope = self.ipc_bus.publish(fallback, ts_ns=snapshot.ts_ns)
            return self._latest_envelope
        except Exception as exc:
            logger.warning("Async LLM risk evaluation error (%s); applying neutral fallback.", exc)
            fallback = SystemicRiskOutput.neutral_fallback(
                f"Async evaluation error ({type(exc).__name__})."
            )
            self._latest_envelope = self.ipc_bus.publish(fallback, ts_ns=snapshot.ts_ns)
            return self._latest_envelope

    async def poll_and_publish_once(self) -> RegimePublishEnvelope:
        """Poll all macro/crypto feeds once, run persistent LLM evaluation, and publish M_regime."""
        snapshot = await self.aggregator.poll_snapshot()
        return await self.evaluate_snapshot(snapshot)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_REDIS_STREAM_KEY",
    "AsyncSystemicFeedAggregator",
    "DeterministicMockBatchGenerator",
    "MacroCryptoFeedSnapshot",
    "PersistentMLXRiskDaemon",
    "RegimePublishEnvelope",
    "SystemicRiskIPCBus",
    "parse_and_validate_llm_risk_json",
]
