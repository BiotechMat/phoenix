"""P2.4 tests: option greeks, surface metrics on a recorded Deribit surface, HLP child-account positions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.contracts import HyperliquidWhaleVaultMetrics
from core.options_surface import black_scholes_delta_gamma, build_option_points, compute_surface_metrics, instrument_specs_from_names
from sleeves.sleeve_1_derivatives_alpha import DerivativesAlphaSleeve

FIX = Path(__file__).resolve().parent / "fixtures"


def _surface() -> dict[str, object]:
    data: dict[str, object] = json.loads((FIX / "deribit_btc_surface.json").read_text(encoding="utf-8"))
    return data


def test_black_scholes_greeks_basic_properties() -> None:
    c_delta, c_gamma = black_scholes_delta_gamma(100.0, 100.0, 0.25, 0.5, True)
    p_delta, p_gamma = black_scholes_delta_gamma(100.0, 100.0, 0.25, 0.5, False)
    assert 0.5 < c_delta < 0.56 and c_delta - p_delta == pytest.approx(1.0)
    assert c_gamma == pytest.approx(p_gamma) and c_gamma > 0.0
    sd = 0.5 * math.sqrt(0.25)
    assert c_gamma == pytest.approx(math.exp(-0.5 * (0.5 * sd) ** 2) / math.sqrt(2 * math.pi) / (100.0 * sd))
    assert black_scholes_delta_gamma(100.0, 50.0, 0.0, 0.5, True) == (1.0, 0.0)  # expired in the money


def test_specs_parsed_from_names_match_deribit_instruments() -> None:
    fx = _surface()
    book = fx["book_summary"]
    instruments = {i["instrument_name"]: i for i in fx["instruments"]}  # type: ignore[union-attr,index]
    specs = instrument_specs_from_names(book)  # type: ignore[arg-type]
    assert len(specs) == len(book)  # type: ignore[arg-type]
    for spec in specs:
        ref = instruments[spec["instrument_name"]]
        assert spec["expiration_timestamp"] == ref["expiration_timestamp"]
        assert spec["strike"] == ref["strike"] and spec["option_type"] == ref["option_type"]


def test_surface_metrics_on_recorded_deribit_surface() -> None:
    fx = _surface()
    now_ms = int(fx["recorded_ms"])  # type: ignore[call-overload]
    m = compute_surface_metrics(fx["book_summary"], fx["instruments"], now_ms)  # type: ignore[arg-type]
    assert m.n_options == 966
    assert m.rr25_30d == pytest.approx(-4.49, abs=0.01)  # puts bid over calls: negative risk reversal
    assert m.rr25_90d == pytest.approx(-4.25, abs=0.01)
    assert m.atm_iv_near == pytest.approx(40.93) and m.atm_iv_far == pytest.approx(37.84)
    assert m.term_structure_slope == pytest.approx((37.84 - 40.93) / 40.93)
    assert m.net_gex_usd_per_1pct == pytest.approx(m.call_gex_usd_per_1pct - m.put_gex_usd_per_1pct)
    assert 30e6 < m.net_gex_usd_per_1pct < 60e6
    assert m.expiry_30d_ms is not None and 20.0 <= (m.expiry_30d_ms - now_ms) / 86_400_000 <= 40.0
    # Our delta agrees with Deribit's own ticker for the same option (0.55927) within 0.01.
    pt = next(p for p in build_option_points(fx["book_summary"], fx["instruments"], now_ms) if p.instrument_name == "BTC-27NOV26-77000-C")  # type: ignore[arg-type]
    assert pt.delta == pytest.approx(0.55927, abs=0.01) and pt.gamma == pytest.approx(5e-05, rel=0.05)


def test_sleeve_parser_uses_real_surface_metrics() -> None:
    fx = _surface()
    sleeve = DerivativesAlphaSleeve()
    surf = sleeve._parse_deribit_book_summary("BTC", fx["book_summary"], int(fx["recorded_ms"]) * 1_000_000)  # type: ignore[arg-type,call-overload]
    assert surf.skew_25d == pytest.approx(-0.0449, abs=1e-4)
    assert surf.term_structure_slope == pytest.approx((37.84 - 40.93) / 40.93, abs=1e-5)
    assert surf.dvol_annualized == pytest.approx(40.93)
    assert sleeve.last_surface_metrics["BTC"].rr25_90d == pytest.approx(-4.25, abs=0.01)
    sig = sleeve.generate_alpha_signal(options_surface=surf, instrument_id="BTC", timestamp_ns=1)
    assert sig.skew_25d == pytest.approx(-0.0449, abs=1e-4)
    assert sig.primary_direction in (-1, 0)  # put skew below the -1.5 vol threshold never reads bullish


def test_hlp_positions_are_summed_across_child_accounts_at_mark() -> None:
    fx = json.loads((FIX / "hlp_vault_children.json").read_text(encoding="utf-8"))
    sleeve = DerivativesAlphaSleeve()
    metrics = sleeve._parse_hlp_vault_details("0xdfc24b077bc1425ad1dea75bcb6f8158e10df303", {"details": fx["vault_details"], "states": fx["children"]}, 1)
    assert metrics.accounts_read == 7
    assert metrics.net_delta_by_coin_usd["BTC"] == pytest.approx(8876.143168 - 4676.6928)
    assert metrics.net_delta_by_coin_usd["SOL"] == pytest.approx(41882.361402 - 39677.884484)
    assert metrics.hlp_crowding == "BALANCED"  # a few thousand dollars against a $2M threshold
    assert metrics.tvl_usd > 1_000_000.0


def test_crowding_is_classified_per_coin_when_available() -> None:
    sleeve = DerivativesAlphaSleeve()
    vault = HyperliquidWhaleVaultMetrics(net_delta_usd=0.0, hlp_crowding="BALANCED", net_delta_by_coin_usd={"BTC": -5_000_000.0, "ETH": 3_000_000.0}, accounts_read=3)
    btc = sleeve.generate_alpha_signal(vault_metrics=vault, instrument_id="BTC", timestamp_ns=1)
    eth = sleeve.generate_alpha_signal(vault_metrics=vault, instrument_id="ETH", timestamp_ns=1)
    sol = sleeve.generate_alpha_signal(vault_metrics=vault, instrument_id="SOL", timestamp_ns=1)
    assert btc.hlp_vault_crowding == "CROWDED_SHORT" and btc.primary_direction == -1  # HLP short: retail crowded long
    assert eth.hlp_vault_crowding == "CROWDED_LONG" and eth.primary_direction == 1
    assert sol.hlp_vault_crowding == "BALANCED"
