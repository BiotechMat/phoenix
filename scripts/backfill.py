#!/usr/bin/env python3
"""Backfill real 15m and 1h bars for BTC, ETH and SOL (REINFORCEMENT_PLAN task P1.2).

Examples::

    .venv/bin/python scripts/backfill.py --months 12
    .venv/bin/python scripts/backfill.py --start 2025-09-01 --end 2025-10-15 --venues okx --horizons 15m

Candles come from Bybit (primary) or OKX, funding from Hyperliquid, open interest from Bybit or OKX.
The exit code is 1 when any asset has a gap longer than two bars or no rows at all.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.backfill_storage import (  # noqa: E402
    DATA_DIR,
    Horizon,
    Venue,
    refresh_funding,
    run_backfill,
)


def _parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coins", default="BTC,ETH,SOL")
    parser.add_argument("--horizons", default="15m,1h")
    parser.add_argument("--months", type=int, default=12, help="Window length when --start is not given")
    parser.add_argument("--start", type=str, default=None, help="ISO date/time (UTC)")
    parser.add_argument("--end", type=str, default=None, help="ISO date/time (UTC), default now")
    parser.add_argument("--venues", default="bybit,okx", help="Candle/OI venue priority, comma separated")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pause", type=float, default=0.12, help="Seconds between paged requests")
    parser.add_argument("--no-mark-index", action="store_true", help="Fetch trade candles only (mark/spot fall back to close)")
    parser.add_argument("--no-open-interest", action="store_true", help="Skip open-interest history")
    parser.add_argument("--funding-only", action="store_true", help="Only re-fetch Hyperliquid funding into existing files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    end_dt = _parse_date(args.end) if args.end else datetime.now(timezone.utc)
    start_dt = _parse_date(args.start) if args.start else end_dt - timedelta(days=30 * args.months)
    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    horizons: list[Horizon] = []
    for h in args.horizons.split(","):
        h = h.strip()
        if h in ("15m", "1h"):
            horizons.append(h)
    venues: list[Venue] = []
    for v in args.venues.split(","):
        v = v.strip().lower()
        if v in ("bybit", "okx"):
            venues.append(v)

    if args.funding_only:
        for coin in coins:
            for h in horizons:
                n = refresh_funding(coin, h, data_dir=args.data_dir, pause_sec=args.pause)
                print(f"{coin}:{h} funding attached to {n} bars")
        return 0

    reports = run_backfill(
        coins,
        horizons,
        start_ms=int(start_dt.timestamp() * 1000),
        end_ms=int(end_dt.timestamp() * 1000),
        data_dir=args.data_dir,
        venues=venues,
        pause_sec=args.pause,
        include_mark_index=not args.no_mark_index,
        include_open_interest=not args.no_open_interest,
    )

    ok = True
    print(f"{'asset':<10}{'rows':>8}{'expected':>10}{'missing':>9}{'max gap':>9}{'OI rows':>9}{'fund rows':>11}  candles  OI      first -> last (UTC)")
    for key, rep in reports.items():
        first = datetime.fromtimestamp(rep.first_ts_ms / 1000, tz=timezone.utc).isoformat(timespec="minutes") if rep.first_ts_ms else "-"
        last = datetime.fromtimestamp(rep.last_ts_ms / 1000, tz=timezone.utc).isoformat(timespec="minutes") if rep.last_ts_ms else "-"
        print(
            f"{key:<10}{rep.rows:>8}{rep.expected_rows:>10}{rep.missing_rows:>9}{rep.max_gap_bars:>9}{rep.oi_rows:>9}{rep.funding_rows:>11}  "
            f"{rep.candle_source or '-':<8} {rep.oi_source or '-':<7} {first} -> {last}"
        )
        if rep.rows == 0 or not rep.passes_gap_rule:
            ok = False
    if not ok:
        print("Backfill incomplete: at least one asset has no rows or a gap longer than 2 bars.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
