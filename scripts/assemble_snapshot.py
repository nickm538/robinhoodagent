#!/usr/bin/env python3
"""Assemble a normalised rh-agent snapshot from captured provider responses.

Reads:
  * a directory of FinancialDatasets `get_stock_prices` JSON dumps (one array
    of OHLCV rows per file; ticker is read from the rows themselves), and
  * an optional fundamentals JSON ({"tickers": {TICKER: {sector, market_cap,
    roe, ...}}}).

Writes a snapshot consumable by rh_agent.providers.snapshot.SnapshotProvider.

This is the bridge used to validate the engine on genuinely live data inside a
restricted-egress sandbox: the inputs are real API responses captured at run
time, not fabricated values.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_BENCH = {"SPY", "RSP", "VIX", "DIA", "QQQ", "^VIX"}


def load_price_files(prices_dir: str) -> dict[str, dict]:
    """Return {ticker: {date: row}} merged & de-duplicated across files."""
    rows_by_ticker: dict[str, dict] = defaultdict(dict)
    patterns = ["*get_stock_prices*", "*get_stock_prices*.txt", "*prices*.json", "*.json", "*.txt"]
    files: set[str] = set()
    for p in patterns:
        files.update(glob.glob(os.path.join(prices_dir, p)))
    for fp in sorted(files):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        rows = data.get("prices") if isinstance(data, dict) else data
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        if "close" not in rows[0] or "time" not in rows[0]:
            continue
        for r in rows:
            tk = r.get("ticker")
            t = r.get("time")
            if not tk or not t:
                continue
            rows_by_ticker[tk][t] = {
                "time": t, "open": r.get("open"), "high": r.get("high"),
                "low": r.get("low"), "close": r.get("close"),
                "adj_close": r.get("adj_close", r.get("close")), "volume": r.get("volume"),
            }
    return rows_by_ticker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-dir", required=True)
    ap.add_argument("--fundamentals", help="fundamentals JSON path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmarks", default=",".join(sorted(DEFAULT_BENCH)))
    args = ap.parse_args()

    bench_syms = {s.strip() for s in args.benchmarks.split(",") if s.strip()}
    price_rows = load_price_files(args.prices_dir)
    fund = {}
    if args.fundamentals and os.path.exists(args.fundamentals):
        fund = json.load(open(args.fundamentals)).get("tickers", {})

    snap = {"_captured_iso": datetime.now(timezone.utc).isoformat(),
            "_source": "FinancialDatasets.AI (live via MCP) — prices + fundamentals",
            "tickers": {}, "benchmarks": {}, "macro": {}}

    for tk, rowmap in price_rows.items():
        rows = [rowmap[d] for d in sorted(rowmap)]
        if len(rows) < 60:
            continue
        if tk in bench_syms:
            snap["benchmarks"][tk] = {"prices": rows}
            continue
        last = rows[-1]
        f = fund.get(tk, {})
        snap["tickers"][tk] = {
            "company": {"name": f.get("name", tk), "sector": f.get("sector", "Unknown"),
                        "market_cap": f.get("market_cap")},
            "quote": {"price": last["close"], "volume": last["volume"],
                      "prev_close": rows[-2]["close"] if len(rows) > 1 else None},
            "prices": rows,
            "fundamentals": {k: v for k, v in f.items()
                             if k not in ("name", "sector") and v is not None},
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(snap, open(args.out, "w"))
    print(f"snapshot written: {args.out}")
    print(f"  tickers:    {len(snap['tickers'])} ({', '.join(sorted(snap['tickers']))})")
    print(f"  benchmarks: {', '.join(sorted(snap['benchmarks']))}")
    for tk, t in sorted(snap["tickers"].items()):
        print(f"    {tk:5} bars={len(t['prices'])} last={t['quote']['price']} "
              f"sector={t['company']['sector']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
