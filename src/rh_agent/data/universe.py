"""Universe construction: the eligible tradable set after liquidity filters,
plus a cheap prescreen funnel so deep multi-factor scoring only runs on the
most promising survivors.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..logging_setup import get_logger
from .market_data import MarketData

log = get_logger("universe")


@dataclass
class Candidate:
    ticker: str
    price: float
    market_cap: float
    adv: float            # avg dollar volume
    hist_days: int
    mom_63: float         # ~3-month price return (prescreen rank)
    sector: str = "Unknown"


def _light(md: MarketData, ticker: str) -> Candidate | None:
    q = md.get_quote(ticker)
    px = q.price if q else None
    df = md.get_prices(ticker)
    if px is None and df is not None and len(df):
        px = float(df["close"].iloc[-1])
    if px is None or df is None or len(df) < 2:
        return None
    comp = md.get_company(ticker)
    mcap = comp.get("market_cap") or 0
    vol = df["volume"].iloc[-21:].mean() if "volume" in df.columns else 0
    adv = float(px * (vol or 0))
    hist = len(df)
    mom = float(df["close"].iloc[-1] / df["close"].iloc[-63] - 1) if hist >= 63 else 0.0
    return Candidate(ticker, float(px), float(mcap or 0), adv, hist, mom,
                     sector=comp.get("sector") or "Unknown")


def build_universe(md: MarketData, cfg: Config, raw: list[str] | None = None) -> list[str]:
    u = cfg.get("universe", {})
    liq = u.get("liquidity", {})
    blacklist = set(u.get("blacklist", []))
    excl_sectors = set(u.get("exclude_sectors", []))
    tickers = raw if raw is not None else md.list_universe()
    if not tickers:
        log.warning("empty raw universe; provide a snapshot or enable a universe provider")
        return []

    # Cap the raw scan for a single pass (production should pre-filter via screener).
    hard_cap = int(u.get("scan_cap", 1500))
    if len(tickers) > hard_cap:
        log.info("raw universe %d > scan_cap %d; truncating this pass", len(tickers), hard_cap)
        tickers = tickers[:hard_cap]

    passed: list[Candidate] = []
    for t in tickers:
        if t in blacklist:
            continue
        try:
            c = _light(md, t)
        except Exception as e:
            log.debug("light fetch %s failed: %s", t, e)
            c = None
        if not c:
            continue
        if c.sector in excl_sectors:
            continue
        if c.price < liq.get("min_price", 5):
            continue
        if c.price > liq.get("max_price", 1e9):
            continue
        if c.market_cap and c.market_cap < liq.get("min_market_cap", 0):
            continue
        if c.adv and c.adv < liq.get("min_avg_dollar_volume", 0):
            continue
        if c.hist_days < liq.get("min_history_days", 200):
            continue
        passed.append(c)

    log.info("universe: %d/%d names pass liquidity filters", len(passed), len(tickers))

    pre = u.get("prescreen", {})
    if pre.get("enabled", True) and len(passed) > pre.get("max_candidates", 250):
        passed.sort(key=lambda c: c.mom_63, reverse=True)
        passed = passed[: pre.get("max_candidates", 250)]
        log.info("prescreen kept top %d by 3-month momentum", len(passed))

    return [c.ticker for c in passed]
