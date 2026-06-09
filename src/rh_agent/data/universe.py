"""Universe construction: the eligible tradable set after liquidity filters,
plus a cheap prescreen funnel so deep multi-factor scoring only runs on the
most promising survivors.
"""
from __future__ import annotations

import math
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
    adv: float            # average dollar volume
    avg_volume: float
    hist_days: int
    mom_63: float         # ~3-month price return (prescreen rank)
    day_change_pct: float = 0.0
    rel_volume: float = 0.0
    dollar_volume_today: float = 0.0
    breakout_20: float = 0.0
    intraday_score: float = 0.0
    sector: str = "Unknown"


def _light(md: MarketData, ticker: str, *, max_quote_age_seconds: float | None = None) -> Candidate | None:
    q = (
        md.get_quote_for_risk(ticker, max_age_seconds=max_quote_age_seconds)
        if max_quote_age_seconds is not None
        else md.get_quote(ticker)
    )
    px = q.price if q else None
    df = md.get_prices(ticker)
    if px is None and df is not None and len(df):
        px = float(df["close"].iloc[-1])
    if px is None or df is None or len(df) < 2:
        return None
    comp = md.get_company(ticker)
    mcap = comp.get("market_cap") or 0
    if "volume" in df.columns and len(df["volume"]):
        recent_volume = df["volume"].iloc[-22:-1] if len(df) >= 22 else df["volume"].iloc[-21:]
        avg_volume = float(recent_volume.mean() or 0)
        latest_volume = float((q.volume if q else 0) or df["volume"].iloc[-1] or 0)
    else:
        avg_volume = 0.0
        latest_volume = float(q.volume if q else 0)
    adv = float(px * (avg_volume or 0))
    dollar_volume_today = float(px * (latest_volume or 0))
    rel_volume = float(latest_volume / avg_volume) if avg_volume else 0.0
    day_change = q.day_change_pct if q and q.day_change_pct is not None else None
    if day_change is None and q and q.prev_close:
        day_change = (px / float(q.prev_close) - 1.0) * 100
    hist = len(df)
    # Guard against a zero/NaN historical close (dirty data in the wider universe)
    # — an unguarded divide yields inf momentum that would sort garbage to the top.
    prev_close = float(df["close"].iloc[-63]) if hist >= 63 else 0.0
    mom = (float(df["close"].iloc[-1]) / prev_close - 1.0) if prev_close > 0 else 0.0
    if not math.isfinite(mom):
        mom = 0.0
    high_20 = float(df["close"].iloc[-20:].max()) if hist >= 20 else px
    breakout = (px / high_20 - 1.0) if high_20 else 0.0
    return Candidate(ticker, float(px), float(mcap or 0), adv, avg_volume, hist, mom,
                     day_change_pct=float(day_change or 0.0),
                     rel_volume=rel_volume,
                     dollar_volume_today=dollar_volume_today,
                     breakout_20=breakout,
                     sector=comp.get("sector") or "Unknown")


def _intraday_score(c: Candidate) -> float:
    """Fast opportunity score for live discovery before expensive deep scoring."""
    positive_move = max(c.day_change_pct, 0.0)
    rel_volume = min(max(c.rel_volume, 0.0), 8.0)
    breakout = max(c.breakout_20, 0.0)
    momentum = max(c.mom_63, 0.0)
    return round((positive_move * 2.0) + (rel_volume * 10.0) + (breakout * 100.0)
                 + (momentum * 30.0), 4)


def _apply_intraday_radar(candidates: list[Candidate], cfg: Config) -> list[Candidate]:
    intraday = (cfg.get("universe.intraday", {}) or {})
    if not intraday.get("enabled", False):
        return candidates

    max_candidates = int(intraday.get("max_candidates", 120))
    min_candidates = int(intraday.get("min_candidates", 30))
    min_day_change = float(intraday.get("min_day_change_pct", 2.0))
    min_positive_change = float(intraday.get("min_positive_day_change_pct", 0.5))
    min_rel_volume = float(intraday.get("min_relative_volume", 1.5))
    min_dollar_volume = float(intraday.get("min_dollar_volume_today", 2_000_000))
    min_breakout = float(intraday.get("min_breakout_pct", 0.0))
    fallback = bool(intraday.get("fallback_to_liquid_universe", True))

    scored: list[Candidate] = []
    for c in candidates:
        c.intraday_score = _intraday_score(c)
        if c.dollar_volume_today < min_dollar_volume:
            continue
        price_runner = c.day_change_pct >= min_day_change
        volume_runner = c.rel_volume >= min_rel_volume and c.day_change_pct >= min_positive_change
        breakout_runner = c.breakout_20 >= min_breakout and c.day_change_pct >= min_positive_change
        if price_runner or volume_runner or breakout_runner:
            scored.append(c)

    if len(scored) < min_candidates and fallback:
        log.warning("intraday radar found %d candidates (<%d); using liquid universe fallback",
                    len(scored), min_candidates)
        scored = candidates

    scored.sort(key=lambda c: (c.intraday_score, c.day_change_pct, c.rel_volume), reverse=True)
    kept = scored[:max_candidates]
    log.info("intraday radar kept %d/%d candidates", len(kept), len(candidates))
    return kept


def build_universe(md: MarketData, cfg: Config, raw: list[str] | None = None) -> list[str]:
    u = cfg.get("universe", {})
    liq = u.get("liquidity", {})
    blacklist = set(u.get("blacklist", []))
    excl_sectors = set(u.get("exclude_sectors", []))
    tickers = raw if raw is not None else md.list_universe()

    # Seed today's market movers (top gainers / most-active) ahead of the base
    # list so the intraday radar actually has live runners to hunt — otherwise it
    # only ever sees the static liquid set and falls back every cycle.
    intraday_cfg = u.get("intraday", {}) or {}
    if (raw is None and intraday_cfg.get("enabled", False)
            and intraday_cfg.get("use_movers_feed", True)):
        try:
            movers = md.get_market_movers(int(intraday_cfg.get("movers_limit", 60))) or []
        except Exception as e:
            log.debug("movers feed unavailable: %s", e)
            movers = []
        if movers:
            seen: set[str] = set()
            merged: list[str] = []
            for t in list(movers) + list(tickers or []):
                if t and t not in seen:
                    seen.add(t)
                    merged.append(t)
            tickers = merged
            log.info("seeded %d market movers into the scan universe", len(movers))

    if not tickers:
        log.warning("empty raw universe; provide a snapshot or enable a universe provider")
        return []

    # Cap the raw scan for a single pass (production should pre-filter via screener).
    hard_cap = int(u.get("scan_cap", 1500))
    if len(tickers) > hard_cap:
        log.info("raw universe %d > scan_cap %d; truncating this pass", len(tickers), hard_cap)
        tickers = tickers[:hard_cap]

    intraday_cfg = u.get("intraday", {}) or {}
    if (intraday_cfg.get("enabled", False) and intraday_cfg.get("batch_quote_prefetch", True)
            and hasattr(md, "prefetch_quotes")):
        if hasattr(md, "clear_quote_prefetch"):
            md.clear_quote_prefetch()
        md.prefetch_quotes(tickers)

    quote_age = (
        float(intraday_cfg.get("quote_max_age_seconds", 60))
        if intraday_cfg.get("enabled", False)
        else None
    )

    def light(t: str) -> Candidate | None:
        try:
            return _light(md, t, max_quote_age_seconds=quote_age)
        except Exception as e:
            log.debug("light fetch %s failed: %s", t, e)
            return None

    # The light pass is I/O-bound (each provider rate-limits itself); fan out
    # like agent._gather does. Serially, a 400-name pass alone could outlast
    # the rebalance cadence and trip the scan watchdog.
    names = [t for t in tickers if t not in blacklist]
    workers = int(cfg.get("data.max_workers", 8))
    if workers > 1 and len(names) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            candidates = list(ex.map(light, names))
    else:
        candidates = [light(t) for t in names]

    passed: list[Candidate] = []
    for c in candidates:
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
        if c.adv < liq.get("min_avg_dollar_volume", 0):
            continue
        if c.avg_volume < liq.get("min_avg_volume_shares", 0):
            continue
        if c.hist_days < liq.get("min_history_days", 200):
            continue
        passed.append(c)

    log.info("universe: %d/%d names pass liquidity filters", len(passed), len(tickers))
    passed = _apply_intraday_radar(passed, cfg)

    pre = u.get("prescreen", {})
    if pre.get("enabled", True) and len(passed) > pre.get("max_candidates", 250):
        passed.sort(key=lambda c: c.mom_63, reverse=True)
        passed = passed[: pre.get("max_candidates", 250)]
        log.info("prescreen kept top %d by 3-month momentum", len(passed))

    return [c.ticker for c in passed]
