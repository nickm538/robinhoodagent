"""MarketData facade — merges providers with per-section priority/fallback
and assembles a fully-populated TickerData for the engine.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ..config import Config
from ..factors.indicators import compute_indicators
from ..logging_setup import get_logger
from ..models import Quote, TickerData, utcnow
from ..providers.base import DataProvider, ProviderUnsupported

log = get_logger("market_data")


def _is_empty(res: Any) -> bool:
    if res is None:
        return True
    if isinstance(res, (pd.DataFrame, pd.Series)):
        return res.empty
    if isinstance(res, (dict, list, str, tuple)):
        return len(res) == 0
    return False


class MarketData:
    def __init__(self, config: Config, providers: dict[str, DataProvider]):
        self.cfg = config
        self.providers = {n: p for n, p in providers.items() if p and getattr(p, "enabled", True)}
        self.priority: dict = config.get("providers", {}) or {}
        self._quote_prefetch: dict[str, Quote] = {}

    def _order(self, section: str) -> list[str]:
        return [n for n in self.priority.get(section, []) if n in self.providers]

    def clear_quote_prefetch(self) -> None:
        self._quote_prefetch.clear()

    def _try(self, section: str, method: str, *args, **kw) -> Any:
        for name in self._order(section):
            p = self.providers[name]
            fn: Callable | None = getattr(p, method, None)
            if fn is None:
                continue
            try:
                res = fn(*args, **kw)
                if not _is_empty(res):
                    return res
            except ProviderUnsupported:
                continue
            except Exception as e:
                log.debug("%s.%s(%s) failed: %s", name, method, args, e)
        return None

    def _merge(self, section: str, method: str, *args) -> dict:
        """Fill-missing merge across providers (used for fundamentals/analyst)."""
        merged: dict = {}
        for name in self._order(section):
            p = self.providers[name]
            fn = getattr(p, method, None)
            if not fn:
                continue
            try:
                d = fn(*args)
            except (ProviderUnsupported, Exception):
                continue
            if isinstance(d, dict):
                for k, v in d.items():
                    if k != "source" and merged.get(k) in (None,) and v is not None:
                        merged[k] = v
                merged.setdefault("sources", []).append(name)
        return merged

    # ---- direct section accessors ----
    def _cached_quote(self, t: str) -> Quote | None:
        return self._quote_prefetch.get(t.upper())

    def prefetch_quotes(self, tickers: list[str]) -> int:
        """Batch-warm quotes via TwelveData when available for intraday radar."""
        td = self.providers.get("twelvedata")
        fn = getattr(td, "get_quotes_batch", None) if td else None
        if not fn or not tickers:
            return 0
        req = len([t for t in tickers if t])
        try:
            batch = fn([t.upper() for t in tickers if t])
        except Exception as e:
            log.warning("twelvedata batch quote prefetch failed (%d names): %s", req, e)
            return 0
        self._quote_prefetch.update(batch)
        if batch:
            log.info("prefetched %d/%d quotes via twelvedata batch", len(batch), req)
        else:
            log.warning("twelvedata batch prefetch returned 0/%d quotes", req)
        return len(batch)

    def get_quote(self, t: str) -> Quote | None:
        hit = self._cached_quote(t)
        if hit and hit.price:
            return hit
        return self._try("quote", "get_quote", t)

    def get_quote_for_risk(self, t: str, max_age_seconds: float = 180) -> Quote | None:
        """Fetch a quote for stop/TP decisions, bypassing stale disk cache when needed."""
        hit = self._cached_quote(t)
        if hit and hit.price:
            age = (utcnow() - hit.asof).total_seconds()
            if age <= max_age_seconds:
                return hit
        # Provider caches store raw payloads, not Quote.asof, so a cached payload
        # can otherwise be rewrapped with a fresh timestamp. Force a live read on
        # this path; it is used only where stale prices are dangerous.
        self._invalidate_quote_cache(t)
        section = "quote_risk" if self._order("quote_risk") else "quote"
        q = self._try(section, "get_quote", t)
        if q and q.price:
            age = (utcnow() - q.asof).total_seconds()
            if age <= max_age_seconds:
                return q
        # missing/stale -> invalidate disk cache and retry once
        self._invalidate_quote_cache(t)
        q = self._try(section, "get_quote", t)
        return q if q and q.price else None

    def _invalidate_quote_cache(self, ticker: str) -> None:
        key = ticker.upper()
        for p in self.providers.values():
            if hasattr(p, "invalidate_quote"):
                try:
                    p.invalidate_quote(key)
                    continue
                except Exception:
                    pass
            cache = getattr(p, "cache", None)
            if not cache:
                continue
            for ns in ("snap", "quote", "quotes", "mboum_quote"):
                try:
                    path = cache._path(ns, key)
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass

    def get_prices(self, t: str, start=None, end=None, interval="day") -> pd.DataFrame | None:
        return self._try("prices", "get_prices", t, start, end, interval)

    def get_company(self, t: str) -> dict:
        return self._try("fundamentals", "get_company", t) or {}

    def get_macro(self) -> dict:
        return self._try("macro", "get_macro") or {}

    def get_index_prices(self, symbol: str) -> pd.DataFrame | None:
        # prefer a real index series, fall back to the ETF proxy price series
        for name in self._order("prices"):
            p = self.providers[name]
            if hasattr(p, "get_index_prices"):
                try:
                    df = p.get_index_prices(symbol)
                    if df is not None and len(df):
                        return df
                except Exception:
                    pass
        return self.get_prices(symbol)

    def get_market_movers(self, limit: int = 60) -> list[str]:
        """Today's top gainers / most-active symbols merged across capable providers."""
        order = self._order("movers") or self._order("universe") or list(self.providers)
        out: list[str] = []
        for name in order:
            p = self.providers.get(name)
            fn = getattr(p, "get_market_movers", None) if p else None
            if fn is None:
                continue
            try:
                for sym in fn(limit):
                    if sym and sym not in out:
                        out.append(sym)
            except Exception as e:
                log.debug("%s movers failed: %s", name, e)
            if len(out) >= limit:
                break
        return out[:limit]

    def list_universe(self) -> list[str]:
        for name in (self.priority.get("universe") or list(self.providers)):
            p = self.providers.get(name)
            if p and hasattr(p, "list_universe"):
                try:
                    u = p.list_universe()
                    if u:
                        return u
                except Exception:
                    continue
        # any provider that can list
        for p in self.providers.values():
            try:
                u = p.list_universe()
                if u:
                    return u
            except Exception:
                continue
        return []

    # ---- news headlines (for the AI analyst) ----
    def headlines(self, ticker: str, limit: int = 6) -> list:
        order = self._order("news_headlines") or ["alphavantage", "web"]
        out: list[str] = []
        for name in order:
            p = self.providers.get(name)
            fn = getattr(p, "get_headlines", None) if p else None
            if fn is None:
                continue
            try:
                for h in fn(ticker, limit):
                    if h and h not in out:
                        out.append(h)
            except Exception as e:
                log.debug("%s headlines(%s) failed: %s", name, ticker, e)
            if len(out) >= limit:
                break
        return out[:limit]

    def market_news(self, limit: int = 8) -> str:
        order = self._order("news_headlines") or ["alphavantage", "web"]
        out: list[str] = []
        for name in order:
            p = self.providers.get(name)
            fn = getattr(p, "get_headlines", None) if p else None
            if fn is None:
                continue
            try:
                for h in fn(None, limit):
                    if h and h not in out:
                        out.append(h)
            except Exception as e:
                log.debug("%s market headlines failed: %s", name, e)
            if len(out) >= limit:
                break
        return " | ".join(out[:limit])

    # ---- full assembly ----
    def build(self, ticker: str, *, deep: bool = True,
              price_start: str | None = None) -> TickerData:
        td = TickerData(ticker=ticker)
        td.company = self.get_company(ticker)
        intraday = self.cfg.get("universe.intraday", {}) or {}
        if intraday.get("enabled", False):
            td.quote = self.get_quote_for_risk(
                ticker,
                max_age_seconds=float(intraday.get("quote_max_age_seconds", 60)),
            )
        else:
            td.quote = self.get_quote(ticker)
        td.prices = self.get_prices(ticker, start=price_start)
        td.fundamentals = self._merge("fundamentals", "get_fundamentals", ticker)
        # market cap fallback chain
        if not td.company.get("market_cap"):
            td.company["market_cap"] = td.fundamentals.get("market_cap")

        # technicals: local compute from prices, enriched by any provider extras
        tech = compute_indicators(td.prices) if td.prices is not None else {}
        prov_tech = self._try("technicals", "get_technicals", ticker)
        if isinstance(prov_tech, dict):
            for k, v in prov_tech.items():
                tech.setdefault(k, v)
        td.technicals = tech

        if deep:
            td.insider = self._try("insider", "get_insider", ticker) or []
            td.institutional = self._try("institutional", "get_institutional", ticker) or {}
            td.news_sentiment = self._try("news_sentiment", "get_news_sentiment", ticker) or {}
            td.analyst = self._merge("analyst_ratings", "get_analyst", ticker)
            td.short_interest = self._try("short_interest", "get_short_interest", ticker) or {}
            td.options = self._try("options_flow", "get_options", ticker) or {}
            td.earnings = self._try("fundamentals", "get_earnings", ticker) or \
                self._try("news_sentiment", "get_earnings", ticker) or {}
            if self._web_research_enabled():
                td.pro_scores = self._pro_scores(ticker, td.company.get("name")) or {}

        td.meta = {"captured_at": utcnow().isoformat(),
                   "has_prices": td.prices is not None and len(td.prices) > 0}
        return td

    def _web_research_enabled(self) -> bool:
        wr = self.cfg.get("web_research", {}) or {}
        if wr.get("enabled", True) is False:
            return False
        return "web" in self.providers

    def _pro_scores(self, ticker: str, name: str | None) -> dict:
        for pname in self._order("pro_scores"):
            p = self.providers[pname]
            if hasattr(p, "get_pro_scores"):
                try:
                    return p.get_pro_scores(ticker, name)
                except Exception:
                    continue
        return {}
