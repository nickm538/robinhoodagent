"""FinancialDatasets.AI provider (priority source).

Docs: https://docs.financialdatasets.ai  — auth via X-API-KEY header.
Response shapes here match the live API (verified against the same data the
FinancialDatasets MCP server returns).
"""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd

from ..logging_setup import get_logger
from ..models import Quote
from .base import (DataProvider, DiskCache, HttpClient, ProviderUnsupported,
                   RateLimitError, prices_to_df)

log = get_logger("financialdatasets")

BASE = "https://api.financialdatasets.ai"

# FD limits are per-minute (unlike Mboum's monthly cap) — when the primary
# provider 429s under load, briefly stop hammering it and let the fallback
# chain serve; a short re-probe restores it as soon as the window clears.
RATE_LIMIT_COOLDOWN_SECONDS = float(
    os.getenv("FINANCIALDATASETS_RATE_LIMIT_COOLDOWN_SECONDS", "60"))

# FinancialDatasets metric name -> our canonical fundamentals key.
_FUND_MAP = {
    "return_on_equity": "roe",
    "return_on_invested_capital": "roic",
    "return_on_assets": "roa",
    "gross_margin": "gross_margin",
    "operating_margin": "operating_margin",
    "net_margin": "net_margin",
    "revenue_growth": "revenue_growth",
    "earnings_growth": "earnings_growth",
    "earnings_per_share_growth": "eps_growth",
    "free_cash_flow_growth": "fcf_growth",
    "ebitda_growth": "ebitda_growth",
    "free_cash_flow_yield": "fcf_yield",
    "price_to_earnings_ratio": "pe_ratio",
    "peg_ratio": "peg_ratio",
    "price_to_sales_ratio": "ps_ratio",
    "price_to_book_ratio": "pb_ratio",
    "enterprise_value_to_ebitda_ratio": "ev_ebitda",
    "debt_to_equity": "debt_to_equity",
    "debt_to_assets": "debt_to_assets",
    "current_ratio": "current_ratio",
    "quick_ratio": "quick_ratio",
    "interest_coverage": "interest_coverage",
    "market_cap": "market_cap",
    "payout_ratio": "payout_ratio",
    "dividend_yield": "dividend_yield",
    "earnings_per_share": "eps",
    "book_value_per_share": "book_value_per_share",
    "free_cash_flow_per_share": "fcf_per_share",
}


class FinancialDatasetsProvider(DataProvider):
    name = "financialdatasets"

    _TTL_KEY = {
        "snap": "quote",
        "facts": "fundamentals",
        "metrics": "fundamentals",
        "inst": "institutional",
        "news": "news_sentiment",
    }

    def __init__(self, api_key: str, cache: DiskCache | None = None, *,
                 cache_ttls: dict | None = None):
        super().__init__(cache)
        # Pace requests UNDER the plan's per-minute ceiling and the 429/Retry-After
        # waves disappear entirely (credits are volume; the 429s are velocity).
        # e.g. a 240/min plan -> FINANCIALDATASETS_MAX_PER_SEC=4.
        rate = float(os.getenv("FINANCIALDATASETS_MAX_PER_SEC", "8"))
        self.http = HttpClient(BASE, max_per_sec=rate, default_headers={"X-API-KEY": api_key})
        self._cache_ttls = cache_ttls or {}
        self._rate_limited_until = 0.0

    def _rate_limit_active(self) -> bool:
        return time.time() < getattr(self, "_rate_limited_until", 0.0)

    def _enter_cooldown(self, seconds: float | None, reason: str) -> None:
        cooldown = float(seconds) if seconds and seconds > 0 else RATE_LIMIT_COOLDOWN_SECONDS
        until = time.time() + cooldown
        if until > getattr(self, "_rate_limited_until", 0.0):
            self._rate_limited_until = until
        log.warning("financialdatasets rate limited (%s) — cooling down %.0fs; "
                    "fallback providers serve meanwhile", reason,
                    max(self._rate_limited_until - time.time(), 0.0))

    def _ttl(self, section: str, default: float) -> float:
        key = self._TTL_KEY.get(section, section)
        val = (getattr(self, "_cache_ttls", None) or {}).get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _cached(self, section: str, ticker: str, ttl: float, path: str, params: dict) -> Any:
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"fd/{section}", key, ttl)
        if hit is not None:
            return hit
        if self._rate_limit_active():
            raise ProviderUnsupported("financialdatasets cooling down after rate limit")
        try:
            data = self.http.get_json(path, params)
        except RateLimitError as e:
            self._enter_cooldown(e.retry_after_seconds, path)
            raise ProviderUnsupported("financialdatasets rate limited") from e
        self.cache.set(f"fd/{section}", key, data, source=self.name)
        return data

    def get_company(self, ticker: str) -> dict:
        d = self._cached("facts", ticker, self._ttl("facts", 1440), "/company/facts/", {"ticker": ticker})
        f = d.get("company_facts", d) if isinstance(d, dict) else {}
        return {
            "name": f.get("name"),
            "sector": f.get("sector"),
            "industry": f.get("industry"),
            "market_cap": f.get("market_cap"),
            "exchange": f.get("exchange"),
            "employees": f.get("number_of_employees"),
            "source": self.name,
        }

    def get_quote(self, ticker: str) -> Quote:
        d = self._cached("snap", ticker, self._ttl("snap", 10), "/prices/snapshot/", {"ticker": ticker})
        s = d.get("snapshot", d) if isinstance(d, dict) else {}
        price = s.get("price")
        if price is None:
            raise ProviderUnsupported
        px = float(price)
        pct = s.get("day_change_percent")
        day_change_pct = float(pct) if pct is not None else None
        prev_close = None
        if day_change_pct is not None and day_change_pct != -100.0:
            prev_close = px / (1.0 + day_change_pct / 100.0)
        elif s.get("day_change") is not None:
            # Fallback: absolute dollar change when percent is absent.
            prev_close = px - float(s["day_change"])
        return Quote(
            ticker=ticker, price=px, volume=float(s.get("volume") or 0),
            day_change_pct=day_change_pct,
            prev_close=prev_close,
            source=self.name,
        )

    def invalidate_quote(self, ticker: str) -> None:
        key = f"/prices/snapshot/|{sorted({'ticker': ticker}.items())}"
        self.cache._path("fd/snap", key).unlink(missing_ok=True)

    def get_prices(self, ticker: str, start: str | None = None, end: str | None = None,
                   interval: str = "day") -> pd.DataFrame:
        # /prices/ REQUIRES start_date AND end_date — omitting them returns HTTP 400.
        # Anchor the default ~2y window on the *resolved end* (not always today) so an
        # end-only call can't yield start > end (which would 400 / return empty).
        from datetime import date, timedelta
        end = end or date.today().isoformat()
        if not start:
            try:
                anchor = date.fromisoformat(end)
            except ValueError:
                anchor = date.today()
            start = (anchor - timedelta(days=730)).isoformat()
        params = {"ticker": ticker, "interval": interval, "interval_multiplier": 1,
                  "start_date": start, "end_date": end}
        d = self._cached("prices", ticker, self._ttl("prices", 720), "/prices/", params)
        recs = d.get("prices", d) if isinstance(d, dict) else d
        return prices_to_df(recs or [])

    def get_fundamentals(self, ticker: str) -> dict:
        d = self._cached("metrics", ticker, self._ttl("metrics", 1440),
                          "/financial-metrics/snapshot/", {"ticker": ticker})
        s = d.get("snapshot", d) if isinstance(d, dict) else {}
        out = {"source": self.name}
        for k, v in _FUND_MAP.items():
            if k in s and s[k] is not None:
                out[v] = s[k]
        return out

    def get_insider(self, ticker: str) -> list:
        d = self._cached("insider", ticker, self._ttl("insider", 720),
                          "/insider-trades/", {"ticker": ticker, "limit": 100})
        trades = d.get("insider_trades", d) if isinstance(d, dict) else d
        out = []
        for t in (trades or []):
            shares = t.get("transaction_shares") or t.get("shares")
            value = t.get("transaction_value") or t.get("value")
            if shares is None and value is None:
                continue
            out.append({
                "date": t.get("filing_date") or t.get("transaction_date"),
                "name": t.get("name"),
                "title": t.get("title"),
                "shares": shares,
                "value": value,
                "is_buy": (shares or 0) > 0 or (t.get("transaction_type", "").lower().startswith("buy")),
            })
        return out

    def get_institutional(self, ticker: str) -> dict:
        try:
            d = self._cached("inst", ticker, self._ttl("inst", 1440),
                             "/institutional-ownership/", {"ticker": ticker, "limit": 50})
        except Exception:
            raise ProviderUnsupported
        recs = d.get("institutional_ownership", d) if isinstance(d, dict) else d
        if not recs:
            raise ProviderUnsupported
        # Aggregate latest reported share change across reporting investors.
        total_shares = sum((r.get("shares") or 0) for r in recs)
        delta = sum((r.get("shares") or 0) - (r.get("shares_previous") or r.get("shares") or 0)
                    for r in recs)
        return {"holders": len(recs), "total_shares": total_shares,
                "net_share_change": delta,
                "net_change_pct": (delta / total_shares) if total_shares else None,
                "source": self.name}

    # The live /news/ endpoint rejects limit > 10 with HTTP 400 ("Invalid limit").
    _NEWS_LIMIT_CAP = 10

    def get_headlines(self, ticker: str | None = None, limit: int = 8) -> list:
        """Recent headlines for a ticker, or broad-market news when ticker is None."""
        capped = min(limit, self._NEWS_LIMIT_CAP)
        params = {"limit": capped}
        if ticker:
            params["ticker"] = ticker
        d = self._cached("news", ticker or "_market", self._ttl("news", 120), "/news/", params)
        items = d.get("news", d) if isinstance(d, dict) else d
        return [it.get("title") or it.get("headline") for it in (items or [])[:limit]
                if isinstance(it, dict) and (it.get("title") or it.get("headline"))]

    def get_news_sentiment(self, ticker: str) -> dict:
        d = self._cached("news", ticker, self._ttl("news", 120), "/news/",
                         {"ticker": ticker, "limit": self._NEWS_LIMIT_CAP})
        items = d.get("news", d) if isinstance(d, dict) else d
        if not items:
            raise ProviderUnsupported
        # Some plans label articles with a 'sentiment'; count them when present.
        score, n = 0.0, 0
        for it in items:
            s = (it.get("sentiment") or "").lower()
            if s in ("positive", "bullish"):
                score += 1
                n += 1
            elif s in ("negative", "bearish"):
                score -= 1
                n += 1
            elif s in ("neutral",):
                n += 1
        if n == 0:
            # No sentiment labels on this plan/ticker: defer to the next provider
            # instead of returning a scoreless dict that would block the chain.
            raise ProviderUnsupported
        return {"score": score / n, "article_count": len(items), "source": self.name}
