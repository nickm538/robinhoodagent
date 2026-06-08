"""FinancialDatasets.AI provider (priority source).

Docs: https://docs.financialdatasets.ai  — auth via X-API-KEY header.
Response shapes here match the live API (verified against the same data the
FinancialDatasets MCP server returns).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..models import Quote
from .base import DataProvider, DiskCache, HttpClient, ProviderUnsupported, prices_to_df

BASE = "https://api.financialdatasets.ai"

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

    def __init__(self, api_key: str, cache: DiskCache | None = None):
        super().__init__(cache)
        self.http = HttpClient(BASE, max_per_sec=8, default_headers={"X-API-KEY": api_key})

    def _cached(self, section: str, ticker: str, ttl: float, path: str, params: dict) -> Any:
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"fd/{section}", key, ttl)
        if hit is not None:
            return hit
        data = self.http.get_json(path, params)
        self.cache.set(f"fd/{section}", key, data, source=self.name)
        return data

    def get_company(self, ticker: str) -> dict:
        d = self._cached("facts", ticker, 1440, "/company/facts/", {"ticker": ticker})
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
        d = self._cached("snap", ticker, 10, "/prices/snapshot/", {"ticker": ticker})
        s = d.get("snapshot", d) if isinstance(d, dict) else {}
        price = s.get("price")
        if price is None:
            raise ProviderUnsupported
        return Quote(
            ticker=ticker, price=float(price), volume=float(s.get("volume") or 0),
            day_change_pct=s.get("day_change_percent"),
            prev_close=(float(price) - s.get("day_change")) if s.get("day_change") else None,
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
        d = self._cached("prices", ticker, 720, "/prices/", params)
        recs = d.get("prices", d) if isinstance(d, dict) else d
        return prices_to_df(recs or [])

    def get_fundamentals(self, ticker: str) -> dict:
        d = self._cached("metrics", ticker, 1440, "/financial-metrics/snapshot/", {"ticker": ticker})
        s = d.get("snapshot", d) if isinstance(d, dict) else {}
        out = {"source": self.name}
        for k, v in _FUND_MAP.items():
            if k in s and s[k] is not None:
                out[v] = s[k]
        return out

    def get_insider(self, ticker: str) -> list:
        d = self._cached("insider", ticker, 720, "/insider-trades/", {"ticker": ticker, "limit": 100})
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
            d = self._cached("inst", ticker, 1440, "/institutional-ownership/",
                             {"ticker": ticker, "limit": 50})
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

    def get_headlines(self, ticker: str | None = None, limit: int = 8) -> list:
        """Recent headlines for a ticker, or broad-market news when ticker is None."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        d = self._cached("news", ticker or "_market", 120, "/news/", params)
        items = d.get("news", d) if isinstance(d, dict) else d
        return [it.get("title") or it.get("headline") for it in (items or [])[:limit]
                if isinstance(it, dict) and (it.get("title") or it.get("headline"))]

    def get_news_sentiment(self, ticker: str) -> dict:
        d = self._cached("news", ticker, 120, "/news/", {"ticker": ticker, "limit": 50})
        items = d.get("news", d) if isinstance(d, dict) else d
        if not items:
            raise ProviderUnsupported
        # FinancialDatasets news has a 'sentiment' label per article on many tickers.
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
        return {"score": (score / n) if n else None, "article_count": len(items),
                "source": self.name}
