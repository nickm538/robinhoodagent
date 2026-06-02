"""Alpha Vantage provider.

Auth: ?apikey= query param. Base https://www.alphavantage.co/query.
Strong for: real-time quote, adjusted daily prices, NEWS_SENTIMENT, earnings
estimates/revisions, company OVERVIEW (rich ratios), insider tx, macro/regime
(treasury yields, fed funds, index data), and a full LISTING_STATUS universe.

Note: free tier is heavily rate limited (~5/min). We cache aggressively and
compute most technicals locally from prices instead of hitting AV per metric.
"""
from __future__ import annotations

import csv
import io
from typing import Any

import pandas as pd

from ..models import Quote
from .base import (CACHE_DIR, DataProvider, DiskCache, HttpClient, OFFLINE,
                   ProviderError, ProviderUnsupported, prices_to_df)

BASE = "https://www.alphavantage.co"


def _num(x: Any) -> float | None:
    try:
        if x in (None, "None", "-", ""):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


class AlphaVantageProvider(DataProvider):
    name = "alphavantage"

    def __init__(self, api_key: str, cache: DiskCache | None = None):
        super().__init__(cache)
        self.api_key = api_key
        self.http = HttpClient(BASE, max_per_sec=1.0)  # be gentle with AV limits

    def _q(self, section: str, ttl: float, params: dict) -> Any:
        p = dict(params)
        key = f"{sorted(p.items())}"
        hit = self.cache.get(f"av/{section}", key, ttl)
        if hit is not None:
            return hit
        p["apikey"] = self.api_key
        data = self.http.get_json("/query", p)
        if isinstance(data, dict) and ("Note" in data or "Information" in data):
            raise ProviderError(f"AlphaVantage throttled: {data.get('Note') or data.get('Information')}")
        self.cache.set(f"av/{section}", key, data, source=self.name)
        return data

    def get_quote(self, ticker: str) -> Quote:
        d = self._q("quote", 10, {"function": "GLOBAL_QUOTE", "symbol": ticker})
        g = d.get("Global Quote", {}) if isinstance(d, dict) else {}
        price = _num(g.get("05. price"))
        if price is None:
            raise ProviderUnsupported
        return Quote(ticker=ticker, price=price, volume=_num(g.get("06. volume")) or 0,
                     prev_close=_num(g.get("08. previous close")),
                     day_change_pct=_num((g.get("10. change percent") or "0").rstrip("%")),
                     source=self.name)

    def get_prices(self, ticker: str, start: str | None = None, end: str | None = None,
                   interval: str = "day") -> pd.DataFrame:
        fn = {"day": "TIME_SERIES_DAILY_ADJUSTED", "week": "TIME_SERIES_WEEKLY_ADJUSTED",
              "month": "TIME_SERIES_MONTHLY_ADJUSTED"}.get(interval, "TIME_SERIES_DAILY_ADJUSTED")
        d = self._q("prices", 720, {"function": fn, "symbol": ticker, "outputsize": "full"})
        ts_key = next((k for k in d if "Time Series" in k), None) if isinstance(d, dict) else None
        if not ts_key:
            raise ProviderUnsupported
        recs = []
        for date, row in d[ts_key].items():
            recs.append({
                "time": date, "open": row.get("1. open"), "high": row.get("2. high"),
                "low": row.get("3. low"), "close": row.get("4. close"),
                "adj_close": row.get("5. adjusted close", row.get("4. close")),
                "volume": row.get("6. volume", row.get("5. volume")),
            })
        df = prices_to_df(recs)
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def get_company(self, ticker: str) -> dict:
        d = self._q("overview", 1440, {"function": "OVERVIEW", "symbol": ticker})
        if not isinstance(d, dict) or not d.get("Symbol"):
            raise ProviderUnsupported
        return {"name": d.get("Name"), "sector": _title(d.get("Sector")),
                "industry": d.get("Industry"), "market_cap": _num(d.get("MarketCapitalization")),
                "beta": _num(d.get("Beta")), "source": self.name, "_overview": d}

    def get_fundamentals(self, ticker: str) -> dict:
        d = self._q("overview", 1440, {"function": "OVERVIEW", "symbol": ticker})
        if not isinstance(d, dict) or not d.get("Symbol"):
            raise ProviderUnsupported
        gp, rev = _num(d.get("GrossProfitTTM")), _num(d.get("RevenueTTM"))
        gross_margin = (gp / rev) if (gp is not None and rev) else None
        return {
            "roe": _num(d.get("ReturnOnEquityTTM")), "roa": _num(d.get("ReturnOnAssetsTTM")),
            "gross_margin": gross_margin,
            "operating_margin": _num(d.get("OperatingMarginTTM")),
            "net_margin": _num(d.get("ProfitMargin")),
            "revenue_growth": _num(d.get("QuarterlyRevenueGrowthYOY")),
            "earnings_growth": _num(d.get("QuarterlyEarningsGrowthYOY")),
            "pe_ratio": _num(d.get("PERatio")), "peg_ratio": _num(d.get("PEGRatio")),
            "ps_ratio": _num(d.get("PriceToSalesRatioTTM")), "pb_ratio": _num(d.get("PriceToBookRatio")),
            "ev_ebitda": _num(d.get("EVToEBITDA")), "dividend_yield": _num(d.get("DividendYield")),
            "eps": _num(d.get("EPS")), "market_cap": _num(d.get("MarketCapitalization")),
            "beta": _num(d.get("Beta")), "analyst_target": _num(d.get("AnalystTargetPrice")),
            "ma50": _num(d.get("50DayMovingAverage")), "ma200": _num(d.get("200DayMovingAverage")),
            "high_52w": _num(d.get("52WeekHigh")), "low_52w": _num(d.get("52WeekLow")),
            "source": self.name,
        }

    def get_news_sentiment(self, ticker: str) -> dict:
        d = self._q("news", 120, {"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": 200})
        feed = d.get("feed", []) if isinstance(d, dict) else []
        if not feed:
            raise ProviderUnsupported
        wsum, w = 0.0, 0.0
        for art in feed:
            for ts in art.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    rel = _num(ts.get("relevance_score")) or 0
                    sc = _num(ts.get("ticker_sentiment_score")) or 0
                    wsum += rel * sc
                    w += rel
        return {"score": (wsum / w) if w else None, "article_count": len(feed),
                "relevance_weighted": True, "source": self.name}

    def get_earnings(self, ticker: str) -> dict:
        out: dict = {"source": self.name}
        try:
            e = self._q("earnings", 720, {"function": "EARNINGS", "symbol": ticker})
            q = e.get("quarterlyEarnings", []) if isinstance(e, dict) else []
            surps = [_num(x.get("surprisePercentage")) for x in q[:8]]
            surps = [s for s in surps if s is not None]
            if surps:
                out["avg_surprise_pct"] = sum(surps) / len(surps)
                out["beat_rate"] = sum(1 for s in surps if s > 0) / len(surps)
        except Exception:
            pass
        try:
            est = self._q("est", 720, {"function": "EARNINGS_ESTIMATES", "symbol": ticker})
            qe = est.get("quarterlyEstimates", est.get("estimates", [])) if isinstance(est, dict) else []
            if qe:
                up = _num(qe[0].get("epsEstimateRevisionUpLast30Days") or qe[0].get("revisionUp"))
                dn = _num(qe[0].get("epsEstimateRevisionDownLast30Days") or qe[0].get("revisionDown"))
                if up is not None and dn is not None and (up + dn) > 0:
                    out["eps_revision_ratio"] = (up - dn) / (up + dn)
        except Exception:
            pass
        if len(out) == 1:
            raise ProviderUnsupported
        return out

    def get_options(self, ticker: str) -> dict:
        d = self._q("pcr", 60, {"function": "REALTIME_PUT_CALL_RATIO", "symbol": ticker})
        pcr = None
        if isinstance(d, dict):
            pcr = _num(d.get("put_call_ratio") or d.get("putCallRatio")
                       or (d.get("data", [{}])[0].get("put_call_ratio") if d.get("data") else None))
        if pcr is None:
            raise ProviderUnsupported
        return {"put_call_ratio": pcr, "source": self.name}

    # -------- macro / regime --------
    def get_macro(self) -> dict:
        out: dict = {"source": self.name}
        try:
            t10 = self._q("t10", 720, {"function": "TREASURY_YIELD", "interval": "daily",
                                       "maturity": "10year"})
            t2 = self._q("t2", 720, {"function": "TREASURY_YIELD", "interval": "daily",
                                     "maturity": "2year"})
            out["yield_10y"] = _num(t10["data"][0]["value"]) if t10.get("data") else None
            out["yield_2y"] = _num(t2["data"][0]["value"]) if t2.get("data") else None
            if out.get("yield_10y") and out.get("yield_2y"):
                out["yield_curve_10_2"] = out["yield_10y"] - out["yield_2y"]
        except Exception:
            pass
        return out

    def get_index_prices(self, symbol: str) -> pd.DataFrame:
        d = self._q(f"idx_{symbol}", 720, {"function": "INDEX_DATA", "symbol": symbol,
                                           "interval": "daily"})
        ts_key = next((k for k in d if "Time Series" in k or "data" == k), None) if isinstance(d, dict) else None
        if not ts_key:
            raise ProviderUnsupported
        block = d[ts_key]
        recs = ([{"time": r.get("date"), "close": r.get("close") or r.get("value")} for r in block]
                if isinstance(block, list)
                else [{"time": k, "close": v.get("4. close", v.get("close"))} for k, v in block.items()])
        return prices_to_df(recs)

    # -------- universe --------
    def list_universe(self) -> list[str]:
        if OFFLINE:
            raise ProviderUnsupported
        hit = self.cache.get("av/listing", "active", 1440)
        if hit is None:
            url = (f"{BASE}/query?function=LISTING_STATUS&state=active&apikey={self.api_key}")
            self.http.limiter.wait()
            txt = self.http.session.get(url, timeout=self.http.timeout).text
            hit = txt
            self.cache.set("av/listing", "active", txt, source=self.name)
        rows = list(csv.DictReader(io.StringIO(hit)))
        return [r["symbol"] for r in rows
                if r.get("assetType") == "Stock" and "-" not in r.get("symbol", "-")]


def _title(s: Any) -> Any:
    return s.title() if isinstance(s, str) else s
