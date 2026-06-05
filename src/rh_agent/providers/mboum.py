"""Mboum provider (priority source).

Docs: https://docs.mboum.com  — base https://api.mboum.com, auth via
``Authorization: Bearer <key>``, 15 req/sec rate limit. Endpoint paths and
parameters below are taken verbatim from the documented example requests.

Mboum wraps payloads inconsistently ({"body": ...} / {"data": ...} / raw),
so every parser unwraps defensively and tolerates missing fields.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..models import Quote
from .base import DataProvider, DiskCache, HttpClient, ProviderUnsupported, prices_to_df

BASE = "https://api.mboum.com"


def _unwrap(d: Any) -> Any:
    if isinstance(d, dict):
        for k in ("body", "data", "result", "results", "quotes"):
            if k in d and d[k] is not None:
                return d[k]
    return d


def _first(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


class MboumProvider(DataProvider):
    name = "mboum"

    def __init__(self, api_key: str, cache: DiskCache | None = None):
        super().__init__(cache)
        self.http = HttpClient(BASE, max_per_sec=12,
                               default_headers={"Authorization": f"Bearer {api_key}",
                                                "Accept": "application/json"})

    def _cached(self, section: str, ttl: float, path: str, params: dict) -> Any:
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"mboum/{section}", key, ttl)
        if hit is not None:
            return hit
        data = self.http.get_json(path, params)
        self.cache.set(f"mboum/{section}", key, data, source=self.name)
        return data

    # ------------------------------------------------------------------ quote
    def get_quote(self, ticker: str) -> Quote:
        d = _unwrap(self._cached("quote", 10, "/v1/markets/quote", {"ticker": ticker, "type": "STOCKS"}))
        q = d[0] if isinstance(d, list) and d else (d if isinstance(d, dict) else {})
        price = _first(q, "regularMarketPrice", "price", "last", "regular_market_price")
        if price is None:
            raise ProviderUnsupported
        return Quote(
            ticker=ticker, price=float(price),
            volume=float(_first(q, "regularMarketVolume", "volume", default=0) or 0),
            prev_close=_first(q, "regularMarketPreviousClose", "previousClose"),
            day_change_pct=_first(q, "regularMarketChangePercent", "changePercent"),
            source=self.name,
        )

    # ------------------------------------------------------------------ prices
    def get_prices(self, ticker: str, start: str | None = None, end: str | None = None,
                   interval: str = "day") -> pd.DataFrame:
        iv = {"day": "1d", "week": "1wk", "month": "1mo"}.get(interval, "1d")
        d = self._cached("history", 720, "/v1/markets/stock/history",
                         {"ticker": ticker, "interval": iv, "diffandsplits": "false"})
        body = _unwrap(d)
        # body is often {"<timestamp>": {date, open, high, low, close, volume}, ...}
        recs = []
        if isinstance(body, dict):
            for v in body.values():
                if isinstance(v, dict) and ("close" in v or "Close" in v):
                    recs.append({k.lower(): val for k, val in v.items()})
        elif isinstance(body, list):
            recs = [{k.lower(): val for k, val in r.items()} for r in body]
        return prices_to_df(recs)

    # ------------------------------------------------------------------ company / fundamentals
    def get_company(self, ticker: str) -> dict:
        # ticker-summary REQUIRES `type` (STOCKS|ETF|MUTUALFUNDS|FUTURES) — omitting it -> HTTP 422
        d = _unwrap(self._cached("summary", 1440, "/v2/markets/stock/ticker-summary",
                                 {"ticker": ticker, "type": "STOCKS"}))
        if not isinstance(d, dict):
            raise ProviderUnsupported
        return {
            "name": _first(d, "longName", "shortName", "name"),
            "sector": _first(d, "sector"),
            "industry": _first(d, "industry"),
            "market_cap": _first(d, "marketCap", "market_cap"),
            "source": self.name,
        }

    def get_fundamentals(self, ticker: str) -> dict:
        d = _unwrap(self._cached("financials", 1440, "/v2/markets/stock/financials",
                                 {"ticker": ticker}))
        if not isinstance(d, dict):
            raise ProviderUnsupported
        out = {"source": self.name}
        mapping = {
            "returnOnEquity": "roe", "returnOnAssets": "roa",
            "grossMargins": "gross_margin", "operatingMargins": "operating_margin",
            "profitMargins": "net_margin", "revenueGrowth": "revenue_growth",
            "earningsGrowth": "earnings_growth", "trailingPE": "pe_ratio",
            "pegRatio": "peg_ratio", "priceToSalesTrailing12Months": "ps_ratio",
            "priceToBook": "pb_ratio", "debtToEquity": "debt_to_equity",
            "currentRatio": "current_ratio", "quickRatio": "quick_ratio",
            "marketCap": "market_cap", "dividendYield": "dividend_yield",
            "trailingEps": "eps", "enterpriseToEbitda": "ev_ebitda",
        }
        for src, dst in mapping.items():
            v = d.get(src)
            if v is not None:
                # mboum sometimes nests as {"raw": x, "fmt": "..."}
                out[dst] = v.get("raw") if isinstance(v, dict) else v
        if "debt_to_equity" in out and out["debt_to_equity"] and out["debt_to_equity"] > 5:
            out["debt_to_equity"] = out["debt_to_equity"] / 100.0  # mboum reports as percent
        return out

    # ------------------------------------------------------------------ analyst
    def get_analyst(self, ticker: str) -> dict:
        out: dict = {"source": self.name}
        try:
            r = _unwrap(self._cached("ratings", 720, "/v1/markets/stock/analyst-ratings",
                                     {"ticker": ticker}))
            if isinstance(r, dict):
                out["buy"] = _first(r, "strongBuy", "buy")
                out["hold"] = _first(r, "hold")
                out["sell"] = _first(r, "sell", "strongSell")
                out["mean_rating"] = _first(r, "recommendationMean", "rating")
        except Exception:
            pass
        try:
            pt = _unwrap(self._cached("pt", 720, "/v2/markets/stock/price-targets",
                                      {"ticker": ticker}))
            if isinstance(pt, dict):
                out["target_mean"] = _first(pt, "targetMeanPrice", "targetMean", "mean")
                out["target_high"] = _first(pt, "targetHighPrice", "targetHigh")
                out["target_low"] = _first(pt, "targetLowPrice", "targetLow")
                out["num_analysts"] = _first(pt, "numberOfAnalystOpinions", "numberOfAnalysts")
        except Exception:
            pass
        if len(out) == 1:
            raise ProviderUnsupported
        return out

    # ------------------------------------------------------------------ short interest
    def get_short_interest(self, ticker: str) -> dict:
        # This endpoint REQUIRES `type` — omitting it returns HTTP 422 — and it
        # answers with an ARRAY of historical reports (index 0 = most recent).
        d = _unwrap(self._cached("short", 1440, "/v2/markets/stock/short-interest",
                                 {"ticker": ticker, "type": "STOCKS"}))
        if isinstance(d, list):
            d = d[0] if d else {}
        if not isinstance(d, dict) or not d:
            raise ProviderUnsupported
        return {
            "short_pct_float": _first(d, "shortPercentOfFloat", "shortPercentFloat",
                                      "percentFloat", "shortFloatPercent"),
            "days_to_cover": _first(d, "daysToCover", "shortRatio"),
            "short_shares": _first(d, "interest", "sharesShort", "shortInterest"),
            "avg_daily_volume": _first(d, "avgDailyShareVolume", "averageDailyVolume"),
            "settlement_date": _first(d, "settlementDate", "date"),
            "source": self.name,
        }

    # ------------------------------------------------------------------ institutional
    def get_institutional(self, ticker: str) -> dict:
        # institutional-holdings REQUIRES `type` (TOTAL|INCREASED|NEW) — omitting it -> HTTP 422.
        d = _unwrap(self._cached("inst", 1440, "/v2/markets/stock/institutional-holdings",
                                 {"ticker": ticker, "type": "TOTAL"}))
        # Documented shape: {ownershipSummary{...}, activePositions{increased,decreased,new}}.
        if isinstance(d, dict) and ("activePositions" in d or "ownershipSummary" in d):
            active = d.get("activePositions") if isinstance(d.get("activePositions"), dict) else {}
            summ = d.get("ownershipSummary") if isinstance(d.get("ownershipSummary"), dict) else {}
            inc = _first(active, "increasedPositions", "increased", "positionsIncreased", default=0) or 0
            dec = _first(active, "decreasedPositions", "decreased", "positionsDecreased", default=0) or 0
            new = _first(active, "newPositions", "new", default=0) or 0
            holders = _first(summ, "institutionsCount", "totalInstitutions", "totalHolders") or (inc + dec + new)
            net = ((inc - dec) / (inc + dec)) if isinstance(inc, (int, float)) \
                and isinstance(dec, (int, float)) and (inc + dec) > 0 else None
            if net is None:           # no usable signal -> fall through to the next provider
                raise ProviderUnsupported
            return {"holders": holders, "net_change_pct": net, "source": self.name}
        # Fallback: a flat list of holders with per-row change.
        recs = d if isinstance(d, list) else (d.get("ownershipList") if isinstance(d, dict) else None)
        if not recs:
            raise ProviderUnsupported
        delta = 0.0
        for r in recs:
            ch = _first(r, "pctChange", "change") or 0
            delta += ch if isinstance(ch, (int, float)) else 0
        return {"holders": len(recs), "net_change_pct": delta / max(len(recs), 1),
                "source": self.name}

    # ------------------------------------------------------------------ insider
    def get_insider(self, ticker: str) -> list:
        d = _unwrap(self._cached("insider", 720, "/v1/markets/insider-trades",
                                 {"ticker": ticker}))
        recs = d if isinstance(d, list) else (d.get("insiderTransactions") if isinstance(d, dict) else [])
        out = []
        for t in (recs or []):
            shares = _first(t, "shares", "transactionShares")
            ttype = (_first(t, "transactionText", "transactionType", default="") or "").lower()
            out.append({
                "date": _first(t, "startDate", "date", "filingDate"),
                "name": _first(t, "filerName", "name"),
                "shares": shares,
                "value": _first(t, "value"),
                "is_buy": "buy" in ttype or "purchase" in ttype or "acqui" in ttype,
            })
        return out

    # ------------------------------------------------------------------ technicals
    def get_technicals(self, ticker: str) -> dict:
        out: dict = {"source": self.name}
        try:
            r = _unwrap(self._cached("rsi", 240, "/v1/markets/indicators/rsi",
                                     {"ticker": ticker, "interval": "1d",
                                      "series_type": "close", "time_period": 14}))
            if isinstance(r, dict):
                vals = list(r.values())
                if vals and isinstance(vals[0], (int, float)):
                    out["rsi"] = float(vals[0])
        except Exception:
            pass
        if len(out) == 1:
            raise ProviderUnsupported
        return out

    # ------------------------------------------------------------------ universe
    def list_universe(self) -> list[str]:
        d = _unwrap(self._cached("tickers", 1440, "/v2/markets/tickers",
                                 {"type": "STOCKS", "page": 1}))
        recs = d if isinstance(d, list) else []
        return [r.get("symbol") for r in recs if isinstance(r, dict) and r.get("symbol")]
