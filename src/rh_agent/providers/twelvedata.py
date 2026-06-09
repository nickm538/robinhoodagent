"""Twelve Data provider — real-time quotes, batch prefetch, price history,
and (on Pro+) market movers.

Auth: ?apikey= . Base https://api.twelvedata.com.

Paid plans unlock higher rate limits and Pro endpoints such as market movers.
Configure throughput with RH_TD_MAX_PER_SEC (default 8).
"""
from __future__ import annotations

import os

import pandas as pd

from ..models import Quote
from .base import DataProvider, DiskCache, HttpClient, ProviderUnsupported, prices_to_df

BASE = "https://api.twelvedata.com"
_BATCH_CHUNK = 120  # Twelve Data batch quote limit


class TwelveDataProvider(DataProvider):
    name = "twelvedata"

    def __init__(self, api_key: str, cache: DiskCache | None = None):
        super().__init__(cache)
        self.api_key = api_key
        max_per_sec = float(os.getenv("RH_TD_MAX_PER_SEC", "8"))
        self.http = HttpClient(BASE, max_per_sec=max_per_sec)

    def _q(self, section: str, ttl: float, path: str, params: dict):
        p = dict(params, apikey=self.api_key)
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"td/{section}", key, ttl)
        if hit is not None:
            return hit
        data = self.http.get_json(path, p)
        if isinstance(data, dict) and data.get("status") == "error":
            msg = str(data.get("message", ""))
            if "api key" in msg.lower() or data.get("code") == 401:
                raise ProviderUnsupported(f"twelvedata auth failed: {msg}")
            raise ProviderUnsupported(msg or "twelvedata error")
        self.cache.set(f"td/{section}", key, data, source=self.name)
        return data

    def _parse_quote(self, ticker: str, d: dict) -> Quote | None:
        if not isinstance(d, dict):
            return None
        price = d.get("close") or d.get("price") or d.get("last")
        if price is None:
            return None
        pct = d.get("percent_change")
        return Quote(
            ticker=ticker,
            price=float(price),
            volume=float(d.get("volume") or 0),
            prev_close=float(d["previous_close"]) if d.get("previous_close") else None,
            day_change_pct=float(pct) if pct is not None else None,
            source=self.name,
        )

    def get_quote(self, ticker: str) -> Quote:
        d = self._q("quote", 10, "/quote", {"symbol": ticker})
        q = self._parse_quote(ticker, d)
        if q is None:
            raise ProviderUnsupported
        return q

    def get_quotes_batch(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch up to 120 symbols per request (comma-separated /quote)."""
        out: dict[str, Quote] = {}
        symbols = [t.upper() for t in tickers if t]
        for i in range(0, len(symbols), _BATCH_CHUNK):
            chunk = symbols[i:i + _BATCH_CHUNK]
            if not chunk:
                continue
            d = self._q("quote", 3, "/quote", {"symbol": ",".join(chunk)})
            if isinstance(d, dict) and d.get("symbol"):
                q = self._parse_quote(str(d["symbol"]).upper(), d)
                if q:
                    out[q.ticker] = q
                continue
            if isinstance(d, dict):
                for sym, payload in d.items():
                    if sym in ("status", "code", "message"):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    tk = str(payload.get("symbol") or sym).upper()
                    q = self._parse_quote(tk, payload)
                    if q:
                        out[tk] = q
        return out

    def invalidate_quote(self, ticker: str) -> None:
        key = f"/quote|{sorted({'symbol': ticker}.items())}"
        self.cache._path("td/quote", key).unlink(missing_ok=True)

    def get_prices(self, ticker: str, start=None, end=None, interval="day") -> pd.DataFrame:
        iv = {"day": "1day", "week": "1week", "month": "1month"}.get(interval, "1day")
        d = self._q("ts", 720, "/time_series",
                    {"symbol": ticker, "interval": iv, "outputsize": 5000})
        vals = d.get("values") if isinstance(d, dict) else None
        if not vals:
            raise ProviderUnsupported
        df = prices_to_df([{"time": v["datetime"], **v} for v in vals])
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def get_company(self, ticker: str) -> dict:
        """Light profile for sector / market-cap fallback during universe prescreen."""
        d = self._q("profile", 1440, "/profile", {"symbol": ticker})
        if not isinstance(d, dict) or not d.get("symbol"):
            raise ProviderUnsupported
        mcap = d.get("market_capitalization") or d.get("market_cap")
        return {
            "name": d.get("name"),
            "sector": d.get("sector"),
            "industry": d.get("industry"),
            "market_cap": float(mcap) if mcap not in (None, "") else None,
            "source": self.name,
        }

    def get_market_movers(self, limit: int = 60) -> list[str]:
        """Top gainers from /market_movers/stocks (Pro plan, 100 credits/request)."""
        d = self._q("movers", 5, "/market_movers/stocks",
                    {"direction": "gainers", "outputsize": min(50, limit)})
        vals = d.get("values") if isinstance(d, dict) else None
        if not vals:
            raise ProviderUnsupported
        out: list[str] = []
        for row in vals:
            if not isinstance(row, dict):
                continue
            sym = row.get("symbol")
            if sym and "-" not in sym and sym not in out:
                out.append(sym)
        if not out:
            raise ProviderUnsupported
        return out[:limit]
