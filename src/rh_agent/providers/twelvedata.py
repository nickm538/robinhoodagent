"""Twelve Data provider — lightweight quote/price fallback.

Auth: ?apikey= . Base https://api.twelvedata.com.
"""
from __future__ import annotations

import pandas as pd

from ..models import Quote
from .base import DataProvider, DiskCache, HttpClient, ProviderUnsupported, prices_to_df

BASE = "https://api.twelvedata.com"


class TwelveDataProvider(DataProvider):
    name = "twelvedata"

    def __init__(self, api_key: str, cache: DiskCache | None = None):
        super().__init__(cache)
        self.api_key = api_key
        self.http = HttpClient(BASE, max_per_sec=2.0)

    def _q(self, section: str, ttl: float, path: str, params: dict):
        p = dict(params, apikey=self.api_key)
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"td/{section}", key, ttl)
        if hit is not None:
            return hit
        data = self.http.get_json(path, p)
        if isinstance(data, dict) and data.get("status") == "error":
            raise ProviderUnsupported
        self.cache.set(f"td/{section}", key, data, source=self.name)
        return data

    def get_quote(self, ticker: str) -> Quote:
        d = self._q("quote", 10, "/quote", {"symbol": ticker})
        price = d.get("close") or d.get("price")
        if price is None:
            raise ProviderUnsupported
        return Quote(ticker=ticker, price=float(price),
                     volume=float(d.get("volume") or 0),
                     prev_close=float(d["previous_close"]) if d.get("previous_close") else None,
                     day_change_pct=float(d["percent_change"]) if d.get("percent_change") else None,
                     source=self.name)

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
