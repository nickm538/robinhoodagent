"""Twelve Data provider — paid-plan market-data source.

Auth: ?apikey= . Base https://api.twelvedata.com.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from ..models import Quote
from .base import DataProvider, DiskCache, HttpClient, ProviderUnsupported, prices_to_df

BASE = "https://api.twelvedata.com"
_BATCH_CHUNK = 120


def _num(x: Any) -> float | None:
    try:
        if x in (None, "None", "-", ""):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


class TwelveDataProvider(DataProvider):
    name = "twelvedata"

    def __init__(
        self,
        api_key: str,
        cache: DiskCache | None = None,
        *,
        max_per_sec: float | None = None,
        enable_market_movers: bool = False,
    ):
        super().__init__(cache)
        self.api_key = api_key
        rate = max_per_sec
        if rate is None:
            rate = float(os.getenv("TWELVEDATA_MAX_PER_SEC", "8"))
        self.http = HttpClient(BASE, max_per_sec=rate)
        self.enable_market_movers = enable_market_movers or os.getenv(
            "TWELVEDATA_ENABLE_MARKET_MOVERS", ""
        ).lower() in ("1", "true", "yes")

    def _q(self, section: str, ttl: float, path: str, params: dict):
        p = dict(params, apikey=self.api_key)
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"td/{section}", key, ttl)
        if hit is not None:
            return hit
        data = self.http.get_json(path, p)
        if isinstance(data, dict) and data.get("status") == "error":
            msg = str(data.get("message", ""))
            raise ProviderUnsupported(msg or "twelvedata error")
        self.cache.set(f"td/{section}", key, data, source=self.name)
        return data

    def _parse_quote(self, ticker: str, d: dict) -> Quote | None:
        if not isinstance(d, dict):
            return None
        symbol = (d.get("symbol") or ticker).upper()
        price = _num(d.get("close") or d.get("last") or d.get("price"))
        if price is None:
            return None
        return Quote(ticker=symbol, price=price,
                     volume=_num(d.get("volume")) or 0,
                     prev_close=_num(d.get("previous_close")),
                     day_change_pct=_num(d.get("percent_change")),
                     source=self.name)

    def get_quote(self, ticker: str) -> Quote:
        symbol = ticker.upper()
        d = self._q("quote", 1, "/quote", {"symbol": symbol, "country": "United States"})
        q = self._parse_quote(symbol, d)
        if q is None:
            raise ProviderUnsupported
        return q

    def get_quotes_batch(self, tickers: list[str]) -> dict[str, Quote]:
        """Fetch up to 120 symbols per request via comma-separated /quote."""
        out: dict[str, Quote] = {}
        symbols = [t.strip().upper() for t in tickers if t and t.strip()]
        for i in range(0, len(symbols), _BATCH_CHUNK):
            chunk = symbols[i:i + _BATCH_CHUNK]
            if not chunk:
                continue
            d = self._q("quote", 1, "/quote",
                        {"symbol": ",".join(chunk), "country": "United States"})
            if isinstance(d, dict) and d.get("symbol"):
                q = self._parse_quote(str(d["symbol"]).upper(), d)
                if q:
                    out[q.ticker] = q
                continue
            if isinstance(d, dict):
                for sym, payload in d.items():
                    if sym in ("status", "code", "message") or not isinstance(payload, dict):
                        continue
                    q = self._parse_quote(str(payload.get("symbol") or sym).upper(), payload)
                    if q:
                        out[q.ticker] = q
        return out

    def invalidate_quote(self, ticker: str) -> None:
        key = f"/quote|{sorted({'symbol': ticker.upper(), 'country': 'United States'}.items())}"
        self.cache._path("td/quote", key).unlink(missing_ok=True)

    def get_prices(self, ticker: str, start=None, end=None, interval="day") -> pd.DataFrame:
        symbol = ticker.upper()
        iv = {"minute": "1min", "hour": "1h", "day": "1day",
              "week": "1week", "month": "1month"}.get(interval, "1day")
        d = self._q("ts", 720, "/time_series",
                    {"symbol": symbol, "interval": iv, "outputsize": 5000,
                     "country": "United States", "order": "asc"})
        vals = d.get("values") if isinstance(d, dict) else None
        if not vals:
            raise ProviderUnsupported
        recs = []
        for v in vals:
            rec = {k: val for k, val in v.items() if k != "datetime"}
            rec["time"] = v.get("datetime")
            recs.append(rec)
        df = prices_to_df(recs)
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def get_company(self, ticker: str) -> dict:
        symbol = ticker.upper()
        d = self._q("profile", 1440, "/profile", {"symbol": symbol, "country": "United States"})
        if not isinstance(d, dict) or not d.get("symbol"):
            raise ProviderUnsupported
        return {
            "name": d.get("name"),
            "sector": d.get("sector"),
            "industry": d.get("industry"),
            "exchange": d.get("exchange"),
            "market_cap": _num(d.get("market_capitalization") or d.get("market_cap")),
            "source": self.name,
        }

    def get_technicals(self, ticker: str) -> dict:
        symbol = ticker.upper()
        out: dict = {"source": self.name}
        specs = [
            ("rsi", "/rsi", {"time_period": 14}, "rsi"),
            ("ema_9", "/ema", {"time_period": 9}, "ema"),
            ("ema_21", "/ema", {"time_period": 21}, "ema"),
        ]
        for key, path, extra, field in specs:
            try:
                params = {"symbol": symbol, "interval": "1day", "outputsize": 1,
                          "country": "United States", **extra}
                d = self._q(key, 240, path, params)
                vals = d.get("values") if isinstance(d, dict) else None
                val = _num(vals[0].get(field)) if vals else None
                if val is not None:
                    out[key] = val
            except Exception:
                pass
        try:
            d = self._q("macd", 240, "/macd",
                        {"symbol": symbol, "interval": "1day", "outputsize": 1,
                         "country": "United States"})
            vals = d.get("values") if isinstance(d, dict) else None
            if vals:
                row = vals[0]
                for src, dst in (("macd", "macd"), ("macd_signal", "macd_signal"),
                                 ("macd_hist", "macd_hist")):
                    val = _num(row.get(src))
                    if val is not None:
                        out[dst] = val
        except Exception:
            pass
        if len(out) == 1:
            raise ProviderUnsupported
        return out

    def list_universe(self) -> list[str]:
        d = self._q("stocks", 1440, "/stocks", {"country": "United States"})
        recs = d.get("data") if isinstance(d, dict) else None
        out: list[str] = []
        seen: set[str] = set()
        for r in recs or []:
            if not isinstance(r, dict):
                continue
            sym = (r.get("symbol") or "").strip().upper()
            typ = (r.get("type") or "").lower()
            currency = (r.get("currency") or "").upper()
            if not sym or sym in seen:
                continue
            if currency and currency != "USD":
                continue
            if typ and typ != "common stock":
                continue
            if any(ch in sym for ch in ("-", ".", "/")):
                continue
            seen.add(sym)
            out.append(sym)
        if not out:
            raise ProviderUnsupported
        return out

    def get_market_movers(self, limit: int = 60) -> list[str]:
        # This endpoint is useful but costs 100 credits/request, so it stays opt-in.
        if not self.enable_market_movers:
            raise ProviderUnsupported
        d = self._q("movers", 5, "/market_movers/stocks", {"country": "USA", "outputsize": limit})
        recs = d.get("values") or d.get("data") if isinstance(d, dict) else None
        out: list[str] = []
        for r in recs or []:
            sym = (r.get("symbol") or "").strip().upper() if isinstance(r, dict) else ""
            if sym and sym not in out and not any(ch in sym for ch in ("-", ".", "/")):
                out.append(sym)
        if not out:
            raise ProviderUnsupported
        return out[:limit]
