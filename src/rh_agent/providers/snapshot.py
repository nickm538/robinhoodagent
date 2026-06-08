"""SnapshotProvider — serves a normalised, timestamped snapshot of real data.

A snapshot is a JSON file produced by capturing real provider responses (e.g.
during a live scan, or via the bundled capture utility). It is NOT mock data:
every value originated from a live API/feed and carries a capture timestamp.

Used for: (a) reproducible/offline runs, (b) backtest price history, and
(c) validating the scoring engine on genuine numbers in restricted networks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..models import Quote
from .base import DataProvider, ProviderUnsupported, prices_to_df


class SnapshotProvider(DataProvider):
    name = "snapshot"

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"snapshot not found: {self.path}")
        self._d = json.loads(self.path.read_text())
        self.tickers: dict = self._d.get("tickers", {})
        self.captured_at = self._d.get("_captured_iso") or self._d.get("_captured_at")

    def _t(self, ticker: str) -> dict:
        t = self.tickers.get(ticker)
        if t is None:
            raise ProviderUnsupported
        return t

    def list_universe(self) -> list[str]:
        return list(self.tickers.keys())

    def get_company(self, ticker: str) -> dict:
        return dict(self._t(ticker).get("company", {}), source="snapshot")

    def get_quote(self, ticker: str) -> Quote:
        q = self._t(ticker).get("quote")
        if not q:
            raise ProviderUnsupported
        return Quote(ticker=ticker, price=float(q["price"]), volume=float(q.get("volume") or 0),
                     prev_close=q.get("prev_close"), day_change_pct=q.get("day_change_pct"),
                     source="snapshot")

    def get_prices(self, ticker: str, start=None, end=None, interval="day") -> pd.DataFrame:
        t = self.tickers.get(ticker)
        recs = (t or {}).get("prices")
        if not recs:
            # benchmarks (SPY/RSP/VIX...) live under a separate key
            recs = self._d.get("benchmarks", {}).get(ticker, {}).get("prices")
        if not recs:
            raise ProviderUnsupported
        df = prices_to_df(recs)
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def _section(self, ticker: str, key: str):
        v = self._t(ticker).get(key)
        if v in (None, {}, []):
            raise ProviderUnsupported
        return v

    def get_fundamentals(self, ticker: str) -> dict:
        return self._section(ticker, "fundamentals")

    def get_technicals(self, ticker: str) -> dict:
        return self._section(ticker, "technicals")

    def get_insider(self, ticker: str) -> list:
        return self._section(ticker, "insider")

    def get_institutional(self, ticker: str) -> dict:
        return self._section(ticker, "institutional")

    def get_news_sentiment(self, ticker: str) -> dict:
        return self._section(ticker, "news_sentiment")

    def get_analyst(self, ticker: str) -> dict:
        return self._section(ticker, "analyst")

    def get_short_interest(self, ticker: str) -> dict:
        return self._section(ticker, "short_interest")

    def get_options(self, ticker: str) -> dict:
        return self._section(ticker, "options")

    def get_earnings(self, ticker: str) -> dict:
        return self._section(ticker, "earnings")

    def get_pro_scores(self, ticker: str, name: str | None = None) -> dict:
        return self._section(ticker, "pro_scores")

    def get_macro(self) -> dict:
        return self._d.get("macro", {})
