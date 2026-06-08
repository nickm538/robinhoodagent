from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from rh_agent.agent import TradingAgent
from rh_agent.config import load_config
from rh_agent.data.market_data import MarketData
from rh_agent.data.universe import build_universe
from rh_agent.models import Quote, utcnow


def _prices(price: float, *, avg_volume: float = 100_000, days: int = 220) -> pd.DataFrame:
    close = np.linspace(price * 0.9, price, days)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": [avg_volume] * days,
        },
        index=pd.date_range("2025-01-01", periods=days, freq="B"),
    )


class _FakeMarketData:
    def __init__(self):
        self.max_ages: list[float] = []
        self.quotes = {
            "RUN": Quote("RUN", 10.0, volume=500_000, day_change_pct=4.0),
            "VOL": Quote("VOL", 20.0, volume=300_000, day_change_pct=0.8),
            "THIN": Quote("THIN", 10.0, volume=1_000, day_change_pct=12.0),
            "DOWN": Quote("DOWN", 15.0, volume=500_000, day_change_pct=-3.0),
            "SLOW": Quote("SLOW", 30.0, volume=100_000, day_change_pct=0.2),
        }

    def get_quote(self, ticker: str) -> Quote | None:
        return self.quotes.get(ticker)

    def get_quote_for_risk(self, ticker: str, max_age_seconds: float = 180) -> Quote | None:
        self.max_ages.append(max_age_seconds)
        return self.get_quote(ticker)

    def get_prices(self, ticker: str):
        price = self.quotes[ticker].price
        return _prices(price)

    def get_company(self, ticker: str) -> dict:
        return {"market_cap": 500_000_000, "sector": "Technology"}


def _intraday_cfg():
    cfg = load_config()
    u = cfg.raw["universe"]
    u["liquidity"].update({
        "min_market_cap": 300_000_000,
        "min_avg_dollar_volume": 0,
        "min_avg_volume_shares": 0,
        "min_history_days": 60,
    })
    u["prescreen"] = {"enabled": False}
    u["intraday"].update({
        "enabled": True,
        "quote_max_age_seconds": 60,
        "max_candidates": 2,
        "min_candidates": 1,
        "min_day_change_pct": 2.0,
        "min_positive_day_change_pct": 0.5,
        "min_relative_volume": 1.5,
        "min_dollar_volume_today": 2_000_000,
        "min_breakout_pct": 0.01,
        "fallback_to_liquid_universe": False,
    })
    return cfg


def test_intraday_radar_keeps_fast_liquid_runners():
    md = _FakeMarketData()
    cfg = _intraday_cfg()

    names = build_universe(md, cfg, raw=["SLOW", "THIN", "DOWN", "VOL", "RUN"])

    assert names == ["RUN", "VOL"]
    assert md.max_ages and set(md.max_ages) == {60}


def test_intraday_radar_falls_back_to_liquid_universe_when_too_sparse():
    md = _FakeMarketData()
    cfg = _intraday_cfg()
    cfg.raw["universe"]["intraday"].update({
        "min_candidates": 3,
        "min_day_change_pct": 25.0,
        "min_relative_volume": 20.0,
        "fallback_to_liquid_universe": True,
    })

    names = build_universe(md, cfg, raw=["SLOW", "VOL"])

    assert set(names) == {"SLOW", "VOL"}


def test_universe_appends_held_tickers_to_dynamic_scan():
    cfg = load_config()
    cfg.raw["universe"]["watchlist"] = ["AAA"]
    agent = TradingAgent.__new__(TradingAgent)
    agent.cfg = cfg
    agent.providers = {}

    assert agent.universe(include_tickers=["BBB"]) == ["AAA", "BBB"]


class _FreshnessProvider:
    enabled = True

    def __init__(self):
        self.invalidated = False

    def get_quote(self, ticker: str) -> Quote:
        if self.invalidated:
            return Quote(ticker, 2.0)
        return Quote(ticker, 1.0, asof=utcnow() - timedelta(minutes=10))

    def invalidate_quote(self, ticker: str) -> None:
        self.invalidated = True


def test_quote_for_risk_invalidates_provider_quote_cache():
    cfg = load_config()
    cfg.raw["providers"]["quote"] = ["fresh"]
    provider = _FreshnessProvider()
    md = MarketData(cfg, {"fresh": provider})

    quote = md.get_quote_for_risk("AAA", max_age_seconds=60)

    assert provider.invalidated is True
    assert quote and quote.price == 2.0
