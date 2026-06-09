"""Tests for the intraday movers feed and universe seeding (the sniping fix)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rh_agent.config import load_config
from rh_agent.data import universe as U


def test_light_guards_zero_historical_close():
    """A zero close 63 bars back must not produce inf momentum (dirty data in
    the wider universe) — it would otherwise sort garbage to the top of the funnel."""
    closes = list(np.linspace(100.0, 120.0, 100))
    closes[-63] = 0.0   # the divisor
    df = pd.DataFrame({"close": closes, "volume": [1e6] * 100},
                      index=pd.date_range("2024-01-01", periods=100, freq="B"))

    class FakeMD:
        def get_quote(self, t):
            return None                      # px falls back to last close
        def get_prices(self, t):
            return df
        def get_company(self, t):
            return {"market_cap": 1e10, "sector": "Technology"}

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        c = U._light(FakeMD(), "AAA")
    assert c is not None
    assert math.isfinite(c.mom_63)
    assert c.mom_63 == 0.0


def test_alpha_vantage_movers_parsing(monkeypatch):
    import rh_agent.providers.alpha_vantage as av
    monkeypatch.setattr(av, "OFFLINE", False)
    p = av.AlphaVantageProvider.__new__(av.AlphaVantageProvider)
    p._q = lambda section, ttl, params: {
        "top_gainers": [{"ticker": "AAA"}, {"ticker": "BBB"}],
        "top_losers": [{"ticker": "ZZZ"}],
        "most_actively_traded": [{"ticker": "BBB"}, {"ticker": "CCC"}],
    }
    movers = p.get_market_movers(limit=10)
    # gainers + most-active, de-duped, order preserved; losers excluded
    assert movers == ["AAA", "BBB", "CCC"]
    assert "ZZZ" not in movers


def test_mboum_list_universe_paginates():
    import rh_agent.providers.mboum as mb
    p = mb.MboumProvider.__new__(mb.MboumProvider)
    pages = {1: [{"symbol": "A"}, {"symbol": "B"}], 2: [{"symbol": "C"}], 3: []}
    p._cached = lambda section, ttl, path, params: pages.get(params["page"], [])
    syms = p.list_universe(max_pages=5)
    assert syms == ["A", "B", "C"]   # walked pages 1+2, stopped on empty page 3


def test_twelvedata_quote_and_invalidation_use_us_symbol_key():
    import rh_agent.providers.twelvedata as td

    p = td.TwelveDataProvider.__new__(td.TwelveDataProvider)
    p._q = lambda section, ttl, path, params: {
        "close": "212.34",
        "volume": "123456",
        "previous_close": "210.00",
        "percent_change": "1.11",
    }
    q = p.get_quote("aapl")
    assert q.ticker == "AAPL"
    assert q.price == 212.34
    assert q.volume == 123456
    assert q.prev_close == 210.00
    assert q.day_change_pct == 1.11

    class _Cache:
        def __init__(self):
            self.ns = self.key = None

        def _path(self, ns, key):
            self.ns, self.key = ns, key

            class _Path:
                def unlink(self, missing_ok=False):
                    pass

            return _Path()

    p.cache = _Cache()
    p.invalidate_quote("aapl")
    assert p.cache.ns == "td/quote"
    assert "AAPL" in p.cache.key
    assert "United States" in p.cache.key


def test_twelvedata_prices_normalize_datetime_without_duplicate_time_column():
    import rh_agent.providers.twelvedata as td

    p = td.TwelveDataProvider.__new__(td.TwelveDataProvider)
    p._q = lambda section, ttl, path, params: {
        "values": [
            {"datetime": "2026-06-06", "open": "10", "high": "11",
             "low": "9", "close": "10.5", "volume": "1000"},
            {"datetime": "2026-06-09", "open": "11", "high": "12",
             "low": "10", "close": "11.5", "volume": "1200"},
        ]
    }
    prices = p.get_prices("AAPL")
    assert list(prices.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert len(prices) == 2
    assert prices["close"].iloc[-1] == 11.5


def test_twelvedata_technicals_use_current_indicator_endpoints():
    import rh_agent.providers.twelvedata as td

    p = td.TwelveDataProvider.__new__(td.TwelveDataProvider)
    calls = []

    def _q(section, ttl, path, params):
        calls.append((path, params))
        if path == "/rsi":
            return {"values": [{"rsi": "61.5"}]}
        if path == "/ema":
            period = params["time_period"]
            return {"values": [{"ema": "101.0" if period == 9 else "99.5"}]}
        if path == "/macd":
            return {"values": [{"macd": "1.2", "macd_signal": "0.8", "macd_hist": "0.4"}]}
        raise AssertionError(path)

    p._q = _q
    out = p.get_technicals("MSFT")
    assert out["rsi"] == 61.5
    assert out["ema_9"] == 101.0
    assert out["ema_21"] == 99.5
    assert out["macd"] == 1.2
    assert [path for path, _ in calls] == ["/rsi", "/ema", "/ema", "/macd"]


def test_twelvedata_list_universe_filters_us_common_stocks():
    import rh_agent.providers.twelvedata as td

    p = td.TwelveDataProvider.__new__(td.TwelveDataProvider)
    p._q = lambda section, ttl, path, params: {
        "data": [
            {"symbol": "AAPL", "currency": "USD", "type": "Common Stock"},
            {"symbol": "SPY", "currency": "USD", "type": "ETF"},
            {"symbol": "B.B", "currency": "USD", "type": "Common Stock"},
            {"symbol": "SHOP", "currency": "CAD", "type": "Common Stock"},
            {"symbol": "MSFT", "currency": "USD", "type": "Common Stock"},
        ]
    }
    assert p.list_universe() == ["AAPL", "MSFT"]


def test_twelvedata_market_movers_are_opt_in():
    import pytest
    import rh_agent.providers.twelvedata as td
    from rh_agent.providers.base import ProviderUnsupported

    p = td.TwelveDataProvider.__new__(td.TwelveDataProvider)
    p.enable_market_movers = False
    with pytest.raises(ProviderUnsupported):
        p.get_market_movers()

    p.enable_market_movers = True
    p._q = lambda section, ttl, path, params: {
        "values": [{"symbol": "AAA"}, {"symbol": "B.B"}, {"symbol": "CCC"}]
    }
    assert p.get_market_movers(limit=10) == ["AAA", "CCC"]


def test_market_data_get_market_movers_facade():
    from rh_agent.data.market_data import MarketData

    class _AV:
        def get_market_movers(self, limit):
            return ["X", "Y"][:limit]

    md = MarketData.__new__(MarketData)
    md.providers = {"alphavantage": _AV()}
    md.priority = {"universe": ["alphavantage"]}
    assert md.get_market_movers(limit=5) == ["X", "Y"]


def _fake_candidate(t):
    return U.Candidate(t, price=50.0, market_cap=1e10, adv=1e7, avg_volume=1e6,
                       hist_days=300, mom_63=0.1, day_change_pct=0.0, rel_volume=1.0,
                       dollar_volume_today=1e7, breakout_20=0.0, sector="Technology")


def test_build_universe_seeds_movers_ahead_of_base(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(U, "_light", lambda md, t, max_quote_age_seconds=None: _fake_candidate(t))

    class FakeMD:
        def get_market_movers(self, limit):
            return ["MOV1", "MOV2"]

        def list_universe(self):
            return ["BASE1", "BASE2"]

    out = U.build_universe(FakeMD(), cfg)
    assert {"MOV1", "MOV2", "BASE1", "BASE2"} <= set(out)
    # movers are seeded ahead of the static base set
    assert out.index("MOV1") < out.index("BASE1")


def test_build_universe_tolerates_movers_feed_failure(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(U, "_light", lambda md, t, max_quote_age_seconds=None: _fake_candidate(t))

    class FakeMD:
        def get_market_movers(self, limit):
            raise RuntimeError("feed down")

        def list_universe(self):
            return ["BASE1", "BASE2"]

    out = U.build_universe(FakeMD(), cfg)
    assert set(out) == {"BASE1", "BASE2"}   # falls back to the base universe, no crash
