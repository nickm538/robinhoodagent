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
