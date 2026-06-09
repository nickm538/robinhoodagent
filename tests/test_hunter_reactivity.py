"""Reactivity fixes: the hunter must not choke itself.

Covers the chokepoints that silently nullified the aggressive config on the
live VM: scan latency aging fresh quotes into stale_quote vetoes, the 60s risk
tick rebuilding full TickerData per held name, the serial universe light pass,
provider-technicals calls on light builds, and Alpha Vantage throttle spam.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from rh_agent.config import load_config
from rh_agent.models import Account, Position, Quote, TickerData, Verdict, utcnow
from rh_agent.scoring import Scorer


def _prices(days: int = 260, start: float = 100.0, end: float = 130.0) -> pd.DataFrame:
    close = np.linspace(start, end, days)
    return pd.DataFrame(
        {"close": close, "adj_close": close, "high": close * 1.01,
         "low": close * 0.99, "volume": [1_000_000] * days},
        index=pd.date_range(end=pd.Timestamp.now("UTC").normalize(), periods=days, freq="B"),
    )


# ---------------- stale_quote measures data age at build, not scan latency ----------------

def test_scan_latency_does_not_stale_flag_fresh_quotes():
    """A quote fetched fresh at build time must stay eligible even when scoring
    happens many minutes later (deep gather + web research + AI overlay)."""
    scorer = Scorer(load_config())
    built = utcnow() - timedelta(minutes=10)
    td = TickerData("FRESH",
                    quote=Quote("FRESH", 100.0, asof=built),
                    meta={"captured_at": built.isoformat()})
    v = Verdict("FRESH", 80.0, {}, pillars_passing=5)
    scorer._add_flags(v, td)
    assert "stale_quote" not in v.flags


def test_genuinely_old_quote_is_still_flagged():
    scorer = Scorer(load_config())
    captured = utcnow()
    td = TickerData("OLD",
                    quote=Quote("OLD", 100.0, asof=captured - timedelta(minutes=10)),
                    meta={"captured_at": captured.isoformat()})
    v = Verdict("OLD", 80.0, {}, pillars_passing=5)
    scorer._add_flags(v, td)
    assert "stale_quote" in v.flags


def test_missing_capture_time_falls_back_to_now():
    scorer = Scorer(load_config())
    td = TickerData("X", quote=Quote("X", 100.0, asof=utcnow() - timedelta(minutes=10)))
    v = Verdict("X", 80.0, {}, pillars_passing=5)
    scorer._add_flags(v, td)
    assert "stale_quote" in v.flags


# ---------------- daemon risk tick: cheap ATR, no full TickerData rebuild ----------------

class _RiskMD:
    """MarketData stub: cached prices OK, full build forbidden."""

    def __init__(self):
        self.prefetched: list[list[str]] = []

    def build(self, *a, **k):
        raise AssertionError("risk tick must NOT rebuild full TickerData")

    def get_prices(self, ticker, *a, **k):
        return _prices()

    def prefetch_quotes(self, tickers):
        self.prefetched.append(list(tickers))
        return len(tickers)


class _RiskAgent:
    def __init__(self, prices):
        self._prices = prices
        self.md = _RiskMD()

    def price_fn(self, ticker, for_risk=False):
        return self._prices.get(ticker)


def _risk_daemon(price: float):
    import rh_agent.daemon as daemon
    cfg = load_config()
    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.cfg = cfg
    d.agent = _RiskAgent({"AAA": price})
    d.state = daemon.DaemonState(stops={"AAA": 95.0}, take_profits={},
                                 high_water={"AAA": price}, pending_risk={})
    d.journal = daemon.Journal(cfg)
    d.journal.enabled = False
    return d


def test_trailing_stop_update_does_not_full_build():
    d = _risk_daemon(price=130.0)

    class _Brk:
        def place_order(self, order, dry_run=True):
            raise AssertionError("no order expected")

    acct = Account(equity=1_000.0, cash=0.0, buying_power=0.0, source="paper",
                   positions=[Position("AAA", 1.0, 100.0, current_price=130.0)])
    triggered = d._manage_risk(_Brk(), acct, execute=False)
    assert triggered == set()
    # trailing stop ratcheted up from cached-price ATR (no full build needed)
    assert d.state.stops["AAA"] > 95.0


def test_ensure_stops_synthesizes_from_cached_prices():
    d = _risk_daemon(price=130.0)
    d.state.stops = {}
    acct = Account(equity=1_000.0, cash=0.0, buying_power=0.0, source="paper",
                   positions=[Position("AAA", 1.0, 100.0, current_price=130.0)])
    d._ensure_stops_for_held(None, acct)
    assert d.state.stops.get("AAA")


def test_held_quotes_prefetched_in_one_batch():
    d = _risk_daemon(price=100.0)
    acct = Account(equity=1_000.0, cash=0.0, buying_power=0.0, source="paper",
                   positions=[Position("AAA", 1.0, 100.0), Position("BBB", 2.0, 50.0)])
    d._prefetch_held_quotes(acct)
    assert d.agent.md.prefetched == [["AAA", "BBB"]]


# ---------------- universe light pass: parallel fan-out, same filtering ----------------

def test_build_universe_parallel_matches_filters():
    from rh_agent.data.universe import build_universe

    class _UniMD:
        def get_quote(self, t):
            return Quote(t, 50.0, volume=2_000_000)

        def get_quote_for_risk(self, t, max_age_seconds=None):
            return self.get_quote(t)

        def get_prices(self, t, *a, **k):
            return _prices(days=260)

        def get_company(self, t):
            return {"market_cap": 5e9, "sector": "Tech"}

    cfg = load_config()
    cfg.raw["universe"]["intraday"]["enabled"] = False
    cfg.raw["universe"]["blacklist"] = ["BAD"]
    cfg.raw["data"]["max_workers"] = 4
    out = build_universe(_UniMD(), cfg, raw=["AAA", "BAD", "BBB", "CCC"])
    assert "BAD" not in out
    assert set(out) == {"AAA", "BBB", "CCC"}


# ---------------- provider technicals only on deep builds ----------------

def _tech_md():
    from rh_agent.data.market_data import MarketData

    class _Prov:
        name = "stub"
        tech_calls = 0

        def get_quote(self, t):
            return Quote(t, 100.0, volume=1e6)

        def get_prices(self, t, *a, **k):
            return _prices()

        def get_company(self, t):
            return {"sector": "Tech", "market_cap": 5e9}

        def get_fundamentals(self, t):
            return {"roe": 0.2}

        def get_technicals(self, t):
            type(self).tech_calls += 1
            return {"rsi": 55.0}

    cfg = load_config()
    cfg.raw["providers"] = {
        "quote": ["stub"], "quote_risk": ["stub"], "prices": ["stub"],
        "fundamentals": ["stub"], "technicals": ["stub"],
    }
    prov = _Prov()
    type(prov).tech_calls = 0
    return MarketData(cfg, {"stub": prov}), prov


def test_light_build_skips_provider_technicals():
    md, prov = _tech_md()
    td = md.build("AAA", deep=False)
    assert type(prov).tech_calls == 0
    assert td.technicals.get("atr") is not None   # local indicators still computed


def test_deep_build_enriches_with_provider_technicals():
    md, prov = _tech_md()
    md.build("AAA", deep=True)
    assert type(prov).tech_calls == 1


# ---------------- Alpha Vantage throttle circuit breaker ----------------

def test_alpha_vantage_cooldown_fails_fast_after_throttle(tmp_path):
    from rh_agent.providers.alpha_vantage import AlphaVantageProvider
    from rh_agent.providers.base import DiskCache, ProviderError, ProviderUnsupported

    av = AlphaVantageProvider("k", DiskCache(tmp_path))
    calls = {"n": 0}

    class _Http:
        def get_json(self, path, params=None, **kw):
            calls["n"] += 1
            return {"Note": "rate limit"}

    av.http = _Http()
    with pytest.raises(ProviderError):
        av._q("quote", 1, {"function": "GLOBAL_QUOTE", "symbol": "AAA"})
    assert calls["n"] == 1
    # cooling down: the next uncached call must NOT hit the network
    with pytest.raises(ProviderUnsupported):
        av._q("quote", 1, {"function": "GLOBAL_QUOTE", "symbol": "BBB"})
    assert calls["n"] == 1
