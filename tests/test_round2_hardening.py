"""Round-2 hardening: Mboum quota circuit breaker, intraday regime tape shock,
monotonic stops across rebalances, buy-only cooldown exclusion, and the
cooldown-aware background scan."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rh_agent.config import load_config
from rh_agent.execution import build_orders
from rh_agent.models import Account, Position, Quote, TargetPosition
from rh_agent.regime import detect_regime


# ----------------------------- mboum quota breaker -------------------------

def _mboum(monkeypatch=None):
    import rh_agent.providers.mboum as mb

    p = mb.MboumProvider.__new__(mb.MboumProvider)
    p._rate_limited_until = 0.0

    class _Cache:
        def __init__(self):
            self.store = {}

        def get(self, ns, key, ttl):
            return self.store.get((ns, key))

        def set(self, ns, key, data, source=""):
            self.store[(ns, key)] = data

    p.cache = _Cache()
    return p, mb


def test_mboum_quota_body_detection():
    _, mb = _mboum()
    assert mb._is_quota_payload({"message": "You have reached your monthly limit"})
    assert mb._is_quota_payload({"error": "Rate limit exceeded, upgrade your plan"})
    # real payloads always carry a data section — never flagged
    assert not mb._is_quota_payload({"body": [{"symbol": "AAPL"}],
                                     "message": "limit info in metadata"})
    assert not mb._is_quota_payload(["raw", "list"])
    assert not mb._is_quota_payload({"message": "ok"})


def test_mboum_429_trips_cooldown_and_stops_calling():
    from rh_agent.providers.base import ProviderUnsupported, RateLimitError

    p, _ = _mboum()
    calls = {"n": 0}

    class _Http:
        def get_json(self, path, params):
            calls["n"] += 1
            raise RateLimitError("429", retry_after_seconds=None)

    p.http = _Http()
    with pytest.raises(ProviderUnsupported):
        p._cached("quote", 10, "/v1/markets/quote", {"ticker": "AAPL"})
    assert calls["n"] == 1
    assert p._rate_limit_active()
    # while cooling down, no further HTTP calls are made at all
    with pytest.raises(ProviderUnsupported):
        p._cached("quote", 10, "/v1/markets/quote", {"ticker": "MSFT"})
    assert calls["n"] == 1


def test_mboum_quota_error_body_not_cached_and_trips_cooldown():
    from rh_agent.providers.base import ProviderUnsupported

    p, _ = _mboum()

    class _Http:
        def get_json(self, path, params):
            return {"message": "Monthly limit reached. Upgrade your plan."}

    p.http = _Http()
    with pytest.raises(ProviderUnsupported):
        p._cached("quote", 10, "/v1/markets/quote", {"ticker": "AAPL"})
    assert p.cache.store == {}          # poison body never cached
    assert p._rate_limit_active()


def test_mboum_previously_cached_quota_body_is_ignored():
    from rh_agent.providers.base import ProviderUnsupported

    p, _ = _mboum()
    key = "/v1/markets/quote|[('ticker', 'AAPL')]"
    p.cache.store[("mboum/quote", key)] = {"message": "rate limit exceeded"}

    class _Http:
        def get_json(self, path, params):
            return {"message": "rate limit exceeded"}

    p.http = _Http()
    with pytest.raises(ProviderUnsupported):   # refetched (hit ignored), then trips
        p._cached("quote", 10, "/v1/markets/quote", {"ticker": "AAPL"})


def test_mboum_normal_payload_flows_and_caches():
    p, _ = _mboum()

    class _Http:
        def get_json(self, path, params):
            return {"body": [{"symbol": "AAPL", "regularMarketPrice": 200.0}]}

    p.http = _Http()
    out = p._cached("quote", 10, "/v1/markets/quote", {"ticker": "AAPL"})
    assert out["body"][0]["regularMarketPrice"] == 200.0
    assert len(p.cache.store) == 1
    assert not p._rate_limit_active()


# ----------------------------- intraday tape shock -------------------------

def _df(values):
    return pd.DataFrame({"close": values},
                        index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


class _TapeMD:
    """Uptrend + calm VIX => base regime risk_on_trend; SPY day move injectable."""

    def __init__(self, spy_day_change=None, vix=12.0):
        rising = list(np.linspace(300.0, 420.0, 260))
        self._series = {"SPY": _df(rising), "RSP": _df(rising), "VIX": _df([vix] * 5)}
        self._dc = spy_day_change

    def get_index_prices(self, symbol):
        return self._series.get(symbol)

    def get_macro(self):
        return {}

    def get_quote(self, t):
        if self._dc is None:
            return None
        return Quote(t, 500.0, day_change_pct=self._dc)


def test_tape_shock_downgrades_risk_on_to_neutral():
    res = detect_regime(_TapeMD(spy_day_change=-1.8), load_config())
    assert res.name == "neutral"
    assert res.signals["spy_day_change_pct"] == pytest.approx(-1.8)
    assert "SPY today" in res.describe()


def test_tape_crash_forces_risk_off():
    res = detect_regime(_TapeMD(spy_day_change=-3.0), load_config())
    assert res.name == "risk_off"
    assert res.exposure == pytest.approx(0.50)


def test_tape_shock_never_upgrades_high_volatility():
    res = detect_regime(_TapeMD(spy_day_change=-3.0, vix=30.0), load_config())
    assert res.name == "high_volatility"


def test_tape_quiet_or_missing_quote_leaves_regime_alone():
    assert detect_regime(_TapeMD(spy_day_change=0.4), load_config()).name == "risk_on_trend"
    assert detect_regime(_TapeMD(spy_day_change=None), load_config()).name == "risk_on_trend"


def test_tape_shock_disabled_via_config():
    cfg = load_config()
    cfg.raw["regime"]["intraday"] = {"enabled": False}
    assert detect_regime(_TapeMD(spy_day_change=-5.0), cfg).name == "risk_on_trend"


# ----------------------------- monotonic stops ------------------------------

def test_rebalance_never_lowers_a_ratcheted_stop(tmp_path, monkeypatch):
    import rh_agent.daemon as daemon
    from rh_agent.agent import RunResult, ScanResult
    from rh_agent.regime import RegimeResult

    monkeypatch.setattr(daemon, "STATE", tmp_path / "daemon_state.json")
    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.state = daemon.DaemonState.load()
    d.journal = MagicMock()
    d.agent = MagicMock()
    d.agent.price_fn = lambda t: 100.0
    d.state.stops = {"AAA": 95.19}          # breakeven-ratcheted earlier today
    d.state.high_water = {"AAA": 100.0}

    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("AAA", 1.0, 95.0, current_price=100.0)])
    scan = ScanResult(regime=RegimeResult("neutral", {}, 0.85), verdicts=[], eligible=[],
                      targets=[TargetPosition("AAA", 0.3, 80.0, stop_price=90.0,
                                              take_profit=120.0, sector="Tech")],
                      equity=1000.0, universe_size=1, scored_size=1)
    run = RunResult(scan=scan, account=acct, orders=[], fills=[], executed=True,
                    mode="paper", post_account=acct)
    d._apply_rebalance_result(run, acct, now=datetime.now(timezone.utc))
    assert d.state.stops["AAA"] == pytest.approx(95.19)   # NOT lowered to 90

    # ...but a HIGHER fresh stop still ratchets up
    scan.targets[0].stop_price = 97.0
    d._apply_rebalance_result(run, acct, now=datetime.now(timezone.utc))
    assert d.state.stops["AAA"] == pytest.approx(97.0)


# ----------------------------- buy-only cooldown exclusion ------------------

def test_exclude_buys_blocks_rebuy_but_never_an_exit():
    cfg = load_config()
    # held name NOT in targets -> exit sell must fire even while on cooldown
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("COOL", 2.0, 100.0, current_price=100.0)])
    orders = build_orders(acct, [], cfg, lambda t: 100.0, exclude_buys={"COOL"})
    assert [o.side for o in orders] == ["sell"]

    # fresh target on cooldown -> buy suppressed with an explain note
    acct2 = Account(equity=1000.0, cash=1000.0, buying_power=1000.0, positions=[])
    notes: list = []
    orders2 = build_orders(acct2, [TargetPosition("COOL", 0.10, 80.0, sector="Tech")],
                           cfg, lambda t: 50.0, exclude_buys={"COOL"}, explain=notes)
    assert orders2 == []
    assert any(n["action"] == "excluded" for n in notes)


def test_exclude_tickers_still_blocks_both_sides():
    cfg = load_config()
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("PEND", 2.0, 100.0, current_price=100.0)])
    orders = build_orders(acct, [], cfg, lambda t: 100.0, exclude_tickers={"PEND"})
    assert orders == []                      # unresolved protective order: hands off


# ----------------------------- cooldown-aware scan ---------------------------

def test_safe_scan_passes_exclusions_to_fresh_agent(monkeypatch):
    import rh_agent.daemon as daemon

    captured = {}

    class _FakeAgent:
        def __init__(self, cfg, snapshot_path=None):
            pass

        def scan(self, equity=None, include_tickers=None, exclude_tickers=None):
            captured.update(equity=equity, include=include_tickers,
                            exclude=exclude_tickers)
            return "scan-result"

    monkeypatch.setattr(daemon, "TradingAgent", _FakeAgent)
    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.cfg = load_config()
    out = d._safe_scan(500.0, ["HELD"], {"COOL"})
    assert out == "scan-result"
    assert captured == {"equity": 500.0, "include": ["HELD"], "exclude": {"COOL"}}
