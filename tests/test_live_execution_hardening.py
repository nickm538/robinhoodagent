"""Tests for dual-agent institutional review hardening and overnight upgrades."""
from __future__ import annotations

from datetime import datetime, timezone
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from rh_agent.agent import ScanResult, TradingAgent
from rh_agent.config import load_config
from rh_agent.execution import execute_orders
from rh_agent.models import Account, Order, Position, TargetPosition, Verdict
from rh_agent.regime import RegimeResult


class _FakeBroker:
    supports_live = True
    name = "fake"

    def __init__(self, statuses: list[str] | None = None):
        self.statuses = list(statuses or ["submitted"])
        self.calls: list[Order] = []

    def place_order(self, order: Order, dry_run: bool = True) -> dict:
        self.calls.append(order)
        status = self.statuses.pop(0) if self.statuses else "submitted"
        return {"status": status, "ticker": order.ticker}

    def get_account(self) -> Account:
        return Account(
            equity=10_000.0, cash=5_000.0, buying_power=5_000.0, positions=[],
            source="robinhood", portfolio_confirmed=True, positions_confirmed=True,
            buying_power_confirmed=True,
        )


def test_execute_orders_refreshes_before_each_live_buy(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_UNDERSTAND_REAL_MONEY")
    broker = _FakeBroker()
    refreshes = {"n": 0}
    real_get = broker.get_account

    def counting_get():
        refreshes["n"] += 1
        return real_get()

    orders = [
        Order("AAA", "sell", 1.0),
        Order("BBB", "buy", 1.0, notional=50.0),
        Order("CCC", "buy", 1.0, notional=50.0),
    ]
    fills, post = execute_orders(
        broker, orders, cfg, get_account=counting_get,
    )
    assert len(fills) == 3
    assert refreshes["n"] >= 3  # before each buy + post-trade
    assert post is not None


def test_execute_orders_skips_buy_on_unreliable_account(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_UNDERSTAND_REAL_MONEY")
    broker = _FakeBroker()

    def bad_account():
        return Account(equity=10_000.0, cash=0.0, buying_power=0.0, positions=[],
                       source="robinhood", portfolio_confirmed=False)

    orders = [Order("AAA", "buy", 1.0, notional=50.0)]
    fills, _ = execute_orders(broker, orders, cfg, get_account=bad_account)
    assert fills[0]["status"] == "skipped"
    assert broker.calls == []


def test_execute_orders_skips_live_buy_when_refresh_raises(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_UNDERSTAND_REAL_MONEY")
    broker = _FakeBroker()

    def broken_account():
        raise RuntimeError("broker timeout")

    fills, _ = execute_orders(
        broker, [Order("AAA", "buy", 1.0, notional=50.0)], cfg,
        get_account=broken_account,
    )
    assert fills == [{"status": "skipped", "ticker": "AAA", "side": "buy",
                      "reason": "account_refresh_failed"}]
    assert broker.calls == []


def test_execute_orders_caps_live_buy_to_fresh_buying_power(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_UNDERSTAND_REAL_MONEY")
    broker = _FakeBroker()

    def small_account():
        return Account(
            equity=10_000.0, cash=30.0, buying_power=30.0, positions=[],
            source="robinhood", portfolio_confirmed=True, positions_confirmed=True,
            buying_power_confirmed=True,
        )

    fills, _ = execute_orders(
        broker, [Order("AAA", "buy", 1.0, notional=100.0)], cfg,
        get_account=small_account,
    )
    assert fills[0]["status"] == "submitted"
    assert broker.calls[0].notional == pytest.approx(29.10)


def test_reconcile_and_execute_places_paper_orders():
    cfg = load_config()
    agent = TradingAgent.__new__(TradingAgent)
    agent.cfg = cfg
    agent.providers = {"snapshot": object()}
    agent._quote_cache = {}
    agent.price_fn = lambda ticker, for_risk=False: 10.0
    agent.default_equity = lambda: 1000.0
    agent._apply_hold_discipline = lambda scan, account, equity: scan
    agent._refresh_execution_quotes = lambda scan, account: None

    class _PaperBroker:
        supports_live = False

        def __init__(self):
            self.calls: list[Order] = []

        def get_account(self):
            return Account(equity=1000.0, cash=1000.0, buying_power=1000.0, positions=[])

        def place_order(self, order: Order, dry_run: bool = True):
            self.calls.append(order)
            return {"status": "filled", "ticker": order.ticker, "side": order.side}

    scan = ScanResult(
        regime=RegimeResult("neutral", {}, 1.0),
        verdicts=[], eligible=[],
        targets=[TargetPosition("AAA", 0.10, 80, sector="Tech")],
        equity=1000.0, universe_size=1, scored_size=1,
    )
    broker = _PaperBroker()
    run = agent.reconcile_and_execute(
        scan, execute=True, broker=broker, account=broker.get_account(), equity=1000.0,
    )
    assert len(run.orders) == 1
    assert len(run.fills) == 1
    assert broker.calls[0].ticker == "AAA"


def test_fd_prev_close_from_percent():
    from rh_agent.providers.financial_datasets import FinancialDatasetsProvider

    fd = FinancialDatasetsProvider.__new__(FinancialDatasetsProvider)
    fd._ttl = lambda section, default: default  # type: ignore[method-assign]
    fd.cache = MagicMock()
    fd.http = MagicMock()
    fd._cached = lambda *a, **k: {  # type: ignore[method-assign]
        "snapshot": {"price": 110.0, "day_change_percent": 10.0, "volume": 1_000_000},
    }["snapshot"]
    q = fd.get_quote("TEST")
    assert q.day_change_pct == pytest.approx(10.0)
    assert q.prev_close == pytest.approx(100.0)


def test_av_sentiment_defers_when_no_ticker_relevance(monkeypatch):
    from rh_agent.providers.alpha_vantage import AlphaVantageProvider
    from rh_agent.providers.base import ProviderUnsupported

    av = AlphaVantageProvider.__new__(AlphaVantageProvider)
    av._q = lambda *a, **k: {  # type: ignore[method-assign]
        "feed": [{"title": "macro", "ticker_sentiment": []}],
    }
    with pytest.raises(ProviderUnsupported):
        av.get_news_sentiment("ZZZZ")


def test_agent_refresh_execution_quotes_clears_cache(monkeypatch):
    cfg = load_config()
    agent = TradingAgent(cfg)
    agent._quote_cache["AAA"] = 100.0
    refreshed: list[list[str]] = []

    def fake_refresh(tickers, max_age_seconds=120):
        refreshed.append(list(tickers))

    agent.md.refresh_quotes = fake_refresh  # type: ignore[method-assign]
    scan = ScanResult(
        regime=RegimeResult("neutral", {}, 0.85),
        verdicts=[], eligible=[],
        targets=[TargetPosition("BBB", 0.1, 80, sector="Tech")],
        equity=10_000, universe_size=1, scored_size=1,
    )
    acct = Account(equity=10_000, cash=5_000, buying_power=5_000,
                   positions=[Position("AAA", 1, 50, current_price=50)])
    agent._refresh_execution_quotes(scan, acct)
    assert refreshed
    assert set(refreshed[0]) == {"AAA", "BBB"}
    assert "AAA" not in agent._quote_cache


def test_daemon_last_rebalance_set_on_completion(tmp_path, monkeypatch):
    import rh_agent.daemon as daemon
    from rh_agent.agent import RunResult

    p = tmp_path / "daemon_state.json"
    monkeypatch.setattr(daemon, "STATE", p)
    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.state = daemon.DaemonState.load()
    d.journal = MagicMock()
    d.agent = MagicMock()
    d.agent.price_fn = lambda t: 100.0
    d.agent.make_broker = MagicMock()

    now = __import__("datetime").datetime(2026, 6, 10, 15, 0, tzinfo=__import__("datetime").timezone.utc)
    scan = ScanResult(
        regime=RegimeResult("neutral", {}, 0.85),
        verdicts=[Verdict("AAA", 80, {}, 3)],
        eligible=[], targets=[], equity=10_000, universe_size=1, scored_size=1,
    )
    run = RunResult(scan=scan, account=Account(equity=10_000, cash=0, buying_power=0, positions=[]),
                    orders=[], fills=[], executed=True, mode="paper")
    d._apply_rebalance_result(run, run.account, now=now)
    assert d.state.last_rebalance == now.isoformat()


def test_daemon_preserves_pending_scan_when_live_refresh_fails(tmp_path, monkeypatch):
    import rh_agent.daemon as daemon

    monkeypatch.setattr(daemon, "STATE", tmp_path / "daemon_state.json")
    cfg = load_config()
    scan = ScanResult(
        regime=RegimeResult("neutral", {}, 0.85),
        verdicts=[], eligible=[], targets=[], equity=10_000,
        universe_size=1, scored_size=1,
    )

    class _LiveBroker:
        supports_live = True

        def __init__(self):
            self.calls = 0

        def get_account(self):
            self.calls += 1
            if self.calls == 1:
                return Account(
                    equity=10_000, cash=10_000, buying_power=10_000, positions=[],
                    source="robinhood", portfolio_confirmed=True,
                    positions_confirmed=True, buying_power_confirmed=True,
                )
            raise RuntimeError("refresh failed")

    broker = _LiveBroker()

    class _Agent:
        providers = {}

        def clear_price_cache(self):
            pass

        def make_broker(self):
            return broker

        def reconcile_and_execute(self, *args, **kwargs):
            raise AssertionError("must not reconcile with a failed live refresh")

    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.cfg = cfg
    d.agent = _Agent()
    d.state = daemon.DaemonState.load()
    d._scan_future = None
    d._scan_started_at = None
    d._pending_scan = scan
    d._pending_scan_at = datetime.now(timezone.utc)
    d._pending_scan_max_age = 300.0
    d.journal = MagicMock()
    d._ensure_stops_for_held = lambda *a, **k: None
    d._manage_risk = lambda *a, **k: set()
    d._due_for_rebalance = lambda now: False

    d.tick(execute=True)

    assert d._pending_scan is scan


def test_loop_refusal_explains_unarmed_live_env(monkeypatch, capsys):
    from rh_agent.cli import cmd_loop

    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    code = cmd_loop(Namespace(config=None, execute=True, snapshot=None, once=False, max_cycles=None))
    out = capsys.readouterr().out

    assert code == 2
    assert "EXECUTION_MODE=live but live trading is not armed" in out
    assert "LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY" in out
