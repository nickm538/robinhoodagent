"""Tests for multi-model review hardening fixes."""
from __future__ import annotations

import pytest

from rh_agent.agent import TradingAgent
from rh_agent.broker.errors import LiveBrokerUnavailable
from rh_agent.broker.orders import order_succeeded, stable_ref_id
from rh_agent.broker.robinhood_mcp import parse_account
from rh_agent.config import load_config
from rh_agent.execution import build_orders
from rh_agent.models import Account, Order, Position, TargetPosition, Verdict
from rh_agent.risk import risk_capped_weight, trailing_stop
from rh_agent.scoring import Scorer


def test_parse_account_never_defaults_bp_to_equity():
    prof = {"data": {"total_value": "1000.00", "cash": "200.00",
                     "buying_power": {"buying_power": "150.00"}}}
    acct = parse_account(prof, {"data": {"positions": []}}, "A1",
                         portfolio_ok=True, positions_ok=True)
    assert acct.buying_power == 150.0
    assert acct.buying_power_confirmed is True
    assert acct.reliable is True

    # missing BP -> cash, NOT equity
    prof2 = {"data": {"total_value": "1000.00", "cash": "200.00"}}
    acct2 = parse_account(prof2, {"data": {"positions": []}}, "A1",
                          portfolio_ok=True, positions_ok=True)
    assert acct2.buying_power == 200.0
    assert acct2.buying_power_confirmed is False
    assert acct2.reliable is False


def test_parse_account_unreliable_when_positions_fetch_failed():
    prof = {"data": {"total_value": "500.00", "cash": "500.00",
                     "buying_power": {"buying_power": "500.00"}}}
    acct = parse_account(prof, None, "A1", portfolio_ok=True, positions_ok=False)
    assert acct.positions == []
    assert acct.positions_confirmed is False
    assert acct.reliable is False


def test_order_succeeded_semantics():
    assert order_succeeded({"status": "submitted"}, executing=True)
    assert order_succeeded({"status": "filled"}, executing=True)
    assert not order_succeeded({"status": "error", "error": "x"}, executing=True)
    assert not order_succeeded({"status": "preview"}, executing=True)
    assert order_succeeded({"status": "preview"}, executing=False)


def test_stable_ref_id_is_deterministic():
    o = Order("AAPL", "buy", None, "market", notional=100.0, reason="enter")
    a = stable_ref_id(o, "ACCT", day_key="2026-06-07")
    b = stable_ref_id(o, "ACCT", day_key="2026-06-07")
    c = stable_ref_id(o, "ACCT", day_key="2026-06-08")
    assert a == b
    assert a != c


def test_risk_capped_weight():
    # 10% weight, 10% stop distance, 1% risk budget -> cap at 10%
    assert risk_capped_weight(100, 90, 0.10, 0.01) == pytest.approx(0.10)
    # 20% weight with same stop -> cap at 10%
    assert risk_capped_weight(100, 90, 0.20, 0.01) == pytest.approx(0.10)


def test_trailing_stop_uses_atr():
    assert trailing_stop(100, atr=2.0, mult=2.5, hard_pct=0.18) == pytest.approx(95.0)
    assert trailing_stop(100, atr=None, mult=2.5, hard_pct=0.18) == pytest.approx(82.0)


def test_eligible_blocks_ai_caution():
    cfg = load_config()
    cfg.raw["portfolio"]["risk_controls"]["block_ai_caution"] = True
    scorer = Scorer(cfg)
    v = Verdict("X", 80, {}, 3, flags=["ai_caution"])
    assert scorer.eligible([v]) == []


def test_build_orders_respects_allow_buys_false():
    cfg = load_config()
    acct = Account(equity=10_000, cash=10_000, buying_power=10_000, positions=[])
    targets = [TargetPosition("AAA", 0.12, 80, sector="Tech")]
    orders = build_orders(acct, targets, cfg, lambda t: 50.0, allow_buys=False)
    assert orders == []


def test_build_orders_excludes_tickers():
    cfg = load_config()
    acct = Account(equity=10_000, cash=10_000, buying_power=10_000,
                   positions=[Position("AAA", 10, 50, current_price=50, market_value=500)])
    targets = [TargetPosition("AAA", 0.12, 80, sector="Tech")]
    orders = build_orders(acct, targets, cfg, lambda t: 50.0, exclude_tickers={"AAA"})
    assert orders == []


def test_make_broker_raises_when_live_armed_without_auth(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_UNDERSTAND_REAL_MONEY")
    monkeypatch.delenv("ROBINHOOD_MCP_TOKEN", raising=False)
    agent = TradingAgent(cfg)

    class NoTokens:
        def has_tokens(self):
            return False

    monkeypatch.setattr("rh_agent.broker.oauth.FileTokenStorage", NoTokens)
    with pytest.raises(LiveBrokerUnavailable):
        agent.make_broker()


def test_daemon_state_invalid_last_rebalance_reset(tmp_path, monkeypatch):
    import rh_agent.daemon as daemon
    p = tmp_path / "daemon_state.json"
    monkeypatch.setattr(daemon, "STATE", p)
    p.write_text('{"last_rebalance":"not-a-date","stops":{},"take_profits":{}}')
    st = daemon.DaemonState.load()
    assert st.last_rebalance == ""
