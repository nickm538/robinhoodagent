"""Tests for portfolio caps and the paper broker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rh_agent.broker.paper import PaperBroker
from rh_agent.config import load_config
from rh_agent.models import Account, Order, Position, TargetPosition, TickerData, Verdict
from rh_agent.portfolio import PortfolioBuilder
from rh_agent.regime import RegimeResult


def _td(ticker, sector, vol=0.3):
    df = pd.DataFrame({"close": np.linspace(100, 120, 60),
                       "high": np.linspace(101, 121, 60),
                       "low": np.linspace(99, 119, 60),
                       "adj_close": np.linspace(100, 120, 60),
                       "volume": [1e6] * 60},
                      index=pd.date_range("2024-01-01", periods=60, freq="B"))
    td = TickerData(ticker, company={"sector": sector, "market_cap": 5e10},
                    prices=df, technicals={"volatility": vol, "atr": 2.0, "price": 120.0})
    return td


def test_position_and_sector_caps():
    cfg = load_config()
    cfg.raw["portfolio"]["max_position_weight"] = 0.10
    cfg.raw["portfolio"]["max_sector_weight"] = 0.35
    cfg.raw["portfolio"]["target_positions"] = 12
    cfg.raw["portfolio"]["autoscale"] = {"enabled": False}   # exercise the static caps
    builder = PortfolioBuilder(cfg)
    # 8 tech names + 2 others -> tech must be capped at 35%
    verdicts, td_map = [], {}
    for i in range(8):
        t = f"T{i}"
        verdicts.append(Verdict(t, 80 - i, {}, 5))
        td_map[t] = _td(t, "Information Technology")
    for t in ["FIN", "HLTH"]:
        verdicts.append(Verdict(t, 70, {}, 5))
        td_map[t] = _td(t, "Financials" if t == "FIN" else "Health Care")
    regime = RegimeResult("risk_on_trend", {}, 1.0)
    targets = builder.build(verdicts, td_map, regime, 100_000)

    assert all(t.weight <= 0.10 + 1e-6 for t in targets)        # per-name cap
    tech = sum(t.weight for t in targets if t.sector == "Information Technology")
    assert tech <= 0.35 + 1e-6                                   # sector cap
    assert sum(t.weight for t in targets) <= 1.0 + 1e-6


def test_paper_broker_buy_sell(tmp_path):
    prices = {"AAA": 100.0}
    broker = PaperBroker(lambda t: prices.get(t), starting_cash=10_000,
                         slippage_bps=0, state_path=tmp_path / "acct.json")
    broker.place_order(Order("AAA", "buy", 10), dry_run=False)
    acct = broker.get_account()
    assert acct.cash == 9_000
    assert acct.position_map()["AAA"].quantity == 10
    broker.place_order(Order("AAA", "sell", 10), dry_run=False)
    acct = broker.get_account()
    assert acct.cash == 10_000
    assert "AAA" not in acct.position_map()


def test_paper_broker_dry_run_does_not_fill(tmp_path):
    broker = PaperBroker(lambda t: 50.0, starting_cash=1_000,
                         state_path=tmp_path / "a.json")
    res = broker.place_order(Order("X", "buy", 1), dry_run=True)
    assert res["status"] == "preview"
    assert broker.get_account().cash == 1_000


def test_paper_broker_rejects_bad_side_and_unheld_sell(tmp_path):
    broker = PaperBroker(lambda t: 10.0, starting_cash=1_000, state_path=tmp_path / "a.json")
    assert broker.place_order(Order("X", "hold", 1), dry_run=False)["status"] == "rejected"
    assert broker.place_order(Order("X", "sell", 1), dry_run=False)["status"] == "rejected"


def test_robinhood_tool_mapping_official_names():
    from rh_agent.broker.robinhood_mcp import discover_tool_map, pick_account_number
    real = ["cancel_equity_order", "get_accounts", "get_equity_orders", "get_equity_positions",
            "get_equity_quotes", "get_equity_tradability", "get_portfolio",
            "place_equity_order", "review_equity_order", "search"]
    m = discover_tool_map(real)
    assert m["positions"] == "get_equity_positions"
    assert m["place_order"] == "place_equity_order"
    assert m["orders"] == "get_equity_orders"
    assert m["quote"] == "get_equity_quotes"
    assert m["accounts"] == "get_accounts"
    assert m["buying_power"] == "get_portfolio"
    assert m["cancel"] == "cancel_equity_order"
    assert pick_account_number(
        {"accounts": [{"account_number": "A1", "agentic_allowed": False},
                      {"account_number": "A2", "agentic_allowed": True}]}) == "A2"


def test_robinhood_account_pick_and_order_args():
    from rh_agent.broker.robinhood_mcp import order_args, pick_account_number
    # real get_accounts shape: nested data.accounts; pick agentic_allowed=true
    accts = {"data": {"accounts": [
        {"account_number": "NONAGENTIC", "agentic_allowed": False},
        {"account_number": "AGENT123", "agentic_allowed": True, "deactivated": False}]}}
    assert pick_account_number(accts) == "AGENT123"
    # single-account object (some servers) — not a list/envelope
    assert pick_account_number({"account_number": "SOLO", "agentic_allowed": True}) == "SOLO"

    allowed = {"account_number", "symbol", "side", "type", "quantity", "dollar_amount",
               "limit_price", "stop_price", "time_in_force", "market_hours", "ref_id"}
    # market $ buy -> dollar_amount STRING; live -> ref_id; only allowed keys
    a = order_args(Order("AAPL", "buy", None, "market", notional=100.0),
                   dry_run=False, account_number="AGENT123")
    assert a["dollar_amount"] == "100.00" and a["type"] == "market"
    assert a["account_number"] == "AGENT123" and "ref_id" in a
    assert set(a) <= allowed, set(a) - allowed
    assert not ({"ticker", "shares", "notional", "action", "order_type"} & set(a))
    # sell -> quantity STRING; dry-run -> no ref_id
    s = order_args(Order("MSFT", "sell", 1.5, "market"), dry_run=True, account_number="X")
    assert s["quantity"] == "1.5" and "ref_id" not in s


def test_robinhood_sdk_place_order_returns_error_dict():
    # a failing order must return an error dict, not crash the rebalance loop
    from rh_agent.broker.robinhood_sdk import RobinhoodSDKBroker
    b = RobinhoodSDKBroker.__new__(RobinhoodSDKBroker)   # skip __init__ (no SDK/token needed)
    b.url, b.account_number, b._map = "x", "A", {"place_order": "place_equity_order"}
    b._run = lambda fn: (_ for _ in ()).throw(RuntimeError("upstream 400"))
    res = b.place_order(Order("AAPL", "buy", None, "market", notional=70.0), dry_run=False)
    assert res["status"] == "error" and "upstream 400" in res["error"]
    assert res["ticker"] == "AAPL"


def test_robinhood_parse_account_portfolio_shape():
    # real get_portfolio shape: nested data with string values; buying_power nested
    from rh_agent.broker.robinhood_mcp import parse_account
    prof = {"data": {"total_value": "412.50", "equity_value": "0.00", "cash": "412.50",
                     "buying_power": {"buying_power": "400.00",
                                      "unleveraged_buying_power": "400.00"}}, "guide": "..."}
    acct = parse_account(prof, {"data": {"positions": []}}, "AGENT123",
                         portfolio_ok=True, positions_ok=True)
    assert acct.equity == 412.50          # total_value, NOT equity_value (0)
    assert acct.cash == 412.50
    assert acct.buying_power == 400.00
    assert acct.account_number == "AGENT123" and not acct.positions
    # a genuinely empty account: total_value "0.00" must stay 0.0, not be masked
    flat = parse_account({"data": {"total_value": "0.00", "cash": "0.00",
                                   "buying_power": {"buying_power": "0.00"}}},
                         {"data": {"positions": []}}, "AGENT123",
                         portfolio_ok=True, positions_ok=True)
    assert flat.equity == 0.0 and flat.buying_power == 0.0
    assert flat.reliable is False


def test_build_orders_respects_buying_power():
    # A buy must never exceed available buying power, even when the target weight
    # (sized against equity) implies a far larger order — else the broker 400s
    # with "Not enough buying power".
    from rh_agent.execution import build_orders
    from rh_agent.models import Account, TargetPosition
    cfg = load_config()
    acct = Account(equity=10_000, cash=100.0, buying_power=100.0, positions=[])
    targets = [TargetPosition("AAA", 0.12, 80, sector="Tech")]   # weight implies ~$1200
    orders = build_orders(acct, targets, cfg, lambda t: 50.0)
    buys = [o for o in orders if o.side == "buy"]
    assert len(buys) == 1
    assert buys[0].notional <= 100.0                      # capped to buying power...
    assert buys[0].notional == round(0.97 * 100.0, 2)     # ...minus the slippage cushion
    assert buys[0].quantity == round(buys[0].notional / 50.0, 4)


def test_build_orders_no_cap_when_cash_ample():
    from rh_agent.execution import build_orders
    from rh_agent.models import Account, TargetPosition
    cfg = load_config()
    acct = Account(equity=10_000, cash=10_000.0, buying_power=10_000.0, positions=[])
    targets = [TargetPosition("AAA", 0.12, 80, sector="Tech")]
    orders = build_orders(acct, targets, cfg, lambda t: 50.0)
    buys = [o for o in orders if o.side == "buy"]
    assert len(buys) == 1 and buys[0].notional == 1200.0   # full target, no cap


def test_autoscale_tiers_by_equity():
    # the book widens + caps tighten as the account grows across equity tiers
    cfg = load_config()
    cfg.raw["portfolio"]["autoscale"] = {
        "enabled": True,
        "tiers": [[0, 3, 0.40, 0.70], [400, 5, 0.30, 0.60],
                  [2000, 8, 0.18, 0.45], [25000, 12, 0.12, 0.35]],
    }
    b = PortfolioBuilder(cfg)
    assert b._autoscale_params(150) == (3, 0.40, 0.70)
    assert b._autoscale_params(513) == (5, 0.30, 0.60)
    assert b._autoscale_params(5000) == (8, 0.18, 0.45)
    assert b._autoscale_params(30000) == (12, 0.12, 0.35)
    # disabled -> None (builder falls back to the static portfolio caps)
    cfg.raw["portfolio"]["autoscale"]["enabled"] = False
    assert PortfolioBuilder(cfg)._autoscale_params(513) is None


def test_per_trade_risk_cap_limits_position_weight():
    cfg = load_config()
    cfg.raw["portfolio"]["autoscale"] = {"enabled": False}
    cfg.raw["portfolio"]["target_positions"] = 1
    cfg.raw["portfolio"]["max_position_weight"] = 0.50
    cfg.raw["portfolio"]["risk_controls"]["per_trade_risk_pct"] = 0.01
    b = PortfolioBuilder(cfg)
    td = _td("AAA", "Information Technology")
    td.technicals["atr"] = 20.0
    regime = RegimeResult("risk_on_trend", {}, 1.0)
    targets = b.build([Verdict("AAA", 90, {}, 5)], {"AAA": td}, regime, 100_000)
    assert len(targets) == 1
    assert targets[0].weight <= 0.056


def test_account_is_agentic():
    from rh_agent.broker.robinhood_mcp import account_is_agentic
    accts = {"data": {"accounts": [
        {"account_number": "NONAGENTIC", "agentic_allowed": False},
        {"account_number": "AGENT123", "agentic_allowed": True, "deactivated": False}]}}
    assert account_is_agentic(accts, "AGENT123") is True
    assert account_is_agentic(accts, "NONAGENTIC") is False


class _DummyAgent:
    def __init__(self, prices):
        self._prices = prices

    def price_fn(self, ticker, for_risk=False):
        return self._prices.get(ticker)


class _FakeBroker:
    def __init__(self, status):
        self.status = status

    def place_order(self, order, dry_run=True):
        return {"status": self.status}


def _daemon_for_risk_test(price: float):
    import rh_agent.daemon as daemon
    cfg = load_config()
    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.cfg = cfg
    d.agent = _DummyAgent({"AAA": price})
    d.state = daemon.DaemonState(stops={"AAA": 95.0}, take_profits={},
                                 high_water={"AAA": price}, pending_risk={})
    return d


def test_daemon_manage_risk_keeps_stop_in_dry_run():
    d = _daemon_for_risk_test(price=90.0)
    acct = Account(equity=1_000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("AAA", 1.0, 100.0, current_price=90.0)])
    hits = d._manage_risk(_FakeBroker("preview"), acct, execute=False)
    assert hits == set()
    assert d.state.stops["AAA"] == 95.0


def test_daemon_manage_risk_keeps_stop_on_order_error():
    d = _daemon_for_risk_test(price=90.0)
    acct = Account(equity=1_000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("AAA", 1.0, 100.0, current_price=90.0)])
    hits = d._manage_risk(_FakeBroker("error"), acct, execute=True)
    assert hits == set()
    assert d.state.stops["AAA"] == 95.0


def test_daemon_manage_risk_clears_stop_on_confirmed_order():
    d = _daemon_for_risk_test(price=90.0)
    acct = Account(equity=1_000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("AAA", 1.0, 100.0, current_price=90.0)])
    hits = d._manage_risk(_FakeBroker("submitted"), acct, execute=True)
    assert hits == {"AAA"}
    assert "AAA" not in d.state.stops


def test_daemon_state_load_tolerates_corruption(tmp_path, monkeypatch):
    # a corrupt/garbage state file must never crash the 24/7 loop at boot
    import rh_agent.daemon as daemon
    p = tmp_path / "daemon_state.json"
    monkeypatch.setattr(daemon, "STATE", p)
    # non-dict stops/take_profits + an unknown key
    p.write_text('{"last_rebalance":"x","stops":"GARBAGE","take_profits":42,"bogus":1}')
    st = daemon.DaemonState.load()
    assert st.stops == {} and st.take_profits == {}      # coerced to dicts
    assert st.last_rebalance == ""                        # invalid iso reset
    # totally invalid json -> fresh state, no crash
    p.write_text("{not valid json")
    st2 = daemon.DaemonState.load()
    assert st2.stops == {} and st2.take_profits == {}
    # invalid last_rebalance is reset (not preserved as "x")
    p.write_text('{"last_rebalance":"x","stops":{},"take_profits":{}}')
    st3 = daemon.DaemonState.load()
    assert st3.last_rebalance == ""
