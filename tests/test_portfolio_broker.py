"""Tests for portfolio caps and the paper broker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rh_agent.broker.paper import PaperBroker
from rh_agent.config import load_config
from rh_agent.models import Order, TargetPosition, TickerData, Verdict
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


def test_robinhood_parse_account_portfolio_shape():
    # real get_portfolio shape: nested data with string values; buying_power nested
    from rh_agent.broker.robinhood_mcp import parse_account
    prof = {"data": {"total_value": "412.50", "equity_value": "0.00", "cash": "412.50",
                     "buying_power": {"buying_power": "400.00",
                                      "unleveraged_buying_power": "400.00"}}, "guide": "..."}
    acct = parse_account(prof, {"data": {"positions": []}}, "AGENT123")
    assert acct.equity == 412.50          # total_value, NOT equity_value (0)
    assert acct.cash == 412.50
    assert acct.buying_power == 400.00
    assert acct.account_number == "AGENT123" and not acct.positions
    # a genuinely empty account: total_value "0.00" must stay 0.0, not be masked
    flat = parse_account({"data": {"total_value": "0.00", "cash": "0.00",
                                   "buying_power": {"buying_power": "0.00"}}},
                         {"data": {"positions": []}}, "AGENT123")
    assert flat.equity == 0.0 and flat.buying_power == 0.0
