"""Tests for the self-improving trade journal + memory."""
from __future__ import annotations

import json

import pytest

from rh_agent.agent import RunResult, ScanResult
from rh_agent.config import load_config
from rh_agent.journal import Journal
from rh_agent.models import Account, Order, Position, Verdict
from rh_agent.regime import RegimeResult


def _journal(tmp_path):
    cfg = load_config()
    j = Journal(cfg)
    j.path = tmp_path / "journal.jsonl"
    j.memory_path = tmp_path / "memory.md"
    return j


def test_realized_pnl_average_cost_matching(tmp_path):
    j = _journal(tmp_path)
    # AAA: buy 1 @100, sell 1 @120 -> +20 (win)
    j.record_order(ticker="AAA", side="buy", qty=1, price=100, reason="enter", status="submitted")
    j.record_order(ticker="AAA", side="sell", qty=1, price=120, reason="take-profit", status="submitted")
    # BBB: buy 2 @50, sell 2 @40 -> -20 (loss)
    j.record_order(ticker="BBB", side="buy", qty=2, price=50, status="submitted")
    j.record_order(ticker="BBB", side="sell", qty=2, price=40, status="submitted")

    s = j.stats()
    assert s["closed_trades"] == 2
    assert s["net_realized"] == pytest.approx(0.0)        # +20 and -20
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["avg_win"] == pytest.approx(20.0)
    assert s["avg_loss"] == pytest.approx(-20.0)

    summary = j.performance_summary()
    assert "2 closed trades" in summary
    assert "hit-rate 50%" in summary


def test_partial_sell_realizes_proportional_pnl(tmp_path):
    j = _journal(tmp_path)
    j.record_order(ticker="AAA", side="buy", qty=4, price=100, status="submitted")
    j.record_order(ticker="AAA", side="sell", qty=2, price=110, status="submitted")  # +10*2
    s = j.stats()
    assert s["closed_trades"] == 1
    assert s["net_realized"] == pytest.approx(20.0)


def test_disabled_journal_is_noop(tmp_path):
    cfg = load_config()
    cfg.raw["journal"]["enabled"] = False
    j = Journal(cfg)
    j.path = tmp_path / "j.jsonl"
    j.record_order(ticker="X", side="buy", qty=1, price=10, status="submitted")
    assert not j.path.exists()
    assert j.performance_summary() == ""


def test_record_rebalance_captures_context(tmp_path):
    j = _journal(tmp_path)
    scan = ScanResult(
        regime=RegimeResult("risk_on_trend", {}, 1.0),
        verdicts=[Verdict("AAA", 80.0, {"momentum_trader": 70.0, "ai_analyst": 65.0}, 4)],
        eligible=[], targets=[], equity=1000.0, universe_size=1, scored_size=1)
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("AAA", 1.0, 100.0, current_price=100.0)])
    run = RunResult(scan=scan, account=acct, post_account=acct,
                    orders=[Order("AAA", "buy", notional=50.0, reason="enter")],
                    fills=[{"status": "submitted"}], executed=True, mode="live")
    j.record_rebalance(run, acct, price_fn=lambda t: 100.0)

    recs = [json.loads(line) for line in j.path.read_text().splitlines() if line.strip()]
    assert len(recs) == 1
    r = recs[0]
    assert r["ticker"] == "AAA" and r["side"] == "buy"
    assert r["price"] == 100.0 and r["notional"] == 50.0
    assert r["regime"] == "risk_on_trend"
    assert r["composite"] == 80.0 and r["pillars"] == 4
    assert r["ai_score"] == 65.0
    # memory.md gets written with the holdings table
    assert j.memory_path.exists()
    assert "Current holdings" in j.memory_path.read_text()


def test_record_rebalance_skips_rejected_and_dryruns(tmp_path):
    j = _journal(tmp_path)
    scan = ScanResult(regime=RegimeResult("neutral", {}, 0.0), verdicts=[], eligible=[],
                      targets=[], equity=1000.0, universe_size=0, scored_size=0)
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0, positions=[])
    # not executed -> nothing recorded
    run = RunResult(scan=scan, account=acct, orders=[Order("AAA", "buy", notional=5.0)],
                    fills=[{"status": "submitted"}], executed=False, mode="paper")
    j.record_rebalance(run, acct, price_fn=lambda t: 1.0)
    assert not j.path.exists()
    # executed but rejected -> skipped
    run2 = RunResult(scan=scan, account=acct, orders=[Order("AAA", "buy", notional=5.0)],
                     fills=[{"status": "rejected"}], executed=True, mode="live")
    j.record_rebalance(run2, acct, price_fn=lambda t: 1.0)
    recs = [line for line in (j.path.read_text().splitlines() if j.path.exists() else []) if line.strip()]
    assert recs == []
