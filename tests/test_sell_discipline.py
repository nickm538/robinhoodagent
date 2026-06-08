from __future__ import annotations

from rh_agent.agent import ScanResult, TradingAgent
from rh_agent.config import load_config
from rh_agent.execution import build_orders
from rh_agent.models import Account, Position, TargetPosition, TickerData, Verdict
from rh_agent.regime import RegimeResult


def _agent():
    cfg = load_config()
    a = TradingAgent.__new__(TradingAgent)
    a.cfg = cfg
    a.price_fn = lambda ticker, for_risk=False: {"HEALTHY": 100.0, "WEAK": 100.0, "MISS": 100.0}[ticker]
    return a


def _scan(verdicts: list[Verdict], targets: list[TargetPosition] | None = None) -> ScanResult:
    td_map = {
        v.ticker: TickerData(v.ticker, company={"sector": "Technology"},
                             technicals={"atr": 2.0})
        for v in verdicts
    }
    return ScanResult(
        regime=RegimeResult("neutral", {}, 0.85),
        verdicts=verdicts,
        eligible=[],
        targets=targets or [],
        equity=10_000.0,
        universe_size=len(verdicts),
        scored_size=len(verdicts),
        td_map=td_map,
    )


def _account(ticker: str) -> Account:
    return Account(
        equity=10_000.0,
        cash=9_000.0,
        buying_power=9_000.0,
        positions=[Position(ticker, 10.0, 100.0, current_price=100.0)],
    )


def test_hold_discipline_keeps_held_name_above_exit_bar():
    agent = _agent()
    acct = _account("HEALTHY")
    scan = _scan([Verdict("HEALTHY", 52.0, {}, pillars_passing=1)])

    scan = agent._apply_hold_discipline(scan, acct, acct.equity)
    orders = build_orders(acct, scan.targets, agent.cfg, agent.price_fn)

    assert [t.ticker for t in scan.targets] == ["HEALTHY"]
    assert orders == []


def test_hold_discipline_allows_exit_on_real_deterioration():
    agent = _agent()
    acct = _account("WEAK")
    scan = _scan([Verdict("WEAK", 30.0, {}, pillars_passing=1)])

    scan = agent._apply_hold_discipline(scan, acct, acct.equity)
    orders = build_orders(acct, scan.targets, agent.cfg, agent.price_fn)

    assert scan.targets == []
    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].ticker == "WEAK"


def test_hold_discipline_does_not_blind_sell_when_scan_data_missing():
    agent = _agent()
    acct = _account("MISS")
    scan = _scan([])

    scan = agent._apply_hold_discipline(scan, acct, acct.equity)
    orders = build_orders(acct, scan.targets, agent.cfg, agent.price_fn)

    assert [t.ticker for t in scan.targets] == ["MISS"]
    assert orders == []
