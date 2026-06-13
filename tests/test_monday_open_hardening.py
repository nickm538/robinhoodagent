"""Monday-open hardening: stale scans must die at harvest, hunts must not be
born straddling the close, and a stock split must never masquerade as a stop.

All three were observed live on 2026-06-12: a hunt kicked at 19:55 UTC was
still computing at the bell (would have fired at Monday's open on Friday
conviction), and KLAC's 10:1 split false-triggered its pre-split stop."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rh_agent.activity import ActivityLog
from rh_agent.config import load_config
from rh_agent.market_calendar import minutes_to_close
from rh_agent.models import Account, Position


def _utcnow():
    return datetime.now(timezone.utc)


def _mk_daemon(tmp_path, monkeypatch, cfg=None):
    import rh_agent.daemon as daemon
    monkeypatch.setattr(daemon, "STATE", tmp_path / "daemon_state.json")
    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.cfg = cfg or load_config()
    d.state = daemon.DaemonState.load()
    d.journal = MagicMock()
    d.activity = ActivityLog(path=tmp_path / "activity.jsonl")
    return d


class _Broker:
    supports_live = False

    def __init__(self):
        self.orders = []

    def get_account(self):
        return Account(equity=1000.0, cash=1000.0, buying_power=1000.0, positions=[])

    def place_order(self, order, dry_run=True):
        self.orders.append(order)
        return {"status": "submitted", "ticker": order.ticker}


# ------------------------- stale scan dies at harvest -------------------------

def test_harvest_discards_scan_finished_over_the_weekend(tmp_path, monkeypatch):
    d = _mk_daemon(tmp_path, monkeypatch)

    class _Agent:
        providers = {}

        def clear_price_cache(self):
            pass

        def make_broker(self):
            return _Broker()

        def default_equity(self):
            return 1000.0

        def reconcile_and_execute(self, *a, **k):
            raise AssertionError("Friday-evening conviction must NOT trade Monday's open")

    d.agent = _Agent()
    d._scan_future = SimpleNamespace(done=lambda: True, result=lambda: "stale-scan")
    d._scan_started_at = _utcnow() - timedelta(hours=65)     # kicked Friday 19:55
    d._pending_scan = None
    d._pending_scan_at = None
    d._pending_scan_expiry = 3600.0
    d._ensure_stops_for_held = lambda *a, **k: None
    d._manage_risk = lambda *a, **k: set()
    d._due_for_rebalance = lambda now: False                 # isolate the harvest path

    d.tick(execute=True)

    assert d._pending_scan is None
    assert d._scan_future is None
    events = d.activity.tail()
    expired = [e for e in events if e["event"] == "scan_expired"]
    assert expired and expired[0].get("at") == "harvest"
    assert d.state.last_scan_seconds == 0.0                  # stale run never pollutes cadence stats


def test_stale_discard_rekicks_fresh_hunt_same_tick(tmp_path, monkeypatch):
    """After discarding a stale harvest, the kick block (an elif of the
    PENDING-execution `if`, not of the harvest block) still runs in the SAME
    tick — Monday 09:30 discards Friday's scan and immediately hunts fresh."""
    import rh_agent.daemon as daemon

    d = _mk_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr(daemon, "minutes_to_close", lambda now=None: None)

    class _Agent:
        providers = {}

        def clear_price_cache(self):
            pass

        def make_broker(self):
            return _Broker()

        def default_equity(self):
            return 1000.0

        def reconcile_and_execute(self, *a, **k):
            raise AssertionError("stale Friday scan must not execute")

    d.agent = _Agent()
    d._scan_future = SimpleNamespace(done=lambda: True, result=lambda: "stale-scan")
    d._scan_started_at = _utcnow() - timedelta(hours=65)
    d._pending_scan = None
    d._pending_scan_at = None
    d._pending_scan_expiry = 3600.0
    d._scan_pool = MagicMock()
    d._ensure_stops_for_held = lambda *a, **k: None
    d._manage_risk = lambda *a, **k: set()
    d._due_for_rebalance = lambda now: True          # Monday open: long overdue

    d.tick(execute=True)

    d._scan_pool.submit.assert_called_once()         # fresh hunt, same tick
    events = [e["event"] for e in d.activity.tail()]
    assert "scan_expired" in events and "scan_started" in events


def test_harvest_keeps_fresh_scan(tmp_path, monkeypatch):
    d = _mk_daemon(tmp_path, monkeypatch)
    reconciled = {"n": 0}
    from rh_agent.agent import RunResult, ScanResult
    from rh_agent.regime import RegimeResult

    scan = ScanResult(regime=RegimeResult("neutral", {}, 0.85), verdicts=[], eligible=[],
                      targets=[], equity=1000.0, universe_size=3, scored_size=3)
    broker = _Broker()
    acct = broker.get_account()

    class _Agent:
        providers = {}

        def clear_price_cache(self):
            pass

        def make_broker(self):
            return broker

        def default_equity(self):
            return 1000.0

        def price_fn(self, t, for_risk=False):
            return 10.0

        def reconcile_and_execute(self, s, **k):
            reconciled["n"] += 1
            return RunResult(scan=s, account=acct, orders=[], fills=[], executed=True,
                             mode="paper", post_account=acct)

    d.agent = _Agent()
    d._scan_future = SimpleNamespace(done=lambda: True, result=lambda: scan)
    d._scan_started_at = _utcnow() - timedelta(minutes=8)
    d._pending_scan = None
    d._pending_scan_at = None
    d._pending_scan_expiry = 3600.0
    d._ensure_stops_for_held = lambda *a, **k: None
    d._manage_risk = lambda *a, **k: set()
    d._due_for_rebalance = lambda now: False

    d.tick(execute=True)

    assert reconciled["n"] == 1                              # fresh scan executes
    assert d.state.last_scan_seconds == pytest.approx(480, abs=5)
    assert any(e["event"] == "scan_done" for e in d.activity.tail())


# ------------------------- pre-close hunt blackout ---------------------------

def test_no_new_hunt_inside_close_blackout(tmp_path, monkeypatch):
    import rh_agent.daemon as daemon

    d = _mk_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr(daemon, "minutes_to_close", lambda now=None: 7.0)   # 15:53 ET

    class _Agent:
        providers = {}

        def clear_price_cache(self):
            pass

        def make_broker(self):
            return _Broker()

        def default_equity(self):
            return 1000.0

    d.agent = _Agent()
    d._scan_future = None
    d._scan_started_at = None
    d._pending_scan = None
    d._pending_scan_at = None
    d._scan_pool = MagicMock()
    d._ensure_stops_for_held = lambda *a, **k: None
    d._manage_risk = lambda *a, **k: set()
    d._due_for_rebalance = lambda now: True                  # due, but too late in the day

    d.tick(execute=True)

    assert d._scan_future is None
    d._scan_pool.submit.assert_not_called()

    # mid-session (or closed-market test harness): hunts proceed normally
    monkeypatch.setattr(daemon, "minutes_to_close", lambda now=None: 180.0)
    d.tick(execute=True)
    d._scan_pool.submit.assert_called_once()


def test_minutes_to_close_math():
    # Mon 2026-06-08 14:00 UTC = 10:00 ET -> 360 minutes to the 16:00 bell
    assert minutes_to_close(datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc)) \
        == pytest.approx(360.0)
    # Early close (day after Thanksgiving): 17:30 UTC = 12:30 ET -> 30m to 13:00
    assert minutes_to_close(datetime(2026, 11, 27, 17, 30, tzinfo=timezone.utc)) \
        == pytest.approx(30.0)
    # closed -> None
    assert minutes_to_close(datetime(2026, 6, 13, 15, 0, tzinfo=timezone.utc)) is None


# ------------------------- split guard ---------------------------------------

def test_split_guard_reanchors_instead_of_selling(tmp_path, monkeypatch):
    """The live KLAC case: 10:1 split -> px 239 vs stored stop 2071, broker avg
    cost already split-adjusted to ~232. Must re-anchor, never sell."""
    d = _mk_daemon(tmp_path, monkeypatch)
    d.agent = SimpleNamespace(
        price_fn=lambda t, for_risk=False: 239.36,
        md=SimpleNamespace(build=lambda tk, deep=False: SimpleNamespace(
            technicals={"atr": 8.0})),
    )
    d.state.stops = {"KLAC": 2071.02}
    d.state.take_profits = {"KLAC": 2600.0}
    d.state.high_water = {"KLAC": 2350.0}
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("KLAC", 0.3924, 232.0, current_price=239.36)])
    broker = _Broker()

    triggered = d._manage_risk(broker, acct, execute=True)

    assert triggered == set()
    assert broker.orders == []                              # the false stop never fires
    assert d.state.stops["KLAC"] == pytest.approx(219.36)   # 239.36 - 2.5*8 re-anchor
    assert d.state.take_profits["KLAC"] == pytest.approx(287.36)
    assert d.state.high_water["KLAC"] == pytest.approx(239.36)
    assert any(e.get("kind") == "split_guard" for e in d.activity.tail())


def test_real_crash_still_fires_stop(tmp_path, monkeypatch):
    """A genuine -45% gap keeps avg cost at the same scale as the stop — the
    guard must NOT block the protective sell."""
    d = _mk_daemon(tmp_path, monkeypatch)
    d.agent = SimpleNamespace(
        price_fn=lambda t, for_risk=False: 50.0,
        md=SimpleNamespace(build=lambda tk, deep=False: SimpleNamespace(technicals={})),
    )
    d.state.stops = {"BAD": 92.0}                            # stop/px = 1.84 >= 1.8 ...
    d.state.high_water = {"BAD": 110.0}
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("BAD", 5.0, 100.0, current_price=50.0)])
    broker = _Broker()                                       # ...but stop/avg = 0.92 < 1.8

    triggered = d._manage_risk(broker, acct, execute=True)

    assert triggered == {"BAD"}
    assert len(broker.orders) == 1 and broker.orders[0].side == "sell"
