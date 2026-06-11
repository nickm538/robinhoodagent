"""Tests for the cadence/observability strengthening pass: activity ledger +
`why` diagnosis, breakeven ratchet, stop re-entry cooldowns, pending-scan
expiry, order-decision notes, and the threaded universe light pass."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rh_agent.activity import ActivityLog, diagnose
from rh_agent.config import load_config
from rh_agent.execution import build_orders
from rh_agent.models import Account, Position, TargetPosition
from rh_agent.risk import breakeven_stop


def _utcnow():
    return datetime.now(timezone.utc)


# ----------------------------- activity ledger -----------------------------

def test_activity_log_record_and_tail(tmp_path):
    log = ActivityLog(path=tmp_path / "activity.jsonl")
    log.record("scan_done", seconds=120.0, targets=5)
    log.record("rebalance", orders=0, notes=[{"ticker": "AAA", "action": "hold_within_band"}])
    events = log.tail(hours=1)
    assert [e["event"] for e in events] == ["scan_done", "rebalance"]
    assert events[0]["seconds"] == 120.0
    # None-valued fields are dropped, ts always present
    assert all("ts" in e for e in events)


def test_activity_log_never_raises(tmp_path):
    log = ActivityLog(path=tmp_path / "nope" / "deep" / "activity.jsonl")
    log.record("heartbeat", equity=1.0)          # parent dirs created on demand
    assert log.tail(hours=1)[0]["event"] == "heartbeat"
    disabled = ActivityLog(path=tmp_path / "off.jsonl")
    disabled.enabled = False
    disabled.record("x")
    assert disabled.tail() == []


def test_diagnose_explains_quiet_hunts_and_scan_bound_cadence(tmp_path, monkeypatch):
    import rh_agent.activity as activity
    monkeypatch.setattr(activity, "DAEMON_STATE", tmp_path / "daemon_state.json")
    cfg = load_config()
    led = ActivityLog(path=tmp_path / "activity.jsonl")
    monkeypatch.setattr(activity, "ActivityLog", lambda c=None, path=None: led)

    led.record("scan_started", held=3)
    led.record("scan_done", seconds=1500.0, universe=380, scored=35, eligible=8, targets=5)
    led.record("rebalance", orders=0, mode="live", allow_buys=True,
               notes=[{"ticker": "AAA", "action": "hold_within_band", "detail": ""},
                      {"ticker": "BBB", "action": "hold_within_band", "detail": ""}])
    led.record("scan_abandoned", seconds=1801.0, timeout=1800.0)
    led.record("risk", kind="stop_filled", ticker="CCC", price=9.5, stop=9.6)

    out = diagnose(cfg, hours=24)
    assert "Median scan duration" in out
    assert "scan-bound" in out                       # 1500s > 20m interval
    assert "discipline working" in out               # zero-order hunts explained
    assert "abandoned 1" in out
    assert "watchdog" in out
    assert "CCC" in out                              # protective exit surfaced


def test_diagnose_with_no_ledger_is_helpful(tmp_path, monkeypatch):
    import rh_agent.activity as activity
    monkeypatch.setattr(activity, "DAEMON_STATE", tmp_path / "daemon_state.json")
    led = ActivityLog(path=tmp_path / "missing.jsonl")
    monkeypatch.setattr(activity, "ActivityLog", lambda c=None, path=None: led)
    out = diagnose(load_config(), hours=24)
    assert "No activity ledger" in out
    assert "Market:" in out


# ----------------------------- breakeven ratchet ---------------------------

def test_breakeven_stop_math():
    # below trigger distance -> no floor yet
    assert breakeven_stop(100.0, 103.0, 4.0, 1.0) is None
    # at/above trigger -> entry (+buffer)
    assert breakeven_stop(100.0, 104.0, 4.0, 1.0) == pytest.approx(100.0)
    assert breakeven_stop(100.0, 110.0, 4.0, 1.0, buffer_pct=0.002) == pytest.approx(100.2)
    # unusable inputs -> None
    assert breakeven_stop(None, 110.0, 4.0, 1.0) is None
    assert breakeven_stop(100.0, 110.0, None, 1.0) is None
    assert breakeven_stop(100.0, 110.0, 4.0, 0.0) is None


def _mk_daemon(tmp_path, monkeypatch, cfg):
    import rh_agent.daemon as daemon
    monkeypatch.setattr(daemon, "STATE", tmp_path / "daemon_state.json")
    d = daemon.AlwaysOnAgent.__new__(daemon.AlwaysOnAgent)
    d.cfg = cfg
    d.state = daemon.DaemonState.load()
    d.journal = MagicMock()
    d.activity = ActivityLog(path=tmp_path / "activity.jsonl")
    return d


class _SellBroker:
    supports_live = False

    def __init__(self, status="submitted"):
        self.status = status
        self.orders = []

    def place_order(self, order, dry_run=True):
        self.orders.append(order)
        return {"status": self.status, "ticker": order.ticker}


def test_manage_risk_breakeven_ratchets_stop_to_entry(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.raw["portfolio"]["risk_controls"].update({
        "stop_loss_atr_mult": 2.5, "hard_stop_pct": 0.18,
        "breakeven_after_atr": 1.0, "breakeven_buffer_pct": 0.002,
    })
    d = _mk_daemon(tmp_path, monkeypatch, cfg)
    d.agent = SimpleNamespace(
        price_fn=lambda t, for_risk=False: 100.0,
        md=SimpleNamespace(build=lambda tk, deep=False: SimpleNamespace(
            technicals={"atr": 4.0})),
    )
    d.state.stops = {"AAA": 85.0}
    d.state.high_water = {"AAA": 100.0}
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("AAA", 1.0, 95.0, current_price=100.0)])
    triggered = d._manage_risk(_SellBroker(), acct, execute=True)
    assert triggered == set()
    # trail = 100 - 2.5*4 = 90; breakeven floor = 95*1.002 = 95.19 wins
    assert d.state.stops["AAA"] == pytest.approx(95.19)
    kinds = [e.get("kind") for e in d.activity.tail()]
    assert "breakeven_set" in kinds


def test_stop_fill_starts_reentry_cooldown_and_it_expires(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.raw["portfolio"]["risk_controls"]["reentry_cooldown_hours_after_stop"] = 6.0
    d = _mk_daemon(tmp_path, monkeypatch, cfg)
    d.agent = SimpleNamespace(
        price_fn=lambda t, for_risk=False: 95.0,
        md=SimpleNamespace(build=lambda tk, deep=False: SimpleNamespace(technicals={})),
    )
    d.state.stops = {"AAA": 100.0}
    d.state.high_water = {"AAA": 110.0}
    acct = Account(equity=1000.0, cash=0.0, buying_power=0.0,
                   positions=[Position("AAA", 2.0, 105.0, current_price=95.0)])
    broker = _SellBroker()
    triggered = d._manage_risk(broker, acct, execute=True)
    assert triggered == {"AAA"}
    assert broker.orders and broker.orders[0].side == "sell"
    assert "AAA" in d.state.cooldowns
    assert d._active_cooldowns(_utcnow()) == {"AAA"}
    # expired entries purge themselves
    d.state.cooldowns["AAA"] = (_utcnow() - timedelta(minutes=1)).isoformat()
    assert d._active_cooldowns(_utcnow()) == set()
    assert d.state.cooldowns == {}
    # invalid entries are dropped, never crash
    d.state.cooldowns["BBB"] = "garbage"
    assert d._active_cooldowns(_utcnow()) == set()


def test_cooldown_disabled_when_config_zero(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.raw["portfolio"]["risk_controls"]["reentry_cooldown_hours_after_stop"] = 0.0
    d = _mk_daemon(tmp_path, monkeypatch, cfg)
    d._start_cooldown("AAA")
    assert d.state.cooldowns == {}


# ----------------------------- pending-scan expiry -------------------------

def test_tick_discards_overnight_stale_pending_scan(tmp_path, monkeypatch):
    from rh_agent.agent import ScanResult
    from rh_agent.regime import RegimeResult

    cfg = load_config()
    d = _mk_daemon(tmp_path, monkeypatch, cfg)
    scan = ScanResult(regime=RegimeResult("neutral", {}, 0.85), verdicts=[], eligible=[],
                      targets=[], equity=1000.0, universe_size=1, scored_size=1)

    class _Broker:
        supports_live = False

        def get_account(self):
            return Account(equity=1000.0, cash=1000.0, buying_power=1000.0, positions=[])

    class _Agent:
        providers = {}

        def clear_price_cache(self):
            pass

        def make_broker(self):
            return _Broker()

        def default_equity(self):
            return 1000.0

        def reconcile_and_execute(self, *a, **k):
            raise AssertionError("stale overnight scan must not be executed")

        def scan(self, *a, **k):  # pragma: no cover - background pool unused here
            raise AssertionError("no new scan in this test")

    d.agent = _Agent()
    d._scan_future = None
    d._scan_started_at = None
    d._pending_scan = scan
    d._pending_scan_at = _utcnow() - timedelta(hours=17)   # finished yesterday
    d._pending_scan_max_age = 300.0
    d._pending_scan_expiry = 3600.0
    d._ensure_stops_for_held = lambda *a, **k: None
    d._manage_risk = lambda *a, **k: set()
    d._due_for_rebalance = lambda now: False

    d.tick(execute=True)

    assert d._pending_scan is None
    assert "scan_expired" in [e["event"] for e in d.activity.tail()]


# ----------------------------- order notes ---------------------------------

def test_build_orders_explains_quiet_cycle():
    cfg = load_config()
    cfg.raw["portfolio"]["rebalance"].update(
        {"no_trade_band": 0.02, "min_order_notional": 15.0})
    acct = Account(equity=1000.0, cash=100.0, buying_power=100.0, positions=[
        Position("HOLD", 2.0, 100.0, current_price=100.0),   # 20% weight
    ])
    targets = [TargetPosition("HOLD", weight=0.21, score=70.0, sector="Tech"),
               TargetPosition("TINY", weight=0.005, score=66.0, sector="Tech")]
    notes: list = []
    orders = build_orders(acct, targets, cfg, lambda t: 100.0, explain=notes)
    assert orders == []
    actions = {n["ticker"]: n["action"] for n in notes}
    assert actions["HOLD"] == "hold_within_band"
    assert actions["TINY"] == "skipped_min_notional"


def test_build_orders_notes_excluded_and_halted():
    cfg = load_config()
    acct = Account(equity=1000.0, cash=1000.0, buying_power=1000.0, positions=[])
    targets = [TargetPosition("COOL", weight=0.10, score=80.0, sector="Tech")]
    notes: list = []
    orders = build_orders(acct, targets, cfg, lambda t: 50.0,
                          exclude_tickers={"COOL"}, explain=notes)
    assert orders == []
    assert any(n["action"] == "excluded" and n["ticker"] == "COOL" for n in notes)

    notes2: list = []
    orders2 = build_orders(acct, targets, cfg, lambda t: 50.0,
                           allow_buys=False, explain=notes2)
    assert orders2 == []
    assert any(n["action"] == "buys_halted" for n in notes2)


def test_build_orders_explain_default_is_noop():
    cfg = load_config()
    acct = Account(equity=1000.0, cash=1000.0, buying_power=1000.0, positions=[])
    targets = [TargetPosition("AAA", weight=0.10, score=80.0, sector="Tech")]
    orders = build_orders(acct, targets, cfg, lambda t: 50.0)
    assert len(orders) == 1     # unchanged behaviour without explain


# ----------------------------- threaded light pass --------------------------

def test_build_universe_threaded_matches_serial(monkeypatch):
    from rh_agent.data import universe as U

    cfg = load_config()
    cfg.raw["universe"]["intraday"]["enabled"] = False
    cfg.raw["universe"]["liquidity"].update({
        "min_market_cap": 0, "min_avg_dollar_volume": 0,
        "min_avg_volume_shares": 0, "min_history_days": 0,
    })

    def fake_light(md, t, max_quote_age_seconds=None):
        return U.Candidate(t, price=50.0, market_cap=1e10, adv=1e7, avg_volume=1e6,
                           hist_days=300, mom_63=0.1, sector="Technology")

    monkeypatch.setattr(U, "_light", fake_light)

    class FakeMD:
        def list_universe(self):
            return [f"T{i}" for i in range(40)]

    cfg.raw["data"] = {"max_workers": 1}
    serial = U.build_universe(FakeMD(), cfg)
    cfg.raw["data"] = {"max_workers": 8}
    threaded = U.build_universe(FakeMD(), cfg)
    assert serial == threaded == [f"T{i}" for i in range(40)]   # order preserved
