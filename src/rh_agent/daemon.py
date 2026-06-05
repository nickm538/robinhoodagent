"""Always-on autonomous loop.

Runs hands-off and non-stop. Each cycle it:
  1. refreshes the account from the broker;
  2. manages risk on open positions every tick (trailing/hard stops, take-profits)
     — this happens intraday, not just at rebalance;
  3. on the configured cadence (e.g. weekly), re-scans the universe, rebuilds the
     target book, and reconciles/executes orders;
  4. enforces a daily-drawdown circuit breaker that suspends new buying;
  5. persists state and sleeps until the next tick.

The loop is crash-resistant: any exception in a cycle is logged and the loop
continues. It honours EXECUTION_MODE (paper vs live) exactly like the rest of
the system — it will not place live orders unless the user has armed them.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

from .agent import TradingAgent
from .config import REPO_ROOT, Config
from .logging_setup import get_logger
from .models import Order

log = get_logger("daemon")
STATE = REPO_ROOT / "state" / "daemon_state.json"

# Minimal US market-holiday set (extend as needed). Half-days treated as open.
_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}


def is_market_open(now_utc: datetime | None = None) -> bool:
    now = now_utc or datetime.now(timezone.utc)
    et = now.astimezone(_ET) if _ET else now
    if et.weekday() >= 5:
        return False
    if et.strftime("%Y-%m-%d") in _HOLIDAYS_2026:
        return False
    minutes = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


@dataclass
class DaemonState:
    last_rebalance: str = ""
    day: str = ""
    day_start_equity: float = 0.0
    stops: dict = None
    take_profits: dict = None

    @classmethod
    def load(cls) -> "DaemonState":
        if STATE.exists():
            try:
                d = json.loads(STATE.read_text())
                known = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in d.items() if k in known})
            except Exception as e:
                log.warning("daemon_state.json unreadable (%s) — starting fresh", e)
        return cls(stops={}, take_profits={})

    def save(self) -> None:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(self.__dict__, indent=2, default=str))


class AlwaysOnAgent:
    def __init__(self, cfg: Config, snapshot_path: str | None = None):
        self.cfg = cfg
        self.agent = TradingAgent(cfg, snapshot_path=snapshot_path)
        self.d = cfg.get("daemon", {}) or {}
        self.state = DaemonState.load()
        if self.state.stops is None:
            self.state.stops = {}
        if self.state.take_profits is None:
            self.state.take_profits = {}

    # -- cadence --
    def _due_for_rebalance(self, now: datetime) -> bool:
        if not self.state.last_rebalance:
            return True
        last = datetime.fromisoformat(self.state.last_rebalance)
        elapsed = (now - last).total_seconds()
        sched = self.cfg.get("portfolio.rebalance.schedule", "weekly")
        if sched == "intraday":   # re-rank several times a day (also covers 'daily')
            hours = float(self.cfg.get("portfolio.rebalance.intraday_hours", 2))
            return elapsed >= hours * 3600
        days = {"daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30}.get(sched, 7)
        return elapsed >= days * 86400

    # -- one cycle --
    def tick(self, execute: bool = False) -> None:
        now = datetime.now(timezone.utc)
        self.agent.clear_price_cache()   # fresh quotes this tick — stops must not see stale prices
        broker = self.agent.make_broker()
        account = broker.get_account()

        # A live account whose balance fails to read (transient fetch hiccup -> equity 0)
        # must not drive risk/anchoring/sizing. Skip the whole tick; the next one recovers.
        if broker.supports_live and not (account.equity and account.equity > 0):
            log.error("live account equity read as 0 — skipping tick entirely (no risk/rebalance)")
            return

        # reset the daily anchor at the first tick of a new ET day
        today = (now.astimezone(_ET) if _ET else now).strftime("%Y-%m-%d")
        if self.state.day != today:
            self.state.day = today
            self.state.day_start_equity = account.equity

        halted = False
        dd_limit = self.cfg.get("portfolio.risk_controls.max_daily_drawdown_halt", 0.06)
        if self.state.day_start_equity > 0:
            dd = account.equity / self.state.day_start_equity - 1.0
            if dd <= -abs(dd_limit):
                halted = True
                log.warning("DAILY DRAWDOWN HALT: %.1f%% <= -%.1f%% — suspending new buys",
                            100 * dd, 100 * dd_limit)

        # 1) risk management on open positions — every tick
        self._manage_risk(broker, account, execute)

        # 2) scheduled rebalance (skipped while halted)
        if self._due_for_rebalance(now) and not halted:
            log.info("rebalance due — scanning")
            run = self.agent.run(execute=execute)
            for tp in run.scan.targets:
                if tp.stop_price:
                    self.state.stops[tp.ticker] = tp.stop_price
                if tp.take_profit:
                    self.state.take_profits[tp.ticker] = tp.take_profit
            # forget stops/take-profits for names no longer held
            held = {p.ticker for p in broker.get_account().positions}
            self.state.stops = {k: v for k, v in self.state.stops.items() if k in held}
            self.state.take_profits = {k: v for k, v in self.state.take_profits.items() if k in held}
            self.state.last_rebalance = now.isoformat()
            log.info("rebalanced: %d targets, %d orders (%s)",
                     len(run.scan.targets), len(run.orders), run.mode)
        else:
            log.info("monitoring %d positions (next rebalance pending)", len(account.positions))

        self.state.save()

    def _manage_risk(self, broker, account, execute: bool) -> None:
        for pos in account.positions:
            px = self.agent.price_fn(pos.ticker)
            if not px:
                continue
            stop = self.state.stops.get(pos.ticker)
            tp = self.state.take_profits.get(pos.ticker)
            if stop and px <= stop:
                log.warning("STOP hit %s @ %.2f (stop %.2f) — selling", pos.ticker, px, stop)
                broker.place_order(Order(pos.ticker, "sell", pos.quantity,
                                         reason=f"stop {stop}"), dry_run=not execute)
                self.state.stops.pop(pos.ticker, None)
            elif tp and px >= tp:
                log.info("TAKE-PROFIT %s @ %.2f (tp %.2f) — trimming half", pos.ticker, px, tp)
                broker.place_order(Order(pos.ticker, "sell", pos.quantity * 0.5,
                                         reason=f"take-profit {tp}"), dry_run=not execute)
                self.state.take_profits.pop(pos.ticker, None)
            else:
                # ratchet a trailing stop upward as price rises
                if stop:
                    atr_mult = self.cfg.get("portfolio.risk_controls.stop_loss_atr_mult", 2.5)
                    trail = px * (1 - 0.02 * atr_mult)
                    if trail > stop:
                        self.state.stops[pos.ticker] = round(trail, 2)

    # -- main loop --
    def run_forever(self, execute: bool = False, once: bool = False,
                    max_cycles: int | None = None) -> None:
        poll = int(self.d.get("poll_seconds", 900))
        trade_only_open = self.d.get("trade_only_when_open", True)
        log.info("AlwaysOnAgent started | poll=%ss | mode=%s | live_armed=%s | execute=%s",
                 poll, self.cfg.execution_mode, self.cfg.live_trading_armed, execute)
        if _ET is None:
            log.error("zoneinfo/tzdata unavailable — market-hours check falls back to UTC, "
                      "which is WRONG by ~4-5h. Install tzdata on this host.")
        cycles = 0
        while True:
            try:
                if trade_only_open and not is_market_open() and "snapshot" not in self.agent.providers:
                    log.info("market closed — idle")
                else:
                    self.tick(execute=execute)
            except Exception as e:
                log.error("cycle error (continuing): %s", e, exc_info=True)
            cycles += 1
            if once or (max_cycles and cycles >= max_cycles):
                break
            time.sleep(poll)
