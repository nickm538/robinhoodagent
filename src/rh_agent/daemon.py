"""Always-on autonomous loop.

Runs hands-off and non-stop. Each cycle it:
  1. refreshes the account from the broker;
  2. manages risk on open positions every tick (trailing/hard stops, take-profits)
     — this happens intraday, not just at rebalance;
  3. on the configured cadence (intraday by default), re-scans the universe, rebuilds the
     target book, and reconciles/executes orders;
  4. enforces a daily-drawdown circuit breaker that suspends new buying (sells still run);
  5. persists state and sleeps until the next tick.

The loop is crash-resistant: any exception in a cycle is logged and the loop
continues. It honours EXECUTION_MODE (paper vs live) exactly like the rest of
the system — it will not place live orders unless the user has armed them.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

from .activity import ActivityLog
from .agent import TradingAgent
from .broker.orders import order_succeeded
from .config import REPO_ROOT, Config, write_private
from .journal import Journal
from .logging_setup import get_logger
from .market_calendar import is_market_open, to_eastern
from .models import Order
from .process_lock import ProcessLockError, daemon_lock
from .risk import atr_stop, breakeven_stop, trailing_stop

log = get_logger("daemon")
STATE = REPO_ROOT / "state" / "daemon_state.json"


@dataclass
class DaemonState:
    last_rebalance: str = ""
    day: str = ""
    day_start_equity: float = 0.0
    stops: dict = None
    take_profits: dict = None
    high_water: dict = None
    pending_risk: dict = None
    # ticker -> ISO time until which re-entry is blocked after a stop-out
    cooldowns: dict = None
    last_scan_seconds: float = 0.0

    @classmethod
    def load(cls) -> "DaemonState":
        if STATE.exists():
            try:
                d = json.loads(STATE.read_text())
                known = {f.name for f in fields(cls)}
                st = cls(**{k: v for k, v in d.items() if k in known})
                if not isinstance(st.stops, dict):
                    st.stops = {}
                if not isinstance(st.take_profits, dict):
                    st.take_profits = {}
                if not isinstance(st.high_water, dict):
                    st.high_water = {}
                if not isinstance(st.pending_risk, dict):
                    st.pending_risk = {}
                if not isinstance(st.cooldowns, dict):
                    st.cooldowns = {}
                if st.last_rebalance and not _valid_iso(st.last_rebalance):
                    log.warning("invalid last_rebalance %r — resetting", st.last_rebalance)
                    st.last_rebalance = ""
                return st
            except Exception as e:
                log.warning("daemon_state.json unreadable (%s) — starting fresh", e)
        return cls(stops={}, take_profits={}, high_water={}, pending_risk={}, cooldowns={})

    def save(self) -> None:
        write_private(STATE, json.dumps(self.__dict__, indent=2, default=str))


def _valid_iso(s: str) -> bool:
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


class AlwaysOnAgent:
    # Class-level fallbacks so instances assembled piecemeal (tests use __new__)
    # keep working; __init__ overrides them with configured values.
    activity = ActivityLog()
    _last_heartbeat: datetime | None = None
    _halt_announced_day: str | None = None
    _pending_scan_expiry: float = 3600.0

    def __init__(self, cfg: Config, snapshot_path: str | None = None):
        self.cfg = cfg
        self._snapshot_path = snapshot_path
        self.agent = TradingAgent(cfg, snapshot_path=snapshot_path)
        self.d = cfg.get("daemon", {}) or {}
        self.state = DaemonState.load()
        for attr in ("stops", "take_profits", "high_water", "pending_risk", "cooldowns"):
            if getattr(self.state, attr) is None:
                setattr(self.state, attr, {})
        self.activity = ActivityLog(cfg)
        self._last_heartbeat: datetime | None = None
        # Background scan worker: the slow rebalance scan (data + AI, ~minutes)
        # runs here so stop/take-profit risk checks keep firing every tick on the
        # main thread. Each scan gets its own TradingAgent/data stack so an
        # orphaned worker cannot race the main agent's caches/state. The worker
        # NEVER touches the broker — all order placement and state writes happen
        # on the main thread when we consume its result.
        self._scan_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rh-scan")
        self._scan_future = None      # in-flight scan, or None
        self._scan_started_at = None  # when the in-flight scan was kicked (watchdog)
        self._pending_scan = None     # finished scan awaiting a safe tick to execute
        # Watchdog: a scan that never returns would otherwise pin _scan_future and
        # silently halt all hunting until a manual restart. Abandon one that runs
        # longer than this and let a fresh scan start next cycle.
        self._scan_timeout = float(self.cfg.get("daemon.scan_timeout_seconds", 1800))
        # A finished scan must not wait forever behind a protective-sell tick.
        self._pending_scan_max_age = float(self.cfg.get("daemon.pending_scan_max_seconds", 300))
        # ...but a scan that sat pending across a market close (or any long gap)
        # is yesterday's conviction — discard rather than trade on it at the open.
        self._pending_scan_expiry = float(self.cfg.get("daemon.pending_scan_expiry_seconds", 3600))
        self._pending_scan_at: datetime | None = None
        self.journal = Journal(self.cfg)

    # -- cadence --
    def _interval_seconds(self) -> float:
        sched = self.cfg.get("portfolio.rebalance.schedule", "hourly")
        if sched == "hourly":
            return 3600.0
        if sched == "intraday":
            return float(self.cfg.get("portfolio.rebalance.intraday_hours", 2)) * 3600.0
        days = {"daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30}.get(sched, 1 / 24)
        return days * 86400.0

    def _due_for_rebalance(self, now: datetime) -> bool:
        if not self.state.last_rebalance:
            return True
        try:
            last = datetime.fromisoformat(self.state.last_rebalance)
        except ValueError:
            return True
        return (now - last).total_seconds() >= self._interval_seconds()

    def _ensure_stops_for_held(self, broker, account) -> None:
        """On boot / each tick, every held name gets a stop if none is recorded."""
        rc = self.cfg.get("portfolio.risk_controls", {})
        atr_mult = float(rc.get("stop_loss_atr_mult", 2.5))
        hard_pct = float(rc.get("hard_stop_pct", 0.18))
        for pos in account.positions:
            tk = pos.ticker
            if self.state.stops.get(tk):
                continue
            px = self.agent.price_fn(tk, for_risk=True) or pos.current_price or pos.avg_price
            if not px:
                continue
            atr = None
            try:
                td = self.agent.md.build(tk, deep=False)
                atr = td.technicals.get("atr") if td else None
            except Exception:
                pass
            stop = atr_stop(px, atr, atr_mult, hard_pct)
            self.state.stops[tk] = stop
            self.state.high_water[tk] = px
            log.info("synthesized stop for held %s @ %.2f", tk, stop)

    # -- re-entry cooldowns (anti-churn after a stop-out) --
    def _start_cooldown(self, ticker: str) -> None:
        """After a protective stop sells a name, block re-buying it for a while.
        The intraday radar would otherwise re-surface the same volatile name on
        the very next hunt (it just WAS a runner) and buy the falling knife back."""
        hours = float(self.cfg.get(
            "portfolio.risk_controls.reentry_cooldown_hours_after_stop", 0.0) or 0.0)
        if hours <= 0:
            return
        if not isinstance(self.state.cooldowns, dict):
            self.state.cooldowns = {}
        until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.state.cooldowns[ticker] = until.isoformat()
        log.info("re-entry cooldown for %s until %s (%.1fh)", ticker, until.isoformat(), hours)
        self.activity.record("risk", kind="cooldown_set", ticker=ticker,
                             until=until.isoformat())

    def _active_cooldowns(self, now: datetime) -> set[str]:
        """Still-active cooldown tickers; expired/invalid entries are purged."""
        if not isinstance(self.state.cooldowns, dict):
            self.state.cooldowns = {}
            return set()
        active: set[str] = set()
        for tk, until in list(self.state.cooldowns.items()):
            try:
                expires = datetime.fromisoformat(until)
            except (TypeError, ValueError):
                self.state.cooldowns.pop(tk, None)
                continue
            if now >= expires:
                self.state.cooldowns.pop(tk, None)
            else:
                active.add(tk)
        return active

    # -- one cycle --
    def tick(self, execute: bool = False) -> None:
        now = datetime.now(timezone.utc)
        # Don't clear the in-memory quote cache out from under an in-flight
        # background scan; risk checks use for_risk quotes that bypass it anyway.
        if self._scan_future is None:
            self.agent.clear_price_cache()
        broker = self.agent.make_broker()
        account = broker.get_account()

        if broker.supports_live and not account.reliable:
            log.error("live account snapshot unreliable — skipping tick entirely")
            return

        today = to_eastern(now).strftime("%Y-%m-%d")
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
                if getattr(self, "_halt_announced_day", None) != today:
                    self._halt_announced_day = today
                    self.activity.record("halt", day=today, drawdown_pct=round(100 * dd, 2),
                                         equity=round(account.equity, 2))

        self._ensure_stops_for_held(broker, account)

        # 1) risk management on open positions — EVERY tick, never blocked by a scan
        risk_actions = self._manage_risk(broker, account, execute)
        # Unresolved protective sells block BOTH order sides; re-entry cooldowns
        # block re-buys only (an exit must never be suppressed by a cooldown).
        pending = set(self.state.pending_risk.keys())
        cooldowns = self._active_cooldowns(now)
        due_for_rebalance = self._due_for_rebalance(now)

        # 2) harvest a finished background scan, or abandon a stuck one (watchdog)
        if self._scan_future is not None:
            scan_age = ((now - self._scan_started_at).total_seconds()
                        if self._scan_started_at else 0.0)
            if self._scan_future.done():
                try:
                    self._pending_scan = self._scan_future.result()
                    self._pending_scan_at = now
                    self.state.last_scan_seconds = round(scan_age, 1)
                    s = self._pending_scan
                    self.activity.record(
                        "scan_done", seconds=round(scan_age, 1),
                        universe=getattr(s, "universe_size", None),
                        scored=getattr(s, "scored_size", None),
                        eligible=len(getattr(s, "eligible", []) or []),
                        targets=len(getattr(s, "targets", []) or []),
                        regime=getattr(getattr(s, "regime", None), "name", None))
                except Exception as e:
                    log.error("background scan failed: %s", e, exc_info=True)
                    self.activity.record("scan_failed", seconds=round(scan_age, 1),
                                         error=str(e)[:300])
                    self._pending_scan = None
                    self._pending_scan_at = None
                self._scan_future = None
                self._scan_started_at = None
            elif scan_age > self._scan_timeout:
                log.error("background scan exceeded %.0fs without finishing — abandoning it; "
                          "a fresh scan will start next cycle", self._scan_timeout)
                self.activity.record("scan_abandoned", seconds=round(scan_age, 1),
                                     timeout=self._scan_timeout)
                # A stuck worker thread can't be force-killed; retire the whole pool
                # (the orphan dies when its HTTP calls time out) and start a clean one.
                self._scan_pool.shutdown(wait=False, cancel_futures=True)
                self._scan_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rh-scan")
                self._scan_future = None
                self._scan_started_at = None
                self._pending_scan = None

        # 2b) expire a pending scan that sat un-executed across a long gap (e.g.
        # finished after the close and idled overnight). Trading the next open on
        # yesterday-afternoon conviction is exactly the "mystery trade at 9:30"
        # failure mode — drop it and let a fresh hunt start since rebalance is due.
        if self._pending_scan is not None and self._pending_scan_at is not None:
            age = (now - self._pending_scan_at).total_seconds()
            if age > self._pending_scan_expiry:
                log.warning("pending scan aged %.0fs > expiry %.0fs — discarding stale "
                            "conviction; a fresh hunt starts next cycle",
                            age, self._pending_scan_expiry)
                self.activity.record("scan_expired", age_seconds=round(age, 1))
                self._pending_scan = None
                self._pending_scan_at = None

        # 3) execute a harvested scan — prefer a clean tick after a protective
        #    sell, but never let a finished hunt rot while risk orders retry.
        pending_at = getattr(self, "_pending_scan_at", None)
        pending_age = (now - pending_at).total_seconds() if pending_at else 0.0
        pending_stale = pending_age > float(getattr(self, "_pending_scan_max_age", 300))
        if self._pending_scan is not None and (not risk_actions or pending_stale):
            if pending_stale and risk_actions:
                log.warning("pending scan aged %.0fs — executing despite risk actions",
                            pending_age)
            scan = self._pending_scan
            # Drop cached quotes so order sizing/reconciliation uses a fresh
            # main-thread price snapshot.
            self.agent.clear_price_cache()
            # Refresh account after the background scan delay so buying-power
            # and positions reflect the live book at execution time.
            try:
                account = broker.get_account()
            except Exception as e:
                log.warning("pre-reconcile account refresh failed: %s", e)
                if broker.supports_live:
                    log.error("live account refresh failed — skipping stale-scan execution")
                    self.state.save()
                    return
            self._pending_scan = None
            self._pending_scan_at = None
            run = self.agent.reconcile_and_execute(
                scan, execute=execute, allow_buys=not halted, exclude_tickers=pending,
                exclude_buy_tickers=cooldowns, broker=broker, account=account)
            self._apply_rebalance_result(run, account, now=now)
            self.activity.record(
                "rebalance", orders=len(run.orders),
                sells=sum(1 for o in run.orders if o.side == "sell"),
                buys=sum(1 for o in run.orders if o.side == "buy"),
                mode=run.mode, allow_buys=not halted,
                targets=len(run.scan.targets),
                notes=(run.order_notes or [])[:20] or None)
        # 4) otherwise, kick off a new scan when due (and none is in flight/pending)
        elif (due_for_rebalance and self._scan_future is None
              and self._pending_scan is None and not risk_actions):
            equity = account.equity if (account.equity and account.equity > 0) \
                else self.agent.default_equity()
            held = [p.ticker for p in account.positions]
            log.info("rebalance due — scanning in background (allow_buys=%s)", not halted)
            self._scan_future = self._scan_pool.submit(self._safe_scan, equity, held,
                                                       set(cooldowns))
            self._scan_started_at = now
            self.activity.record("scan_started", held=len(held), equity=round(equity, 2),
                                 cooldowns=sorted(cooldowns) or None)
        else:
            log.info("monitoring %d positions (next rebalance pending)", len(account.positions))

        self._heartbeat(now, account, halted)
        self.state.save()

    def _heartbeat(self, now: datetime, account, halted: bool) -> None:
        """Cadence-aware pulse: one INFO line per tick + a ledger event every ~15m
        so `rh-agent why` can prove the loop was alive between rebalances."""
        try:
            last = self.state.last_rebalance
            next_due = "now"
            if last:
                remain = self._interval_seconds() \
                    - (now - datetime.fromisoformat(last)).total_seconds()
                next_due = f"{max(remain, 0) / 60:.0f}m"
            scanning = ""
            if self._scan_started_at is not None:
                scanning = f" | scan in flight {(now - self._scan_started_at).total_seconds() / 60:.0f}m"
            log.info("pulse: equity $%.2f | %d positions | next hunt %s%s%s",
                     account.equity, len(account.positions), next_due, scanning,
                     " | BUYS HALTED" if halted else "")
            if (self._last_heartbeat is None
                    or (now - self._last_heartbeat).total_seconds() >= 900):
                self._last_heartbeat = now
                self.activity.record("heartbeat", equity=round(account.equity, 2),
                                     positions=len(account.positions), halted=halted,
                                     scan_in_flight=bool(self._scan_future is not None),
                                     last_scan_seconds=self.state.last_scan_seconds or None)
        except Exception as e:
            log.debug("heartbeat failed: %s", e)

    def _mode(self) -> str:
        return "live" if self.cfg.live_trading_armed else "paper"

    def _safe_scan(self, equity: float, held: list[str], exclude: set[str] | None = None):
        """Background worker: the slow scan only (data + AI). No broker, no state."""
        scan_agent = TradingAgent(
            self.cfg,
            snapshot_path=getattr(self, "_snapshot_path", None),
        )
        return scan_agent.scan(equity=equity, include_tickers=held,
                               exclude_tickers=set(exclude or ()))

    def _apply_rebalance_result(self, run, account, *, now: datetime | None = None) -> None:
        for tp in run.scan.targets:
            if tp.stop_price:
                # Stops are monotonic for a held name: a rebalance recomputing from
                # the current price must never undo a breakeven/trailing ratchet.
                prev = self.state.stops.get(tp.ticker)
                self.state.stops[tp.ticker] = max(prev, tp.stop_price) if prev \
                    else tp.stop_price
            if tp.take_profit:
                self.state.take_profits[tp.ticker] = tp.take_profit
            if tp.ticker not in self.state.high_water:
                px = self.agent.price_fn(tp.ticker) or 0.0
                if px:
                    self.state.high_water[tp.ticker] = px
        held_account = run.post_account
        if held_account is None:
            try:
                held_account = self.agent.make_broker().get_account()
            except Exception as e:
                log.warning("post-rebalance account refresh failed: %s", e)
        if held_account is not None:
            held = {p.ticker for p in held_account.positions}
        else:
            held = {p.ticker for p in account.positions}
        self.state.stops = {k: v for k, v in self.state.stops.items() if k in held}
        self.state.take_profits = {k: v for k, v in self.state.take_profits.items() if k in held}
        self.state.high_water = {k: v for k, v in self.state.high_water.items() if k in held}
        # Prune pending-risk flags for names no longer held, so a name that left
        # the book another way (manual sale, unrecognized fill status) is not
        # permanently excluded from future re-entry.
        self.state.pending_risk = {k: v for k, v in self.state.pending_risk.items() if k in held}
        # Journal the executed orders with their decision context (self-improving log).
        self.journal.record_rebalance(run, held_account or account, self.agent.price_fn)
        if now is not None:
            # Anchor cadence to scan COMPLETION so a slow hunt doesn't compress the interval.
            self.state.last_rebalance = now.isoformat()
        log.info("rebalanced: %d targets, %d orders (%s)",
                 len(run.scan.targets), len(run.orders), run.mode)

    def _manage_risk(self, broker, account, execute: bool) -> set[str]:
        """Manage stops/TPs. Returns tickers where a protective sell was confirmed."""
        triggered: set[str] = set()
        rc = self.cfg.get("portfolio.risk_controls", {})
        atr_mult = float(rc.get("stop_loss_atr_mult", 2.5))
        hard_pct = float(rc.get("hard_stop_pct", 0.18))
        be_after_atr = float(rc.get("breakeven_after_atr", 0.0) or 0.0)
        be_buffer = float(rc.get("breakeven_buffer_pct", 0.0) or 0.0)
        for pos in account.positions:
            tk = pos.ticker
            # Pending risk orders are excluded from rebalance, but we still evaluate stops/TPs each tick.
            try:
                px = self.agent.price_fn(tk, for_risk=True)
            except Exception as e:
                log.warning("quote failed for %s risk check: %s", tk, e)
                continue
            if not px:
                continue
            hw = max(self.state.high_water.get(tk, px), px)
            self.state.high_water[tk] = hw
            stop = self.state.stops.get(tk)
            tp = self.state.take_profits.get(tk)
            if stop and px <= stop:
                log.warning("STOP hit %s @ %.2f (stop %.2f) — selling", tk, px, stop)
                res = broker.place_order(
                    Order(tk, "sell", pos.quantity, reason=f"stop {stop}"),
                    dry_run=not execute,
                )
                if execute and order_succeeded(res, executing=True):
                    self.state.stops.pop(tk, None)
                    self.state.take_profits.pop(tk, None)
                    self.state.high_water.pop(tk, None)
                    self.state.pending_risk.pop(tk, None)
                    triggered.add(tk)
                    self.journal.record_order(ticker=tk, side="sell", qty=pos.quantity,
                                              price=px, reason=f"stop {stop}",
                                              status=(res or {}).get("status", "submitted"),
                                              mode=self._mode())
                    self.activity.record("risk", kind="stop_filled", ticker=tk,
                                         price=round(px, 4), stop=stop)
                    self._start_cooldown(tk)
                elif not execute:
                    log.info("STOP preview for %s — keeping stop state unchanged", tk)
                else:
                    self.state.pending_risk[tk] = "stop"
                    log.error("STOP sell failed for %s — keeping stop active", tk)
                    self.activity.record("risk", kind="stop_failed", ticker=tk,
                                         price=round(px, 4), stop=stop)
            elif tp and px >= tp:
                log.info("TAKE-PROFIT %s @ %.2f (tp %.2f) — trimming half", tk, px, tp)
                res = broker.place_order(
                    Order(tk, "sell", pos.quantity * 0.5, reason=f"take-profit {tp}"),
                    dry_run=not execute,
                )
                if execute and order_succeeded(res, executing=True):
                    self.state.take_profits.pop(tk, None)
                    self.state.pending_risk.pop(tk, None)
                    triggered.add(tk)
                    self.journal.record_order(ticker=tk, side="sell", qty=pos.quantity * 0.5,
                                              price=px, reason=f"take-profit {tp}",
                                              status=(res or {}).get("status", "submitted"),
                                              mode=self._mode())
                    self.activity.record("risk", kind="tp_filled", ticker=tk,
                                         price=round(px, 4), take_profit=tp)
                elif not execute:
                    log.info("TP preview for %s — keeping TP state unchanged", tk)
                else:
                    self.state.pending_risk[tk] = "take_profit"
                    log.error("TP sell failed for %s — keeping TP active", tk)
                    self.activity.record("risk", kind="tp_failed", ticker=tk,
                                         price=round(px, 4), take_profit=tp)
            elif stop:
                atr = None
                try:
                    td = self.agent.md.build(tk, deep=False)
                    atr = td.technicals.get("atr") if td else None
                except Exception:
                    pass
                trail = trailing_stop(hw, atr, atr_mult, hard_pct)
                # Breakeven ratchet: once the trade has worked by >= N×ATR, the
                # stop never sits below entry again — a winner can't become a loser.
                be = breakeven_stop(pos.avg_price, hw, atr, be_after_atr, be_buffer)
                candidate = max(trail, be) if be is not None else trail
                if candidate > stop:
                    self.state.stops[tk] = candidate
                    if be is not None and candidate == be and stop < be:
                        log.info("breakeven ratchet %s: stop %.2f -> %.2f (entry %.2f)",
                                 tk, stop, candidate, pos.avg_price)
                        self.activity.record("risk", kind="breakeven_set", ticker=tk,
                                             stop=candidate, entry=round(pos.avg_price, 4))
        return triggered

    # -- main loop --
    def run_forever(self, execute: bool = False, once: bool = False,
                    max_cycles: int | None = None) -> None:
        poll = int(self.d.get("poll_seconds", 900))
        trade_only_open = self.d.get("trade_only_when_open", True)
        log.info("AlwaysOnAgent started | poll=%ss | mode=%s | live_armed=%s | execute=%s",
                 poll, self.cfg.execution_mode, self.cfg.live_trading_armed, execute)
        if _ET is None:
            log.warning("zoneinfo/tzdata unavailable — using the built-in US-Eastern DST "
                        "fallback for market hours. Install tzdata for belt-and-braces.")
        cycles = 0
        try:
            with daemon_lock():
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
        except ProcessLockError as e:
            log.error("%s", e)
            raise
        finally:
            self._scan_pool.shutdown(wait=False, cancel_futures=True)
