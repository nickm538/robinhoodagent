"""Activity ledger + self-diagnosis — the agent's flight recorder.

The daemon appends one compact JSONL event per notable action (scan started /
finished / abandoned, rebalance outcomes with per-name order reasoning, stop and
take-profit fires, halts, heartbeats) to ``state/activity.jsonl``. The ledger
makes the difference between "the agent is hunting but the held book still
ranks best" and "the agent is stuck" visible at a glance.

``rh-agent why`` renders a human diagnosis from the ledger + daemon state so the
operator never has to guess whether a quiet day is discipline or a malfunction.
Everything here is best-effort and must NEVER break the trading loop.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from statistics import median

from .config import REPO_ROOT
from .logging_setup import get_logger
from .market_calendar import is_market_open, session_state

log = get_logger("activity")

DAEMON_STATE = REPO_ROOT / "state" / "daemon_state.json"

# Rotation bounds: the ledger is an operational tail, not an archive (the
# journal is the durable trade record). Keep it small enough to read whole.
_MAX_BYTES = 8 * 1024 * 1024
_KEEP_LINES = 4000


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ActivityLog:
    def __init__(self, cfg=None, path=None):
        d = (cfg.get("daemon", {}) if cfg else {}) or {}
        self.enabled = bool(d.get("activity_log", True))
        self.path = path or (REPO_ROOT / "state" / "activity.jsonl")

    def record(self, event: str, **fields) -> None:
        """Append one event. Best-effort; never raises into the trading loop."""
        if not self.enabled:
            return
        try:
            rec = {"ts": utcnow().isoformat(), "event": event}
            for k, v in fields.items():
                if v is not None:
                    rec[k] = v
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
        except Exception as e:
            log.debug("activity record failed: %s", e)

    def _rotate_if_needed(self) -> None:
        try:
            if not self.path.exists() or self.path.stat().st_size < _MAX_BYTES:
                return
            lines = self.path.read_text().splitlines()[-_KEEP_LINES:]
            self.path.write_text("\n".join(lines) + "\n")
        except Exception as e:
            log.debug("activity rotate failed: %s", e)

    def tail(self, hours: float | None = 24.0) -> list[dict]:
        """Events from the last N hours (all events when hours is None)."""
        if not self.path.exists():
            return []
        cutoff = utcnow() - timedelta(hours=hours) if hours else None
        out: list[dict] = []
        try:
            with open(self.path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if cutoff is not None:
                        try:
                            ts = datetime.fromisoformat(rec.get("ts", ""))
                        except ValueError:
                            continue
                        if ts < cutoff:
                            continue
                    out.append(rec)
        except OSError as e:
            log.warning("activity read failed: %s", e)
        return out


# --------------------------------------------------------------------------
# Diagnosis ("rh-agent why")
# --------------------------------------------------------------------------

def _fmt_minutes(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def _load_daemon_state() -> dict:
    if not DAEMON_STATE.exists():
        return {}
    try:
        return json.loads(DAEMON_STATE.read_text())
    except Exception:
        return {}


def _scan_lines(events: list[dict], cfg, hours: float) -> list[str]:
    started = [e for e in events if e.get("event") == "scan_started"]
    done = [e for e in events if e.get("event") == "scan_done"]
    abandoned = [e for e in events if e.get("event") == "scan_abandoned"]
    failed = [e for e in events if e.get("event") == "scan_failed"]
    expired = [e for e in events if e.get("event") == "scan_expired"]
    broker_unavailable = [e for e in events if e.get("event") == "broker_unavailable"]
    lines = [f"Hunts (last {hours:.0f}h): started {len(started)}, completed {len(done)}, "
             f"abandoned {len(abandoned)}, failed {len(failed)}, expired {len(expired)}."]
    durations = [float(e.get("seconds", 0)) for e in done if e.get("seconds")]
    interval_s = float(cfg.get("portfolio.rebalance.intraday_hours", 0.33)) * 3600
    if durations:
        med = median(durations)
        lines.append(f"Median scan duration {_fmt_minutes(med)} -> effective hunt cadence "
                     f"~{_fmt_minutes(med + interval_s)} (configured interval "
                     f"{_fmt_minutes(interval_s)} + scan time).")
        if med > interval_s:
            lines.append("NOTE: scan time exceeds the configured interval — cadence is "
                         "scan-bound. Speed it up via universe.scan_cap / deep_top_k, or "
                         "accept the slower hunt.")
    if abandoned:
        timeout = float(cfg.get("daemon.scan_timeout_seconds", 1800))
        lines.append(f"WARNING: {len(abandoned)} scan(s) hit the {_fmt_minutes(timeout)} "
                     "watchdog and were abandoned — no rebalance happened those cycles. "
                     "Check provider rate limits / API keys (rh-agent doctor), or raise "
                     "daemon.scan_timeout_seconds / lower universe.scan_cap.")
    if failed:
        last = failed[-1]
        lines.append(f"WARNING: last scan failure: {str(last.get('error', ''))[:140]}")
    if broker_unavailable:
        lines.append(f"WARNING: {len(broker_unavailable)} tick(s) skipped because the live "
                     "broker account snapshot was unreliable. Check OAuth and connectivity.")
    if not (started or done):
        lines.append("No hunts recorded in this window — if the market was open, check the "
                     "daemon is running (systemctl status rh-agent) and logs/loop.log.")
    return lines


def _rebalance_lines(events: list[dict]) -> list[str]:
    rebs = [e for e in events if e.get("event") == "rebalance"]
    if not rebs:
        return ["No completed rebalances in this window."]
    placed = sum(int(e.get("orders", 0)) for e in rebs)
    quiet = [e for e in rebs if not e.get("orders")]
    lines = [f"Rebalances: {len(rebs)} completed, {placed} orders placed, "
             f"{len(quiet)} produced no orders."]
    if quiet:
        reasons: dict[str, int] = {}
        for e in quiet:
            for n in e.get("notes", []) or []:
                a = n.get("action", "other") if isinstance(n, dict) else "other"
                reasons[a] = reasons.get(a, 0) + 1
        if reasons.get("hold_within_band"):
            lines.append("Zero-order hunts mean the held book still ranked best (drift inside "
                         "the no-trade band / exit hysteresis held). That is discipline "
                         "working, not a stall.")
        if reasons:
            top = ", ".join(f"{k}×{v}" for k, v in
                            sorted(reasons.items(), key=lambda kv: -kv[1])[:4])
            lines.append(f"Quiet-hunt reasons: {top}.")
    last = rebs[-1]
    lines.append(f"Last rebalance {last.get('ts', '?')}: {last.get('orders', 0)} orders "
                 f"({last.get('mode', '?')} mode, buys "
                 f"{'allowed' if last.get('allow_buys', True) else 'HALTED'}).")
    return lines


def _risk_lines(events: list[dict], state: dict) -> list[str]:
    stops = [e for e in events if e.get("event") == "risk" and e.get("kind") == "stop_filled"]
    tps = [e for e in events if e.get("event") == "risk" and e.get("kind") == "tp_filled"]
    fails = [e for e in events if e.get("event") == "risk"
             and str(e.get("kind", "")).endswith("_failed")]
    be = [e for e in events if e.get("event") == "risk" and e.get("kind") == "breakeven_set"]
    lines = []
    if stops or tps:
        what = []
        if stops:
            what.append(f"{len(stops)} stop(s): "
                        + ", ".join(str(e.get("ticker")) for e in stops[-6:]))
        if tps:
            what.append(f"{len(tps)} take-profit trim(s): "
                        + ", ".join(str(e.get("ticker")) for e in tps[-6:]))
        lines.append("Protective exits — " + "; ".join(what) + ".")
    if be:
        lines.append("Breakeven ratchets set on: "
                     + ", ".join(str(e.get("ticker")) for e in be[-8:]) + ".")
    if fails:
        lines.append(f"WARNING: {len(fails)} protective order(s) were rejected — see "
                     "logs/loop.log and state/daemon_state.json pending_risk.")
    cools = state.get("cooldowns") or {}
    if cools:
        lines.append("Re-entry cooldowns active: "
                     + ", ".join(f"{t} until {u}" for t, u in list(cools.items())[:8]) + ".")
    stops_map = state.get("stops") or {}
    if stops_map:
        lines.append("Live stops: " + ", ".join(f"{t}@{v}" for t, v in
                                                list(stops_map.items())[:10]) + ".")
    return lines or ["No protective exits in this window."]


def diagnose(cfg, *, hours: float = 24.0, now: datetime | None = None) -> str:
    """Human-readable answer to 'what has the agent been doing, and is that normal?'"""
    now = now or utcnow()
    sess = session_state(now)
    state = _load_daemon_state()
    events = ActivityLog(cfg).tail(hours)

    lines = ["== rh-agent why ==", ""]
    lines.append(f"Market: {sess['phase']} (ET {sess['et']}; clock source {sess['tz_source']}). "
                 + ("Trading window is OPEN." if is_market_open(now) else
                    "Outside regular hours — the loop idles by design "
                    "(daemon.trade_only_when_open)."))
    last_reb = state.get("last_rebalance") or ""
    if last_reb:
        try:
            age = (now - datetime.fromisoformat(last_reb)).total_seconds()
            lines.append(f"Last completed hunt: {last_reb} ({_fmt_minutes(age)} ago).")
        except ValueError:
            pass
    if state.get("day_start_equity"):
        lines.append(f"Session baseline equity (day start): ${state['day_start_equity']:.2f}.")
    lines.append("")
    if not events:
        lines.append("No activity ledger found for this window. The ledger starts recording "
                     "after the daemon restarts on a build that includes it "
                     "(state/activity.jsonl).")
        return "\n".join(lines)
    lines += _scan_lines(events, cfg, hours)
    lines.append("")
    lines += _rebalance_lines(events)
    lines.append("")
    lines += _risk_lines(events, state)
    try:
        from .journal import Journal
        s = Journal(cfg).stats()
        if s["closed_trades"]:
            lines.append("")
            lines.append(f"Realized so far: {s['closed_trades']} closed trades, hit-rate "
                         f"{s['hit_rate']:.0%}, net ${s['net_realized']:+.2f}.")
    except Exception:
        pass
    return "\n".join(lines)
