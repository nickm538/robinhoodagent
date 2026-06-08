"""Trade journal + self-improvement memory.

Records every executed order with its decision context (regime, scores, AI
stance) to an append-only JSONL log, reconstructs realized P&L by average-cost
matching, and produces:

  * a compact ``performance_summary()`` fed into the AI overlay so Claude
    reasons with awareness of its own recent realized results, and
  * a human-readable ``memory.md`` digest for the operator.

It is deliberately read-only with respect to strategy: it informs decisions and
the operator, and NEVER changes parameters on its own.
"""
from __future__ import annotations

import json
from statistics import mean

from .config import REPO_ROOT, write_private
from .logging_setup import get_logger
from .models import utcnow

log = get_logger("journal")


def _status(fill) -> str:
    if not isinstance(fill, dict):
        return "unknown"
    return (fill.get("status") or "unknown")


def _accepted(fill) -> bool:
    return _status(fill).lower() in ("submitted", "filled")


def _fill_price(fill):
    if not isinstance(fill, dict):
        return None
    for k in ("fill", "price", "est_fill"):
        v = fill.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    res = fill.get("result")
    if isinstance(res, dict):
        for k in ("price", "average_price", "filled_price"):
            v = res.get(k)
            try:
                if v and float(v) > 0:
                    return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _short(reason: str, n: int = 40) -> str:
    reason = (reason or "").strip()
    return reason[:n]


class Journal:
    def __init__(self, cfg):
        j = (cfg.get("journal", {}) or {})
        self.enabled = bool(j.get("enabled", True))
        self.path = REPO_ROOT / j.get("path", "state/journal.jsonl")
        self.memory_path = REPO_ROOT / j.get("memory_md", "state/memory.md")
        self.summary_trades = int(j.get("summary_trades", 20))

    # ---------------------------------------------------------------- writing
    def record_order(self, *, ticker, side, qty=None, notional=None, price=None,
                     reason="", status="unknown", mode="live", context=None) -> None:
        """Append one executed order to the journal. Best-effort; never raises."""
        if not self.enabled:
            return
        rec = {
            "ts": utcnow().isoformat(),
            "ticker": (ticker or "").upper(),
            "side": (side or "").lower(),
            "qty": round(float(qty), 6) if qty is not None else None,
            "notional": round(float(notional), 2) if notional is not None else None,
            "price": round(float(price), 4) if price is not None else None,
            "reason": _short(reason, 120),
            "status": status,
            "mode": mode,
        }
        if context:
            rec.update({k: v for k, v in context.items() if v is not None})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
        except Exception as e:  # journaling must never break the trading loop
            log.warning("journal write failed: %s", e)

    def record_rebalance(self, run, account, price_fn=None) -> None:
        """Journal the accepted orders from a rebalance with their decision context."""
        if not self.enabled or not getattr(run, "executed", False):
            return
        scan = getattr(run, "scan", None)
        regime = getattr(scan.regime, "name", "") if scan else ""
        vmap = {v.ticker: v for v in (scan.verdicts if scan else [])}
        equity = round(account.equity, 2) if account and account.equity else None
        fills = run.fills or []
        for i, o in enumerate(run.orders):
            fill = fills[i] if i < len(fills) else None
            if not _accepted(fill):
                continue
            v = vmap.get(o.ticker)
            price = _fill_price(fill)
            if price is None and price_fn is not None:
                try:
                    price = price_fn(o.ticker)
                except Exception:
                    price = None
            ctx = {"regime": regime, "equity": equity}
            if v is not None:
                ctx["composite"] = round(v.composite, 1)
                ctx["pillars"] = v.pillars_passing
                ai = (v.analyst_scores or {}).get("ai_analyst")
                if ai is not None:
                    ctx["ai_score"] = round(float(ai), 1)
            self.record_order(ticker=o.ticker, side=o.side, qty=o.quantity,
                              notional=o.notional, price=price, reason=o.reason,
                              status=_status(fill), mode=getattr(run, "mode", "live"),
                              context=ctx)
        try:
            self.write_memory_md(account)
        except Exception as e:
            log.warning("memory.md update failed: %s", e)

    # ---------------------------------------------------------------- reading
    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        try:
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError as e:
            log.warning("journal read failed: %s", e)
        return out

    def _ledger(self, records):
        """Average-cost match of buys→sells → realized P&L per closed lot."""
        pos: dict[str, dict] = {}
        realized: list[dict] = []
        for r in records:
            tk, side, qty, price = (r.get("ticker"), r.get("side"),
                                    r.get("qty"), r.get("price"))
            if not tk or not qty or price is None:
                continue
            p = pos.setdefault(tk, {"qty": 0.0, "cost": 0.0})
            if side == "buy":
                p["qty"] += qty
                p["cost"] += qty * price
            elif side == "sell" and p["qty"] > 1e-9:
                avg = p["cost"] / p["qty"]
                sq = min(qty, p["qty"])
                realized.append({
                    "ticker": tk,
                    "pnl": (price - avg) * sq,
                    "ret_pct": (price / avg - 1) * 100 if avg else 0.0,
                    "reason": r.get("reason", ""),
                    "ts": r.get("ts"),
                })
                p["qty"] -= sq
                p["cost"] -= avg * sq
                if p["qty"] <= 1e-9:
                    p["qty"], p["cost"] = 0.0, 0.0
        return pos, realized

    def stats(self) -> dict:
        _, realized = self._ledger(self._read())
        n = len(realized)
        wins = [r for r in realized if r["pnl"] > 0]
        losses = [r for r in realized if r["pnl"] <= 0]
        return {
            "closed_trades": n,
            "hit_rate": (len(wins) / n) if n else 0.0,
            "net_realized": sum(r["pnl"] for r in realized),
            "avg_win": mean([r["pnl"] for r in wins]) if wins else 0.0,
            "avg_loss": mean([r["pnl"] for r in losses]) if losses else 0.0,
            "recent": realized[-self.summary_trades:],
        }

    def performance_summary(self) -> str:
        """Compact, bounded text for the AI overlay's context (self-awareness)."""
        if not self.enabled:
            return ""
        s = self.stats()
        if s["closed_trades"] == 0:
            return "Trade journal: no closed trades yet (no realized P&L history to learn from)."
        recent = "; ".join(
            f"{r['ticker']} {r['ret_pct']:+.1f}% ({_short(r['reason'], 24)})"
            for r in s["recent"][-6:]
        )
        return (
            "Your realized track record so far: {n} closed trades, hit-rate {hr:.0%}, "
            "net ${net:+.2f}, avg win ${aw:+.2f} / avg loss ${al:+.2f}. "
            "Recent exits: {recent}. Weigh this when judging similar setups."
        ).format(n=s["closed_trades"], hr=s["hit_rate"], net=s["net_realized"],
                 aw=s["avg_win"], al=s["avg_loss"], recent=recent or "—")

    def write_memory_md(self, account=None) -> None:
        if not self.enabled:
            return
        s = self.stats()
        lines = ["# rh-agent memory", "", f"_updated {utcnow().isoformat()}_", "",
                 "## Realized performance", ""]
        if s["closed_trades"]:
            lines += [
                f"- Closed trades: **{s['closed_trades']}**",
                f"- Hit rate: **{s['hit_rate']:.0%}**",
                f"- Net realized P&L: **${s['net_realized']:+.2f}**",
                f"- Avg win / loss: ${s['avg_win']:+.2f} / ${s['avg_loss']:+.2f}",
            ]
        else:
            lines.append("- No closed trades yet.")
        if account is not None and getattr(account, "positions", None):
            lines += ["", "## Current holdings", "",
                      "| Ticker | Qty | Avg | Last | Unreal P&L |",
                      "|---|---|---|---|---|"]
            for p in account.positions:
                lines.append(
                    f"| {p.ticker} | {p.quantity:.4f} | {p.avg_price:.2f} | "
                    f"{p.current_price:.2f} | {getattr(p, 'unrealized_pnl', 0) or 0:+.2f} |")
        if s["recent"]:
            lines += ["", "## Recent closed trades", "",
                      "| Ticker | Return | Reason |", "|---|---|---|"]
            for r in reversed(s["recent"][-12:]):
                lines.append(f"| {r['ticker']} | {r['ret_pct']:+.1f}% | {_short(r['reason'])} |")
        try:
            write_private(self.memory_path, "\n".join(lines) + "\n")
        except Exception as e:
            log.warning("memory.md write failed: %s", e)
