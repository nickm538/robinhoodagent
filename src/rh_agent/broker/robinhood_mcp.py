"""RobinhoodMCPBroker — places real orders through the Robinhood Agentic
Trading MCP (https://agent.robinhood.com/mcp/trading).

Robinhood's agent design only lets an agent *trade* inside the separate
"Agentic" account (it can read others but not touch them), which is itself a
safety wall. On top of that, this broker:
  * auto-discovers the server's tools (names differ across MCP builds) and maps
    them to {accounts, positions, buying_power, place_order, cancel, orders};
  * defaults every order to dry-run;
  * refuses to send a live order unless explicitly armed by the caller.

Tool-name auto-discovery matches the official server and the common community
servers (place_stock_order / buy_stock / robinhood_place_trade / ...).
"""
from __future__ import annotations

import re

from ..logging_setup import get_logger
from ..models import Account, Order, Position
from .base import Broker
from .mcp_client import MCPHttpClient, MCPError
from .orders import stable_ref_id

log = get_logger("broker.rh")

# Ordered regex patterns -> logical capability. First matching tool wins.
# Official Robinhood Agentic names (*_equity_*) are listed first, then common
# community-server names as fallbacks.
_PATTERNS = {
    "place_order":  [r"place_equity_order", r"place_stock_order", r"buy_stock",
                     r"place_order", r"place_trade", r"submit_order", r"create_order"],
    "review_order": [r"review_equity_order", r"review_order", r"preview_order"],
    "positions":    [r"get_equity_positions", r"get_stock_positions", r"get_positions", r"positions"],
    "portfolio":    [r"get_portfolio", r"portfolio"],
    "accounts":     [r"get_accounts", r"account_profile", r"get_account\b"],
    "buying_power": [r"get_portfolio", r"buying_power", r"get_account_balance", r"get_accounts"],
    "orders":       [r"get_equity_orders", r"get_orders", r"recent_stock_orders",
                     r"open_orders", r"order_history"],
    "cancel":       [r"cancel_equity_order", r"cancel_stock_order", r"cancel_order", r"cancel"],
    "quote":        [r"get_equity_quotes", r"get_stock_quote", r"get_quote"],
}


def _to_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _unwrap_rows(raw, keys: list[str]) -> list:
    """Pull a list out of Robinhood's {data: {<key>: [...]}} envelopes."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        d = raw.get("data", raw)
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            for k in keys:
                if isinstance(d.get(k), list):
                    return d[k]
    return []


def _walk_num(obj, *subs):
    """First numeric whose key contains any of the substrings (recursive)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and any(s in k.lower() for s in subs):
                return float(v)
            if isinstance(v, str) and any(s in k.lower() for s in subs):
                n = _to_num(v)
                if n is not None:
                    return n
        for v in obj.values():
            r = _walk_num(v, *subs)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _walk_num(v, *subs)
            if r is not None:
                return r
    return None


def pick_account_number(raw) -> str | None:
    """Select the Agentic account (agentic_allowed=true) — the only one the
    agent may trade — from a get_accounts response."""
    rows = [r for r in _unwrap_rows(raw, ["accounts", "results"]) if isinstance(r, dict)]
    if not rows:  # some servers return a single account object, not a list/envelope
        d = raw.get("data", raw) if isinstance(raw, dict) else raw
        if isinstance(d, dict) and (d.get("account_number") or d.get("account_id") or d.get("id")):
            rows = [d]
    agentic = [r for r in rows if r.get("agentic_allowed") is True and not r.get("deactivated")]
    for r in agentic:
        num = (r.get("account_number") or r.get("rhs_account_number")
               or r.get("account_id") or r.get("id"))
        if num:
            return str(num)
    return None


def account_is_agentic(raw, account_number: str | None) -> bool:
    """True when account_number refers to an active Agentic-trading account."""
    if not account_number:
        return False
    rows = [r for r in _unwrap_rows(raw, ["accounts", "results"]) if isinstance(r, dict)]
    if not rows:
        d = raw.get("data", raw) if isinstance(raw, dict) else raw
        if isinstance(d, dict):
            rows = [d]
    for r in rows:
        num = (r.get("account_number") or r.get("rhs_account_number")
               or r.get("account_id") or r.get("id"))
        if num and str(num) == str(account_number):
            return r.get("agentic_allowed") is True and not r.get("deactivated")
    return False


def parse_account(prof, positions_raw, account_number: str | None, *,
                  portfolio_ok: bool = True, positions_ok: bool = True) -> Account:
    # Robinhood get_portfolio: data.total_value / data.cash / data.buying_power.buying_power
    # (all strings). get_equity_positions: data.positions[].
    d = prof.get("data", prof) if isinstance(prof, dict) else {}
    d = d if isinstance(d, dict) else {}
    equity = _to_num(d.get("total_value"))
    if equity is None:
        equity = _walk_num(prof, "total_value", "total_equity", "portfolio_value")
    cash = _to_num(d.get("cash"))
    if cash is None:
        cash = _walk_num(prof, "cash")
    bp_obj = d.get("buying_power")
    bp = _to_num(bp_obj.get("buying_power")) if isinstance(bp_obj, dict) else _to_num(bp_obj)
    if bp is None:
        bp = _walk_num(prof, "buying_power")
    bp_confirmed = bp is not None
    # Never default buying power to total equity — that over-authorizes buys.
    if bp is None:
        bp = cash if cash is not None else 0.0

    positions = []
    if positions_ok and positions_raw is not None:
        for p in _unwrap_rows(positions_raw, ["positions", "results"]):
            if not isinstance(p, dict):
                continue
            qty = _to_num(p.get("quantity") or p.get("shares"))
            if not qty:
                continue
            avg = _to_num(p.get("average_buy_price") or p.get("average_cost")
                          or p.get("cost_basis") or p.get("avg_price")) or 0.0
            px = _to_num(p.get("price") or p.get("current_price") or p.get("last_price")) or avg
            positions.append(Position(ticker=p.get("symbol") or p.get("ticker"), quantity=qty,
                                      avg_price=avg, current_price=px,
                                      market_value=round(qty * px, 2)))
    if portfolio_ok and not equity and positions_ok:
        equity = (cash or 0.0) + sum(p.market_value for p in positions)
    return Account(
        equity=round(equity or 0.0, 2),
        cash=round(cash or 0.0, 2),
        buying_power=round(bp or 0.0, 2),
        positions=positions,
        account_number=account_number or "agentic",
        source="robinhood",
        portfolio_confirmed=portfolio_ok and prof is not None,
        positions_confirmed=positions_ok and positions_raw is not None,
        buying_power_confirmed=bp_confirmed,
    )


def _fmt_num(x) -> str:
    return f"{float(x):.6f}".rstrip("0").rstrip(".") or "0"


def order_args(order: Order, dry_run: bool, account_number: str | None) -> dict:
    """Build args matching Robinhood's place/review_equity_order schema exactly
    (additionalProperties:false — only these keys; quantities are strings)."""
    otype = order.order_type or "market"
    args: dict = {"symbol": order.ticker, "side": order.side, "type": otype,
                  "time_in_force": order.time_in_force or "gfd"}
    if account_number:
        args["account_number"] = account_number

    if otype == "market" and order.side == "buy" and order.notional:
        args["dollar_amount"] = f"{float(order.notional):.2f}"   # fractional $ buy
    elif order.quantity is not None:
        args["quantity"] = _fmt_num(order.quantity)              # share quantity
    elif order.notional and otype == "market":
        args["dollar_amount"] = f"{float(order.notional):.2f}"

    if otype in ("limit", "stop_limit") and order.limit_price:
        args["limit_price"] = f"{float(order.limit_price):.2f}"
        args.pop("dollar_amount", None)                          # limit needs quantity, not $
        if "quantity" not in args and order.quantity is not None:
            args["quantity"] = _fmt_num(order.quantity)

    if not dry_run:
        args["ref_id"] = stable_ref_id(order, account_number)
    return args


def discover_tool_map(names: list[str]) -> dict:
    """Map a server's tool names to logical capabilities via the patterns."""
    mp = {}
    for cap, pats in _PATTERNS.items():
        for pat in pats:
            hit = next((n for n in names if re.search(pat, n, re.I)), None)
            if hit:
                mp[cap] = hit
                break
    return mp


class RobinhoodMCPBroker(Broker):
    name = "robinhood"
    supports_live = True

    def __init__(self, url: str, token: str | None, *, account_number: str | None = None):
        if not token:
            raise MCPError("Robinhood MCP requires an OAuth bearer token "
                           "(set ROBINHOOD_MCP_TOKEN after authorising the MCP).")
        self.client = MCPHttpClient(url, token)
        self.client.connect()
        self.tools = {t["name"]: t for t in self.client.list_tools()}
        self.map = self._discover()
        self.account_number = account_number
        log.info("Robinhood MCP tools: %d discovered; mapping=%s",
                 len(self.tools), {k: v for k, v in self.map.items()})

    def _discover(self) -> dict:
        return discover_tool_map(list(self.tools))

    def _call(self, cap: str, args: dict | None = None):
        tool = self.map.get(cap)
        if not tool:
            raise MCPError(f"no Robinhood MCP tool for capability '{cap}'")
        return self.client.call_tool(tool, args or {})

    def _resolve_account_number(self, *, require_agentic: bool = False) -> str | None:
        if not self.map.get("accounts"):
            if require_agentic and not self.account_number:
                raise MCPError("no accounts tool available and no account_number configured")
            return self.account_number
        try:
            accts = self._call("accounts", {})
            if self.account_number:
                if account_is_agentic(accts, self.account_number):
                    return self.account_number
                raise MCPError("configured account_number is not agentic_allowed")
            picked = pick_account_number(accts)
            if picked:
                self.account_number = picked
                log.info("using Robinhood account %s", self.account_number)
                return picked
            if require_agentic:
                raise MCPError("no agentic_allowed account found")
        except MCPError:
            raise
        except Exception as e:
            log.warning("account resolve failed: %s", e)
            if require_agentic:
                raise MCPError(f"account resolve failed: {e}") from e
        return self.account_number

    # ------------------------------------------------------------------ account
    def get_account(self) -> Account:
        acct_no = None
        try:
            acct_no = self._resolve_account_number(require_agentic=False)
        except Exception as e:
            log.warning("account resolve failed: %s", e)
        prof = positions_raw = None
        prof_ok = pos_ok = False
        args = {} if not acct_no else {"account_number": acct_no}
        try:
            prof = self._call("buying_power", args)
            prof_ok = prof is not None
        except Exception as e:
            log.warning("buying_power read failed: %s", e)
        try:
            positions_raw = self._call("positions", args)
            pos_ok = True
        except Exception as e:
            log.warning("positions read failed: %s", e)
        return parse_account(prof, positions_raw, acct_no,
                             portfolio_ok=prof_ok, positions_ok=pos_ok)

    # ------------------------------------------------------------------ orders
    def place_order(self, order: Order, dry_run: bool = True) -> dict:
        args = None
        try:
            acct = self._resolve_account_number(require_agentic=not dry_run)
            args = order_args(order, dry_run, acct)
        except Exception as e:
            log.error("live order FAILED %s %s: %s", order.side, order.ticker, e)
            return {"status": "error", "ticker": order.ticker, "error": str(e)}
        if dry_run:
            if self.map.get("review_order"):
                try:
                    detail = self._call("review_order", args)
                    return {"status": "preview", "ticker": order.ticker, "side": order.side,
                            "qty": order.quantity, "result": detail}
                except Exception as e:
                    log.warning("review_order failed: %s", e)
            return {"status": "preview", "ticker": order.ticker, "side": order.side,
                    "qty": order.quantity, "note": "dry_run (not transmitted)"}
        try:
            res = self._call("place_order", args)
        except Exception as e:
            log.error("live order FAILED %s %s: %s", order.side, order.ticker, e)
            return {"status": "error", "ticker": order.ticker, "error": str(e)}
        return {"status": "submitted", "ticker": order.ticker, "result": res}

    def get_orders(self) -> list:
        try:
            return self._call("orders", {}) or []
        except Exception:
            return []

    def cancel_all(self) -> None:
        try:
            orders = self.get_orders()
        except Exception as e:
            log.warning("cancel_all: could not list open orders: %s", e)
            return
        # Cancel each order independently: this is a safety operation, so one
        # failed cancel must not strand the remaining open orders. (There is no
        # bulk-cancel tool, and the JSON-RPC client is not thread-safe, so these
        # stay sequential.)
        for o in orders:
            oid = o.get("id") if isinstance(o, dict) else None
            if not oid:
                continue
            try:
                self._call("cancel", {"order_id": oid, "id": oid})
            except Exception as e:
                log.warning("cancel_all: failed to cancel order %s: %s", oid, e)
