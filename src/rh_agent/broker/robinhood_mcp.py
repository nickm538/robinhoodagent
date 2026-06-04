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
    for r in (agentic or rows):
        num = (r.get("account_number") or r.get("rhs_account_number")
               or r.get("account_id") or r.get("id"))
        if num:
            return str(num)
    return None


def parse_account(prof, positions_raw, account_number: str | None) -> Account:
    bp = _walk_num(prof, "buying_power") if prof else None
    equity = (_walk_num(prof, "total_equity", "portfolio_value", "total_market_value",
                        "market_value", "equity") if prof else None)
    cash = _walk_num(prof, "cash", "uninvested", "settled_funds") if prof else None

    positions = []
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
    if not equity:
        equity = (cash or 0.0) + sum(p.market_value for p in positions)
    return Account(equity=round(equity or 0.0, 2), cash=round(cash or 0.0, 2),
                   buying_power=round(bp if bp is not None else (cash or equity or 0.0), 2),
                   positions=positions, account_number=account_number or "agentic",
                   source="robinhood")


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

    if not dry_run:                                              # idempotency key for live places
        import uuid
        args["ref_id"] = str(uuid.uuid4())
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

    # ------------------------------------------------------------------ account
    def get_account(self) -> Account:
        acct_no = self.account_number
        prof = positions_raw = None
        try:
            prof = self._call("buying_power", {} if not acct_no else {"account_number": acct_no})
        except Exception as e:
            log.warning("buying_power read failed: %s", e)
        try:
            positions_raw = self._call("positions", {} if not acct_no else {"account_number": acct_no})
        except Exception as e:
            log.warning("positions read failed: %s", e)
        return parse_account(prof, positions_raw, acct_no)

    # ------------------------------------------------------------------ orders
    def place_order(self, order: Order, dry_run: bool = True) -> dict:
        if dry_run:
            # If the live tool can't preview, return our own preview rather than risk a send.
            return {"status": "preview", "ticker": order.ticker, "side": order.side,
                    "qty": order.quantity, "note": "dry_run (not transmitted unless tool previews)"}
        res = self._call("place_order", order_args(order, dry_run, self.account_number))
        return {"status": "submitted", "ticker": order.ticker, "result": res}

    def get_orders(self) -> list:
        try:
            return self._call("orders", {}) or []
        except Exception:
            return []

    def cancel_all(self) -> None:
        try:
            for o in self.get_orders():
                oid = o.get("id") if isinstance(o, dict) else None
                if oid:
                    self._call("cancel", {"order_id": oid, "id": oid})
        except Exception as e:
            log.warning("cancel_all failed: %s", e)
