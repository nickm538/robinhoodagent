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
_PATTERNS = {
    "place_order": [r"place_stock_order", r"buy_stock", r"place_order", r"place_trade",
                    r"submit_order", r"create_order", r"order_stock"],
    "positions":   [r"get_stock_positions", r"get_positions", r"get_portfolio", r"positions"],
    "accounts":    [r"get_accounts?$", r"account_profile", r"get_account\b"],
    "buying_power":[r"buying_power", r"get_account_balance", r"account_profile", r"get_account\b"],
    "orders":      [r"get_orders", r"recent_stock_orders", r"open_orders", r"order_history"],
    "cancel":      [r"cancel_stock_order", r"cancel_order", r"cancel"],
    "quote":       [r"get_stock_quote", r"get_quote\b"],
}


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
        names = list(self.tools)
        mp = {}
        for cap, pats in _PATTERNS.items():
            for pat in pats:
                hit = next((n for n in names if re.search(pat, n, re.I)), None)
                if hit:
                    mp[cap] = hit
                    break
        return mp

    def _call(self, cap: str, args: dict | None = None):
        tool = self.map.get(cap)
        if not tool:
            raise MCPError(f"no Robinhood MCP tool for capability '{cap}'")
        return self.client.call_tool(tool, args or {})

    # ------------------------------------------------------------------ account
    def get_account(self) -> Account:
        acct_no = self.account_number
        cash = bp = equity = 0.0
        try:
            prof = self._call("buying_power", {} if not acct_no else {"account_number": acct_no})
            d = prof[0] if isinstance(prof, list) and prof else prof
            if isinstance(d, dict):
                cash = float(d.get("cash") or d.get("buying_power") or 0)
                bp = float(d.get("buying_power") or cash)
                equity = float(d.get("equity") or d.get("portfolio_equity") or 0)
        except Exception as e:
            log.warning("buying_power read failed: %s", e)
        positions = []
        try:
            raw = self._call("positions", {} if not acct_no else {"account_number": acct_no})
            for p in (raw if isinstance(raw, list) else raw.get("positions", []) if isinstance(raw, dict) else []):
                if not isinstance(p, dict):
                    continue
                qty = float(p.get("quantity") or p.get("shares") or 0)
                if qty == 0:
                    continue
                px = float(p.get("price") or p.get("current_price") or p.get("last_price") or 0)
                positions.append(Position(
                    ticker=p.get("symbol") or p.get("ticker"),
                    quantity=qty, avg_price=float(p.get("average_buy_price") or p.get("avg_price") or 0),
                    current_price=px, market_value=round(qty * px, 2)))
        except Exception as e:
            log.warning("positions read failed: %s", e)
        if equity == 0:
            equity = cash + sum(p.market_value for p in positions)
        return Account(equity=round(equity, 2), cash=round(cash, 2),
                       buying_power=round(bp or cash, 2), positions=positions,
                       account_number=acct_no or "agentic", source="robinhood")

    # ------------------------------------------------------------------ orders
    def place_order(self, order: Order, dry_run: bool = True) -> dict:
        args = {
            "symbol": order.ticker, "ticker": order.ticker,
            "side": order.side, "action": order.side,
            "order_type": order.order_type, "type": order.order_type,
            "time_in_force": order.time_in_force,
            "dry_run": dry_run,
        }
        if order.quantity:
            args["quantity"] = round(order.quantity, 6)
            args["shares"] = round(order.quantity, 6)
        if order.notional:
            args["amount"] = round(order.notional, 2)
            args["notional"] = round(order.notional, 2)
        if order.order_type == "limit" and order.limit_price:
            args["limit_price"] = round(order.limit_price, 2)
            args["price"] = round(order.limit_price, 2)
        if self.account_number:
            args["account_number"] = self.account_number
        if dry_run:
            # If the live tool can't preview, return our own preview rather than risk a send.
            return {"status": "preview", "ticker": order.ticker, "side": order.side,
                    "qty": order.quantity, "note": "dry_run (not transmitted unless tool previews)"}
        res = self._call("place_order", args)
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
