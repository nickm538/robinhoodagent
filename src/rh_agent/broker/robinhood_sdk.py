"""Durable Robinhood broker built on the official ``mcp`` SDK.

Unlike the lightweight token broker, this one authenticates via the persisted
OAuth session (see oauth.py) and lets the SDK **refresh the access token
automatically**, so an always-on bot keeps trading without manual re-auth.
Run ``rh-agent auth`` once on the host, then this broker just works.
"""
from __future__ import annotations

import asyncio
import json

from ..logging_setup import get_logger
from ..models import Account, Order
from .base import Broker
from .oauth import FileTokenStorage, _require_sdk, make_provider
from .robinhood_mcp import discover_tool_map, order_args, parse_account

log = get_logger("broker.rh_sdk")


def _extract(res):
    """Pull a plain value out of an mcp CallToolResult."""
    if getattr(res, "isError", False):
        raise RuntimeError(f"tool error: {getattr(res, 'content', res)}")
    sc = getattr(res, "structuredContent", None)
    if sc:
        return sc
    out = []
    for block in getattr(res, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            out.append(json.loads(text))
        except Exception:
            out.append(text)
    return out[0] if len(out) == 1 else (out or None)


class RobinhoodSDKBroker(Broker):
    name = "robinhood"
    supports_live = True

    def __init__(self, url: str, account_number: str | None = None):
        _require_sdk()
        if not FileTokenStorage().has_tokens():
            raise RuntimeError("No Robinhood OAuth tokens found. Run `rh-agent auth` first.")
        self.url = url
        self.account_number = account_number
        self._map: dict | None = None
        log.info("Robinhood SDK broker ready (auto-refreshing OAuth)")

    # -- async plumbing --
    async def _session(self, coro_fn):
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        provider = make_provider(self.url, interactive=False)
        async with streamablehttp_client(self.url, auth=provider) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if self._map is None:
                    tools = await session.list_tools()
                    self._map = discover_tool_map([t.name for t in tools.tools])
                    log.info("discovered RH tools: %s", self._map)
                return await coro_fn(session)

    def _run(self, coro_fn):
        return asyncio.run(self._session(coro_fn))

    async def _call(self, session, cap: str, args: dict | None = None):
        tool = (self._map or {}).get(cap)
        if not tool:
            raise RuntimeError(f"no Robinhood MCP tool for capability '{cap}'")
        return _extract(await session.call_tool(tool, args or {}))

    # -- Broker API --
    def get_account(self) -> Account:
        async def fn(session):
            a = {} if not self.account_number else {"account_number": self.account_number}
            prof = pos = None
            try:
                prof = await self._call(session, "buying_power", a)
            except Exception as e:
                log.warning("buying_power read failed: %s", e)
            try:
                pos = await self._call(session, "positions", a)
            except Exception as e:
                log.warning("positions read failed: %s", e)
            return parse_account(prof, pos, self.account_number)
        return self._run(fn)

    def place_order(self, order: Order, dry_run: bool = True) -> dict:
        if dry_run:
            return {"status": "preview", "ticker": order.ticker, "side": order.side,
                    "qty": order.quantity, "note": "dry_run (not transmitted)"}

        async def fn(session):
            return await self._call(session, "place_order",
                                    order_args(order, False, self.account_number))
        return {"status": "submitted", "ticker": order.ticker, "result": self._run(fn)}

    def get_orders(self) -> list:
        async def fn(session):
            try:
                return await self._call(session, "orders", {}) or []
            except Exception:
                return []
        return self._run(fn)
