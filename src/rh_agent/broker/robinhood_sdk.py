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
from .mcp_client import validate_mcp_url
from .oauth import FileTokenStorage, _require_sdk, make_provider
from .robinhood_mcp import (
    account_is_agentic,
    discover_tool_map,
    order_args,
    parse_account,
    pick_account_number,
)

log = get_logger("broker.rh_sdk")


def _unwrap_exc(e: BaseException) -> BaseException:
    """Drill into anyio / 3.11 ExceptionGroups to the first underlying error."""
    for _ in range(10):
        subs = getattr(e, "exceptions", None)
        if subs:
            e = subs[0]
        else:
            break
    return e


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
        # Validate the endpoint before anything else (independent of the SDK),
        # so a misconfigured URL fails fast and consistently with the other paths.
        self.url = validate_mcp_url(url)
        _require_sdk()
        if not FileTokenStorage().has_tokens():
            raise RuntimeError("No Robinhood OAuth tokens found. Run `rh-agent auth` first.")
        self.account_number = account_number
        self._map: dict | None = None
        log.info("Robinhood SDK broker ready (auto-refreshing OAuth)")

    # -- async plumbing --
    async def _session(self, coro_fn):
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        provider = make_provider(self.url, interactive=False)
        # terminate_on_close=False: Robinhood rejects the MCP session-cleanup
        # DELETE with a 400, which anyio otherwise surfaces as a TaskGroup crash.
        async with streamablehttp_client(self.url, auth=provider,
                                         terminate_on_close=False) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if self._map is None:
                    tools = await session.list_tools()
                    self._map = discover_tool_map([t.name for t in tools.tools])
                    log.info("discovered RH tools: %s", self._map)
                return await coro_fn(session)

    def _run(self, coro_fn):
        try:
            return asyncio.run(self._session(coro_fn))
        except Exception as e:              # unwrap TaskGroup/ExceptionGroup to the real cause
            raise _unwrap_exc(e) from None  # (Exception still covers anyio ExceptionGroup)

    async def _call(self, session, cap: str, args: dict | None = None):
        tool = (self._map or {}).get(cap)
        if not tool:
            raise RuntimeError(f"no Robinhood MCP tool for capability '{cap}'")
        return _extract(await session.call_tool(tool, args or {}))

    async def _resolve_account(self, session, *, require_agentic: bool = False) -> str | None:
        """Robinhood requires account_number on calls; fetch it once from
        get_accounts (preferring the Agentic account)."""
        if (self._map or {}).get("accounts"):
            try:
                accts = await self._call(session, "accounts", {})
                if self.account_number:
                    if account_is_agentic(accts, self.account_number):
                        return self.account_number
                    raise RuntimeError("configured account_number is not agentic_allowed")
                self.account_number = pick_account_number(accts)
                if self.account_number:
                    log.info("using Robinhood account %s", self.account_number)
                elif require_agentic:
                    log.warning("could not determine an agentic_allowed account from get_accounts")
            except RuntimeError:
                raise
            except Exception as e:
                log.warning("account resolve failed: %s", e)
                if require_agentic:
                    raise
        if require_agentic and not self.account_number:
            raise RuntimeError("no agentic_allowed account available for live order placement")
        return self.account_number

    # -- Broker API --
    def get_account(self) -> Account:
        async def fn(session):
            acct = await self._resolve_account(session, require_agentic=False)
            a = {"account_number": acct} if acct else {}
            prof = pos = None
            prof_ok = pos_ok = False
            try:
                prof = await self._call(session, "buying_power", a)   # -> get_portfolio
                prof_ok = prof is not None
            except Exception as e:
                log.warning("portfolio read failed: %s", e)
            try:
                pos = await self._call(session, "positions", a)       # -> get_equity_positions
                pos_ok = True
            except Exception as e:
                log.warning("positions read failed: %s", e)
            return parse_account(prof, pos, acct, portfolio_ok=prof_ok, positions_ok=pos_ok)
        return self._run(fn)

    def place_order(self, order: Order, dry_run: bool = True) -> dict:
        async def fn(session):
            acct = await self._resolve_account(session, require_agentic=not dry_run)
            args = order_args(order, dry_run, acct)
            if dry_run:
                # validate via the broker's review tool if present (no live order)
                if (self._map or {}).get("review_order"):
                    return await self._call(session, "review_order", args)
                return {"note": "no review tool available; nothing transmitted"}
            return await self._call(session, "place_order", args)

        try:
            detail = self._run(fn)
        except Exception as e:                  # one bad order must not crash the whole run
            log.error("place_order %s failed: %s", order.ticker, e)
            return {"status": "error", "ticker": order.ticker, "side": order.side,
                    "error": str(e)}
        return {"status": "preview" if dry_run else "submitted",
                "ticker": order.ticker, "result": detail}

    def get_orders(self) -> list:
        async def fn(session):
            try:
                return await self._call(session, "orders", {}) or []
            except Exception:
                return []
        return self._run(fn)


def _shape(o, depth: int = 0):
    """Structure (field names + types) of a value, WITHOUT the actual values —
    so we can learn response shapes without exposing balances/account numbers."""
    if depth > 5:
        return "..."
    if isinstance(o, dict):
        return {k: _shape(v, depth + 1) for k, v in list(o.items())[:50]}
    if isinstance(o, list):
        return [_shape(o[0], depth + 1)] if o else []
    return type(o).__name__


def probe(url: str, sample_ticker: str = "AAPL") -> dict:
    """Read-only introspection of the Robinhood MCP: every tool's input schema
    plus the *shape* of account/portfolio/positions/orders/quote responses.
    Places NO orders. Run this and share the output to finalise live wiring."""
    url = validate_mcp_url(url)
    return asyncio.run(_probe(url, sample_ticker))


async def _probe(url: str, sample_ticker: str) -> dict:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    out: dict = {"tools": [], "shapes": {}}
    provider = make_provider(url, interactive=False)
    async with streamablehttp_client(url, auth=provider) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tl = await session.list_tools()
            names = [t.name for t in tl.tools]
            for t in tl.tools:
                out["tools"].append({"name": t.name,
                                     "description": (t.description or "")[:240],
                                     "input_schema": t.inputSchema})
            mp = discover_tool_map(names)
            out["capability_map"] = mp

            async def call(tool, args):
                return _extract(await session.call_tool(tool, args or {}))

            acct = None
            if mp.get("accounts"):
                try:
                    a = await call(mp["accounts"], {})
                    out["shapes"]["accounts"] = _shape(a)
                    acct = pick_account_number(a)
                    out["account_resolved"] = bool(acct)
                except Exception as e:
                    out["shapes"]["accounts_error"] = str(e)
            args = {"account_number": acct} if acct else {}
            for cap in ("portfolio", "positions", "orders"):
                if mp.get(cap):
                    try:
                        out["shapes"][cap] = _shape(await call(mp[cap], args))
                    except Exception as e:
                        out["shapes"][f"{cap}_error"] = str(e)
            if mp.get("quote"):
                try:
                    out["shapes"]["quote"] = _shape(
                        await call(mp["quote"], {"symbols": [sample_ticker]}))
                except Exception as e:
                    out["shapes"]["quote_error"] = str(e)
            # simulate a tiny $1 market buy via review (places NO order) to
            # validate that our order builder matches the real schema
            if mp.get("review_order") and acct:
                sample = order_args(Order(sample_ticker, "buy", None, "market", notional=1.0),
                                    dry_run=True, account_number=acct)
                out["review_sample_args"] = {k: v for k, v in sample.items()
                                             if k != "account_number"}
                try:
                    out["shapes"]["review_order"] = _shape(await call(mp["review_order"], sample))
                    out["review_ok"] = True
                except Exception as e:
                    out["shapes"]["review_error"] = str(e)
    return out
