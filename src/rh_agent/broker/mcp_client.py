"""Minimal MCP client over Streamable HTTP (JSON-RPC 2.0).

Enough of the protocol to: initialize a session, list tools, and call tools.
Handles both ``application/json`` and ``text/event-stream`` responses. Auth is
a bearer token (the OAuth access token obtained when the user authorises the
Robinhood Trading MCP). No third-party MCP SDK is required.
"""
from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import urlparse

import requests

from ..logging_setup import get_logger

log = get_logger("mcp")
PROTOCOL_VERSION = "2025-06-18"

# Loopback hosts where plain http is tolerated (local dev / mock MCP servers).
# Only true loopback names — 0.0.0.0 is a bind-all address, not loopback.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class MCPError(Exception):
    pass


def validate_mcp_url(url: str) -> str:
    """Validate the MCP endpoint before any OAuth bearer token is attached.

    The endpoint comes from operator config (``ROBINHOOD_MCP_URL``), but a typo
    or copy-paste mistake could point it at a plaintext or non-HTTP destination
    and leak the bearer token. Require https for remote hosts (http only for
    loopback), and reject anything that is not http(s) with a host.
    """
    if not url or not isinstance(url, str):
        raise MCPError("MCP URL is empty — set ROBINHOOD_MCP_URL.")
    parsed = urlparse(url.strip())
    host = parsed.hostname or ""
    if parsed.scheme not in ("http", "https") or not host:
        raise MCPError(f"MCP URL must be an http(s) URL with a host, got {url!r}.")
    try:
        parsed.port  # raises ValueError on a non-numeric/out-of-range port
    except ValueError as e:
        raise MCPError(f"MCP URL has an invalid port: {url!r}.") from e
    is_loopback = host in _LOCAL_HOSTS
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
    if parsed.scheme == "http" and not is_loopback:
        raise MCPError(
            "MCP URL must use https for remote hosts (the OAuth bearer token "
            f"would otherwise transit in cleartext): {url!r}."
        )
    return url.strip()


class MCPHttpClient:
    def __init__(self, url: str, token: str | None = None, timeout: int = 30):
        self.url = validate_mcp_url(url)
        self.timeout = timeout
        self.session = requests.Session()
        self.session_id: str | None = None
        self._id = 0
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream",
             "MCP-Protocol-Version": PROTOCOL_VERSION}
        if token:
            h["Authorization"] = f"Bearer {token}"
        self.session.headers.update(h)

    # ---- low-level JSON-RPC ----
    def _rpc(self, method: str, params: dict | None = None, *, notify: bool = False) -> Any:
        msg = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._id += 1
            msg["id"] = self._id
        if params is not None:
            msg["params"] = params
        headers = {}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        r = self.session.post(self.url, json=msg, headers=headers, timeout=self.timeout,
                              stream=True)
        if "Mcp-Session-Id" in r.headers:
            self.session_id = r.headers["Mcp-Session-Id"]
        if notify:
            return None
        if r.status_code >= 400:
            raise MCPError(f"{method} -> HTTP {r.status_code}: {r.text[:300]}")
        return self._parse(r)

    def _parse(self, r: requests.Response) -> Any:
        ctype = r.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
                    if isinstance(data, dict) and ("result" in data or "error" in data):
                        return self._result(data)
            raise MCPError("no result in SSE stream")
        return self._result(r.json())

    @staticmethod
    def _result(data: dict) -> Any:
        if "error" in data:
            raise MCPError(str(data["error"]))
        return data.get("result")

    # ---- handshake ----
    def connect(self) -> dict:
        res = self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "rh-agent", "version": "1.0.0"},
        })
        self._rpc("notifications/initialized", notify=True)
        log.info("MCP connected: %s", (res or {}).get("serverInfo", {}))
        return res or {}

    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        while True:
            params = {"cursor": cursor} if cursor else {}
            res = self._rpc("tools/list", params) or {}
            tools.extend(res.get("tools", []))
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        res = self._rpc("tools/call", {"name": name, "arguments": arguments or {}}) or {}
        # MCP returns content blocks; pull structured/text payloads
        if res.get("isError"):
            raise MCPError(f"tool {name} error: {res.get('content')}")
        if "structuredContent" in res:
            return res["structuredContent"]
        out = []
        for block in res.get("content", []):
            if block.get("type") == "text":
                txt = block.get("text", "")
                try:
                    out.append(json.loads(txt))
                except Exception:
                    out.append(txt)
        return out[0] if len(out) == 1 else (out or res)
