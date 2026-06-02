"""Robinhood OAuth for the standalone bot.

Runs the MCP OAuth flow once (browser approval on the machine that will run the
bot) and persists the tokens to ``state/robinhood_oauth.json`` via the official
``mcp`` SDK token storage. After that the bot authenticates non-interactively
and the SDK refreshes the access token automatically — true hands-off operation.

Requires the optional dependency:  pip install "mcp>=1.2.0"  (the ``live`` extra).
"""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ..config import REPO_ROOT
from ..logging_setup import get_logger

log = get_logger("oauth")
TOKEN_FILE = REPO_ROOT / "state" / "robinhood_oauth.json"


def _require_sdk():
    try:
        import mcp  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The Robinhood live broker needs the MCP SDK. Install it with:\n"
            '    pip install "mcp>=1.2.0"\n'
            "(or `pip install -e \".[live]\"`)."
        ) from e


class FileTokenStorage:
    """Implements the mcp SDK ``TokenStorage`` protocol against a JSON file."""

    def __init__(self, path: Path = TOKEN_FILE):
        self.path = Path(path)

    def _load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=2, default=str))
        try:
            self.path.chmod(0o600)
        except Exception:
            pass

    def has_tokens(self) -> bool:
        return bool(self._load().get("tokens"))

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        d = self._load().get("tokens")
        return OAuthToken(**d) if d else None

    async def set_tokens(self, tokens) -> None:
        data = self._load()
        data["tokens"] = tokens.model_dump(exclude_none=True, mode="json")
        self._save(data)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        d = self._load().get("client_info")
        return OAuthClientInformationFull(**d) if d else None

    async def set_client_info(self, info) -> None:
        data = self._load()
        data["client_info"] = info.model_dump(exclude_none=True, mode="json")
        self._save(data)


async def _wait_for_callback(port: int):
    """Run a one-shot localhost server to capture the OAuth redirect."""
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h3>rh-agent: authorization received. You can close this tab.</h3>")
            if "code" in params and not fut.done():
                loop.call_soon_threadsafe(
                    fut.set_result, (params["code"][0], params.get("state", [None])[0]))

        def log_message(self, *a):  # silence
            return

    srv = HTTPServer(("localhost", port), Handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    try:
        return await fut
    finally:
        srv.server_close()


def make_provider(server_url: str, port: int = 8765, interactive: bool = True):
    """Build an mcp OAuthClientProvider backed by our file storage."""
    _require_sdk()
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    redirect_uri = f"http://localhost:{port}/callback"

    async def redirect_handler(url: str) -> None:
        print("\n=== Authorize rh-agent with Robinhood ===")
        print("Open this URL in your browser and approve:\n")
        print("  " + url + "\n")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    async def callback_handler():
        if not interactive:
            raise RuntimeError("Robinhood token expired and cannot refresh non-interactively; "
                               "re-run `rh-agent auth`.")
        return await _wait_for_callback(port)

    metadata = OAuthClientMetadata(
        client_name="rh-agent",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=FileTokenStorage(),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


async def _authenticate(server_url: str, port: int = 8765) -> list[str]:
    _require_sdk()
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    provider = make_provider(server_url, port, interactive=True)
    async with streamablehttp_client(server_url, auth=provider) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]


def authenticate(server_url: str, port: int = 8765) -> list[str]:
    """Blocking entry point used by `rh-agent auth`. Returns discovered tool names."""
    return asyncio.run(_authenticate(server_url, port))
