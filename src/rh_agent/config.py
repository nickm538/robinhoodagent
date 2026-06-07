"""Configuration loading: YAML strategy file + environment secrets.

Secrets (API keys, tokens) come ONLY from the environment / .env — never
from the committed YAML. The YAML holds strategy parameters only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

# Map provider name -> environment variable holding its key. We accept several
# spellings because different hosts pre-seed differently.
_KEY_ENV = {
    "financialdatasets": ["FINANCIALDATASETS_API_KEY", "FinancialDatasetsAI_API_KEY",
                           "FinancialDatasets_API_Key", "FINANCIAL_DATASETS_API_KEY"],
    "firecrawl": ["FIRECRAWL_API_KEY"],
    "mboum": ["MBOUM_API_KEY", "MBOUM_KEY"],
    "alphavantage": ["ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY", "AV_API_KEY"],
    "twelvedata": ["TWELVEDATA_API_KEY", "TwelveData_API_KEY", "TWELVE_DATA_API_KEY"],
    "exa": ["EXA_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


def _deep_get(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


class Config:
    def __init__(self, data: dict, path: Path | None = None):
        self._d = data
        self.path = path

    # -- strategy params --
    def get(self, dotted: str, default: Any = None) -> Any:
        return _deep_get(self._d, dotted, default)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    @property
    def raw(self) -> dict:
        return self._d

    # -- secrets --
    @staticmethod
    def api_key(provider: str) -> str | None:
        for env_name in _KEY_ENV.get(provider, []):
            val = os.getenv(env_name)
            if val:
                return val.strip().strip("'").strip('"')
        return None

    @staticmethod
    def robinhood_token() -> str | None:
        tok = os.getenv("ROBINHOOD_MCP_TOKEN")
        if tok:
            return tok.strip()
        f = os.getenv("ROBINHOOD_MCP_TOKEN_FILE")
        if f and Path(f).expanduser().exists():
            return Path(f).expanduser().read_text().strip()
        return None

    @staticmethod
    def robinhood_url() -> str:
        return os.getenv("ROBINHOOD_MCP_URL", "https://agent.robinhood.com/mcp/trading").strip()

    @property
    def execution_mode(self) -> str:
        return os.getenv("EXECUTION_MODE", self.get("execution.mode", "live")).lower()

    @property
    def live_trading_armed(self) -> bool:
        """Live orders require explicit mode+confirmation alignment."""
        confirm = os.getenv("LIVE_TRADING_CONFIRM", self.get("execution.live_trading_confirm", ""))
        return (
            self.execution_mode == "live"
            and confirm == "I_UNDERSTAND_REAL_MONEY"
        )

    def available_providers(self) -> dict:
        """Which data providers actually have keys present."""
        return {p: bool(self.api_key(p)) for p in _KEY_ENV}


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return Config(data, p)
