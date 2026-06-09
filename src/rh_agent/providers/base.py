"""Provider base classes: disk cache, rate-limited HTTP client, and the
DataProvider interface every source implements (partially).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from ..config import REPO_ROOT
from ..logging_setup import get_logger
from ..models import Quote

log = get_logger("providers")

CACHE_DIR = Path(os.getenv("RH_CACHE_DIR", REPO_ROOT / "data_cache"))
OFFLINE = os.getenv("RH_OFFLINE", "0") in ("1", "true", "True")


class ProviderUnsupported(Exception):
    """Raised when a provider does not implement a requested data section."""


class ProviderError(Exception):
    pass


class DiskCache:
    """Tiny JSON disk cache. Keys are hashed; values carry capture time."""

    def __init__(self, root: Path = CACHE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()[:24]
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{h}.json"

    def get(self, namespace: str, key: str, ttl_minutes: float | None) -> Optional[Any]:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        try:
            blob = json.loads(p.read_text())
        except Exception:
            return None
        if ttl_minutes is not None and not OFFLINE:
            ts = blob.get("_captured_at", 0)
            if (time.time() - ts) > ttl_minutes * 60:
                return None
        return blob.get("data")

    def set(self, namespace: str, key: str, data: Any, source: str = "") -> None:
        p = self._path(namespace, key)
        payload = {"_captured_at": time.time(),
                   "_captured_iso": datetime.now(timezone.utc).isoformat(),
                   "_source": source, "data": data}
        try:
            p.write_text(json.dumps(payload, default=str))
        except Exception as e:  # pragma: no cover
            log.debug("cache write failed: %s", e)


class RateLimiter:
    def __init__(self, max_per_sec: float):
        self.min_interval = 1.0 / max_per_sec if max_per_sec > 0 else 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            delta = time.time() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.time()


class HttpClient:
    def __init__(self, base_url: str, *, max_per_sec: float = 5, timeout: int = 20,
                 default_headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.limiter = RateLimiter(max_per_sec)
        self.session = requests.Session()
        if default_headers:
            self.session.headers.update(default_headers)
        self.session.headers.setdefault("User-Agent", "rh-agent/1.0")

    def get_json(self, path: str, params: dict | None = None, *, headers: dict | None = None,
                 retries: int = 3) -> Any:
        if OFFLINE:
            raise ProviderError("RH_OFFLINE set: refusing live HTTP request")
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        last_err: Exception | None = None
        for attempt in range(retries):
            self.limiter.wait()
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:  # network / decode
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        raise ProviderError(f"GET {url} failed after {retries} tries: {last_err}")

    def post_json(self, path: str, payload: dict | None = None, *, headers: dict | None = None,
                  retries: int = 3) -> Any:
        if OFFLINE:
            raise ProviderError("RH_OFFLINE set: refusing live HTTP request")
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        last_err: Exception | None = None
        for attempt in range(retries):
            self.limiter.wait()
            try:
                r = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:  # network / decode
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        raise ProviderError(f"POST {url} failed after {retries} tries: {last_err}")


class DataProvider:
    """Interface. Subclasses implement whichever sections they support."""

    name: str = "base"
    enabled: bool = True

    def __init__(self, cache: DiskCache | None = None):
        self.cache = cache or DiskCache()

    # -- universe --
    def list_universe(self) -> list[str]:
        raise ProviderUnsupported

    def get_market_movers(self, limit: int = 60) -> list[str]:
        """Today's top gainers / most-active symbols (intraday discovery)."""
        raise ProviderUnsupported

    # -- per ticker sections --
    def get_company(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_quote(self, ticker: str) -> Quote:
        raise ProviderUnsupported

    def get_prices(self, ticker: str, start: str | None = None, end: str | None = None,
                   interval: str = "day") -> pd.DataFrame:
        raise ProviderUnsupported

    def get_fundamentals(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_technicals(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_insider(self, ticker: str) -> list:
        raise ProviderUnsupported

    def get_institutional(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_news_sentiment(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_analyst(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_short_interest(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_options(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_earnings(self, ticker: str) -> dict:
        raise ProviderUnsupported

    def get_macro(self) -> dict:
        raise ProviderUnsupported


def prices_to_df(records: list[dict]) -> pd.DataFrame:
    """Normalise a list of OHLCV dicts into a clean, date-indexed DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # accept many column spellings
    rename = {
        "t": "time", "timestamp": "time", "date": "time", "datetime": "time",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "adjClose": "adj_close", "adjusted_close": "adj_close",
        "Adj Close": "adj_close", "5. adjusted close": "adj_close",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "time" not in df.columns:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=False)
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    return df[keep].dropna(how="all")
