"""Massive (formerly Polygon.io) provider — paid, high-throughput market data.

Docs: https://massive.com/docs/rest/stocks/overview — base https://api.massive.com
(api.polygon.io stays valid through the transition; override via MASSIVE_BASE_URL).
Auth via ``Authorization: Bearer <key>`` so the key never appears in URLs,
cache keys, or logs.

NICHE (per the desk's source-synergy mandate): paid Massive plans have no
request-count cap, so this provider absorbs the high-frequency per-name scan
traffic that throttles the per-minute FinancialDatasets plan and drained the
monthly-capped Mboum plan. It expertly serves: bulk snapshot quotes (one call
warms the whole intraday radar), deep daily history (aggregates), the
top-gainers movers feed, FINRA short interest (restores the squeeze factor
while Mboum is quota-capped), news with per-ticker sentiment insights, ticker
reference (universe fallback + company facts), and indicator enrichment.

Entitlement-aware: endpoints outside the subscribed plan (HTTP 403) are
remembered per section and skipped instantly afterwards — the priority chain
falls through with zero wasted round trips.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..logging_setup import get_logger
from ..models import Quote
from .base import (DataProvider, DiskCache, HttpClient, ProviderError,
                   ProviderUnsupported, RateLimitError, env_float, prices_to_df)

log = get_logger("massive")

BASE = os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")

# Paid plans are uncapped on request count; this only paces burst politeness.
# env_float: a malformed .env line must never abort daemon startup at import.
DEFAULT_MAX_PER_SEC = env_float("MASSIVE_MAX_PER_SEC", 20.0, minimum=0.1)
RATE_LIMIT_COOLDOWN_SECONDS = env_float("MASSIVE_RATE_LIMIT_COOLDOWN_SECONDS", 60.0,
                                        minimum=1.0)

_SNAPSHOT_CHUNK = 100          # tickers per filtered-snapshot batch request
_UNIVERSE_MAX_PAGES = 6        # 1000/page — plenty above universe.scan_cap

_SENTIMENT_VALUE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


class MassiveProvider(DataProvider):
    name = "massive"

    def __init__(self, api_key: str, cache: DiskCache | None = None, *,
                 max_per_sec: float | None = None):
        super().__init__(cache)
        rate = max_per_sec if max_per_sec and max_per_sec > 0 else DEFAULT_MAX_PER_SEC
        self.http = HttpClient(BASE, max_per_sec=rate,
                               default_headers={"Authorization": f"Bearer {api_key}",
                                                "Accept": "application/json"})
        self._rate_limited_until = 0.0
        self._unsupported: set[str] = set()   # sections the plan is not entitled to

    # ------------------------------------------------------------- guardrails
    def _rate_limit_active(self) -> bool:
        return time.time() < self._rate_limited_until

    def _enter_cooldown(self, seconds: float | None, reason: str) -> None:
        cooldown = float(seconds) if seconds and seconds > 0 else RATE_LIMIT_COOLDOWN_SECONDS
        until = time.time() + cooldown
        if until > self._rate_limited_until:
            self._rate_limited_until = until
        log.warning("massive rate limited (%s) — cooling down %.0fs", reason,
                    max(self._rate_limited_until - time.time(), 0.0))

    def _q(self, section: str, ttl: float, path: str, params: dict | None = None) -> Any:
        if section in self._unsupported:
            raise ProviderUnsupported(f"massive plan not entitled to {section}")
        params = params or {}
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"massive/{section}", key, ttl)
        if hit is not None:
            return hit
        if self._rate_limit_active():
            raise ProviderUnsupported("massive cooling down after rate limit")
        try:
            data = self.http.get_json(path, params)
        except RateLimitError as e:
            self._enter_cooldown(e.retry_after_seconds, path)
            raise ProviderUnsupported("massive rate limited") from e
        except ProviderError as e:
            # Plan-gated endpoint: remember and stop probing this section.
            if "403" in str(e):
                self._unsupported.add(section)
                log.info("massive: %s not in plan (403) — section disabled this run", section)
            raise ProviderUnsupported(str(e)) from e
        self.cache.set(f"massive/{section}", key, data, source=self.name)
        return data

    # ------------------------------------------------------------------ quote
    @staticmethod
    def _parse_snapshot_ticker(tk: dict) -> Quote | None:
        if not isinstance(tk, dict):
            return None
        symbol = str(tk.get("ticker") or "").upper()
        if not symbol:
            return None
        last_trade = tk.get("lastTrade") or {}
        minute = tk.get("min") or {}
        day = tk.get("day") or {}
        prev = tk.get("prevDay") or {}
        price = (_num(last_trade.get("p")) or _num(minute.get("c"))
                 or _num(day.get("c")) or _num(prev.get("c")))
        if not price or price <= 0:
            return None
        asof = None
        upd = _num(tk.get("updated"))
        if upd and upd > 0:
            try:  # nanosecond epoch
                asof = datetime.fromtimestamp(upd / 1e9, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                asof = None
        q = Quote(ticker=symbol, price=price,
                  volume=_num(day.get("v")) or _num(minute.get("av")) or 0,
                  prev_close=_num(prev.get("c")),
                  day_change_pct=_num(tk.get("todaysChangePerc")),
                  source="massive")
        if asof is not None:
            # honest timestamp: on delayed plans the freshness gates downstream
            # (stale_quote flag, risk-quote max age) see the true data age.
            q.asof = asof
        return q

    def get_quote(self, ticker: str) -> Quote:
        symbol = ticker.upper()
        d = self._q("snap", 1, f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        q = self._parse_snapshot_ticker((d or {}).get("ticker") or {})
        if q is None:
            raise ProviderUnsupported
        return q

    def invalidate_quote(self, ticker: str) -> None:
        symbol = ticker.upper()
        key = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}|[]"
        self.cache._path("massive/snap", key).unlink(missing_ok=True)

    def get_quotes_batch(self, tickers: list[str]) -> dict[str, Quote]:
        """Filtered full-market snapshot: one request warms up to 100 radar
        quotes — the cheap bulk feed the intraday hunter wants."""
        out: dict[str, Quote] = {}
        symbols = [t.strip().upper() for t in tickers if t and t.strip()]
        for i in range(0, len(symbols), _SNAPSHOT_CHUNK):
            chunk = symbols[i:i + _SNAPSHOT_CHUNK]
            try:
                d = self._q("snapbatch", 0.5, "/v2/snapshot/locale/us/markets/stocks/tickers",
                            {"tickers": ",".join(chunk)})
            except Exception as e:
                if self._rate_limit_active():
                    log.warning("massive batch stopped during cooldown after %d/%d quotes",
                                len(out), len(symbols))
                    break
                log.debug("massive batch chunk failed: %s", e)
                continue
            for tk in (d or {}).get("tickers") or []:
                q = self._parse_snapshot_ticker(tk)
                if q:
                    out[q.ticker] = q
        return out

    # ----------------------------------------------------------------- prices
    def get_prices(self, ticker: str, start: str | None = None, end: str | None = None,
                   interval: str = "day") -> pd.DataFrame:
        symbol = ticker.upper()
        timespan = {"day": "day", "week": "week", "month": "month"}.get(interval, "day")
        end_d = end or datetime.now(timezone.utc).date().isoformat()
        start_d = start or (datetime.now(timezone.utc).date() - timedelta(days=750)).isoformat()
        d = self._q("history", 720,
                    f"/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_d}/{end_d}",
                    {"adjusted": "true", "sort": "asc", "limit": 50000})
        rows = (d or {}).get("results") or []
        if not rows:
            raise ProviderUnsupported
        records = []
        for r in rows:
            t_ms = _num(r.get("t"))
            if t_ms is None:
                continue
            # epoch-ms must be converted explicitly (prices_to_df would read raw
            # ints as nanoseconds and produce 1970 dates)
            records.append({
                "time": datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
                .date().isoformat(),
                "open": r.get("o"), "high": r.get("h"), "low": r.get("l"),
                "close": r.get("c"), "volume": r.get("v"),
            })
        df = prices_to_df(records)   # adjusted=true -> adj_close mirrors close
        if df is None or not len(df):
            raise ProviderUnsupported
        return df

    # ---------------------------------------------------------------- company
    def get_company(self, ticker: str) -> dict:
        symbol = ticker.upper()
        d = self._q("facts", 1440, f"/v3/reference/tickers/{symbol}")
        res = (d or {}).get("results") or {}
        if not res:
            raise ProviderUnsupported
        # Massive/Polygon reference has no GICS sector — only SIC. Use the SIC
        # description for BOTH sector and industry so company facts can route here
        # (flat-rate) and still drive the portfolio's sector-concentration cap with
        # a consistent grouping (every Massive-sourced name buckets the same way).
        sic = res.get("sic_description")
        return {
            "name": res.get("name"),
            "sector": sic,
            "industry": sic,
            "market_cap": _num(res.get("market_cap")),
            "exchange": res.get("primary_exchange"),
            "employees": res.get("total_employees"),
            "shares_outstanding": _num(res.get("weighted_shares_outstanding"))
            or _num(res.get("share_class_shares_outstanding")),
            "source": self.name,
        }

    # ------------------------------------------------------------------- news
    def _news(self, ticker: str | None, limit: int) -> list[dict]:
        params: dict = {"limit": max(1, min(limit, 50)), "order": "desc"}
        if ticker:
            params["ticker"] = ticker.upper()
        d = self._q("news", 60, "/v2/reference/news", params)
        return (d or {}).get("results") or []

    def get_headlines(self, ticker: str | None = None, limit: int = 8) -> list:
        out: list[str] = []
        for art in self._news(ticker, limit):
            title = art.get("title") if isinstance(art, dict) else None
            if title and title not in out:
                out.append(title)
            if len(out) >= limit:
                break
        return out

    def get_news_sentiment(self, ticker: str) -> dict:
        """Per-ticker sentiment from article insights, mapped to the same
        -1..1 scale Alpha Vantage uses so cross-sectional ranks stay coherent."""
        symbol = ticker.upper()
        values: list[float] = []
        for art in self._news(symbol, 50):
            for ins in (art.get("insights") or []):
                if str(ins.get("ticker", "")).upper() != symbol:
                    continue
                v = _SENTIMENT_VALUE.get(str(ins.get("sentiment", "")).lower())
                if v is not None:
                    values.append(v)
        if not values:
            raise ProviderUnsupported
        score = sum(values) / len(values)
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return {"score": round(score, 3), "article_count": len(values),
                "label": label, "source": self.name}

    # ----------------------------------------------------------------- movers
    def get_market_movers(self, limit: int = 60) -> list[str]:
        """Today's top gainers snapshot (long-only hunter: losers excluded)."""
        d = self._q("movers", 3, "/v2/snapshot/locale/us/markets/stocks/gainers")
        out: list[str] = []
        for tk in (d or {}).get("tickers") or []:
            t = str(tk.get("ticker") or "") if isinstance(tk, dict) else ""
            if t and t.isalpha() and t not in out:
                out.append(t)
            if len(out) >= limit:
                break
        if not out:
            raise ProviderUnsupported
        return out

    # ------------------------------------------------------------- technicals
    def get_technicals(self, ticker: str) -> dict:
        symbol = ticker.upper()
        out: dict = {}

        def _values(path: str, params: dict) -> list[dict]:
            d = self._q("indicators", 240, path, params)
            return ((d or {}).get("results") or {}).get("values") or []

        base = {"timespan": "day", "series_type": "close", "order": "desc", "limit": 1}
        try:
            vals = _values(f"/v1/indicators/rsi/{symbol}", dict(base, window=14))
            if vals:
                out["rsi"] = _num(vals[0].get("value"))
        except ProviderUnsupported:
            raise
        except Exception:
            pass
        for win, key in ((9, "ema_9"), (21, "ema_21")):
            try:
                vals = _values(f"/v1/indicators/ema/{symbol}", dict(base, window=win))
                if vals:
                    out[key] = _num(vals[0].get("value"))
            except Exception:
                pass
        try:
            vals = _values(f"/v1/indicators/macd/{symbol}",
                           dict(base, short_window=12, long_window=26, signal_window=9))
            if vals:
                out["macd"] = _num(vals[0].get("value"))
                out["macd_signal"] = _num(vals[0].get("signal"))
                out["macd_hist"] = _num(vals[0].get("histogram"))
        except Exception:
            pass
        out = {k: v for k, v in out.items() if v is not None}
        if not out:
            raise ProviderUnsupported
        return out

    # --------------------------------------------------------- short interest
    def get_short_interest(self, ticker: str) -> dict:
        """FINRA short interest — restores the squeeze factor while Mboum's
        monthly quota is exhausted. short_pct_float is a fraction (Yahoo-style,
        e.g. 0.052) computed against shares outstanding when available."""
        symbol = ticker.upper()
        d = self._q("short", 1440, "/stocks/v1/short-interest",
                    {"ticker": symbol, "limit": 1, "sort": "settlement_date", "order": "desc"})
        rows = (d or {}).get("results") or []
        if not rows:
            raise ProviderUnsupported
        r = rows[0]
        short_shares = _num(r.get("short_interest"))
        out = {
            "short_shares": short_shares,
            "days_to_cover": _num(r.get("days_to_cover")),
            "avg_daily_volume": _num(r.get("avg_daily_volume")),
            "settlement_date": r.get("settlement_date"),
            "source": self.name,
        }
        if short_shares:
            try:
                shares_out = (self.get_company(symbol) or {}).get("shares_outstanding")
            except Exception:
                shares_out = None
            if shares_out and shares_out > 0:
                out["short_pct_float"] = round(short_shares / shares_out, 4)
        return out

    # --------------------------------------------------------------- universe
    def list_universe(self, max_pages: int = _UNIVERSE_MAX_PAGES) -> list[str]:
        """Active US common stocks from the reference tickers feed (paginated)."""
        out: list[str] = []
        path: str = "/v3/reference/tickers"
        params: dict | None = {"market": "stocks", "active": "true",
                               "type": "CS", "limit": 1000}
        for _ in range(max_pages):
            d = self._q("tickers", 1440, path, params)
            for row in (d or {}).get("results") or []:
                t = str(row.get("ticker") or "")
                if t and t.isalpha() and t == t.upper():
                    out.append(t)
            nxt = (d or {}).get("next_url")
            if not nxt:
                break
            path, params = nxt, None     # cursor URL carries the continuation
        return out
