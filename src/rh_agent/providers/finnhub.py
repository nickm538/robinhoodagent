"""Finnhub provider — free, broad-coverage resilience layer.

Docs: https://finnhub.io/docs/api — base https://finnhub.io/api/v1, auth via the
``X-Finnhub-Token`` header (kept out of URLs/logs). Free tier ≈ 60 calls/min.

NICHE (per the desk's source-synergy mandate): a zero-cost SAFETY NET that keeps
the smart-money and sentiment pillars alive when the paid sources throttle or
hit quotas. It is wired as a LATE fallback everywhere — except analyst ratings,
where it sits right behind Mboum and RESTORES that pillar while Mboum's monthly
quota is exhausted (Finnhub's recommendation-trends are free).

Deliberately NOT wired for:
  * prices — Finnhub moved historical candles to a paid tier; free 403s, so it
    must never enter the price/technicals chains (would starve momentum factors).
  * fundamentals ratios — Finnhub reports margins/growth/ROE in PERCENT while the
    rest of the stack uses fractions; mixing conventions would corrupt the
    cross-sectional quality ranks. Held out until units are live-verified
    (get_fundamentals intentionally not implemented -> base raises Unsupported).
    Company facts (name/sector/market cap, absolute values) ARE provided.
  * news sentiment SCORES — the free tier has headlines but no usable score.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from ..logging_setup import get_logger
from ..models import Quote
from .base import (DataProvider, DiskCache, HttpClient, ProviderUnsupported,
                   RateLimitError, env_float)

log = get_logger("finnhub")

BASE = "https://finnhub.io/api/v1"

DEFAULT_MAX_PER_SEC = env_float("FINNHUB_MAX_PER_SEC", 1.0, minimum=0.1)   # free ~60/min
RATE_LIMIT_COOLDOWN_SECONDS = env_float("FINNHUB_RATE_LIMIT_COOLDOWN_SECONDS", 60.0,
                                        minimum=1.0)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None   # drop NaN


class FinnhubProvider(DataProvider):
    name = "finnhub"

    def __init__(self, api_key: str, cache: DiskCache | None = None, *,
                 max_per_sec: float | None = None):
        super().__init__(cache)
        rate = max_per_sec if max_per_sec and max_per_sec > 0 else DEFAULT_MAX_PER_SEC
        self.http = HttpClient(BASE, max_per_sec=rate,
                               default_headers={"X-Finnhub-Token": api_key,
                                                "Accept": "application/json"})
        self._rate_limited_until = 0.0

    # ------------------------------------------------------------- guardrails
    def _rate_limit_active(self) -> bool:
        return time.time() < self._rate_limited_until

    def _enter_cooldown(self, seconds: float | None, reason: str) -> None:
        cooldown = float(seconds) if seconds and seconds > 0 else RATE_LIMIT_COOLDOWN_SECONDS
        until = time.time() + cooldown
        if until > self._rate_limited_until:
            self._rate_limited_until = until
        log.warning("finnhub rate limited (%s) — cooling down %.0fs; other providers serve",
                    reason, max(self._rate_limited_until - time.time(), 0.0))

    def _get(self, section: str, ttl: float, path: str, params: dict | None = None) -> Any:
        params = params or {}
        key = f"{path}|{sorted(params.items())}"
        hit = self.cache.get(f"finnhub/{section}", key, ttl)
        if hit is not None:
            return hit
        if self._rate_limit_active():
            raise ProviderUnsupported("finnhub cooling down after rate limit")
        try:
            data = self.http.get_json(path, params)
        except RateLimitError as e:
            self._enter_cooldown(e.retry_after_seconds, path)
            raise ProviderUnsupported("finnhub rate limited") from e
        self.cache.set(f"finnhub/{section}", key, data, source=self.name)
        return data

    # ------------------------------------------------------------------ quote
    def get_quote(self, ticker: str) -> Quote:
        symbol = ticker.upper()
        d = self._get("quote", 1, "/quote", {"symbol": symbol})
        price = _num((d or {}).get("c"))
        if not price or price <= 0:        # 0 = unknown/closed symbol on free tier
            raise ProviderUnsupported
        q = Quote(ticker=symbol, price=price,
                  volume=0.0,               # free /quote carries no volume
                  prev_close=_num(d.get("pc")),
                  day_change_pct=_num(d.get("dp")),
                  source=self.name)
        ts = _num(d.get("t"))
        if ts and ts > 0:
            try:
                q.asof = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                pass
        return q

    def invalidate_quote(self, ticker: str) -> None:
        key = f"/quote|{sorted({'symbol': ticker.upper()}.items())}"
        self.cache._path("finnhub/quote", key).unlink(missing_ok=True)

    # ---------------------------------------------------------------- company
    def get_company(self, ticker: str) -> dict:
        d = self._get("profile", 1440, "/stock/profile2", {"symbol": ticker.upper()})
        if not isinstance(d, dict) or not d:
            raise ProviderUnsupported
        mcap = _num(d.get("marketCapitalization"))
        shares = _num(d.get("shareOutstanding"))
        industry = d.get("finnhubIndustry")
        return {
            "name": d.get("name"),
            "sector": industry,             # Finnhub has one industry field; use for both
            "industry": industry,
            "market_cap": mcap * 1e6 if mcap else None,         # reported in millions
            "shares_outstanding": shares * 1e6 if shares else None,
            "exchange": d.get("exchange"),
            "source": self.name,
        }

    # ----------------------------------------------------- analyst (the niche)
    def get_analyst(self, ticker: str) -> dict:
        """Recommendation trends -> buy/hold/sell counts. Free, so this keeps the
        analyst factor alive while Mboum's monthly quota is exhausted. No price
        targets on the free tier (target_mean omitted -> upside neutralizes)."""
        d = self._get("reco", 720, "/stock/recommendation", {"symbol": ticker.upper()})
        rows = d if isinstance(d, list) else []
        if not rows:
            raise ProviderUnsupported
        r = rows[0]    # most recent period first
        strong_buy = int(_num(r.get("strongBuy")) or 0)
        buy = int(_num(r.get("buy")) or 0)
        hold = int(_num(r.get("hold")) or 0)
        sell = int(_num(r.get("sell")) or 0)
        strong_sell = int(_num(r.get("strongSell")) or 0)
        total = strong_buy + buy + hold + sell + strong_sell
        if total <= 0:
            raise ProviderUnsupported
        return {
            "buy": strong_buy + buy,
            "hold": hold,
            "sell": sell + strong_sell,
            "period": r.get("period"),
            "source": self.name,
        }

    # ----------------------------------------------------------- insider flow
    def get_insider(self, ticker: str) -> list:
        d = self._get("insider", 720, "/stock/insider-transactions",
                      {"symbol": ticker.upper(), "limit": 60})
        rows = (d or {}).get("data") if isinstance(d, dict) else None
        if not rows:
            raise ProviderUnsupported
        out: list[dict] = []
        for tr in rows:
            if not isinstance(tr, dict):
                continue
            change = _num(tr.get("change"))      # +shares acquired / -disposed
            if not change:
                continue
            px = _num(tr.get("transactionPrice"))
            out.append({
                "shares": abs(change),
                "value": abs(change) * px if px else None,
                "is_buy": change > 0,
                "date": tr.get("transactionDate") or tr.get("filingDate"),
            })
        if not out:
            raise ProviderUnsupported
        return out

    # -------------------------------------------------------------- earnings
    def get_earnings(self, ticker: str) -> dict:
        """Earnings-surprise history (free) -> feeds the catalyst pillar's
        earnings_surprise_history factor. No estimate revisions on free."""
        d = self._get("earnings", 720, "/stock/earnings", {"symbol": ticker.upper()})
        rows = d if isinstance(d, list) else []
        surprises = [s for s in (_num(r.get("surprisePercent")) for r in rows
                                 if isinstance(r, dict)) if s is not None]
        surprises = surprises[:4]          # last ~4 quarters
        if not surprises:
            raise ProviderUnsupported
        beats = sum(1 for s in surprises if s > 0)
        return {
            "avg_surprise_pct": sum(surprises) / len(surprises),
            "beat_rate": beats / len(surprises),
            "source": self.name,
        }

    # --------------------------------------------------------------- headlines
    def get_headlines(self, ticker: str | None = None, limit: int = 8) -> list:
        today = datetime.now(timezone.utc).date()
        if ticker:
            d = self._get("cnews", 60, "/company-news",
                          {"symbol": ticker.upper(),
                           "from": (today - timedelta(days=14)).isoformat(),
                           "to": today.isoformat()})
        else:
            d = self._get("mnews", 30, "/news", {"category": "general"})
        rows = d if isinstance(d, list) else []
        out: list[str] = []
        for art in rows:
            title = art.get("headline") if isinstance(art, dict) else None
            if title and title not in out:
                out.append(title)
            if len(out) >= limit:
                break
        return out

    # --------------------------------------------------------------- universe
    def list_universe(self) -> list[str]:
        d = self._get("symbols", 1440, "/stock/symbol", {"exchange": "US"})
        rows = d if isinstance(d, list) else []
        out: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol") or "")
            if (row.get("type") == "Common Stock" and sym
                    and sym.isalpha() and sym == sym.upper()):
                out.append(sym)
        return out
