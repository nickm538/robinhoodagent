"""Web-research provider for *pro-source* scores, headline context, and sentiment fallback.

Wires Firecrawl (search + markdown scrape) and Exa (neural search with recency).
Used where no clean API exists: Zacks Rank, Morningstar stars, Danelfin AI Score,
TipRanks Smart Score; headline enrichment for the AI analyst; news-tone fallback
when API sentiment is missing.

REAL-MONEY DISCIPLINE (no look-ahead, minimal noise):
  * Runs only on deep-scored names (MarketData.build deep=True) — never the 400-name
    intraday light pass.
  * Parses only point-in-time published ratings — never scraped price targets or
    forward EPS estimates (those stay on API analyst data with known as-of dates).
  * Exa queries carry a recency window (default 3 days) so stale or speculative
    "outlook" pieces are deprioritized.
  * Keyword sentiment is a LAST-RESORT fallback with a minimum hit count; omitted
    when signal is too thin (factor neutralises).
  * Never invents a score — fields populate only when a concrete number is parsed.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ..logging_setup import get_logger
from .base import DataProvider, DiskCache, HttpClient, ProviderUnsupported

log = get_logger("providers.web")

FIRECRAWL = "https://api.firecrawl.dev"
EXA = "https://api.exa.ai"

_POS = [r"\bbeat\b", r"\bsurge\b", r"\bsoar\b", r"\bupgrade\b", r"\brecord\b",
        r"\brally\b", r"\braises\b", r"\btops\b", r"\boutperform\b",
        r"\bjumps\b", r"\bstrong\b"]
_NEG = [r"\bmiss\b", r"\bplunge\b", r"\bdowngrade\b", r"\blawsuit\b", r"\bprobe\b",
        r"\bcut\b", r"\bwarns\b", r"\bslump\b", r"\bfalls\b", r"\bweak\b", r"\bhalts\b"]
# Speculative / forward language — down-weight for sentiment (not hard block).
_SPECULATIVE = [r"\bwill\b", r"\bcould\b", r"\bforecast\b", r"\bexpects?\b",
                r"\boutlook\b", r"\btarget price\b", r"\bprice target\b"]


class WebResearchProvider(DataProvider):
    name = "web"

    def __init__(self, firecrawl_key: str | None = None, exa_key: str | None = None,
                 cache: DiskCache | None = None, settings: dict | None = None):
        super().__init__(cache)
        self.firecrawl_key = firecrawl_key
        self.exa_key = exa_key
        s = settings or {}
        self.combine_engines = bool(s.get("combine_engines", True))
        self.max_results = int(s.get("max_search_results", 5))
        self.max_chars = int(s.get("max_chars_per_search", 8000))
        self.exa_recency_days = int(s.get("exa_recency_days", 3))
        self.min_sentiment_hits = int(s.get("min_sentiment_hits", 4))
        self.pro_scores_ttl = float(s.get("pro_scores_ttl_minutes", 720))
        self.headlines_ttl = float(s.get("headlines_ttl_minutes", 60))
        self.sentiment_ttl = float(s.get("sentiment_ttl_minutes", 120))
        self.fc = (HttpClient(FIRECRAWL, max_per_sec=2,
                              default_headers={"Authorization": f"Bearer {firecrawl_key}"})
                   if firecrawl_key else None)
        self.exa = (HttpClient(EXA, max_per_sec=4, default_headers={"x-api-key": exa_key})
                    if exa_key else None)
        if not self.fc and not self.exa:
            self.enabled = False

    def _exa_start_date(self) -> str:
        since = datetime.now(timezone.utc) - timedelta(days=max(1, self.exa_recency_days))
        return since.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _firecrawl_snippets(self, query: str, limit: int) -> list[dict[str, str]]:
        if self.fc is None:
            return []
        out: list[dict[str, str]] = []
        try:
            r = self.fc.session.post(
                f"{FIRECRAWL}/v1/search",
                json={"query": query, "limit": limit,
                      "scrapeOptions": {"formats": ["markdown"]}},
                timeout=self.fc.timeout)
            if not r.ok:
                return out
            for item in (r.json().get("data") or []):
                title = (item.get("title") or item.get("metadata", {}).get("title") or "").strip()
                body = (item.get("markdown") or item.get("description") or "").strip()
                url = (item.get("url") or item.get("metadata", {}).get("sourceURL") or "").strip()
                if title or body:
                    out.append({"title": title, "text": body, "url": url, "source": "firecrawl"})
        except Exception as e:
            log.debug("firecrawl search failed: %s", e)
        return out

    def _exa_snippets(self, query: str, limit: int) -> list[dict[str, str]]:
        if self.exa is None:
            return []
        out: list[dict[str, str]] = []
        try:
            payload: dict[str, Any] = {
                "query": query,
                "numResults": limit,
                "contents": {"text": {"maxCharacters": 2000}},
                "startPublishedDate": self._exa_start_date(),
            }
            r = self.exa.session.post(f"{EXA}/search", json=payload, timeout=self.exa.timeout)
            if not r.ok:
                return out
            for item in (r.json().get("results") or []):
                title = (item.get("title") or "").strip()
                body = (item.get("text") or "").strip()
                url = (item.get("url") or "").strip()
                if title or body:
                    out.append({"title": title, "text": body, "url": url, "source": "exa"})
        except Exception as e:
            log.debug("exa search failed: %s", e)
        return out

    def _merge_snippets(self, chunks: list[list[dict[str, str]]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        merged: list[dict[str, str]] = []
        for group in chunks:
            for sn in group:
                key = (sn.get("title") or sn.get("url") or sn.get("text", "")[:80]).lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(sn)
        return merged

    def _search_snippets(self, query: str, ttl: float, limit: int | None = None) -> list[dict[str, str]]:
        lim = limit or self.max_results
        hit = self.cache.get("web/snippets", query, ttl)
        if hit is not None:
            return hit
        fc = self._firecrawl_snippets(query, lim)
        exa = self._exa_snippets(query, lim)
        if self.combine_engines:
            snippets = self._merge_snippets([fc, exa])
        elif fc:
            snippets = fc
        else:
            snippets = exa
        self.cache.set("web/snippets", query, snippets, source="web")
        return snippets

    def _search_text(self, query: str, ttl: float = 720, limit: int | None = None) -> str:
        hit = self.cache.get("web/search", query, ttl)
        if hit is not None:
            return hit
        snippets = self._search_snippets(query, ttl, limit)
        parts: list[str] = []
        total = 0
        for sn in snippets:
            block = "\n".join(x for x in (sn.get("title"), sn.get("text")) if x)
            if not block:
                continue
            if total + len(block) > self.max_chars:
                block = block[: max(0, self.max_chars - total)]
            parts.append(block)
            total += len(block)
            if total >= self.max_chars:
                break
        text = "\n".join(parts)
        self.cache.set("web/search", query, text, source="web")
        return text

    def get_headlines(self, ticker: str | None = None, limit: int = 8) -> list[str]:
        """Recent headline strings for the AI analyst (recency-bounded Exa + Firecrawl)."""
        if not self.enabled:
            raise ProviderUnsupported
        if ticker:
            query = f"{ticker} stock news earnings SEC filing"
        else:
            query = "US stock market macro news Federal Reserve tariffs"
        snippets = self._search_snippets(query, self.headlines_ttl, limit)
        out: list[str] = []
        for sn in snippets:
            title = (sn.get("title") or "").strip()
            if not title:
                first = (sn.get("text") or "").split("\n", 1)[0].strip()
                title = first[:160] if first else ""
            if title and title not in out:
                out.append(title[:200])
            if len(out) >= limit:
                break
        if not out:
            raise ProviderUnsupported
        return out

    def _parse_pro_scores(self, text: str) -> dict:
        out: dict = {}
        m = re.search(r"Zacks Rank[^0-9#]{0,12}#?\s*([1-5])\b", text, re.I)
        if m:
            out["zacks_rank"] = int(m.group(1))
        m = re.search(r"AI Score[^0-9]{0,8}(\d{1,2})\s*/\s*10", text, re.I) or \
            re.search(r"AI Score of\s*(\d{1,2})", text, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                out["danelfin_ai"] = v
        m = re.search(r"(\d)\s*-?\s*star", text, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 5:
                out["morningstar_stars"] = v
        m = re.search(r"Smart Score[^0-9]{0,8}(\d{1,2})\b", text, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                out["tipranks_smart_score"] = v
        return out

    def get_pro_scores(self, ticker: str, name: str | None = None) -> dict:
        if not self.enabled:
            raise ProviderUnsupported
        label = f"{ticker} {name or ''}".strip()
        # One combined query (not four) — same engines, fewer round trips, consistent snapshot.
        query = (f"{label} current Zacks Rank Morningstar star rating "
                 f"Danelfin AI Score TipRanks Smart Score analyst rating today")
        text = self._search_text(query, ttl=self.pro_scores_ttl, limit=self.max_results)
        parsed = self._parse_pro_scores(text)
        if not parsed:
            raise ProviderUnsupported
        return {"source": "web", **parsed}

    def get_news_sentiment(self, ticker: str) -> dict:
        """Headline-tone fallback when API sentiment is unavailable."""
        if not self.enabled:
            raise ProviderUnsupported
        text = self._search_text(
            f"{ticker} stock news today earnings revenue",
            ttl=self.sentiment_ttl,
            limit=6,
        ).lower()
        if not text:
            raise ProviderUnsupported
        speculative = sum(len(re.findall(w, text)) for w in _SPECULATIVE)
        pos = sum(len(re.findall(w, text)) for w in _POS)
        neg = sum(len(re.findall(w, text)) for w in _NEG)
        # Deprioritize forward-looking / speculative copy (common source of false positives).
        if speculative > max(pos, neg, 1):
            pos = max(0, pos - speculative // 2)
            neg = max(0, neg - speculative // 2)
        tot = pos + neg
        if tot < self.min_sentiment_hits:
            raise ProviderUnsupported
        score = max(-1.0, min(1.0, (pos - neg) / tot))
        return {"score": score, "article_count": tot, "source": "web",
                "speculative_hits": speculative}
