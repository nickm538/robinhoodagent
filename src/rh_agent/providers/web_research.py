"""Web-research provider for pro scores, headline context, and sentiment fallback.

Wires Firecrawl (search + markdown scrape) and Exa (neural search with recency).
Used where no clean API exists: Zacks Rank, Morningstar stars, Danelfin AI Score,
TipRanks Smart Score; headline enrichment for the AI analyst; news-tone fallback
when API sentiment is missing.

REAL-MONEY DISCIPLINE:
  * Runs only on deep-scored names through MarketData.build(deep=True).
  * Parses only concrete published ratings, never scraped price targets/forecasts.
  * Exa queries use a short recency window for headlines and live context.
  * Generic web sentiment is opt-in and requires enough keyword evidence.
  * Missing/ambiguous data is omitted so factors neutralize instead of being faked.
"""
from __future__ import annotations

import os
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
_SPECULATIVE = [r"\bwill\b", r"\bcould\b", r"\bforecast\b", r"\bexpects?\b",
                r"\boutlook\b", r"\btarget price\b", r"\bprice target\b"]


class WebResearchProvider(DataProvider):
    name = "web"

    def __init__(self, firecrawl_key: str | None = None, exa_key: str | None = None,
                 cache: DiskCache | None = None, *, settings: dict | None = None,
                 max_results: int | None = None, enable_news_sentiment: bool | None = None):
        super().__init__(cache)
        self.firecrawl_key = firecrawl_key
        self.exa_key = exa_key
        s = settings or {}
        if max_results is not None:
            s.setdefault("max_search_results", max_results)
        if enable_news_sentiment is not None:
            s.setdefault("enable_news_sentiment", enable_news_sentiment)
        self.combine_engines = bool(s.get("combine_engines", True))
        self.max_results = max(1, min(int(s.get("max_search_results", 3)), 8))
        self.max_chars = int(s.get("max_chars_per_search", 8000))
        self.exa_recency_days = int(s.get("exa_recency_days", 3))
        self.min_sentiment_hits = int(s.get("min_sentiment_hits", 4))
        self.pro_scores_ttl = float(s.get("pro_scores_ttl_minutes", 720))
        self.headlines_ttl = float(s.get("headlines_ttl_minutes", 60))
        self.sentiment_ttl = float(s.get("sentiment_ttl_minutes", 120))
        self.enable_news_sentiment = bool(s.get("enable_news_sentiment", False)) or os.getenv(
            "WEB_RESEARCH_ENABLE_NEWS_SENTIMENT", ""
        ).lower() in ("1", "true", "yes")
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
        if self.fc is not None:
            try:
                data = self.fc.post_json(
                    "/v1/search",
                    {"query": query, "limit": limit,
                     "scrapeOptions": {"formats": ["markdown"]}},
                )
                out: list[dict[str, str]] = []
                for item in (data.get("data") or []):
                    title = (item.get("title") or item.get("metadata", {}).get("title") or "").strip()
                    body = (item.get("markdown") or item.get("description") or "").strip()
                    url = (item.get("url") or item.get("metadata", {}).get("sourceURL") or "").strip()
                    if title or body:
                        out.append({"title": title, "text": body, "url": url, "source": "firecrawl"})
                return out
            except Exception as e:
                log.debug("firecrawl search failed: %s", e)
        return []

    def _exa_snippets(self, query: str, limit: int, *, recency: bool = True) -> list[dict[str, str]]:
        if self.exa is not None:
            try:
                payload: dict[str, Any] = {
                    "query": query,
                    "numResults": limit,
                    "contents": {"text": {"maxCharacters": 2000}},
                }
                if recency:
                    payload["startPublishedDate"] = self._exa_start_date()
                data = self.exa.post_json("/search", payload)
                out: list[dict[str, str]] = []
                for item in (data.get("results") or []):
                    title = (item.get("title") or "").strip()
                    body = (item.get("text") or "").strip()
                    url = (item.get("url") or "").strip()
                    if title or body:
                        out.append({"title": title, "text": body, "url": url, "source": "exa"})
                return out
            except Exception as e:
                log.debug("exa search failed: %s", e)
        return []

    @staticmethod
    def _merge_snippets(chunks: list[list[dict[str, str]]]) -> list[dict[str, str]]:
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

    def _search_snippets(
        self,
        query: str,
        ttl: float,
        limit: int | None = None,
        *,
        recency: bool = True,
    ) -> list[dict[str, str]]:
        lim = limit or self.max_results
        key = f"{query}|recency={recency}"
        hit = self.cache.get("web/snippets", key, ttl)
        if hit is not None:
            return hit
        fc = self._firecrawl_snippets(query, lim)
        exa = self._exa_snippets(query, lim, recency=recency)
        snippets = self._merge_snippets([fc, exa]) if self.combine_engines else (fc or exa)
        self.cache.set("web/snippets", key, snippets, source="web")
        return snippets

    # ---- low level search returning concatenated text from top results ----
    def _search_text(self, query: str, ttl: float = 720, limit: int | None = None,
                     *, recency: bool = True) -> str:
        key = f"{query}|recency={recency}"
        hit = self.cache.get("web/search", key, ttl)
        if hit is not None:
            return hit
        parts: list[str] = []
        total = 0
        for sn in self._search_snippets(query, ttl, limit, recency=recency):
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
        self.cache.set("web/search", key, text, source="web")
        return text

    @staticmethod
    def _mentions_subject(text: str, ticker: str, name: str | None) -> bool:
        hay = text.lower()
        if re.search(rf"\b{re.escape(ticker.lower())}\b", hay):
            return True
        if name:
            tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", name.lower()) if len(t) >= 4]
            return any(t in hay for t in tokens[:3])
        return False

    def _source_text(self, source: str, query: str, ticker: str, name: str | None) -> str:
        text = self._search_text(query, ttl=self.pro_scores_ttl, recency=False)
        if text and self._mentions_subject(text, ticker, name):
            return text
        log.debug("web %s result ignored for %s: subject not present", source, ticker)
        return ""

    def get_headlines(self, ticker: str | None = None, limit: int = 8) -> list[str]:
        """Recent headline strings for the AI analyst."""
        if not self.enabled:
            raise ProviderUnsupported
        query = (
            f"{ticker} stock news earnings SEC filing"
            if ticker else
            "US stock market macro news Federal Reserve tariffs"
        )
        snippets = self._search_snippets(query, self.headlines_ttl, limit, recency=True)
        out: list[str] = []
        for sn in snippets:
            title = (sn.get("title") or "").strip()
            if not title:
                title = (sn.get("text") or "").split("\n", 1)[0].strip()
            if title and title not in out:
                out.append(title[:200])
            if len(out) >= limit:
                break
        if not out:
            raise ProviderUnsupported
        return out

    # ---- public: pro-source scores ----
    def get_pro_scores(self, ticker: str, name: str | None = None) -> dict:
        if not self.enabled:
            raise ProviderUnsupported
        label = f"{ticker} {name or ''}".strip()
        out: dict = {"source": "web"}

        zacks = self._source_text(
            "zacks", f"site:zacks.com/stock/quote {label} Zacks Rank", ticker, name)
        m = re.search(r"Zacks Rank[^0-9#]{0,12}#?\s*([1-5])\b", zacks, re.I)
        if m:
            out["zacks_rank"] = int(m.group(1))  # 1=Strong Buy .. 5=Strong Sell

        dane = self._source_text(
            "danelfin", f"site:danelfin.com/stock {label} Danelfin AI Score", ticker, name)
        m = re.search(r"AI Score[^0-9]{0,8}(\d{1,2})\s*/\s*10", dane, re.I) or \
            re.search(r"AI Score of\s*(\d{1,2})", dane, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                out["danelfin_ai"] = v

        ms = self._source_text(
            "morningstar",
            f"site:morningstar.com/stocks {label} Morningstar star rating fair value",
            ticker, name)
        m = re.search(r"(\d)\s*-?\s*star", ms, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 5:
                out["morningstar_stars"] = v

        tr = self._source_text(
            "tipranks", f"site:tipranks.com/stocks {label} Smart Score analyst consensus",
            ticker, name)
        m = re.search(r"Smart Score[^0-9]{0,8}(\d{1,2})\b", tr, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                out["tipranks_smart_score"] = v

        if len(out) == 1:
            raise ProviderUnsupported
        return out

    def get_news_sentiment(self, ticker: str) -> dict:
        """Headline-tone fallback when API sentiment is unavailable."""
        if not self.enabled or not self.enable_news_sentiment:
            raise ProviderUnsupported
        text = self._search_text(
            f"{ticker} stock news today earnings revenue",
            ttl=self.sentiment_ttl,
            limit=6,
            recency=True,
        ).lower()
        if not text:
            raise ProviderUnsupported
        speculative = sum(len(re.findall(w, text)) for w in _SPECULATIVE)
        pos = sum(len(re.findall(w, text)) for w in _POS)
        neg = sum(len(re.findall(w, text)) for w in _NEG)
        if speculative > max(pos, neg, 1):
            pos = max(0, pos - speculative // 2)
            neg = max(0, neg - speculative // 2)
        tot = pos + neg
        if tot < self.min_sentiment_hits:
            raise ProviderUnsupported
        return {"score": max(-1.0, min(1.0, (pos - neg) / tot)), "article_count": tot,
                "source": "web", "speculative_hits": speculative}
