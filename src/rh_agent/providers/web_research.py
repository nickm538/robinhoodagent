"""Web-research provider for *pro-source* scores and instant news.

Wires in Firecrawl (search + scrape) and Exa (neural search). Used to read
ratings that have no clean API: Zacks Rank, Morningstar star rating,
Danelfin AI Score, TipRanks Smart Score / analyst consensus.

CRITICAL: this provider never invents a score. A field is populated ONLY if a
concrete number is parsed from fetched page text; otherwise it is omitted and
the corresponding factor neutralises itself.
"""
from __future__ import annotations

import re

from ..logging_setup import get_logger
from .base import DataProvider, DiskCache, HttpClient, ProviderUnsupported

log = get_logger("providers.web")

FIRECRAWL = "https://api.firecrawl.dev"
EXA = "https://api.exa.ai"


class WebResearchProvider(DataProvider):
    name = "web"

    def __init__(self, firecrawl_key: str | None = None, exa_key: str | None = None,
                 cache: DiskCache | None = None):
        super().__init__(cache)
        self.firecrawl_key = firecrawl_key
        self.exa_key = exa_key
        self.fc = (HttpClient(FIRECRAWL, max_per_sec=2,
                              default_headers={"Authorization": f"Bearer {firecrawl_key}"})
                   if firecrawl_key else None)
        self.exa = (HttpClient(EXA, max_per_sec=4, default_headers={"x-api-key": exa_key})
                    if exa_key else None)
        if not self.fc and not self.exa:
            self.enabled = False

    # ---- low level search returning concatenated text from top results ----
    def _search_text(self, query: str, ttl: float = 720, limit: int = 5) -> str:
        hit = self.cache.get("web/search", query, ttl)
        if hit is not None:
            return hit
        text = ""
        if self.fc is not None:
            try:
                r = self.fc.session.post(
                    f"{FIRECRAWL}/v1/search",
                    json={"query": query, "limit": limit,
                          "scrapeOptions": {"formats": ["markdown"]}},
                    timeout=self.fc.timeout)
                if r.ok:
                    js = r.json()
                    for item in (js.get("data") or []):
                        text += "\n" + (item.get("markdown") or item.get("description") or "")
            except Exception as e:
                log.debug("firecrawl search failed: %s", e)
        if not text and self.exa is not None:
            try:
                r = self.exa.session.post(
                    f"{EXA}/search",
                    json={"query": query, "numResults": limit,
                          "contents": {"text": {"maxCharacters": 2000}}},
                    timeout=self.exa.timeout)
                if r.ok:
                    js = r.json()
                    for item in (js.get("results") or []):
                        text += "\n" + (item.get("text") or item.get("title") or "")
            except Exception as e:
                log.debug("exa search failed: %s", e)
        self.cache.set("web/search", query, text, source="web")
        return text

    # ---- public: pro-source scores ----
    def get_pro_scores(self, ticker: str, name: str | None = None) -> dict:
        if not self.enabled:
            raise ProviderUnsupported
        label = f"{ticker} {name or ''}".strip()
        out: dict = {"source": "web"}

        zacks = self._search_text(f"{label} stock Zacks Rank rating")
        m = re.search(r"Zacks Rank[^0-9#]{0,12}#?\s*([1-5])\b", zacks, re.I)
        if m:
            out["zacks_rank"] = int(m.group(1))  # 1=Strong Buy .. 5=Strong Sell

        dane = self._search_text(f"{ticker} Danelfin AI Score")
        m = re.search(r"AI Score[^0-9]{0,8}(\d{1,2})\s*/\s*10", dane, re.I) or \
            re.search(r"AI Score of\s*(\d{1,2})", dane, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                out["danelfin_ai"] = v

        ms = self._search_text(f"{label} Morningstar star rating fair value")
        m = re.search(r"(\d)\s*-?\s*star", ms, re.I)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 5:
                out["morningstar_stars"] = v

        tr = self._search_text(f"{ticker} TipRanks Smart Score analyst consensus")
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
        if not self.enabled:
            raise ProviderUnsupported
        text = self._search_text(f"{ticker} stock news today", ttl=120, limit=6).lower()
        if not text:
            raise ProviderUnsupported
        pos_words = [r"\bbeat\b", r"\bsurge\b", r"\bsoar\b", r"\bupgrade\b", r"\brecord\b",
                     r"\brally\b", r"\braises\b", r"\btops\b", r"\boutperform\b",
                     r"\bjumps\b", r"\bstrong\b"]
        neg_words = [r"\bmiss\b", r"\bplunge\b", r"\bdowngrade\b", r"\blawsuit\b", r"\bprobe\b",
                     r"\bcut\b", r"\bwarns\b", r"\bslump\b", r"\bfalls\b", r"\bweak\b", r"\bhalts\b"]
        pos = sum(len(re.findall(w, text)) for w in pos_words)
        neg = sum(len(re.findall(w, text)) for w in neg_words)
        tot = pos + neg
        if tot == 0:
            raise ProviderUnsupported
        return {"score": (pos - neg) / tot, "article_count": tot, "source": "web"}
