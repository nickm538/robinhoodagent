"""Tests for Firecrawl/Exa web research — offline, no network."""
from __future__ import annotations

import pytest

from rh_agent.providers.web_research import WebResearchProvider


def _provider(**settings):
    p = WebResearchProvider("fc-key", "exa-key", settings=settings)
    p._firecrawl_snippets = lambda q, lim: [
        {"title": "FC headline", "text": "Zacks Rank #2 Strong Buy", "url": "a", "source": "firecrawl"},
    ]
    p._exa_snippets = lambda q, lim: [
        {"title": "Exa headline", "text": "Smart Score 8 analyst consensus", "url": "b", "source": "exa"},
    ]
    return p


def test_combine_engines_merges_both_sources():
    from rh_agent.providers.base import DiskCache
    p = _provider(combine_engines=True)
    p.cache = DiskCache()  # isolate from other tests
    snippets = p._search_snippets("NVDA ratings combine", ttl=60)
    titles = {s["title"] for s in snippets}
    assert "FC headline" in titles and "Exa headline" in titles


def test_fallback_mode_prefers_firecrawl():
    from rh_agent.providers.base import DiskCache
    p = _provider(combine_engines=False)
    p.cache = DiskCache()
    snippets = p._search_snippets("NVDA ratings fallback", ttl=60)
    assert len(snippets) == 1
    assert snippets[0]["source"] == "firecrawl"


def test_pro_scores_single_query_parses_all():
    p = _provider()
    p._search_text = lambda q, ttl=720, limit=None: (
        "Zacks Rank #2 Danelfin AI Score 7/10 4-star Morningstar Smart Score 8"
    )
    out = p.get_pro_scores("NVDA", "NVIDIA")
    assert out["zacks_rank"] == 2
    assert out["danelfin_ai"] == 7
    assert out["morningstar_stars"] == 4
    assert out["tipranks_smart_score"] == 8


def test_pro_scores_raises_when_nothing_parsed():
    p = _provider()
    p._search_text = lambda q, ttl=720, limit=None: "no ratings here"
    with pytest.raises(Exception):
        p.get_pro_scores("ZZZ")


def test_news_sentiment_noise_gate():
    p = _provider(min_sentiment_hits=4)
    p._search_text = lambda q, ttl=120, limit=None: "one beat"  # too thin
    with pytest.raises(Exception):
        p.get_news_sentiment("AAA")


def test_news_sentiment_deprioritizes_speculative_copy():
    p = _provider(min_sentiment_hits=2)
    p._search_text = lambda q, ttl=120, limit=None: (
        "forecast will target price outlook could beat miss"
    )
    with pytest.raises(Exception):
        p.get_news_sentiment("AAA")


def test_get_headlines_dedupes():
    p = _provider()
    p._search_snippets = lambda q, ttl, limit=None: [
        {"title": "Earnings beat", "text": "", "url": "x"},
        {"title": "Earnings beat", "text": "", "url": "y"},
        {"title": "Upgrade note", "text": "", "url": "z"},
    ]
    h = p.get_headlines("NVDA", limit=5)
    assert h == ["Earnings beat", "Upgrade note"]
