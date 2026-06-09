from __future__ import annotations

import pytest

from rh_agent.providers.base import ProviderUnsupported
from rh_agent.providers.web_research import WebResearchProvider


def test_web_pro_scores_are_source_scoped_and_numeric_only():
    p = WebResearchProvider.__new__(WebResearchProvider)
    p.enabled = True
    p.max_results = 3
    p.pro_scores_ttl = 720
    queries = []

    def _search_text(query, ttl=720, limit=None, recency=True):
        queries.append(query)
        if "zacks.com" in query:
            return "AAPL Apple Inc Zacks Rank #2 (Buy)"
        if "danelfin.com" in query:
            return "AAPL AI Score 8/10"
        if "morningstar.com" in query:
            return "Apple Inc has a 4-star Morningstar rating"
        if "tipranks.com" in query:
            return "AAPL TipRanks Smart Score 9 analyst consensus"
        return ""

    p._search_text = _search_text
    out = p.get_pro_scores("AAPL", "Apple Inc")

    assert out == {
        "source": "web",
        "zacks_rank": 2,
        "danelfin_ai": 8,
        "morningstar_stars": 4,
        "tipranks_smart_score": 9,
    }
    assert all("site:" in q for q in queries)


def test_web_pro_scores_ignore_off_subject_results():
    p = WebResearchProvider.__new__(WebResearchProvider)
    p.enabled = True
    p.max_results = 3
    p.pro_scores_ttl = 720
    p._search_text = lambda query, ttl=720, limit=None, recency=True: \
        "MSFT Zacks Rank #1 AI Score 10/10"

    with pytest.raises(ProviderUnsupported):
        p.get_pro_scores("AAPL", "Apple Inc")


def test_web_news_sentiment_is_opt_in():
    p = WebResearchProvider.__new__(WebResearchProvider)
    p.enabled = True
    p.enable_news_sentiment = False

    with pytest.raises(ProviderUnsupported):
        p.get_news_sentiment("AAPL")

    p.enable_news_sentiment = True
    p.max_results = 3
    p.sentiment_ttl = 120
    p.min_sentiment_hits = 2
    p._search_text = lambda query, ttl=120, limit=6, recency=True: \
        "AAPL beat estimates and shares surge"
    out = p.get_news_sentiment("AAPL")
    assert out["score"] > 0
    assert out["source"] == "web"


def test_web_headlines_merge_firecrawl_and_exa_snippets():
    p = WebResearchProvider.__new__(WebResearchProvider)
    p.enabled = True
    p.headlines_ttl = 60
    p.max_results = 3
    p.combine_engines = True
    p._search_snippets = lambda query, ttl, limit=None, recency=True: [
        {"title": "AAPL launches new chip", "text": "", "source": "firecrawl"},
        {"title": "Analysts discuss AAPL demand", "text": "", "source": "exa"},
        {"title": "AAPL launches new chip", "text": "", "source": "exa"},
    ]

    assert p.get_headlines("AAPL", limit=3) == [
        "AAPL launches new chip",
        "Analysts discuss AAPL demand",
    ]


def test_web_snippets_use_http_client_post_json_helpers():
    class _Client:
        timeout = 20

        def __init__(self, payload):
            self.payload = payload
            self.calls = []

        def post_json(self, path, payload):
            self.calls.append((path, payload))
            return self.payload

    p = WebResearchProvider.__new__(WebResearchProvider)
    p.fc = _Client({"data": [{"title": "FC title", "markdown": "FC body", "url": "u"}]})
    p.exa = _Client({"results": [{"title": "EXA title", "text": "EXA body", "url": "e"}]})
    p.exa_recency_days = 3

    assert p._firecrawl_snippets("AAPL news", 2)[0]["source"] == "firecrawl"
    assert p.fc.calls[0][0] == "/v1/search"
    exa = p._exa_snippets("AAPL news", 2)
    assert exa[0]["source"] == "exa"
    assert p.exa.calls[0][0] == "/search"
    assert "startPublishedDate" in p.exa.calls[0][1]
