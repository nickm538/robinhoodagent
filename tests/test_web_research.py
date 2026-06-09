from __future__ import annotations

import pytest

from rh_agent.providers.base import ProviderUnsupported
from rh_agent.providers.web_research import WebResearchProvider


def test_web_pro_scores_are_source_scoped_and_numeric_only():
    p = WebResearchProvider.__new__(WebResearchProvider)
    p.enabled = True
    p.max_results = 3
    queries = []

    def _search_text(query, ttl=720, limit=None):
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
    p._search_text = lambda query, ttl=720, limit=None: "MSFT Zacks Rank #1 AI Score 10/10"

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
    p._search_text = lambda query, ttl=120, limit=6: "AAPL beat estimates and shares surge"
    out = p.get_news_sentiment("AAPL")
    assert out["score"] > 0
    assert out["source"] == "web"
