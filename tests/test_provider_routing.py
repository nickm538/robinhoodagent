"""Shipped-config provider routing mandate.

FinancialDatasets.AI is the PRIMARY source wherever it is capable, Mboum (pro)
runs right behind it, and TwelveData stays COMPLEMENTARY (batch prefetch,
fallback quotes/prices, technicals enrichment). These tests pin the shipped
config so a tuning PR can't silently demote the paid primary providers.
"""
from __future__ import annotations

from rh_agent.config import load_config


def _providers():
    return load_config().get("providers", {})


def test_financialdatasets_is_primary_for_core_sections():
    p = _providers()
    for section in ("fundamentals", "prices", "quote", "quote_risk",
                    "insider", "institutional", "news_headlines"):
        assert p[section][0] == "financialdatasets", (
            f"{section}: financialdatasets must be primary, got {p[section]}")


def test_mboum_backs_up_every_core_section():
    p = _providers()
    for section in ("fundamentals", "prices", "quote", "quote_risk",
                    "technicals", "insider", "institutional", "news_sentiment"):
        assert "mboum" in p[section], f"{section}: mboum (pro) missing from {p[section]}"
    assert p["universe"][0] == "mboum"


def test_twelvedata_stays_complementary_not_primary():
    p = _providers()
    for section in ("prices", "quote", "quote_risk", "technicals"):
        chain = p[section]
        assert "twelvedata" in chain, f"{section}: twelvedata missing from {chain}"
        assert chain[0] != "twelvedata", (
            f"{section}: twelvedata must be complementary, not primary ({chain})")


# ---- FinancialDatasets /news/ quirks (live API caps limit at 10) ----

def _fd_with_news(monkeypatch, items):
    from rh_agent.providers.financial_datasets import FinancialDatasetsProvider
    fd = FinancialDatasetsProvider.__new__(FinancialDatasetsProvider)
    seen = {}

    def fake_cached(section, ticker, ttl, path, params):
        seen.update(params)
        return {"news": items}

    monkeypatch.setattr(fd, "_cached", fake_cached)
    return fd, seen


def test_fd_news_limit_capped_at_10(monkeypatch):
    fd, seen = _fd_with_news(monkeypatch, [{"title": f"h{i}"} for i in range(10)])
    fd.get_headlines("AAPL", limit=50)
    assert seen["limit"] <= 10
    seen.clear()
    try:
        fd.get_news_sentiment("AAPL")
    except Exception:
        pass
    assert seen["limit"] <= 10


def test_fd_sentiment_defers_when_articles_unlabeled(monkeypatch):
    import pytest

    from rh_agent.providers.base import ProviderUnsupported
    fd, _ = _fd_with_news(monkeypatch, [{"title": "no label"}] * 5)
    with pytest.raises(ProviderUnsupported):
        fd.get_news_sentiment("AAPL")


def test_fd_sentiment_scores_labeled_articles(monkeypatch):
    fd, _ = _fd_with_news(monkeypatch, [
        {"title": "a", "sentiment": "positive"},
        {"title": "b", "sentiment": "positive"},
        {"title": "c", "sentiment": "negative"},
        {"title": "d", "sentiment": "neutral"},
    ])
    out = fd.get_news_sentiment("AAPL")
    assert out["score"] == 0.25
    assert out["article_count"] == 4
