"""Shipped-config provider routing mandate (COST-OPTIMIZED for a small account).

FinancialDatasets bills per call, so it is reserved for the low-frequency,
days-cached depth it alone does well (fundamental ratios via merge + 13F
institutional). The high-volume reads (quotes, prices, company, news) run on
flat-rate Massive and free Finnhub. These tests pin that intent so a future
tuning PR can't silently re-promote the per-call provider into the firehose.
"""
from __future__ import annotations

from rh_agent.config import load_config


def _providers():
    return load_config().get("providers", {})


def test_flatrate_massive_is_primary_for_high_volume_reads():
    p = _providers()
    # The per-call leak was quotes/prices/company — those must lead with the
    # flat-rate, uncapped provider, never the per-call one.
    for section in ("quote", "quote_risk", "prices", "fundamentals"):
        assert p[section][0] == "massive", (
            f"{section}: massive (flat-rate) must be primary, got {p[section]}")


def test_financialdatasets_demoted_off_the_firehose_but_retained():
    p = _providers()
    # FD must NOT be primary on the high-volume per-call sections...
    for section in ("quote", "quote_risk", "prices", "news_headlines", "insider"):
        assert p[section][0] != "financialdatasets", (
            f"{section}: FD must not lead the firehose, got {p[section]}")
    # ...but stays available as a fallback for resilience.
    for section in ("quote", "prices", "fundamentals"):
        assert "financialdatasets" in p[section]
    # FD remains the source for the two things only it does (cached for days):
    assert p["institutional"][0] == "financialdatasets"
    assert "financialdatasets" in p["fundamentals"]   # supplies ratios via merge


def test_free_providers_lead_where_they_can():
    p = _providers()
    assert p["insider"][0] == "finnhub"          # free insider, FD fallback
    assert p["analyst_ratings"][0] == "mboum" and "finnhub" in p["analyst_ratings"]
    assert p["news_headlines"][0] == "massive"   # flat-rate headlines first


def test_twelvedata_stays_complementary_not_primary():
    p = _providers()
    for section in ("prices", "quote", "quote_risk", "technicals"):
        chain = p[section]
        assert "twelvedata" in chain, f"{section}: twelvedata missing from {chain}"
        assert chain[0] != "twelvedata", (
            f"{section}: twelvedata must be complementary, not primary ({chain})")


def test_fundamentals_caches_hard_to_bound_fd_spend():
    ttls = load_config().get("providers.cache_ttl_minutes", {})
    # FD's only recurring sections must be cached for days so per-call spend is
    # a handful/day, not thousands. (FD honors these TTLs; see config comment.)
    assert ttls["fundamentals"] >= 10080      # >= 7 days
    assert ttls["institutional"] >= 10080


def test_fd_news_limit_is_capped_at_10(monkeypatch):
    from rh_agent.providers.financial_datasets import FinancialDatasetsProvider

    fd = FinancialDatasetsProvider.__new__(FinancialDatasetsProvider)
    seen = {}

    def fake_cached(section, ticker, ttl, path, params):
        seen.update(params)
        return {"news": [{"title": "h"}] * 10}

    monkeypatch.setattr(fd, "_cached", fake_cached)
    fd.get_headlines("AAPL", limit=50)
    assert seen["limit"] <= 10

def test_mboum_still_present_where_capable():
    p = _providers()
    assert p["universe"][0] == "mboum"
    assert p["news_sentiment"][0] == "alphavantage"
    assert "mboum" not in p.get("news_sentiment", [])   # mboum has no sentiment endpoint
