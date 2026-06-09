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
