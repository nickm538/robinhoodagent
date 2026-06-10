"""Shipped-config provider routing mandate.

MBOUM + FinancialDatasets.AI are the backbone. FD is primary for its deep-data
specialty (fundamentals, insider, institutional, news) — low-frequency, cheap.
MBOUM owns the high-frequency reads (quote/quote_risk/prices); FD is kept OFF
those lists because per-name quotes across the wide universe were the dominant
API spend. TwelveData stays COMPLEMENTARY (batch prefetch, fallback). These
tests pin the shipped routing so a tuning PR can't silently change it.
"""
from __future__ import annotations

from rh_agent.config import load_config


def _providers():
    return load_config().get("providers", {})


def test_financialdatasets_primary_for_deep_data():
    """FD owns its low-frequency deep-data specialty (cheap, daily-cached)."""
    p = _providers()
    for section in ("fundamentals", "insider", "institutional", "news_headlines"):
        assert p[section][0] == "financialdatasets", (
            f"{section}: financialdatasets must be primary, got {p[section]}")


def test_mboum_primary_and_fd_off_high_frequency_quotes():
    """High-frequency reads route to MBOUM first; FD stays OFF those lists to
    avoid the per-name quote spend that dominated cost."""
    p = _providers()
    for section in ("quote", "quote_risk", "prices"):
        assert p[section][0] == "mboum", (
            f"{section}: mboum must be primary, got {p[section]}")
        assert "financialdatasets" not in p[section], (
            f"{section}: FD must stay off the high-frequency list, got {p[section]}")


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
