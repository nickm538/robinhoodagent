"""Finnhub provider: parsing into the engine's factor-ready shapes, the
analyst-ratings restoration (the Mboum-gap niche), guardrails, and routing pins.

Field names are asserted against what factors/library.py actually consumes, so a
parse that silently feeds nothing would fail here."""
from __future__ import annotations

import pytest

from rh_agent.config import load_config
from rh_agent.models import Quote
from rh_agent.providers.base import ProviderUnsupported, RateLimitError
from rh_agent.providers.finnhub import FinnhubProvider


def _provider() -> FinnhubProvider:
    p = FinnhubProvider.__new__(FinnhubProvider)
    p._rate_limited_until = 0.0

    class _Cache:
        def __init__(self):
            self.store = {}

        def get(self, ns, key, ttl):
            return self.store.get((ns, key))

        def set(self, ns, key, data, source=""):
            self.store[(ns, key)] = data

    p.cache = _Cache()
    return p


# --------------------------------------------------------------------- quote

def test_quote_parses_and_rejects_zero_price():
    p = _provider()
    p._get = lambda *a, **k: {"c": 261.74, "d": 1.5, "dp": 0.5766, "pc": 260.24,
                              "t": 1_760_000_000}
    q = p.get_quote("aapl")
    assert isinstance(q, Quote)
    assert q.ticker == "AAPL"
    assert q.price == pytest.approx(261.74)
    assert q.prev_close == pytest.approx(260.24)
    assert q.day_change_pct == pytest.approx(0.5766)
    assert q.source == "finnhub"
    assert q.asof.year >= 2025          # honest timestamp from `t`

    p._get = lambda *a, **k: {"c": 0, "pc": 0}     # unknown/closed symbol on free tier
    with pytest.raises(ProviderUnsupported):
        p.get_quote("ZZZZ")


# ------------------------------------------------------------------- company

def test_company_scales_millions_to_absolute():
    p = _provider()
    p._get = lambda *a, **k: {"name": "Apple Inc", "finnhubIndustry": "Technology",
                              "marketCapitalization": 3_000_000, "shareOutstanding": 15_000.0,
                              "exchange": "NASDAQ"}
    c = p.get_company("AAPL")
    assert c["name"] == "Apple Inc"
    assert c["sector"] == "Technology"
    assert c["market_cap"] == pytest.approx(3e12)         # millions -> absolute
    assert c["shares_outstanding"] == pytest.approx(15e9)


# --------------------------------------------------- analyst (the niche win)

def test_analyst_collapses_recommendation_to_factor_fields():
    p = _provider()
    p._get = lambda *a, **k: [
        {"period": "2026-06-01", "strongBuy": 13, "buy": 11, "hold": 7, "sell": 1, "strongSell": 0},
        {"period": "2026-05-01", "strongBuy": 10, "buy": 9, "hold": 8, "sell": 2, "strongSell": 1},
    ]
    a = p.get_analyst("AAPL")
    # factor analyst_consensus_upside reads buy/hold/sell -> must be present + newest period
    assert a["buy"] == 24          # strongBuy + buy, most recent row only
    assert a["hold"] == 7
    assert a["sell"] == 1          # sell + strongSell
    assert a["source"] == "finnhub"

    p._get = lambda *a, **k: []
    with pytest.raises(ProviderUnsupported):
        p.get_analyst("AAPL")


def test_analyst_feeds_consensus_factor_end_to_end():
    from rh_agent.factors.library import analyst_consensus_upside
    from rh_agent.models import TickerData

    p = _provider()
    p._get = lambda *a, **k: [{"period": "2026-06-01", "strongBuy": 8, "buy": 4,
                               "hold": 2, "sell": 0, "strongSell": 0}]
    td = TickerData("AAA")
    td.analyst = p.get_analyst("AAA")
    # 12 buys of 14 -> ratio .857 -> factor returns 0.3*ratio (no price target) > 0
    assert analyst_consensus_upside(td) == pytest.approx(0.3 * (12 / 14), abs=1e-6)


# ------------------------------------------------------------- insider flow

def test_insider_maps_change_to_buy_sell_with_value():
    from rh_agent.factors.library import insider_net_buying
    from rh_agent.models import TickerData

    p = _provider()
    p._get = lambda *a, **k: {"data": [
        {"change": 1000, "transactionPrice": 50.0, "transactionDate": "2026-06-01"},
        {"change": -400, "transactionPrice": 52.0, "transactionDate": "2026-06-02"},
        {"change": 0, "transactionPrice": 51.0},          # no-op, dropped
    ]}
    rows = p.get_insider("AAA")
    assert len(rows) == 2
    assert rows[0] == {"shares": 1000, "value": 50_000.0, "is_buy": True,
                       "date": "2026-06-01"}
    assert rows[1]["is_buy"] is False
    # end-to-end: net buying = (+50000 - 20800)/mcap
    td = TickerData("AAA", company={"market_cap": 1e9})
    td.insider = rows
    assert insider_net_buying(td) == pytest.approx((50_000 - 20_800) / 1e9)


# ---------------------------------------------------------------- earnings

def test_earnings_surprise_history():
    from rh_agent.factors.library import earnings_surprise_history
    from rh_agent.models import TickerData

    p = _provider()
    p._get = lambda *a, **k: [
        {"surprisePercent": 8.0}, {"surprisePercent": -2.0},
        {"surprisePercent": 5.0}, {"surprisePercent": 1.0}, {"surprisePercent": 99.0},
    ]
    e = p.get_earnings("AAA")
    assert e["avg_surprise_pct"] == pytest.approx((8 - 2 + 5 + 1) / 4)   # last 4 only
    assert e["beat_rate"] == pytest.approx(3 / 4)
    td = TickerData("AAA")
    td.earnings = e
    assert earnings_surprise_history(td) == pytest.approx(3.0)


# --------------------------------------------------------------- headlines

def test_headlines_dedup_and_limit():
    p = _provider()
    p._get = lambda *a, **k: [{"headline": "A"}, {"headline": "B"}, {"headline": "A"},
                              {"headline": "C"}]
    assert p.get_headlines("AAA", limit=2) == ["A", "B"]


# --------------------------------------------------------------- universe

def test_list_universe_filters_common_stock():
    p = _provider()
    p._get = lambda *a, **k: [
        {"symbol": "AAPL", "type": "Common Stock"},
        {"symbol": "SPY", "type": "ETF"},
        {"symbol": "BRK.B", "type": "Common Stock"},
        {"symbol": "MSFT", "type": "Common Stock"},
    ]
    assert p.list_universe() == ["AAPL", "MSFT"]


# -------------------------------------------------------- guards & no-prices

def test_429_trips_cooldown_then_skips_calls():
    p = _provider()
    calls = {"n": 0}

    class _Http:
        def get_json(self, path, params):
            calls["n"] += 1
            raise RateLimitError("429", retry_after_seconds=None)

    p.http = _Http()
    with pytest.raises(ProviderUnsupported):
        p._get("quote", 1, "/quote", {"symbol": "AAPL"})
    assert calls["n"] == 1 and p._rate_limit_active()
    with pytest.raises(ProviderUnsupported):
        p._get("quote", 1, "/quote", {"symbol": "MSFT"})
    assert calls["n"] == 1            # no HTTP during cooldown


def test_finnhub_has_no_price_or_fundamentals_methods():
    # Must NOT enter prices/technicals chains (free tier lacks candles) and must
    # contribute no fundamentals ratios (unit-mix risk) -> base raises Unsupported.
    p = _provider()
    with pytest.raises(ProviderUnsupported):
        p.get_prices("AAPL")
    with pytest.raises(ProviderUnsupported):
        p.get_fundamentals("AAPL")


# ------------------------------------------------------------ wiring + routing

def test_build_providers_wires_finnhub(monkeypatch):
    from rh_agent.config import _KEY_ENV
    from rh_agent.providers import build_providers

    for names in _KEY_ENV.values():
        for var in names:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    providers = build_providers(load_config())
    assert list(providers) == ["finnhub"]


def test_routing_keeps_finnhub_out_of_prices_and_backs_analyst():
    p = load_config().get("providers", {})
    assert "finnhub" not in p["prices"]          # no free candles -> never a price source
    assert "finnhub" not in p["technicals"]
    assert "finnhub" in p["analyst_ratings"] and p["analyst_ratings"][0] == "mboum"
    for section in ("quote", "quote_risk", "insider", "news_headlines"):
        assert "finnhub" in p[section]
        assert p[section][-1] in ("finnhub", "alphavantage", "web")  # always a late fallback
