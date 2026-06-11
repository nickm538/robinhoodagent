"""TwelveData provider and MarketData integration tests."""
from __future__ import annotations

import pytest

from rh_agent.data.market_data import MarketData
from rh_agent.models import Quote
from rh_agent.providers.base import ProviderUnsupported, RateLimitError
from rh_agent.providers.twelvedata import TwelveDataProvider


def test_twelvedata_batch_quote_parsing():
    p = TwelveDataProvider.__new__(TwelveDataProvider)
    p._q = lambda section, ttl, path, params: {
        "AAPL": {
            "symbol": "AAPL",
            "close": "150.0",
            "volume": "1000",
            "previous_close": "148.0",
            "percent_change": "1.35",
        },
        "MSFT": {
            "symbol": "MSFT",
            "close": "400.0",
            "volume": "500",
            "previous_close": "395.0",
            "percent_change": "1.27",
        },
    }
    out = p.get_quotes_batch(["AAPL", "MSFT"])
    assert set(out) == {"AAPL", "MSFT"}
    assert out["AAPL"].price == 150.0
    assert out["AAPL"].day_change_pct == 1.35


def test_twelvedata_single_symbol_batch_shape():
    p = TwelveDataProvider.__new__(TwelveDataProvider)
    p._q = lambda section, ttl, path, params: {
        "symbol": "NVDA",
        "close": "900.0",
        "volume": "1",
        "previous_close": "880.0",
        "percent_change": "2.27",
    }
    out = p.get_quotes_batch(["NVDA"])
    assert out["NVDA"].price == 900.0


def test_twelvedata_market_movers_parsing_when_enabled():
    p = TwelveDataProvider.__new__(TwelveDataProvider)
    p.enable_market_movers = True
    p._q = lambda section, ttl, path, params: {
        "values": [
            {"symbol": "RUN1"},
            {"symbol": "RUN2"},
            {"symbol": "BAD.CLASS"},
        ],
        "status": "ok",
    }
    assert p.get_market_movers(limit=5) == ["RUN1", "RUN2"]


def test_market_data_prefetch_and_quote_risk_order():
    class _TD:
        name = "twelvedata"

        def get_quotes_batch(self, tickers):
            return {t: Quote(ticker=t, price=10.0, source="twelvedata") for t in tickers}

        def get_quote(self, ticker):
            return Quote(ticker=ticker, price=99.0, source="twelvedata")

    class _Mb:
        name = "mboum"

        def get_quote(self, ticker):
            return Quote(ticker=ticker, price=1.0, source="mboum")

    md = MarketData.__new__(MarketData)
    md.cfg = None
    md.providers = {"twelvedata": _TD(), "mboum": _Mb()}
    md.priority = {
        "quote": ["mboum", "twelvedata"],
        "quote_risk": ["twelvedata", "mboum"],
    }
    md._quote_prefetch = {}

    assert md.prefetch_quotes(["AAA", "BBB"]) == 2
    q = md.get_quote("AAA")
    assert q.source == "twelvedata"
    assert q.price == 10.0

    md.clear_quote_prefetch()
    q_risk = md.get_quote_for_risk("ZZZ", max_age_seconds=300)
    assert q_risk.source == "twelvedata"
    assert q_risk.price == 99.0


def test_market_movers_merge_providers():
    class _AV:
        def get_market_movers(self, limit):
            return ["AAA", "BBB"]

    class _TD:
        def get_market_movers(self, limit):
            return ["BBB", "CCC"]

    md = MarketData.__new__(MarketData)
    md.cfg = None
    md.providers = {"alphavantage": _AV(), "twelvedata": _TD()}
    md.priority = {"movers": ["alphavantage", "twelvedata"]}
    md._quote_prefetch = {}
    assert md.get_market_movers(limit=10) == ["AAA", "BBB", "CCC"]


def test_twelvedata_rate_limit_enters_cooldown_and_skips_repeat_calls():
    p = TwelveDataProvider.__new__(TwelveDataProvider)
    p.api_key = "x"
    p.rate_limit_cooldown_seconds = 30.0
    p._rate_limited_until = 0.0

    class _Cache:
        def get(self, namespace, key, ttl):
            return None
        def set(self, namespace, key, data, source=""):
            raise AssertionError("cache.set should not run on rate limit")

    class _Http:
        def __init__(self):
            self.calls = 0
        def get_json(self, path, params):
            self.calls += 1
            raise RateLimitError("GET rate limited (HTTP 429)")

    p.cache = _Cache()
    p.http = _Http()

    with pytest.raises(ProviderUnsupported, match="rate limited"):
        p.get_prices("AAPL")
    assert p.http.calls == 1
    assert p._rate_limited_until > 0

    with pytest.raises(ProviderUnsupported, match="cooling down"):
        p.get_prices("MSFT")
    assert p.http.calls == 1


def test_get_quotes_batch_adapts_and_tolerates_chunk_failure():
    p = TwelveDataProvider.__new__(TwelveDataProvider)
    p._batch_size = 4
    p._rate_limited_until = 0.0
    sizes_seen: list[int] = []

    def fake_q(section, ttl, path, params):
        syms = params["symbol"].split(",")
        sizes_seen.append(len(syms))
        if len(syms) > 2:
            raise ProviderUnsupported("batch too large")
        return {s: {"symbol": s, "close": "100.0", "volume": "1000"} for s in syms}

    p._q = fake_q
    out = p.get_quotes_batch(["AAA", "BBB", "CCC", "DDD", "EEE"])

    assert set(out) == {"AAA", "BBB", "CCC", "DDD", "EEE"}
    assert all(q.price == 100.0 for q in out.values())
    assert p._batch_size <= 2
    assert max(sizes_seen) == 4 and min(sizes_seen) <= 2


def test_market_data_falls_through_scoreless_sentiment_payload():
    class _EmptySentiment:
        def get_news_sentiment(self, ticker):
            return {"article_count": 3, "source": "empty"}

    class _ScoredSentiment:
        def get_news_sentiment(self, ticker):
            return {"score": 0.4, "article_count": 2, "source": "scored"}

    md = MarketData.__new__(MarketData)
    md.providers = {"empty": _EmptySentiment(), "scored": _ScoredSentiment()}
    md.priority = {"news_sentiment": ["empty", "scored"]}

    out = md._try("news_sentiment", "get_news_sentiment", "AAA")
    assert out["score"] == 0.4
