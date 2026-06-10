"""TwelveData provider and MarketData integration tests."""
from __future__ import annotations

from rh_agent.data.market_data import MarketData
from rh_agent.models import Quote
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


def test_get_quotes_batch_adapts_and_tolerates_chunk_failure():
    """A plan that rejects large comma-batches must shrink the request and keep
    the quotes from the chunks that do succeed (not throw the whole batch away)."""
    from rh_agent.providers.base import ProviderUnsupported
    p = TwelveDataProvider.__new__(TwelveDataProvider)
    p._batch_size = 4                      # starts too big for this fake plan
    sizes_seen: list[int] = []

    def fake_q(section, ttl, path, params):
        syms = params["symbol"].split(",")
        sizes_seen.append(len(syms))
        if len(syms) > 2:                  # plan caps at 2 symbols/request
            raise ProviderUnsupported("batch too large")
        return {s: {"symbol": s, "close": "100.0", "volume": "1000"} for s in syms}

    p._q = fake_q
    out = p.get_quotes_batch(["AAA", "BBB", "CCC", "DDD", "EEE"])

    assert set(out) == {"AAA", "BBB", "CCC", "DDD", "EEE"}   # nothing dropped
    assert all(q.price == 100.0 for q in out.values())
    assert p._batch_size <= 2                                # learned the smaller size
    assert max(sizes_seen) == 4 and min(sizes_seen) <= 2     # tried big, then shrank


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
