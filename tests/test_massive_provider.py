"""Massive (ex-Polygon) provider: parsing, entitlement/429 guards, batch
prefetch routing, factory wiring, and the niche-routing config pins."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rh_agent.config import load_config
from rh_agent.models import Quote
from rh_agent.providers.base import ProviderError, ProviderUnsupported, RateLimitError
from rh_agent.providers.massive import MassiveProvider


def _provider() -> MassiveProvider:
    p = MassiveProvider.__new__(MassiveProvider)
    p._rate_limited_until = 0.0
    p._unsupported = set()

    class _Cache:
        def __init__(self):
            self.store = {}

        def get(self, ns, key, ttl):
            return self.store.get((ns, key))

        def set(self, ns, key, data, source=""):
            self.store[(ns, key)] = data

    p.cache = _Cache()
    return p


def _snapshot_ticker(sym="AAPL", last=210.5, day_c=209.0, prev_c=205.0,
                     change_pct=2.68, vol=51_000_000, updated_ns=None):
    return {
        "ticker": sym,
        "todaysChangePerc": change_pct,
        "todaysChange": 5.5,
        "updated": updated_ns if updated_ns is not None
        else int(datetime.now(timezone.utc).timestamp() * 1e9),
        "day": {"o": 206.0, "h": 211.0, "l": 205.5, "c": day_c, "v": vol, "vw": 208.7},
        "min": {"av": 50_000_000, "c": 210.4, "v": 120_000},
        "lastTrade": {"p": last, "s": 100},
        "lastQuote": {"P": 210.6, "p": 210.4},
        "prevDay": {"c": prev_c, "v": 48_000_000},
    }


# ------------------------------------------------------------------- quotes

def test_quote_parses_snapshot_with_honest_timestamp():
    p = _provider()
    ns = int(datetime(2026, 6, 12, 15, 30, tzinfo=timezone.utc).timestamp() * 1e9)
    p._q = lambda section, ttl, path, params=None: {
        "ticker": _snapshot_ticker(updated_ns=ns)}
    q = p.get_quote("aapl")
    assert q.ticker == "AAPL"
    assert q.price == pytest.approx(210.5)        # lastTrade preferred
    assert q.prev_close == pytest.approx(205.0)
    assert q.day_change_pct == pytest.approx(2.68)
    assert q.volume == pytest.approx(51_000_000)
    assert q.source == "massive"
    # asof comes from the snapshot's own `updated` (delayed plans stay honest)
    assert q.asof == datetime(2026, 6, 12, 15, 30, tzinfo=timezone.utc)


def test_quote_price_fallback_chain_and_rejects_empty():
    p = _provider()
    tk = _snapshot_ticker()
    tk["lastTrade"] = {}
    tk["min"] = {}
    tk["day"] = {"c": 0}                           # zero day close pre-market
    p._q = lambda *a, **k: {"ticker": tk}
    assert p.get_quote("AAPL").price == pytest.approx(205.0)   # prevDay fallback

    p._q = lambda *a, **k: {"ticker": {}}
    with pytest.raises(ProviderUnsupported):
        p.get_quote("AAPL")


def test_quotes_batch_chunks_at_100_and_merges():
    p = _provider()
    calls: list[int] = []

    def fake_q(section, ttl, path, params=None):
        syms = params["tickers"].split(",")
        calls.append(len(syms))
        return {"tickers": [_snapshot_ticker(s) for s in syms]}

    p._q = fake_q
    out = p.get_quotes_batch([f"T{i:03d}A" for i in range(230)])
    assert calls == [100, 100, 30]
    assert len(out) == 230
    assert all(isinstance(q, Quote) for q in out.values())


def test_quotes_batch_stops_when_cooldown_trips_midway(monkeypatch):
    import time as _time
    p = _provider()
    state = {"n": 0}

    def fake_q(section, ttl, path, params=None):
        state["n"] += 1
        if state["n"] == 2:
            p._rate_limited_until = _time.time() + 60
            raise ProviderUnsupported("cooling")
        return {"tickers": [_snapshot_ticker(s) for s in params["tickers"].split(",")]}

    p._q = fake_q
    out = p.get_quotes_batch([f"S{i:03d}B" for i in range(250)])
    assert state["n"] == 2                  # chunk 1 ok, chunk 2 trips, chunk 3 never sent
    assert len(out) == 100


# ------------------------------------------------------------------- prices

def test_prices_convert_epoch_ms_to_real_dates():
    p = _provider()
    day_ms = 86_400_000
    base = 1_760_000_000_000                  # ~2025-10 epoch ms
    p._q = lambda section, ttl, path, params=None: {
        "results": [
            {"t": base, "o": 10, "h": 11, "l": 9.5, "c": 10.5, "v": 1_000_000},
            {"t": base + day_ms, "o": 10.5, "h": 12, "l": 10.4, "c": 11.8, "v": 1_500_000},
        ]
    }
    df = p.get_prices("AAPL")
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert len(df) == 2
    assert df.index[0].year >= 2025                      # NOT 1970 (ms vs ns trap)
    assert df["adj_close"].iloc[-1] == pytest.approx(11.8)

    p._q = lambda *a, **k: {"results": []}
    with pytest.raises(ProviderUnsupported):
        p.get_prices("AAPL")


# ------------------------------------------------------------------ company

def test_company_facts_with_shares_outstanding():
    p = _provider()
    p._q = lambda section, ttl, path, params=None: {"results": {
        "name": "Apple Inc.", "market_cap": 3.1e12, "primary_exchange": "XNAS",
        "sic_description": "Electronic Computers", "total_employees": 150_000,
        "weighted_shares_outstanding": 15_000_000_000,
    }}
    c = p.get_company("AAPL")
    assert c["market_cap"] == pytest.approx(3.1e12)
    assert c["shares_outstanding"] == pytest.approx(15e9)
    assert c["industry"] == "Electronic Computers"
    # SIC description doubles as sector so company can route here (flat-rate)
    # and still drive the portfolio sector cap with a consistent grouping.
    assert c["sector"] == "Electronic Computers"


# --------------------------------------------------------------------- news

def test_headlines_and_insight_sentiment_score():
    p = _provider()
    p._q = lambda section, ttl, path, params=None: {"results": [
        {"title": "AAPL beats on earnings",
         "insights": [{"ticker": "AAPL", "sentiment": "positive"},
                      {"ticker": "MSFT", "sentiment": "negative"}]},
        {"title": "Supply chain wobble",
         "insights": [{"ticker": "AAPL", "sentiment": "negative"}]},
        {"title": "Neutral note",
         "insights": [{"ticker": "AAPL", "sentiment": "neutral"}]},
    ]}
    heads = p.get_headlines("AAPL", limit=2)
    assert heads == ["AAPL beats on earnings", "Supply chain wobble"]

    s = p.get_news_sentiment("AAPL")
    assert s["score"] == pytest.approx(0.0)       # (+1 -1 +0)/3 — MSFT insight ignored
    assert s["article_count"] == 3
    assert s["label"] == "neutral"

    p._q = lambda *a, **k: {"results": [{"title": "no insights here"}]}
    with pytest.raises(ProviderUnsupported):      # defer to AV/FD instead of faking 0
        p.get_news_sentiment("AAPL")


# ------------------------------------------------------------------- movers

def test_movers_returns_clean_gainers_only():
    p = _provider()
    p._q = lambda section, ttl, path, params=None: {"tickers": [
        {"ticker": "RUNR"}, {"ticker": "B.WS"}, {"ticker": "GAP-U"},
        {"ticker": "MOON"}, {"ticker": "RUNR"},
    ]}
    assert p.get_market_movers(limit=10) == ["RUNR", "MOON"]

    p._q = lambda *a, **k: {"tickers": []}
    with pytest.raises(ProviderUnsupported):
        p.get_market_movers()


# --------------------------------------------------------------- technicals

def test_technicals_parse_rsi_ema_macd():
    p = _provider()

    def fake_q(section, ttl, path, params=None):
        if "/rsi/" in path:
            return {"results": {"values": [{"value": "62.1"}]}}
        if "/ema/" in path:
            return {"results": {"values": [
                {"value": "101.5" if params["window"] == 9 else "99.2"}]}}
        if "/macd/" in path:
            return {"results": {"values": [
                {"value": "1.4", "signal": "0.9", "histogram": "0.5"}]}}
        raise AssertionError(path)

    p._q = fake_q
    t = p.get_technicals("MSFT")
    assert t["rsi"] == pytest.approx(62.1)
    assert t["ema_9"] == pytest.approx(101.5)
    assert t["ema_21"] == pytest.approx(99.2)
    assert t["macd_hist"] == pytest.approx(0.5)


# ----------------------------------------------------------- short interest

def test_short_interest_with_pct_float_fraction():
    p = _provider()

    def fake_q(section, ttl, path, params=None):
        if "short-interest" in path:
            return {"results": [{"settlement_date": "2026-05-29",
                                 "short_interest": 75_000_000,
                                 "avg_daily_volume": 25_000_000,
                                 "days_to_cover": 3.0}]}
        return {"results": {"weighted_shares_outstanding": 1_500_000_000,
                            "name": "X", "market_cap": 1e10}}

    p._q = fake_q
    si = p.get_short_interest("GME")
    assert si["short_shares"] == pytest.approx(75e6)
    assert si["days_to_cover"] == pytest.approx(3.0)
    assert si["short_pct_float"] == pytest.approx(0.05)   # fraction, Yahoo-style
    assert si["settlement_date"] == "2026-05-29"


def test_short_interest_omits_pct_when_shares_unknown():
    p = _provider()

    def fake_q(section, ttl, path, params=None):
        if "short-interest" in path:
            return {"results": [{"short_interest": 1_000_000, "days_to_cover": 2.0}]}
        raise ProviderUnsupported

    p._q = fake_q
    si = p.get_short_interest("XYZ")
    assert "short_pct_float" not in si
    assert si["days_to_cover"] == pytest.approx(2.0)


# ----------------------------------------------------------------- universe

def test_list_universe_paginates_and_filters():
    p = _provider()
    pages = {
        "/v3/reference/tickers": {
            "results": [{"ticker": "AAPL"}, {"ticker": "B.B"}, {"ticker": "MSFT"}],
            "next_url": "https://api.massive.com/v3/reference/tickers?cursor=abc",
        },
        "https://api.massive.com/v3/reference/tickers?cursor=abc": {
            "results": [{"ticker": "NVDA"}, {"ticker": "GAP-U"}],
        },
    }
    p._q = lambda section, ttl, path, params=None: pages[path]
    assert p.list_universe() == ["AAPL", "MSFT", "NVDA"]


# ------------------------------------------------------ guards & wiring

def test_429_trips_cooldown_and_stops_calling():
    p = _provider()
    calls = {"n": 0}

    class _Http:
        def get_json(self, path, params):
            calls["n"] += 1
            raise RateLimitError("429", retry_after_seconds=None)

    p.http = _Http()
    with pytest.raises(ProviderUnsupported):
        p._q("snap", 1, "/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
    assert calls["n"] == 1 and p._rate_limit_active()
    with pytest.raises(ProviderUnsupported):
        p._q("snap", 1, "/v2/snapshot/locale/us/markets/stocks/tickers/MSFT")
    assert calls["n"] == 1


def test_403_marks_section_unsupported_permanently():
    p = _provider()
    calls = {"n": 0}

    class _Http:
        def get_json(self, path, params):
            calls["n"] += 1
            raise ProviderError("GET x failed after 3 tries: 403 Client Error")

    p.http = _Http()
    with pytest.raises(ProviderUnsupported):
        p._q("short", 1440, "/stocks/v1/short-interest", {"ticker": "AAPL"})
    with pytest.raises(ProviderUnsupported):      # no second HTTP round trip
        p._q("short", 1440, "/stocks/v1/short-interest", {"ticker": "MSFT"})
    assert calls["n"] == 1
    assert "short" in p._unsupported


def test_prefetch_prefers_massive_then_falls_back_to_twelvedata():
    from rh_agent.data.market_data import MarketData

    class _Massive:
        def get_quotes_batch(self, tickers):
            return {t: Quote(t, 10.0, source="massive") for t in tickers}

    class _TD:
        def get_quotes_batch(self, tickers):
            return {t: Quote(t, 11.0, source="twelvedata") for t in tickers}

    md = MarketData.__new__(MarketData)
    md.providers = {"massive": _Massive(), "twelvedata": _TD()}
    md.priority = {"quote": ["financialdatasets", "massive", "twelvedata"]}
    md._quote_prefetch = {}
    assert md.prefetch_quotes(["AAA", "BBB"]) == 2
    assert md._quote_prefetch["AAA"].source == "massive"

    class _DeadMassive:
        def get_quotes_batch(self, tickers):
            raise RuntimeError("down")

    md2 = MarketData.__new__(MarketData)
    md2.providers = {"massive": _DeadMassive(), "twelvedata": _TD()}
    md2.priority = {"quote": ["massive", "twelvedata"]}
    md2._quote_prefetch = {}
    assert md2.prefetch_quotes(["AAA"]) == 1
    assert md2._quote_prefetch["AAA"].source == "twelvedata"


def test_build_providers_wires_massive_from_env(monkeypatch):
    from rh_agent.providers import build_providers

    from rh_agent.config import _KEY_ENV

    for provider, names in _KEY_ENV.items():
        if provider == "massive":
            continue
        for var in names:                          # hosts pre-seed alternate spellings
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    providers = build_providers(load_config())
    assert list(providers) == ["massive"]
    assert providers["massive"].name == "massive"


def test_niche_routing_config_pins():
    p = load_config().get("providers", {})
    # Massive owns movers + short interest, and is now the flat-rate PRIMARY for
    # the high-volume per-call sections (the cost-optimization).
    assert p["movers"][0] == "massive"
    assert p["short_interest"][0] == "massive"
    for section in ("prices", "quote", "quote_risk", "fundamentals"):
        assert p[section][0] == "massive", f"{section}: massive must lead, got {p[section]}"
    for section in ("news_headlines", "news_sentiment", "universe", "technicals"):
        assert "massive" in p[section], f"{section}: massive missing from {p[section]}"
    assert p["universe"][0] == "mboum"   # listing still mboum-first
