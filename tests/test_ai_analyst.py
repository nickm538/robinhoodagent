"""Tests for the AI analyst overlay — all offline (no network / no SDK needed)."""
from __future__ import annotations

from rh_agent.analysts.ai_analyst import AIAnalyst
from rh_agent.config import load_config


def test_ai_analyst_noop_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = AIAnalyst(load_config())
    assert a.enabled is False
    res = a.assess("market ctx", [{"ticker": "NVDA"}])
    assert res.views == {} and res.market_read == ""


class _Blk:
    def __init__(self, t, x=""):
        self.type, self.text = t, x


class _Resp:
    def __init__(self, text):
        self.content = [_Blk("thinking"), _Blk("text", text)]
        self.usage = None


def test_ai_analyst_parses_and_clamps(monkeypatch):
    a = AIAnalyst(load_config())
    a.enabled = True  # force on; inject a fake client (no SDK import)
    payload = ('{"market_read":"narrow risk-on tape",'
               '"assessments":[{"ticker":"nvda","score":150,"stance":"bullish","rationale":"AI capex"},'
               '{"ticker":"XYZ","score":-5,"stance":"bearish","rationale":"guide cut"}]}')

    class _Msgs:
        def create(self, **kw):
            # static system prompt must be cached + sent as a list block
            assert isinstance(kw["system"], list)
            assert kw["system"][0]["cache_control"]["type"] == "ephemeral"
            return _Resp(payload)

    class _Client:
        messages = _Msgs()

    a._client = _Client()
    res = a.assess("ctx", [{"ticker": "NVDA"}, {"ticker": "XYZ"}])
    assert res.market_read == "narrow risk-on tape"
    assert res.views["NVDA"]["score"] == 100.0   # clamped from 150
    assert res.views["XYZ"]["score"] == 0.0       # clamped from -5
    assert res.views["NVDA"]["stance"] == "bullish"


def test_ai_analyst_bad_json_is_safe(monkeypatch):
    a = AIAnalyst(load_config())
    a.enabled = True

    class _Msgs:
        def create(self, **kw):
            return _Resp("not json at all")

    class _Client:
        messages = _Msgs()

    a._client = _Client()
    res = a.assess("ctx", [{"ticker": "NVDA"}])
    assert res.views == {}   # unparseable -> safe empty, no crash
