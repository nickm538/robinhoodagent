from __future__ import annotations

import pytest


def test_http_client_post_json_honors_offline(monkeypatch):
    import rh_agent.providers.base as base

    monkeypatch.setattr(base, "OFFLINE", True)
    client = base.HttpClient("https://example.com", max_per_sec=0)

    with pytest.raises(base.ProviderError, match="RH_OFFLINE"):
        client.post_json("/search", {"query": "AAPL"})


def test_http_client_get_json_raises_rate_limit_error_immediately():
    import rh_agent.providers.base as base

    class _Resp:
        status_code = 429
        headers = {}

    client = base.HttpClient("https://example.com", max_per_sec=0)
    client.session.get = lambda *args, **kwargs: _Resp()

    with pytest.raises(base.RateLimitError, match="HTTP 429"):
        client.get_json("/quote", {"symbol": "AAPL"})
