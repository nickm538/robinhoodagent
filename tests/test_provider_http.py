from __future__ import annotations

import pytest
import requests


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


@pytest.mark.parametrize("method", ["get_json", "post_json"])
def test_http_client_does_not_retry_permanent_4xx(monkeypatch, method):
    import rh_agent.providers.base as base

    calls = 0

    class _Resp:
        status_code = 403
        headers = {}

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp()

    client = base.HttpClient("https://example.com", max_per_sec=0)
    monkeypatch.setattr(client.session, "get" if method == "get_json" else "post", request)

    with pytest.raises(base.ProviderHTTPError) as exc:
        getattr(client, method)("/restricted", {})
    assert exc.value.status_code == 403
    assert calls == 1


def test_http_client_retries_transient_5xx(monkeypatch):
    import rh_agent.providers.base as base

    calls = 0

    class _Resp:
        status_code = 503
        headers = {}

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp()

    client = base.HttpClient("https://example.com", max_per_sec=0)
    monkeypatch.setattr(client.session, "get", request)
    monkeypatch.setattr(base.time, "sleep", lambda _: None)

    with pytest.raises(base.ProviderError, match="after 3 tries"):
        client.get_json("/temporarily-unavailable")
    assert calls == 3
