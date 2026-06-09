from __future__ import annotations

import pytest


def test_http_client_post_json_honors_offline(monkeypatch):
    import rh_agent.providers.base as base

    monkeypatch.setattr(base, "OFFLINE", True)
    client = base.HttpClient("https://example.com", max_per_sec=0)

    with pytest.raises(base.ProviderError, match="RH_OFFLINE"):
        client.post_json("/search", {"query": "AAPL"})
