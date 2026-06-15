"""Headless re-auth: parsing the pasted OAuth redirect URL.

The operator on Cloud Console browser SSH (no port-forward) approves in their
own browser and pastes the failed-to-load redirect URL back; we must recover
(code, state) from whatever form they paste."""
from __future__ import annotations

from rh_agent.broker.oauth import _parse_redirect


def test_parse_full_localhost_redirect():
    url = "http://localhost:8765/callback?code=ABC123&state=xyz789"
    assert _parse_redirect(url) == ("ABC123", "xyz789")


def test_parse_redirect_code_only_url():
    url = "http://localhost:8765/callback?code=ONLYCODE"
    assert _parse_redirect(url) == ("ONLYCODE", None)


def test_parse_redirect_extra_params_and_order():
    url = "http://localhost:8765/callback?state=ST&code=CD&scope=read"
    assert _parse_redirect(url) == ("CD", "ST")


def test_parse_bare_query_string_without_scheme():
    assert _parse_redirect("code=BAREQ&state=S2") == ("BAREQ", "S2")


def test_parse_bare_code_token():
    # operator pasted just the code value
    assert _parse_redirect("justthecode123") == ("justthecode123", None)


def test_parse_whitespace_is_trimmed():
    assert _parse_redirect("  http://localhost:8765/callback?code=C&state=S  ") == ("C", "S")


def test_parse_empty_or_codeless_returns_none():
    assert _parse_redirect("") == (None, None)
    assert _parse_redirect("   ") == (None, None)
    # a URL with no code (e.g. an error redirect) is not mistaken for a code
    assert _parse_redirect("http://localhost:8765/callback?error=access_denied") == (None, None)
