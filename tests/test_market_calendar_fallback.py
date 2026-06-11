"""Built-in US-Eastern fallback: market hours must stay correct without tzdata."""
from __future__ import annotations

from datetime import datetime, timezone

import rh_agent.market_calendar as mc


def test_eastern_offset_statutory_dst_rule():
    # Winter (EST) / summer (EDT)
    assert mc._eastern_offset_hours(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)) == -5
    assert mc._eastern_offset_hours(datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)) == -4
    # Spring-forward: second Sunday of March 2026 is the 8th, 2:00 EST = 7:00 UTC
    assert mc._eastern_offset_hours(datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc)) == -5
    assert mc._eastern_offset_hours(datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)) == -4
    # Fall-back: first Sunday of November 2026 is the 1st, 2:00 EDT = 6:00 UTC
    assert mc._eastern_offset_hours(datetime(2026, 11, 1, 5, 59, tzinfo=timezone.utc)) == -4
    assert mc._eastern_offset_hours(datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)) == -5


def test_is_market_open_correct_without_zoneinfo(monkeypatch):
    monkeypatch.setattr(mc, "_ET", None)
    # Mon 2026-06-08 14:00 UTC = 10:00 EDT -> open
    assert mc.is_market_open(datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc)) is True
    # Mon 2026-06-08 13:00 UTC = 09:00 EDT -> pre-market (the old UTC fallback
    # would have wrongly treated 13:00 as mid-session)
    assert mc.is_market_open(datetime(2026, 6, 8, 13, 0, tzinfo=timezone.utc)) is False
    # Winter session boundary: Thu 2026-01-15 14:30 UTC = 09:30 EST -> open
    assert mc.is_market_open(datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc)) is True
    assert mc.is_market_open(datetime(2026, 1, 15, 14, 29, tzinfo=timezone.utc)) is False
    # Early close (day after Thanksgiving 2026): 12:30 ET open, 14:00 ET closed
    assert mc.is_market_open(datetime(2026, 11, 27, 17, 30, tzinfo=timezone.utc)) is True
    assert mc.is_market_open(datetime(2026, 11, 27, 19, 0, tzinfo=timezone.utc)) is False


def test_fallback_agrees_with_zoneinfo_when_available(monkeypatch):
    if mc._ET is None:  # pragma: no cover - host without tzdata
        return
    probes = [
        datetime(2026, 1, 15, 14, 45, tzinfo=timezone.utc),
        datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 8, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc),
    ]
    with_zone = [mc.is_market_open(p) for p in probes]
    monkeypatch.setattr(mc, "_ET", None)
    without_zone = [mc.is_market_open(p) for p in probes]
    assert with_zone == without_zone


def test_session_state_phases():
    s = mc.session_state(datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc))  # Thu 11:00 ET
    assert s["open"] is True
    assert s["phase"].startswith("regular")
    s2 = mc.session_state(datetime(2026, 6, 13, 15, 0, tzinfo=timezone.utc))  # Saturday
    assert s2["open"] is False
    assert s2["phase"] == "weekend"
    s3 = mc.session_state(datetime(2026, 6, 11, 21, 30, tzinfo=timezone.utc))  # 17:30 ET
    assert s3["phase"] == "after-hours"
