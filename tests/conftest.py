"""Shared test guards.

Tests build daemons via ``__new__`` all over the suite; the class-level
ActivityLog fallback would otherwise write their synthetic events into the
real ``state/activity.jsonl`` — which `rh-agent why` then reports as live
trading activity. Redirect it to a per-session temp file.
"""
from __future__ import annotations

import pytest

from rh_agent.activity import ActivityLog


@pytest.fixture(autouse=True)
def _isolate_activity_ledger(tmp_path, monkeypatch):
    import rh_agent.daemon as daemon
    monkeypatch.setattr(daemon.AlwaysOnAgent, "activity",
                        ActivityLog(path=tmp_path / "activity.jsonl"), raising=False)
    yield
