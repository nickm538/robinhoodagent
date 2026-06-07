"""Account/credential state must be created with owner-only permissions
(addresses the 'Insecure File Creation Permissions' review finding)."""
from __future__ import annotations

import json
import os
import stat

import pytest

from rh_agent.broker.paper import PaperBroker
from rh_agent.config import write_private
from rh_agent.models import Order

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX file permissions not enforced on Windows")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_write_private_creates_owner_only_file(tmp_path):
    p = tmp_path / "nested" / "secret.json"
    write_private(p, '{"token": "x"}')
    assert p.read_text() == '{"token": "x"}'
    assert _mode(p) == 0o600              # rw for owner only
    assert _mode(p.parent) == 0o700       # parent dir tightened


def test_write_private_overwrites_and_keeps_perms(tmp_path):
    p = tmp_path / "s.json"
    write_private(p, "first")
    write_private(p, "second")
    assert p.read_text() == "second"
    assert _mode(p) == 0o600


def test_paper_account_state_is_private(tmp_path):
    state = tmp_path / "paper_account.json"
    broker = PaperBroker(lambda t: 100.0, starting_cash=1_000, state_path=state)
    broker.place_order(Order("AAA", "buy", 1), dry_run=False)
    assert state.exists()
    assert _mode(state) == 0o600
    # sanity: it still round-trips as valid JSON account state
    assert "positions" in json.loads(state.read_text())


def test_daemon_state_is_private(tmp_path, monkeypatch):
    import rh_agent.daemon as daemon
    p = tmp_path / "state" / "daemon_state.json"
    monkeypatch.setattr(daemon, "STATE", p)
    st = daemon.DaemonState(stops={"AAA": 90.0}, take_profits={}, high_water={},
                            pending_risk={})
    st.save()
    assert _mode(p) == 0o600
    assert _mode(p.parent) == 0o700
