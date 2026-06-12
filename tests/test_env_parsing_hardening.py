"""Copilot-review hardening: malformed numeric env vars / config values must
never crash provider construction or module import — a 24/7 daemon that cannot
START because of an empty `.env` line would crash-loop under systemd forever."""
from __future__ import annotations

import importlib

import pytest

from rh_agent.providers.base import env_float


def test_env_float_defensive_parsing(monkeypatch):
    monkeypatch.delenv("RH_TEST_NUM", raising=False)
    assert env_float("RH_TEST_NUM", 7.5) == 7.5            # missing -> default

    monkeypatch.setenv("RH_TEST_NUM", "")
    assert env_float("RH_TEST_NUM", 7.5) == 7.5            # empty -> default

    monkeypatch.setenv("RH_TEST_NUM", "   ")
    assert env_float("RH_TEST_NUM", 7.5) == 7.5            # whitespace -> default

    monkeypatch.setenv("RH_TEST_NUM", "banana")
    assert env_float("RH_TEST_NUM", 7.5) == 7.5            # junk -> default

    monkeypatch.setenv("RH_TEST_NUM", "-3")
    assert env_float("RH_TEST_NUM", 7.5, minimum=0.1) == 7.5   # below floor -> default

    monkeypatch.setenv("RH_TEST_NUM", " 4 ")
    assert env_float("RH_TEST_NUM", 7.5) == 4.0            # valid (trimmed) parses


def test_fd_construction_survives_malformed_pace_env(monkeypatch):
    from rh_agent.providers.financial_datasets import FinancialDatasetsProvider

    monkeypatch.setenv("FINANCIALDATASETS_MAX_PER_SEC", "")
    p = FinancialDatasetsProvider("k")
    assert p.http.limiter.min_interval == pytest.approx(1 / 8)   # default 8/s

    monkeypatch.setenv("FINANCIALDATASETS_MAX_PER_SEC", "not-a-number")
    p2 = FinancialDatasetsProvider("k")
    assert p2.http.limiter.min_interval == pytest.approx(1 / 8)


def test_massive_module_import_survives_malformed_env(monkeypatch):
    import rh_agent.providers.massive as mv

    monkeypatch.setenv("MASSIVE_MAX_PER_SEC", "banana")
    monkeypatch.setenv("MASSIVE_RATE_LIMIT_COOLDOWN_SECONDS", "")
    mv = importlib.reload(mv)                       # must not raise at import
    assert mv.DEFAULT_MAX_PER_SEC == 20.0
    assert mv.RATE_LIMIT_COOLDOWN_SECONDS == 60.0

    monkeypatch.delenv("MASSIVE_MAX_PER_SEC")
    monkeypatch.delenv("MASSIVE_RATE_LIMIT_COOLDOWN_SECONDS")
    mv = importlib.reload(mv)                       # restore module-level state
    assert mv.DEFAULT_MAX_PER_SEC == 20.0


def test_twelvedata_construction_survives_malformed_batch_env(monkeypatch):
    from rh_agent.providers.twelvedata import TwelveDataProvider

    monkeypatch.setenv("TWELVEDATA_BATCH_SIZE", "oops")
    monkeypatch.setenv("TWELVEDATA_RATE_LIMIT_COOLDOWN_SECONDS", "")
    p = TwelveDataProvider("k")
    assert p._batch_size >= 1
    assert p.rate_limit_cooldown_seconds == pytest.approx(300.0)


def test_build_providers_survives_malformed_config_numbers(monkeypatch):
    from rh_agent.config import _KEY_ENV, load_config
    from rh_agent.providers import build_providers

    for provider, names in _KEY_ENV.items():
        for var in names:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TWELVEDATA_API_KEY", "k")
    monkeypatch.setenv("MASSIVE_API_KEY", "k")
    cfg = load_config()
    cfg.raw["providers"]["twelvedata_max_per_sec"] = "fast please"
    cfg.raw["providers"]["massive_max_per_sec"] = {"oops": True}
    providers = build_providers(cfg)                # must not raise
    assert set(providers) == {"twelvedata", "massive"}


def test_scoring_gates_survive_malformed_env(monkeypatch):
    from rh_agent.config import load_config
    from rh_agent.models import Verdict
    from rh_agent.scoring import Scorer

    monkeypatch.setenv("RH_MIN_CONVICTION", "")
    monkeypatch.setenv("RH_MIN_PILLARS", "two")
    scorer = Scorer(load_config())
    v = Verdict("AAA", 99.0, {}, pillars_passing=5)
    assert scorer.eligible([v]) == [v]              # falls back to config gates
