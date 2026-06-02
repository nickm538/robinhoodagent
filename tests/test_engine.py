"""Unit tests for the core quant math. All synthetic, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rh_agent.factors import indicators as ind
from rh_agent.factors.library import _PLAUSIBLE, _f, mom_3_1
from rh_agent.factors.normalize import cross_sectional_scores, weighted_blend
from rh_agent.models import TickerData
from rh_agent.backtest import metrics


def _series(n=300, start=100, drift=0.001, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    px = start * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "adj_close": px, "volume": 1e6}, index=idx)


def test_cross_sectional_ranks_and_neutralises():
    raw = {"A": 1.0, "B": 2.0, "C": 3.0, "D": None}
    s = cross_sectional_scores(raw, min_coverage=0.5)
    assert s["C"] > s["B"] > s["A"]      # monotonic
    assert s["D"] == 50.0                 # missing -> neutral
    assert 0 <= min(s.values()) and max(s.values()) <= 100


def test_low_coverage_neutralises_all():
    raw = {"A": 1.0, "B": None, "C": None, "D": None}
    s = cross_sectional_scores(raw, min_coverage=0.5)
    assert all(v == 50.0 for v in s.values())


def test_weighted_blend_renormalises_missing():
    score, cov = weighted_blend({"x": 80, "y": 60}, {"x": 0.5, "y": 0.25, "z": 0.25})
    assert cov == pytest.approx(2 / 3)
    assert score == pytest.approx((80 * 0.5 + 60 * 0.25) / 0.75)


def test_indicators_reasonable():
    df = _series(seed=1)
    out = ind.compute_indicators(df)
    assert 0 <= out["rsi"] <= 100
    assert out["sma50"] is not None and out["sma200"] is not None
    assert out["atr"] > 0 and out["volatility"] > 0


def test_momentum_orientation():
    # deterministic trends (no noise) so the sign is unambiguous
    strong = TickerData("S", prices=_series(drift=0.003, vol=0.0, seed=2))
    weak = TickerData("W", prices=_series(drift=-0.001, vol=0.0, seed=3))
    assert mom_3_1(strong) > 0 > mom_3_1(weak)


def test_plausibility_clamp_rejects_bad_roe():
    good = TickerData("G", fundamentals={"roe": 0.30})
    bad = TickerData("B", fundamentals={"roe": 1.67})   # 167% -> implausible
    assert _f(good, "roe") == 0.30
    assert _f(bad, "roe") is None
    assert _PLAUSIBLE["roe"][1] < 1.67


def test_metrics_drawdown_and_sharpe():
    idx = pd.date_range("2020-01-01", periods=24, freq="ME")
    eq = pd.Series(np.linspace(100, 150, 24), index=idx)
    assert metrics.max_drawdown(eq) <= 0
    assert metrics.cagr(eq) > 0
    bench = pd.Series(np.linspace(100, 120, 24), index=idx)
    summ = metrics.summarize(eq, bench)
    assert summ["total_return"] > summ["benchmark_total_return"]
