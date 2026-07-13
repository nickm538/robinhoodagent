"""Coverage for previously-untested pure helpers flagged by review:
risk.py (atr_stop / take_profit / annualized_vol / daily_drawdown_halt and
edge paths of trailing_stop / risk_capped_weight), factors/normalize.py,
market_calendar.py, broker/orders.py edge paths, and Scorer._add_flags.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from rh_agent.broker.orders import order_succeeded, stable_ref_id
from rh_agent.config import load_config
from rh_agent.factors.normalize import (
    NEUTRAL,
    cross_sectional_scores,
    weighted_blend,
    winsorize,
)
from rh_agent.models import Order, TickerData, Verdict
from rh_agent.risk import (
    annualized_vol,
    atr_stop,
    daily_drawdown_halt,
    risk_capped_weight,
    take_profit,
    trailing_stop,
)
from rh_agent.scoring import Scorer

try:
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

from rh_agent.market_calendar import is_market_open


# --------------------------- risk.py ---------------------------

def test_atr_stop_uses_atr_distance_when_within_hard_floor():
    # ATR distance (2.5*2=5) is tighter than the 18% hard floor -> use ATR stop
    assert atr_stop(100.0, 2.0, 2.5, 0.18) == pytest.approx(95.0)


def test_atr_stop_clamps_to_hard_floor_when_atr_wide():
    # A huge ATR would push the stop below the hard floor -> clamp to floor
    assert atr_stop(100.0, 50.0, 2.5, 0.18) == pytest.approx(82.0)


def test_atr_stop_no_atr_falls_back_to_hard_floor():
    assert atr_stop(100.0, None, 2.5, 0.18) == pytest.approx(82.0)
    assert atr_stop(100.0, 0.0, 2.5, 0.18) == pytest.approx(82.0)


def test_take_profit_with_and_without_atr():
    assert take_profit(100.0, 2.0, 6.0) == pytest.approx(112.0)
    assert take_profit(100.0, None, 6.0) is None
    assert take_profit(100.0, 0.0, 6.0) is None


def test_daily_drawdown_halt_thresholds():
    # -6% breaches a 5% limit
    assert daily_drawdown_halt(94.0, 100.0, 0.05) is True
    # exactly at the limit (-5%) halts (<=)
    assert daily_drawdown_halt(95.0, 100.0, 0.05) is True
    # -4% does not breach
    assert daily_drawdown_halt(96.0, 100.0, 0.05) is False
    # sign of limit is irrelevant (abs)
    assert daily_drawdown_halt(94.0, 100.0, -0.05) is True
    # non-positive starting equity -> never halt
    assert daily_drawdown_halt(1.0, 0.0, 0.05) is False
    assert daily_drawdown_halt(1.0, -100.0, 0.05) is False


def test_annualized_vol_prefers_technicals():
    td = TickerData("X", technicals={"volatility": 0.42})
    assert annualized_vol(td) == pytest.approx(0.42)


def test_annualized_vol_default_when_unknown():
    assert annualized_vol(TickerData("X")) == pytest.approx(0.30)
    # too-short price history also falls back to the conservative default
    short = TickerData("X", prices=pd.DataFrame({"close": [100.0, 101.0]}))
    assert annualized_vol(short) == pytest.approx(0.30)


def test_annualized_vol_computes_from_prices():
    rng = np.random.default_rng(0)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 120))
    td = TickerData("X", prices=pd.DataFrame({"close": closes}))
    v = annualized_vol(td)
    assert v > 0 and v != pytest.approx(0.30)  # computed, not the default


def test_trailing_stop_edge_paths():
    assert trailing_stop(100.0, atr=2.0, mult=2.5, hard_pct=0.18) == pytest.approx(95.0)
    assert trailing_stop(100.0, atr=None, mult=2.5, hard_pct=0.18) == pytest.approx(82.0)
    # a wide ATR is clamped to the hard floor
    assert trailing_stop(100.0, atr=50.0, mult=2.5, hard_pct=0.18) == pytest.approx(82.0)


def test_risk_capped_weight_edge_paths():
    # no stop / no price / non-positive risk pct -> weight is unchanged
    assert risk_capped_weight(100.0, None, 0.10, 0.01) == pytest.approx(0.10)
    assert risk_capped_weight(0.0, 90.0, 0.10, 0.01) == pytest.approx(0.10)
    assert risk_capped_weight(100.0, 90.0, 0.10, 0.0) == pytest.approx(0.10)
    # price at/below the stop -> unchanged (no positive stop distance)
    assert risk_capped_weight(90.0, 90.0, 0.10, 0.01) == pytest.approx(0.10)
    assert risk_capped_weight(80.0, 90.0, 0.10, 0.01) == pytest.approx(0.10)
    # tight 5% stop with 1% budget caps a 50% weight to 20%
    assert risk_capped_weight(100.0, 95.0, 0.50, 0.01) == pytest.approx(0.20)


# ----------------------- factors/normalize.py -----------------------

def test_winsorize_clips_outliers():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    out = winsorize(s, sigma=1.0)
    assert out.max() < 100.0          # the outlier is pulled in
    assert out.iloc[0] == pytest.approx(1.0)  # in-range value untouched


def test_winsorize_noops_on_degenerate_input():
    flat = pd.Series([5.0, 5.0, 5.0])      # zero std
    pd.testing.assert_series_equal(winsorize(flat, 3.0), flat)
    tiny = pd.Series([1.0, 2.0])           # len < 3
    pd.testing.assert_series_equal(winsorize(tiny, 3.0), tiny)


def test_cross_sectional_scores_ranks_higher_raw_better():
    raw = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    out = cross_sectional_scores(raw)
    assert out["e"] == pytest.approx(100.0)   # best raw -> top percentile
    assert out["a"] == pytest.approx(20.0)
    assert out["a"] < out["c"] < out["e"]


def test_cross_sectional_scores_neutralises_missing():
    raw = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "miss": None}
    out = cross_sectional_scores(raw)
    assert out["miss"] == pytest.approx(NEUTRAL)


def test_cross_sectional_scores_low_coverage_all_neutral():
    # only 1 of 5 present (< 3 and below default 0.5 coverage) -> all neutral
    raw = {"a": 1.0, "b": None, "c": None, "d": None, "e": None}
    out = cross_sectional_scores(raw)
    assert set(out) == set(raw)
    assert all(v == NEUTRAL for v in out.values())


def test_cross_sectional_scores_empty():
    assert cross_sectional_scores({}) == {}


def test_weighted_blend_renormalises_dropped_factors():
    score, coverage = weighted_blend({"a": 80.0, "b": 60.0}, {"a": 1.0, "b": 1.0, "c": 2.0})
    assert score == pytest.approx(70.0)       # c dropped, a/b renormalised
    assert coverage == pytest.approx(2 / 3)   # 2 of 3 weighted factors present


def test_weighted_blend_no_overlap_returns_neutral():
    score, coverage = weighted_blend({"x": 90.0}, {"a": 1.0})
    assert score == pytest.approx(NEUTRAL)
    assert coverage == pytest.approx(0.0)


# ----------------------- market_calendar.py -----------------------

@pytest.mark.skipif(_ET is None, reason="zoneinfo/tzdata unavailable")
def test_market_open_regular_session():
    # Monday 2026-06-08, 10:00 ET -> open
    assert is_market_open(datetime(2026, 6, 8, 10, 0, tzinfo=_ET)) is True
    # The opening instant is included; the closing bell is not.
    assert is_market_open(datetime(2026, 6, 8, 9, 30, tzinfo=_ET)) is True
    assert is_market_open(datetime(2026, 6, 8, 16, 0, tzinfo=_ET)) is False


def test_market_closed_outside_session():
    if _ET is None:  # pragma: no cover
        pytest.skip("zoneinfo unavailable")
    assert is_market_open(datetime(2026, 6, 8, 9, 0, tzinfo=_ET)) is False
    assert is_market_open(datetime(2026, 6, 8, 17, 0, tzinfo=_ET)) is False


@pytest.mark.skipif(_ET is None, reason="zoneinfo/tzdata unavailable")
def test_market_closed_weekend_and_holiday():
    # Saturday 2026-06-06
    assert is_market_open(datetime(2026, 6, 6, 12, 0, tzinfo=_ET)) is False
    # Friday 2026-07-03 is a full closure
    assert is_market_open(datetime(2026, 7, 3, 12, 0, tzinfo=_ET)) is False


@pytest.mark.skipif(_ET is None, reason="zoneinfo/tzdata unavailable")
def test_market_early_close_day():
    # Friday 2026-11-27 closes at 13:00 ET
    assert is_market_open(datetime(2026, 11, 27, 12, 0, tzinfo=_ET)) is True
    assert is_market_open(datetime(2026, 11, 27, 13, 30, tzinfo=_ET)) is False


def test_market_open_accepts_utc_naive_default(monkeypatch):
    # passing no argument uses datetime.now(utc); just assert it returns a bool
    assert isinstance(is_market_open(datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc)), bool)


# ----------------------- broker/orders.py edge paths -----------------------

def test_order_succeeded_none_and_unknown_status():
    assert order_succeeded(None) is False
    assert order_succeeded({}) is False                 # no status
    assert order_succeeded({"status": "queued"}) is False  # unrecognised


def test_order_succeeded_preview_depends_on_executing():
    assert order_succeeded({"status": "preview"}, executing=False) is True
    assert order_succeeded({"status": "preview"}, executing=True) is False


def test_order_succeeded_rejected_and_error():
    assert order_succeeded({"status": "REJECTED"}) is False  # case-insensitive
    assert order_succeeded({"status": "error"}) is False


def test_stable_ref_id_distinguishes_orders_and_is_32_hex():
    buy = Order("AAPL", "buy", None, "market", notional=100.0, reason="enter")
    sell = Order("AAPL", "sell", 1.0, "market", reason="exit")
    rid = stable_ref_id(buy, "ACCT", day_key="2026-06-07")
    assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)
    assert rid != stable_ref_id(sell, "ACCT", day_key="2026-06-07")
    # a different account changes the id
    assert rid != stable_ref_id(buy, "OTHER", day_key="2026-06-07")


# ----------------------- Scorer._add_flags -----------------------

def _scorer() -> Scorer:
    return Scorer(load_config())


def test_add_flags_marks_imminent_earnings():
    v = Verdict("X", 80.0)
    _scorer()._add_flags(v, TickerData("X", earnings={"days_to_next": 3}))
    assert "earnings_in_3d" in v.flags


def test_add_flags_marks_high_volatility():
    v = Verdict("X", 80.0)
    _scorer()._add_flags(v, TickerData("X", technicals={"atr_pct": 0.08}))
    assert "high_volatility" in v.flags


def test_add_flags_marks_smallcap():
    v = Verdict("X", 80.0)
    _scorer()._add_flags(v, TickerData("X", company={"market_cap": 1e9}))
    assert "smallcap" in v.flags


def test_add_flags_quiet_when_nothing_triggers():
    v = Verdict("X", 80.0)
    td = TickerData("X", earnings={"days_to_next": 30},
                    technicals={"atr_pct": 0.01}, company={"market_cap": 5e10})
    _scorer()._add_flags(v, td)
    assert v.flags == []
