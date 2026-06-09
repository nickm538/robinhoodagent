"""Hunter-mode reactivity: stale-quote fix, snipe boost, rotation, entry band."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from rh_agent.agent import ScanResult, TradingAgent
from rh_agent.config import load_config
from rh_agent.data.universe import build_universe
from rh_agent.execution import build_orders
from rh_agent.models import Account, Position, Quote, TargetPosition, TickerData, Verdict, utcnow
from rh_agent.regime import RegimeResult
from rh_agent.scoring import Scorer


def _prices(price: float = 50.0, days: int = 220) -> pd.DataFrame:
    close = np.linspace(price * 0.9, price, days)
    return pd.DataFrame(
        {"close": close, "adj_close": close, "high": close * 1.01,
         "low": close * 0.99, "volume": [500_000] * days},
        index=pd.date_range("2025-01-01", periods=days, freq="B"),
    )


def test_fresh_quotes_after_gather_clear_stale_eligibility():
    cfg = load_config()
    cfg.raw["data"]["freshness"]["quote_max_age_seconds"] = 180
    scorer = Scorer(cfg)

    td = TickerData("RUN", quote=Quote("RUN", 100.0, asof=utcnow() - timedelta(minutes=12)),
                    prices=_prices())
    verdict = Verdict("RUN", 80.0, {}, pillars_passing=5)
    scorer._add_flags(verdict, td)
    assert "stale_quote" in verdict.flags
    assert scorer.eligible([verdict]) == []

    td.quote = Quote("RUN", 100.0, asof=utcnow())
    verdict.flags = []
    scorer._add_flags(verdict, td)
    assert "stale_quote" not in verdict.flags
    assert scorer.eligible([verdict])


def test_entry_no_trade_band_allows_immediate_new_buys():
    cfg = load_config()
    cfg.raw["portfolio"]["rebalance"]["no_trade_band"] = 0.05
    cfg.raw["portfolio"]["rebalance"]["entry_no_trade_band"] = 0.0
    acct = Account(equity=500.0, cash=400.0, buying_power=400.0, positions=[
        Position("AAPL", 0.5, 200.0, 210.0),
    ])
    targets = [TargetPosition(ticker="SNIPE", weight=0.04, score=70.0, sector="Tech")]
    orders = build_orders(acct, targets, cfg, lambda t: 50.0 if t == "SNIPE" else 210.0)
    buys = [o for o in orders if o.side == "buy"]
    assert any(o.ticker == "SNIPE" for o in buys)


def test_intraday_prescreen_prefers_runners_over_slow_momentum():
    cfg = load_config()
    cfg.raw["universe"]["prescreen"] = {"enabled": True, "max_candidates": 1}
    cfg.raw["universe"]["intraday"]["enabled"] = True
    cfg.raw["universe"]["intraday"]["fallback_to_liquid_universe"] = False
    cfg.raw["universe"]["intraday"].update({
        "min_day_change_pct": 0.0,
        "min_positive_day_change_pct": 0.0,
        "min_relative_volume": 0.0,
        "min_dollar_volume_today": 0.0,
        "min_breakout_pct": -1.0,
    })
    cfg.raw["universe"]["liquidity"].update({
        "min_market_cap": 0, "min_avg_dollar_volume": 0,
        "min_avg_volume_shares": 0, "min_history_days": 60,
    })

    class _MD:
        def list_universe(self):
            return ["SLOW", "RUN"]

        def prefetch_quotes(self, tickers):
            return 0

        def clear_quote_prefetch(self):
            pass

        def get_quote(self, t):
            return Quote(t, 10.0, volume=500_000, day_change_pct=4.0 if t == "RUN" else 0.2)

        def get_quote_for_risk(self, t, max_age_seconds=180):
            return self.get_quote(t)

        def get_prices(self, t):
            if t == "RUN":
                close = np.full(220, 10.0)
                return pd.DataFrame(
                    {"close": close, "adj_close": close, "high": close * 1.01,
                     "low": close * 0.99, "volume": [500_000] * 220},
                    index=pd.date_range("2025-01-01", periods=220, freq="B"),
                )
            return _prices(30.0)

        def get_company(self, t):
            return {"market_cap": 1e9, "sector": "Tech"}

    out = build_universe(_MD(), cfg, raw=["SLOW", "RUN"])
    assert out[0] == "RUN"


def test_rotation_sells_stale_hold_for_stronger_runner():
    cfg = load_config()
    cfg.raw["hunter"] = {"rotation_enabled": True, "rotation_score_margin": 5.0}
    agent = TradingAgent.__new__(TradingAgent)
    agent.cfg = cfg

    held_v = Verdict("OLD", 55.0, {}, pillars_passing=2)
    scan = ScanResult(
        regime=RegimeResult("neutral", {}, 0.85),
        verdicts=[held_v],
        eligible=[],
        targets=[TargetPosition(ticker="NEW", weight=0.25, score=65.0, sector="Tech")],
        equity=500.0,
        universe_size=2,
        scored_size=2,
    )
    acct = Account(equity=500.0, cash=100.0, buying_power=100.0, positions=[
        Position("OLD", 1.0, 50.0, 52.0),
    ])
    assert agent._should_rotate_for_runner("OLD", held_v, scan, acct)


def test_intraday_hunter_boost_lifts_day_runners():
    cfg = load_config()
    cfg.raw["universe"]["intraday"]["enabled"] = True
    cfg.raw["universe"]["intraday"]["composite_boost_max"] = 12.0
    agent = TradingAgent.__new__(TradingAgent)
    agent.cfg = cfg

    td = TickerData("RUN", quote=Quote("RUN", 20.0, volume=2_000_000, day_change_pct=5.0),
                    prices=_prices(20.0))
    v = Verdict("RUN", 50.0, {}, pillars_passing=1)
    agent._apply_intraday_hunter_boost([v], {"RUN": td})
    assert v.composite > 50.0
