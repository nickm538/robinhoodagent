from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from rh_agent.agent import TradingAgent
from rh_agent.analysts.ai_analyst import SYSTEM_PROMPT
from rh_agent.config import load_config
from rh_agent.models import Quote, TickerData, Verdict, utcnow
from rh_agent.scoring import Scorer


def _prices(days: int = 260, start: float = 100.0, end: float = 130.0) -> pd.DataFrame:
    close = np.linspace(start, end, days)
    return pd.DataFrame(
        {"close": close, "adj_close": close, "high": close * 1.01,
         "low": close * 0.99, "volume": [1_000_000] * days},
        index=pd.date_range(end=pd.Timestamp.now("UTC").normalize(), periods=days, freq="B"),
    )


def test_stale_quote_flag_blocks_new_eligibility():
    cfg = load_config()
    cfg.raw["portfolio"]["min_conviction_score"] = 50.0
    cfg.raw["normalize"]["min_pillars_passing"] = 1
    scorer = Scorer(cfg)
    td = TickerData(
        "STALE",
        quote=Quote("STALE", 100.0, asof=utcnow() - timedelta(minutes=10)),
        prices=_prices(),
        technicals={"volatility": 0.25, "price": 100.0},
    )
    verdict = Verdict("STALE", 80.0, {}, pillars_passing=5)

    scorer._add_flags(verdict, td)

    assert "stale_quote" in verdict.flags
    assert scorer.eligible([verdict]) == []


def test_ai_prompt_requires_uncertainty_and_causal_skepticism():
    prompt = SYSTEM_PROMPT.lower()

    assert "do not invent" in prompt
    assert "uncertainty" in prompt
    assert "causal" in prompt
    assert "options flow" in prompt
    assert "geopolitical" in prompt


def test_ai_context_helpers_include_history_market_and_data_quality():
    cfg = load_config()
    agent = TradingAgent.__new__(TradingAgent)
    agent.cfg = cfg

    class _MD:
        def get_index_prices(self, symbol):
            return _prices(start=100.0, end=120.0)

    agent.md = _MD()
    td = TickerData(
        "AWARE",
        quote=Quote("AWARE", 130.0, volume=2_000_000, day_change_pct=3.2),
        prices=_prices(),
        options={"put_call_ratio": 0.7},
        short_interest={"short_pct_float": 0.08},
        institutional={"net_change_pct": 0.02},
        news_sentiment={"score": 0.3},
    )

    data_quality = agent._data_quality_context(td)
    pattern = agent._historical_pattern_context(td)
    relationship = agent._market_relationship_context(td)

    assert data_quality["has_options"] is True
    assert data_quality["has_short_interest"] is True
    assert "return_21d" in pattern
    assert "realized_vol_63d" in pattern
    assert "spy_correlation" in relationship
    assert relationship["note"].startswith("Correlation is descriptive")
