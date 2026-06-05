"""Two-stage scan funnel — light-screen the whole universe on fast signals,
then deep-score (and AI-review) only the top survivors. This keeps a wide hunt
tractable on small hardware. Fully offline: scan() is isolated and every heavy
collaborator is stubbed, so no providers, network, or market data are touched.
"""
from __future__ import annotations

import rh_agent.agent as agent_mod
from rh_agent.agent import TradingAgent
from rh_agent.config import load_config
from rh_agent.models import TickerData, Verdict
from rh_agent.regime import RegimeResult


def _num(t: str) -> int:
    digits = "".join(c for c in t if c.isdigit())
    return int(digits) if digits else 0


def _agent(monkeypatch, *, threshold: int, top_k: int, snapshot: bool):
    """An agent whose scan() collaborators are stubbed; records _gather calls."""
    cfg = load_config()
    cfg.raw.setdefault("universe", {})
    cfg.raw["universe"]["two_stage_threshold"] = threshold
    cfg.raw["universe"]["deep_top_k"] = top_k

    a = TradingAgent.__new__(TradingAgent)   # bypass __init__ -> no providers/network
    a.cfg = cfg
    a.md = None                              # detect_regime is stubbed; never dereferenced
    a.providers = {"snapshot": object()} if snapshot else {"financialdatasets": object()}
    a._quote_cache = {}

    monkeypatch.setattr(agent_mod, "detect_regime",
                        lambda md, cfg: RegimeResult("neutral", {}, 0.85))

    calls: list[tuple[bool, list[str]]] = []

    def fake_gather(names, deep=True):
        calls.append((deep, list(names)))
        return [TickerData(t) for t in names]
    a._gather = fake_gather

    class _Scorer:
        # rank by the numeric suffix (descending) so "top-K" is a non-trivial subset
        def score(self, data, regime):
            vs = [Verdict(td.ticker, composite=float(_num(td.ticker))) for td in data]
            vs.sort(key=lambda v: v.composite, reverse=True)
            return vs

        def eligible(self, verdicts):
            return []
    a.scorer = _Scorer()

    class _Builder:
        def build(self, eligible, td_map, regime, equity):
            return []
    a.builder = _Builder()

    class _AI:
        enabled = False          # overlay no-ops
    a.ai = _AI()
    return a, calls


def test_funnel_engages_above_threshold(monkeypatch):
    names = [f"T{i:02d}" for i in range(20)]
    a, calls = _agent(monkeypatch, threshold=5, top_k=3, snapshot=False)
    res = a.scan(equity=100_000, tickers=names)

    assert len(calls) == 2
    # light pass: cheap (deep=False) over the WHOLE universe
    assert calls[0][0] is False and set(calls[0][1]) == set(names)
    # deep pass: full (deep=True) over ONLY the top-K light survivors, in rank order
    assert calls[1] == (True, ["T19", "T18", "T17"])
    # universe_size reflects the full hunt; scored_size only the deep set
    assert res.universe_size == 20 and res.scored_size == 3


def test_funnel_skipped_under_threshold(monkeypatch):
    names = [f"T{i:02d}" for i in range(4)]
    a, calls = _agent(monkeypatch, threshold=5, top_k=3, snapshot=False)
    a.scan(equity=100_000, tickers=names)

    assert len(calls) == 1                       # straight to the deep pass
    assert calls[0][0] is True and set(calls[0][1]) == set(names)


def test_funnel_skipped_in_snapshot_mode(monkeypatch):
    names = [f"T{i:02d}" for i in range(20)]      # 20 > threshold 5, but snapshot is in-memory
    a, calls = _agent(monkeypatch, threshold=5, top_k=3, snapshot=True)
    a.scan(equity=100_000, tickers=names)

    assert len(calls) == 1                        # funnel bypassed for snapshots
    assert calls[0][0] is True and set(calls[0][1]) == set(names)
