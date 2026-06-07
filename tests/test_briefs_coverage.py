"""Coverage for the second batch of review briefs:

- MCP URL validation (the actionable kernel of the "SSRF via unvalidated client
  URL" finding): reject non-https remote endpoints before the OAuth token is sent.
- scoring.Scorer.score / Scorer.eligible edge paths
- analysts.panel.Panel.evaluate
- regime.detect_regime + RegimeResult.describe
- report.render_scan / write_markdown
- process_lock.daemon_lock
- risk.annualized_vol degenerate (flat-price) path
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from rh_agent.agent import ScanResult
from rh_agent.analysts.panel import Panel
from rh_agent.broker.mcp_client import MCPError, validate_mcp_url
from rh_agent.config import load_config
from rh_agent.factors.library import FACTORS
from rh_agent.models import TargetPosition, TickerData, Verdict
from rh_agent.regime import RegimeResult, detect_regime
from rh_agent.report import render_scan, write_markdown
from rh_agent.risk import annualized_vol
from rh_agent.scoring import Scorer


# ----------------------- MCP URL validation (security) -----------------------

def test_validate_mcp_url_allows_https_remote():
    url = "https://agent.robinhood.com/mcp"
    assert validate_mcp_url(url) == url
    assert validate_mcp_url("  https://host/mcp  ") == "https://host/mcp"


def test_validate_mcp_url_allows_http_loopback_only():
    for url in ("http://localhost:8765/mcp", "http://127.0.0.1/mcp", "http://[::1]:9/mcp"):
        assert validate_mcp_url(url) == url


def test_validate_mcp_url_rejects_http_remote():
    # plaintext to a remote host would leak the OAuth bearer token
    with pytest.raises(MCPError):
        validate_mcp_url("http://evil.example.com/mcp")


def test_validate_mcp_url_rejects_bad_scheme_or_empty():
    for bad in ("", "ftp://host/x", "file:///etc/passwd", "not a url", "https://"):
        with pytest.raises(MCPError):
            validate_mcp_url(bad)


# ----------------------------- scoring.Scorer -----------------------------

def _priced_td(ticker: str, slope: float, *, market_cap: float = 5e10,
               vol: float = 0.3) -> TickerData:
    n = 260
    closes = np.linspace(100.0, 100.0 + slope, n)
    df = pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "adj_close": closes, "volume": [1e6] * n},
        index=pd.date_range("2023-01-02", periods=n, freq="B"),
    )
    return TickerData(ticker, company={"sector": "Information Technology",
                                       "market_cap": market_cap},
                      prices=df, technicals={"volatility": vol})


def _risk_on_regime() -> RegimeResult:
    w = load_config().get("regime", {})["weights"]["risk_on_trend"]
    return RegimeResult("risk_on_trend", w, 1.0)


def test_scorer_score_ranks_and_flags():
    scorer = Scorer(load_config())
    data = [_priced_td("AAA", 90.0, market_cap=1e9),   # steepest momentum + smallcap
            _priced_td("BBB", 60.0),
            _priced_td("CCC", 30.0),
            _priced_td("DDD", 10.0)]
    verdicts = scorer.score(data, _risk_on_regime())

    assert {v.ticker for v in verdicts} == {"AAA", "BBB", "CCC", "DDD"}
    # sorted by composite, descending; steepest momentum leads
    comps = [v.composite for v in verdicts]
    assert comps == sorted(comps, reverse=True)
    assert verdicts[0].ticker == "AAA"
    # every verdict carries a score for all five personas
    assert all(set(v.analyst_scores) == set(load_config().get("analysts", {}))
               for v in verdicts)
    # _add_flags ran: the sub-$2B name is flagged smallcap
    aaa = next(v for v in verdicts if v.ticker == "AAA")
    assert "smallcap" in aaa.flags


def test_scorer_score_empty_input():
    assert Scorer(load_config()).score([], _risk_on_regime()) == []


def _eligible_cfg(monkeypatch):
    monkeypatch.delenv("RH_MIN_CONVICTION", raising=False)
    monkeypatch.delenv("RH_MIN_PILLARS", raising=False)
    cfg = load_config()
    cfg.raw["portfolio"]["min_conviction_score"] = 60.0
    cfg.raw["normalize"]["min_pillars_passing"] = 2
    rc = cfg.raw["portfolio"].setdefault("risk_controls", {})
    rc.update({"block_ai_caution": True, "block_high_volatility": False,
               "block_earnings_within_days": 2})
    return cfg


def test_eligible_conviction_and_pillar_floors(monkeypatch):
    scorer = Scorer(_eligible_cfg(monkeypatch))
    passing = Verdict("OK", 75.0, pillars_passing=3)
    low_conv = Verdict("LOWC", 40.0, pillars_passing=3)
    few_pillars = Verdict("FEW", 75.0, pillars_passing=1)
    out = {v.ticker for v in scorer.eligible([passing, low_conv, few_pillars])}
    assert out == {"OK"}


def test_eligible_blocks_high_vol_and_imminent_earnings(monkeypatch):
    cfg = _eligible_cfg(monkeypatch)
    cfg.raw["portfolio"]["risk_controls"]["block_high_volatility"] = True
    scorer = Scorer(cfg)
    hv = Verdict("HV", 80.0, pillars_passing=3, flags=["high_volatility"])
    earn = Verdict("ER", 80.0, pillars_passing=3, flags=["earnings_in_1d"])
    safe = Verdict("OK", 80.0, pillars_passing=3, flags=["earnings_in_9d"])
    out = {v.ticker for v in scorer.eligible([hv, earn, safe])}
    assert out == {"OK"}        # high-vol and earnings-in-1d are filtered


# --------------------------- analysts.panel ---------------------------

def test_panel_evaluate_full_coverage():
    panel = Panel(load_config())
    normalized = {f: {"X": 80.0} for f in FACTORS}
    presence = {f: {"X"} for f in FACTORS}
    v = panel.evaluate("X", normalized, presence, _risk_on_regime())
    assert v.ticker == "X"
    assert v.composite == pytest.approx(80.0, abs=0.5)
    assert set(v.analyst_scores) == set(load_config().get("analysts", {}))
    assert v.pillars_passing == 5      # all five personas clear the 55 threshold


def test_panel_evaluate_partial_presence_limits_pillars():
    panel = Panel(load_config())
    momentum_factors = load_config().get("analysts", {})["momentum_trader"]["factors"]
    normalized = {f: {"X": 90.0} for f in FACTORS}
    # X only has data for the momentum persona's factors
    presence = {f: ({"X"} if f in momentum_factors else set()) for f in FACTORS}
    v = panel.evaluate("X", normalized, presence, _risk_on_regime())
    assert v.pillars_passing == 1      # only momentum has coverage
    assert v.analyst_scores["momentum_trader"] == pytest.approx(90.0, abs=0.5)


# ------------------------------- regime -------------------------------

def _df(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": values},
                        index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


class _FakeMD:
    def __init__(self, series: dict, macro: dict | None = None):
        self._series = series
        self._macro = macro or {}

    def get_index_prices(self, symbol):
        return self._series.get(symbol)

    def get_macro(self):
        return self._macro


def test_detect_regime_risk_on_trend():
    rising = list(np.linspace(300.0, 420.0, 260))
    md = _FakeMD({"SPY": _df(rising), "RSP": _df(rising), "VIX": _df([12.0] * 5)})
    res = detect_regime(md, load_config())
    assert res.name == "risk_on_trend"
    assert res.exposure == pytest.approx(1.0)


def test_detect_regime_risk_off_when_below_trend():
    falling = list(np.linspace(420.0, 300.0, 260))
    md = _FakeMD({"SPY": _df(falling), "RSP": _df(falling), "VIX": _df([12.0] * 5)})
    assert detect_regime(md, load_config()).name == "risk_off"


def test_detect_regime_high_volatility_overrides_trend():
    rising = list(np.linspace(300.0, 420.0, 260))
    md = _FakeMD({"SPY": _df(rising), "RSP": _df(rising), "VIX": _df([30.0] * 5)})
    assert detect_regime(md, load_config()).name == "high_volatility"


def test_detect_regime_neutral_when_vix_elevated_but_not_stressed():
    rising = list(np.linspace(300.0, 420.0, 260))
    md = _FakeMD({"SPY": _df(rising), "RSP": _df(rising), "VIX": _df([20.0] * 5)})
    assert detect_regime(md, load_config()).name == "neutral"


def test_regime_describe_includes_exposure():
    r = RegimeResult("neutral", {}, 0.85,
                     signals={"spx_above_200dma": True, "vix": 18.0,
                              "breadth": 0.01, "yield_curve_10_2": 0.4})
    text = r.describe()
    assert "neutral" in text and "exposure 85%" in text and "VIX 18.0" in text


# ------------------------------- report -------------------------------

def _scan_with_targets() -> ScanResult:
    regime = RegimeResult("neutral", {}, 0.85, signals={"vix": 18.0})
    v = Verdict("AAA", 82.5, analyst_scores={"momentum_trader": 80.0, "quant": 70.0},
                pillars_passing=3, rationale="momentum=80 | quant=70", flags=["smallcap"])
    tp = TargetPosition("AAA", 0.12, 82.5, shares=5.0, dollars=600.0,
                        stop_price=90.0, take_profit=130.0, sector="Tech",
                        rationale="momentum=80")
    return ScanResult(regime=regime, verdicts=[v], eligible=[v], targets=[tp],
                      equity=10_000.0, universe_size=20, scored_size=8)


def test_write_markdown_creates_report(tmp_path):
    out = tmp_path / "scan.md"
    path = write_markdown(_scan_with_targets(), out)
    assert path == out and out.exists()
    text = out.read_text()
    assert "# rh-agent target book" in text
    assert "## Target portfolio" in text
    assert "AAA" in text
    assert "Analyst panel" in text
    assert "Not investment advice" in text


def test_write_markdown_empty_book(tmp_path):
    regime = RegimeResult("risk_off", {}, 0.5, signals={})
    scan = ScanResult(regime=regime, verdicts=[], eligible=[], targets=[],
                      equity=500.0, universe_size=5, scored_size=0)
    text = write_markdown(scan, tmp_path / "empty.md").read_text()
    assert "positions: **0**" in text


def test_render_scan_runs_for_both_branches(capsys):
    # with targets
    render_scan(_scan_with_targets())
    # empty book branch
    empty = ScanResult(regime=RegimeResult("neutral", {}, 0.85, signals={}),
                       verdicts=[], eligible=[], targets=[], equity=0.0,
                       universe_size=0, scored_size=0)
    render_scan(empty)
    out = capsys.readouterr().out
    assert "No positions clear" in out


# ----------------------------- process_lock -----------------------------

def test_daemon_lock_writes_pid_and_excludes_second_holder(tmp_path):
    from rh_agent.process_lock import ProcessLockError, daemon_lock

    p = tmp_path / "daemon.lock"
    with daemon_lock(p):
        assert p.exists()
        assert int(p.read_text()) == os.getpid()
        with pytest.raises(ProcessLockError):
            with daemon_lock(p):
                pass


def test_daemon_lock_reacquired_after_release(tmp_path):
    from rh_agent.process_lock import daemon_lock

    p = tmp_path / "daemon.lock"
    with daemon_lock(p):
        pass
    # lock released on exit -> a fresh acquisition succeeds
    with daemon_lock(p):
        assert p.exists()


# ------------------------------- risk -------------------------------

def test_annualized_vol_flat_prices_returns_default():
    # zero-variance history -> std is 0 -> conservative default, never 0
    flat = TickerData("X", prices=pd.DataFrame({"close": [100.0] * 80}))
    assert annualized_vol(flat) == pytest.approx(0.30)
