"""Factor library.

Each factor is ``f(td) -> float | None`` returning a value *oriented so that
higher is better* (so cross-sectional ranking is uniform). ``None`` means the
data was unavailable and the factor neutralises itself for that name.

Factor names here MUST match the keys under ``analysts.*.factors`` in
config.yaml — that mapping is what assigns factors to the five personas.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..models import TickerData


def _close(td: TickerData) -> Optional[pd.Series]:
    if td.prices is None or "close" not in td.prices or len(td.prices) < 30:
        return None
    return td.prices["adj_close"] if "adj_close" in td.prices else td.prices["close"]


def _ret(c: pd.Series, lag_recent: int, lag_far: int) -> Optional[float]:
    if len(c) <= lag_far:
        return None
    a, b = c.iloc[-lag_recent - 1], c.iloc[-lag_far - 1]
    if b == 0 or pd.isna(a) or pd.isna(b):
        return None
    return float(a / b - 1.0)


# ------------------------------------------------------------------ momentum
def mom_12_1(td):
    c = _close(td)
    return _ret(c, 21, 252) if c is not None else None

def mom_6_1(td):
    c = _close(td)
    return _ret(c, 21, 126) if c is not None else None

def mom_3_1(td):
    c = _close(td)
    return _ret(c, 21, 63) if c is not None else None


def risk_adj_momentum(td):
    c = _close(td)
    if c is None:
        return None
    m = _ret(c, 21, 126)
    vol = td.technicals.get("volatility")
    if m is None or not vol:
        return None
    return float(m / vol)


def trend_sma(td):
    t = td.technicals
    px, s50, s200 = t.get("price"), t.get("sma50"), t.get("sma200")
    if not (px and s200):
        return None
    score = px / s200 - 1.0
    if s50:
        score += 0.05 if s50 > s200 else -0.05
    return float(score)


def dist_52w_high(td):
    t = td.technicals
    px, hi = t.get("price"), t.get("high_52w") or td.fundamentals.get("high_52w")
    if not (px and hi):
        return None
    return float(min(px / hi, 1.02))


# --------------------------------------------------------------- fundamentals
# Data hygiene: reject implausible values (bad share counts / stale feeds
# produce e.g. 1600% ROE). Out-of-range -> treated as missing, not trusted.
_PLAUSIBLE = {
    "roe": (-1.0, 1.5), "roic": (-1.0, 1.5), "roa": (-1.0, 1.0),
    "gross_margin": (-0.5, 1.0), "operating_margin": (-2.0, 1.0), "net_margin": (-2.0, 1.0),
    "revenue_growth": (-0.95, 5.0), "earnings_growth": (-5.0, 10.0), "eps_growth": (-5.0, 10.0),
    "fcf_yield": (-0.5, 0.5), "fcf_growth": (-5.0, 10.0),
    "current_ratio": (0.0, 50.0), "debt_to_equity": (0.0, 20.0),
    "interest_coverage": (-50.0, 2000.0),
}


def _f(td, key):
    v = td.fundamentals.get(key)
    if not (isinstance(v, (int, float)) and np.isfinite(v)):
        return None
    lo, hi = _PLAUSIBLE.get(key, (-1e18, 1e18))
    if v < lo or v > hi:
        return None
    return float(v)


def roe(td):            return _f(td, "roe")
def roic(td):           return _f(td, "roic")
def gross_margin(td):   return _f(td, "gross_margin")
def net_margin(td):     return _f(td, "net_margin")
def revenue_growth(td): return _f(td, "revenue_growth")
def earnings_growth(td):return _f(td, "earnings_growth")
def fcf_yield(td):      return _f(td, "fcf_yield")


def balance_sheet_strength(td):
    cr, de, ic = _f(td, "current_ratio"), _f(td, "debt_to_equity"), _f(td, "interest_coverage")
    if cr is None and de is None and ic is None:
        return None
    score = 0.0
    score += (cr if cr is not None else 1.0)
    score -= 0.5 * min(de, 5.0) if de is not None else 0.0
    score += 0.05 * min(ic, 20.0) if ic is not None else 0.0
    return float(score)


# ------------------------------------------------------------------ catalyst
def eps_estimate_revision(td):     return _e(td, "eps_revision_ratio")
def revenue_estimate_revision(td): return _e(td, "rev_revision_ratio")
def earnings_surprise_history(td):
    v = _e(td, "avg_surprise_pct")
    if v is not None:
        return v
    br = _e(td, "beat_rate")
    return (br - 0.5) * 20 if br is not None else None


def _e(td, key):
    v = td.earnings.get(key)
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def upcoming_catalyst(td):
    d = td.earnings.get("days_to_next")
    if d is None:
        return None
    d = float(d)
    if d < 2:
        return 0.2           # imminent print = event risk
    if 5 <= d <= 35:
        return 1.0           # catalyst within horizon
    return 0.5


# --------------------------------------------------------------- smart money
def insider_net_buying(td):
    if not td.insider:
        return None
    mcap = td.market_cap or 0
    net = 0.0
    for tr in td.insider:
        val = tr.get("value")
        if val is None and tr.get("shares") and td.price:
            val = abs(tr["shares"]) * td.price
        if val is None:
            continue
        net += abs(val) if tr.get("is_buy") else -abs(val)
    return float(net / mcap) if mcap else float(np.sign(net))


def institutional_net_buying(td):
    v = td.institutional.get("net_change_pct")
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def short_squeeze_setup(td):
    si = td.short_interest
    spf, dtc = si.get("short_pct_float"), si.get("days_to_cover")
    if spf is None and dtc is None:
        return None
    c = _close(td)
    mom = _ret(c, 21, 63) if c is not None else 0.0
    base = (spf or 0) * (min(dtc or 0, 10) / 10.0 + 0.3)
    return float(base * (1.0 if (mom or 0) > 0 else 0.3))


# ------------------------------------------------------------------ sentiment
def news_sentiment(td):
    v = td.news_sentiment.get("score")
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def options_positioning(td):
    pcr = td.options.get("put_call_ratio")
    if pcr is None or pcr <= 0:
        return None
    return float(-pcr)        # lower put/call (more call demand) ranks higher


def zacks_rank(td):
    r = td.pro_scores.get("zacks_rank")
    return float(6 - r) if isinstance(r, (int, float)) else None   # 1(best)->5, 5(worst)->1


def analyst_consensus_upside(td):
    a, px = td.analyst, td.price
    tgt = a.get("target_mean")
    up = (tgt / px - 1.0) if (tgt and px) else None
    buy, hold, sell = a.get("buy"), a.get("hold"), a.get("sell")
    ratio = None
    if any(isinstance(x, (int, float)) for x in (buy, hold, sell)):
        tot = (buy or 0) + (hold or 0) + (sell or 0)
        ratio = (buy or 0) / tot if tot else None
    if up is None and ratio is None:
        return None
    return float((up or 0) + 0.3 * (ratio or 0))


def danelfin_ai(td):
    v = td.pro_scores.get("danelfin_ai")
    return float(v) if isinstance(v, (int, float)) else None


def morningstar(td):
    v = td.pro_scores.get("morningstar_stars")
    return float(v) if isinstance(v, (int, float)) else None


FACTORS: dict[str, Callable[[TickerData], Optional[float]]] = {
    "mom_12_1": mom_12_1, "mom_6_1": mom_6_1, "mom_3_1": mom_3_1,
    "risk_adj_momentum": risk_adj_momentum, "trend_sma": trend_sma,
    "dist_52w_high": dist_52w_high,
    "roe": roe, "roic": roic, "gross_margin": gross_margin, "net_margin": net_margin,
    "revenue_growth": revenue_growth, "earnings_growth": earnings_growth,
    "fcf_yield": fcf_yield, "balance_sheet_strength": balance_sheet_strength,
    "eps_estimate_revision": eps_estimate_revision,
    "revenue_estimate_revision": revenue_estimate_revision,
    "earnings_surprise_history": earnings_surprise_history,
    "upcoming_catalyst": upcoming_catalyst,
    "insider_net_buying": insider_net_buying,
    "institutional_net_buying": institutional_net_buying,
    "short_squeeze_setup": short_squeeze_setup,
    "news_sentiment": news_sentiment, "options_positioning": options_positioning,
    "zacks_rank": zacks_rank, "analyst_consensus_upside": analyst_consensus_upside,
    "danelfin_ai": danelfin_ai, "morningstar": morningstar,
}


def compute_raw_factors(td: TickerData) -> dict[str, float]:
    out = {}
    for name, fn in FACTORS.items():
        try:
            v = fn(td)
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            out[name] = float(v)
    return out
