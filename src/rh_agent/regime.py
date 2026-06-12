"""Market-regime detection — the Chief PM's macro lens.

Reads broad-market trend (SPX vs 200dma via SPY), volatility (VIX), breadth
(equal- vs cap-weight, RSP/SPY) and the yield curve, then picks the regime
that decides how the five analysts are weighted and how much capital is deployed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .logging_setup import get_logger

log = get_logger("regime")


@dataclass
class RegimeResult:
    name: str
    weights: dict
    exposure: float
    signals: dict = field(default_factory=dict)

    def describe(self) -> str:
        s = self.signals
        bits = []
        if s.get("spx_above_200dma") is not None:
            bits.append(f"SPX {'uptrend' if s['spx_above_200dma'] else 'downtrend'}")
        if s.get("vix") is not None:
            bits.append(f"VIX {s['vix']:.1f}")
        if s.get("breadth") is not None:
            bits.append(f"breadth {s['breadth']:+.1%}")
        if s.get("spy_day_change_pct") is not None:
            bits.append(f"SPY today {s['spy_day_change_pct']:+.1f}%")
        if s.get("yield_curve_10_2") is not None:
            bits.append(f"10y-2y {s['yield_curve_10_2']:+.2f}")
        return f"{self.name} ({', '.join(bits)}) -> exposure {self.exposure:.0%}"


def _trend_above_200(md, symbol="SPY", lookback: int = 200) -> bool | None:
    df = md.get_index_prices(symbol)
    if df is None or len(df) < lookback:
        return None
    return float(df["close"].iloc[-1]) > float(df["close"].iloc[-lookback:].mean())


def _vix(md) -> float | None:
    for sym in ("VIX", "^VIX"):
        try:
            df = md.get_index_prices(sym)
            if df is not None and len(df):
                return float(df["close"].iloc[-1])
        except Exception:
            continue
    return md.get_macro().get("vix")


def _breadth(md) -> float | None:
    """RSP (equal weight) vs SPY (cap weight) 3-month relative return.
    Positive => broad participation (healthy)."""
    try:
        rsp, spy = md.get_index_prices("RSP"), md.get_index_prices("SPY")
        if rsp is None or spy is None or len(rsp) < 63 or len(spy) < 63:
            return None
        r = rsp["close"].iloc[-1] / rsp["close"].iloc[-63] - 1
        s = spy["close"].iloc[-1] / spy["close"].iloc[-63] - 1
        return float(r - s)
    except Exception:
        return None


def _intraday_tape_adjust(md, cfg: Config, name: str) -> tuple[str, float | None]:
    """Between daily bars, a violent SPY day move should de-risk NOW, not at the
    next close. Downgrades the regime on the live tape; never upgrades it.
    Graceful no-op when disabled or no live SPY quote is available."""
    icfg = cfg.get("regime.intraday", {}) or {}
    if not icfg.get("enabled", False):
        return name, None
    get_quote = getattr(md, "get_quote", None)
    if get_quote is None:
        return name, None
    try:
        q = get_quote("SPY")
        dc = float(q.day_change_pct) if q and q.day_change_pct is not None else None
    except Exception:
        return name, None
    if dc is None:
        return name, None
    risk_off_at = float(icfg.get("spy_drop_risk_off_pct", -2.5))
    neutral_at = float(icfg.get("spy_drop_neutral_pct", -1.5))
    if name != "high_volatility":      # already the most defensive weighting
        if dc <= risk_off_at and name in ("risk_on_trend", "neutral"):
            log.warning("intraday tape shock: SPY %+.2f%% today — downgrading %s -> risk_off",
                        dc, name)
            name = "risk_off"
        elif dc <= neutral_at and name == "risk_on_trend":
            log.info("intraday tape weakness: SPY %+.2f%% today — risk_on_trend -> neutral", dc)
            name = "neutral"
    return name, dc


def detect_regime(md, cfg: Config) -> RegimeResult:
    rc = cfg.get("regime", {})
    sig = rc.get("signals", {})
    weights = rc.get("weights", {})
    exposure = rc.get("exposure", {})

    above = _trend_above_200(md, "SPY", int(sig.get("spx_trend_lookback", 200)))
    vix = _vix(md)
    breadth = _breadth(md)
    macro = md.get_macro()

    signals = {"spx_above_200dma": above, "vix": vix, "breadth": breadth,
               "yield_curve_10_2": macro.get("yield_curve_10_2")}

    calm = sig.get("vix_calm_below", 16)
    stress = sig.get("vix_stress_above", 26)

    if vix is not None and vix >= stress:
        name = "high_volatility"
    elif above is False:
        name = "risk_off"
    elif above is True and (vix is None or vix <= calm) and (breadth is None or breadth > -0.03):
        name = "risk_on_trend"
    else:
        name = "neutral"

    name, spy_dc = _intraday_tape_adjust(md, cfg, name)
    if spy_dc is not None:
        signals["spy_day_change_pct"] = spy_dc

    res = RegimeResult(name=name,
                       weights=weights.get(name, weights.get("neutral", {})),
                       exposure=float(exposure.get(name, 0.85)),
                       signals=signals)
    log.info("regime: %s", res.describe())
    return res
