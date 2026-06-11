"""Risk helpers: volatility, ATR stops, and exposure guards."""
from __future__ import annotations

from .models import TickerData


def annualized_vol(td: TickerData) -> float:
    v = td.technicals.get("volatility")
    if v and v > 0:
        return float(v)
    # fallback from prices
    if td.prices is not None and len(td.prices) > 20:
        r = td.prices["close"].pct_change().dropna()
        if len(r) > 5:
            return float(r.iloc[-63:].std() * (252 ** 0.5)) or 0.30
    return 0.30  # conservative default when unknown


def atr_stop(price: float, atr: float | None, mult: float, hard_pct: float) -> float:
    hard = price * (1 - hard_pct)
    if atr and atr > 0:
        return round(max(price - mult * atr, hard), 2)
    return round(hard, 2)


def take_profit(price: float, atr: float | None, mult: float) -> float | None:
    if atr and atr > 0:
        return round(price + mult * atr, 2)
    return None


def daily_drawdown_halt(equity: float, day_start_equity: float, limit: float) -> bool:
    if day_start_equity <= 0:
        return False
    return (equity / day_start_equity - 1.0) <= -abs(limit)


def trailing_stop(high_water: float, atr: float | None, mult: float, hard_pct: float) -> float:
    """Ratchet a trailing stop upward from the high-water mark using ATR distance."""
    hard = high_water * (1 - hard_pct)
    if atr and atr > 0:
        return round(max(high_water - mult * atr, hard), 2)
    return round(hard, 2)


def breakeven_stop(avg_price: float | None, high_water: float, atr: float | None,
                   after_atr_mult: float, buffer_pct: float = 0.0) -> float | None:
    """Once a position has run >= after_atr_mult×ATR above entry, floor its stop at
    entry (+small buffer) so a confirmed winner can never round-trip into a loss.
    Returns None (no floor) until the trigger distance is reached or inputs are
    unusable; callers only ever ratchet stops UP with this value."""
    if (not avg_price or avg_price <= 0 or not atr or atr <= 0
            or not after_atr_mult or after_atr_mult <= 0):
        return None
    if high_water < avg_price + after_atr_mult * atr:
        return None
    return round(avg_price * (1.0 + max(buffer_pct, 0.0)), 2)


def risk_capped_weight(price: float, stop_price: float | None, equity_weight: float,
                       per_trade_risk_pct: float) -> float:
    """Cap target weight so loss at stop is <= per_trade_risk_pct of equity."""
    if not stop_price or not price or price <= stop_price or per_trade_risk_pct <= 0:
        return equity_weight
    stop_dist = (price - stop_price) / price
    if stop_dist <= 0:
        return equity_weight
    max_w = per_trade_risk_pct / stop_dist
    return min(equity_weight, max_w)
