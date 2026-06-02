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
