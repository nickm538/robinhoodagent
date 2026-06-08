"""Core data structures shared across the pipeline.

These are deliberately plain dataclasses / dicts so that any provider can
populate them and any consumer (factors, risk, broker) can read them without
tight coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Quote:
    ticker: str
    price: float
    volume: float = 0.0
    prev_close: Optional[float] = None
    day_change_pct: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    asof: datetime = field(default_factory=utcnow)
    source: str = ""


@dataclass
class TickerData:
    """Everything the engine knows about one symbol at scan time.

    Each section is filled by whichever provider answers first. Missing data
    is left as ``None``/empty — factors then neutralise that factor rather
    than inventing a value.
    """

    ticker: str
    company: dict = field(default_factory=dict)         # name, sector, industry, market_cap
    quote: Optional[Quote] = None
    prices: Optional[pd.DataFrame] = None               # cols: open high low close adj_close volume
    fundamentals: dict = field(default_factory=dict)    # ROE, margins, growth, ratios...
    technicals: dict = field(default_factory=dict)      # rsi, macd, adx, atr, obv, sma...
    insider: list = field(default_factory=list)         # list of insider transactions
    institutional: dict = field(default_factory=dict)   # ownership + recent change
    news_sentiment: dict = field(default_factory=dict)  # score, article_count, label
    analyst: dict = field(default_factory=dict)         # consensus, price target, buy ratio
    short_interest: dict = field(default_factory=dict)  # pct float, days to cover
    options: dict = field(default_factory=dict)         # put/call ratio, iv rank
    earnings: dict = field(default_factory=dict)        # surprises, estimate revisions, next date
    pro_scores: dict = field(default_factory=dict)      # zacks, morningstar, danelfin, tipranks
    meta: dict = field(default_factory=dict)            # captured_at, per-section sources

    @property
    def price(self) -> Optional[float]:
        if self.quote and self.quote.price:
            return self.quote.price
        if self.prices is not None and len(self.prices):
            return float(self.prices["close"].iloc[-1])
        return None

    @property
    def market_cap(self) -> Optional[float]:
        return self.company.get("market_cap") or self.fundamentals.get("market_cap")

    @property
    def sector(self) -> str:
        return self.company.get("sector") or "Unknown"


@dataclass
class Verdict:
    """The Chief PM's blended decision for a ticker."""

    ticker: str
    composite: float                  # 0..100
    analyst_scores: dict = field(default_factory=dict)   # name -> score
    pillars_passing: int = 0
    rationale: str = ""
    flags: list = field(default_factory=list)            # e.g. ["earnings_in_3d"]


# ----------------------------- broker side -----------------------------

@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    current_price: float = 0.0


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    positions: list = field(default_factory=list)        # list[Position]
    account_number: str = ""
    source: str = ""
    portfolio_confirmed: bool = True   # False when portfolio/equity fetch failed
    positions_confirmed: bool = True     # False when positions fetch failed
    buying_power_confirmed: bool = True  # False when BP could not be parsed

    def position_map(self) -> dict:
        return {p.ticker: p for p in self.positions}

    @property
    def reliable(self) -> bool:
        """Live cycles must not reconcile unless every account field is confirmed."""
        if self.source != "robinhood":
            return True
        return (
            self.portfolio_confirmed
            and self.positions_confirmed
            and self.buying_power_confirmed
            and bool(self.equity and self.equity > 0)
        )


@dataclass
class Order:
    ticker: str
    side: str                         # "buy" | "sell"
    quantity: Optional[float] = None  # shares; None when using notional
    order_type: str = "market"        # market | limit
    limit_price: Optional[float] = None
    time_in_force: str = "gfd"
    notional: Optional[float] = None  # dollar amount alternative to quantity
    reason: str = ""

    def to_dict(self) -> dict:
        d = {
            "ticker": self.ticker,
            "side": self.side,
            "quantity": round(self.quantity, 6) if self.quantity is not None else None,
            "order_type": self.order_type,
            "time_in_force": self.time_in_force,
        }
        if self.limit_price is not None:
            d["limit_price"] = round(self.limit_price, 2)
        if self.notional is not None:
            d["notional"] = round(self.notional, 2)
        if self.reason:
            d["reason"] = self.reason
        return d


@dataclass
class TargetPosition:
    ticker: str
    weight: float                     # target fraction of equity
    score: float
    shares: float = 0.0
    dollars: float = 0.0
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    sector: str = "Unknown"
    rationale: str = ""
