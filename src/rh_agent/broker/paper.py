"""PaperBroker — the safe default.

Fills orders at *live* market prices (supplied by a price callback) plus a
configurable slippage, and tracks a persistent simulated account on disk. No
brokerage is ever contacted. This is real-price paper trading, not a mock data
source: the prices are the same live quotes the live broker would transact on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..config import REPO_ROOT
from ..logging_setup import get_logger
from ..models import Account, Order, Position, utcnow
from .base import Broker

log = get_logger("broker.paper")

STATE_PATH = REPO_ROOT / "state" / "paper_account.json"


class PaperBroker(Broker):
    name = "paper"
    supports_live = False

    def __init__(self, price_fn: Callable[[str], float | None], *,
                 starting_cash: float = 100_000.0, slippage_bps: float = 5.0,
                 state_path: Path = STATE_PATH):
        self.price_fn = price_fn
        self.slippage = slippage_bps / 10_000.0
        self.state_path = Path(state_path)
        self.state = self._load(starting_cash)

    def _load(self, starting_cash: float) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        s = {"cash": starting_cash, "positions": {}, "history": [], "created": utcnow().isoformat()}
        self._save(s)
        return s

    def _save(self, s: dict | None = None) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(s or self.state, indent=2, default=str))

    def get_account(self) -> Account:
        positions = []
        equity = self.state["cash"]
        for t, pos in self.state["positions"].items():
            px = self.price_fn(t) or pos["avg_price"]
            mv = pos["qty"] * px
            equity += mv
            positions.append(Position(ticker=t, quantity=pos["qty"], avg_price=pos["avg_price"],
                                      current_price=px, market_value=round(mv, 2),
                                      unrealized_pnl=round((px - pos["avg_price"]) * pos["qty"], 2)))
        return Account(equity=round(equity, 2), cash=round(self.state["cash"], 2),
                       buying_power=round(self.state["cash"], 2), positions=positions,
                       account_number="PAPER", source="paper")

    def place_order(self, order: Order, dry_run: bool = True) -> dict:
        px = self.price_fn(order.ticker)
        if px is None:
            return {"status": "rejected", "reason": "no price", "order": order.to_dict()}
        qty = order.quantity
        if qty is None and order.notional:
            qty = order.notional / px
        if not qty or qty <= 0:
            return {"status": "rejected", "reason": "bad qty", "order": order.to_dict()}
        fill = px * (1 + self.slippage if order.side == "buy" else 1 - self.slippage)
        if dry_run:
            return {"status": "preview", "ticker": order.ticker, "side": order.side,
                    "qty": round(qty, 4), "est_fill": round(fill, 2)}

        pos = self.state["positions"].get(order.ticker, {"qty": 0.0, "avg_price": 0.0})
        if order.side == "buy":
            cost = qty * fill
            if cost > self.state["cash"] + 1e-6:
                qty = self.state["cash"] / fill
                cost = qty * fill
            new_qty = pos["qty"] + qty
            pos["avg_price"] = ((pos["avg_price"] * pos["qty"]) + cost) / new_qty if new_qty else fill
            pos["qty"] = new_qty
            self.state["cash"] -= cost
        else:  # sell
            qty = min(qty, pos["qty"])
            self.state["cash"] += qty * fill
            pos["qty"] -= qty
        if pos["qty"] <= 1e-6:
            self.state["positions"].pop(order.ticker, None)
        else:
            self.state["positions"][order.ticker] = pos
        rec = {"status": "filled", "ts": utcnow().isoformat(), "ticker": order.ticker,
               "side": order.side, "qty": round(qty, 4), "fill": round(fill, 2),
               "reason": order.reason}
        self.state["history"].append(rec)
        self._save()
        return rec
