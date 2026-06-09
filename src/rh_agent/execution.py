"""Reconcile the current account against target positions and emit the minimal
set of orders, honouring a no-trade band (to cut churn) and a turnover cap.
"""
from __future__ import annotations

from typing import Callable

from .config import Config
from .logging_setup import get_logger
from .models import Account, Order, TargetPosition

log = get_logger("execution")


def build_orders(account: Account, targets: list[TargetPosition], cfg: Config,
                 price_fn: Callable[[str], float | None], *,
                 allow_buys: bool = True,
                 exclude_tickers: set[str] | None = None) -> list[Order]:
    equity = account.equity or 0.0
    if equity <= 0:
        return []
    reb = cfg.get("portfolio.rebalance", {})
    band = float(reb.get("no_trade_band", 0.015))
    entry_band = float(reb.get("entry_no_trade_band", 0.0))
    max_turn = float(reb.get("max_turnover_per_rebalance", 0.30))
    min_notional = float(reb.get("min_order_notional", 20.0))

    exclude = exclude_tickers or set()
    cur = account.position_map()
    tmap = {t.ticker: t for t in targets}
    orders: list[Order] = []

    # --- exits & trims (sells) ---
    for tk, pos in cur.items():
        if tk in exclude:
            continue
        px = price_fn(tk) or pos.current_price or pos.avg_price
        cur_w = (pos.quantity * px) / equity if px else 0.0
        tgt_w = tmap[tk].weight if tk in tmap else 0.0
        if tk not in tmap:
            orders.append(Order(tk, "sell", pos.quantity, reason="exit: dropped from target"))
        elif (cur_w - tgt_w) > band:
            sell_dollars = (cur_w - tgt_w) * equity
            if sell_dollars >= min_notional and px:
                orders.append(Order(tk, "sell", round(sell_dollars / px, 4),
                                    reason=f"trim {cur_w:.1%}->{tgt_w:.1%}"))

    # --- entries & adds (buys) ---
    buys: list[Order] = []
    for tk, t in tmap.items():
        if tk in exclude:
            continue
        if not allow_buys:
            continue
        px = price_fn(tk)
        if not px:
            continue
        cur_w = (cur[tk].quantity * px) / equity if tk in cur else 0.0
        if (t.weight - cur_w) <= band:
            continue
        buy_dollars = (t.weight - cur_w) * equity
        if buy_dollars < min_notional:
            continue
        buys.append(Order(tk, "buy", round(buy_dollars / px, 4), notional=round(buy_dollars, 2),
                          reason=f"{'enter' if tk not in cur else 'add'} score={t.score}"))

    # --- turnover cap: scale buys down if needed (sells always allowed) ---
    buy_turnover = sum((o.notional or 0) for o in buys) / equity
    if buy_turnover > max_turn:
        scale = max_turn / buy_turnover
        for o in buys:
            o.notional = round((o.notional or 0) * scale, 2)
            px = price_fn(o.ticker) or 0
            o.quantity = round(o.notional / px, 4) if px else o.quantity
        log.info("turnover cap hit: scaled buys by %.2f", scale)
        # scaling can push small orders under the minimum — drop those (-> cash)
        buys = [o for o in buys if (o.notional or 0) >= min_notional]

    # --- buying-power cap: never order more cash than the account actually has,
    # or the broker rejects it ("Not enough buying power"). Cushion for the
    # slippage between our quote and the market fill. ---
    bp = float(account.buying_power or 0.0)
    if bp > 0:
        budget = bp * 0.97
        buy_total = sum((o.notional or 0) for o in buys)
        if buy_total > budget:
            scale = (budget / buy_total) if buy_total else 0.0
            for o in buys:
                o.notional = round((o.notional or 0) * scale, 2)
                px = price_fn(o.ticker) or 0
                o.quantity = round(o.notional / px, 4) if px else o.quantity
            log.info("buying-power cap: $%.0f available -> scaled buys by %.2f", bp, scale)
            buys = [o for o in buys if (o.notional or 0) >= min_notional]

    orders.extend(buys)
    log.info("orders: %d (%d sells, %d buys)", len(orders),
             sum(1 for o in orders if o.side == "sell"), len(buys))
    return orders
